import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch.nn.init import trunc_normal_
import math
import numpy as np
from transformers.activations import ACT2FN
from transformers.models.albert.modeling_albert import AlbertAttention, AlbertEmbeddings
from transformers import AlbertPreTrainedModel, AlbertConfig

xops = None

class DFTPooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states.mean(dim=1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.mlp(x)


def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start

def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False




class Fourier_filter(nn.Module):
    def __init__(self, dim, sparsity_threshold=0.01):
        super().__init__()
        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.dim, self.dim))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.dim))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.dim, self.dim))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.dim))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.dim, self.dim))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.dim))
    def forward(self, x: torch.Tensor):
        x = x.float()
        B, N, C = x.shape
        assert C == self.dim
        x = torch.fft.fft(x, dim=1, norm="ortho")  # [bs, N, dim]
        original = x.clone().requires_grad_().to("cuda")
        x_high = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x.real, self.w1[0]) - \
            torch.einsum("bli,ii->bli", x.imag, self.w1[1]) + \
            self.b1[0]
        )
        x_low = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x.real, self.w1[1]) + \
            torch.einsum("bli,ii->bli", x.imag, self.w1[0]) + \
            self.b1[1]
        )
        y = torch.stack([x_high, x_low], dim=-1)# [bs, N, dim, 2]
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        # *************
        x_high1 = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x_high, self.w2[0]) - \
            torch.einsum("bli,ii->bli", x_low, self.w2[1]) + \
            self.b2[0]
        )
        x_low1 = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x_high, self.w2[1]) + \
            torch.einsum("bli,ii->bli", x_low, self.w2[0]) + \
            self.b2[1]
        )
        x = torch.stack([x_high1, x_low1], dim=-1)# [bs, 4, sql//8+1, num_block, block_size, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        x_high2 = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x_high1, self.w3[0]) - \
            torch.einsum("bli,ii->bli", x_low1, self.w3[1]) + \
            self.b3[0]
        )
        x_low2 = ACT2FN["gelu"](
            torch.einsum("bli,ii->bli", x_low1, self.w3[0]) + \
            torch.einsum("bli,ii->bli", x_high1, self.w3[1]) + \
            self.b3[1]
        )

        z = torch.stack([x_high2, x_low2], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z+x
        z = torch.view_as_complex(z)

        z = z + original
        x = torch.fft.ifft(z, dim=1, norm="ortho").real

        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out + 1e-8))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out + 1e-8))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, x):
        x_i = self.fc1(x)
        x_i = self.act1(x_i)
        x_i = self.fc2(x_i)
        x_i = self.drop(x_i)
        x = self.norm(x_i + x)
        # torch.cuda.empty_cache()
        return x

class ClassMix(nn.Module):
    def __init__(self,config: AlbertConfig):
        super().__init__()
        self.fourier = Fourier_filter(config.hidden_size)
        self.attn = AlbertAttention(config)
        self.mlp = PVT2FFN(config.hidden_size, config.intermediate_size)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        fourier_outputs = self.fourier(hidden_states)
        attention_output = self.attn(fourier_outputs,attention_mask)[0]
        attention_output = self.mlp(attention_output)
        return attention_output

class Block(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.attn=AlbertAttention(config)
        self.mlp = PVT2FFN(in_features=config.hidden_size, hidden_features=config.intermediate_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out + 1e-8))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out + 1e-8))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, x, attention_mask):
        x = self.attn(x, attention_mask)[0]
        x = self.mlp(x)
        # torch.cuda.empty_cache()
        return x

class DFT_Trans(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.config = config
        self.bert = AlbertEmbeddings(config)
        self.embedding_hidden_mapping_in=nn.Linear(config.embedding_size, config.hidden_size)
        # weights = self.bert.word_embeddings.weight
        self.mix_block = nn.ModuleList([ClassMix(config) for _ in range(config.num_hidden_layers-10)])
        self.block = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers-6)])

        self.pooler = DFTPooler(config.hidden_size)
        # classification head, CRF
        # self.head = nn.Linear(embed_dims,
        #                       num_classes) if num_classes > 0 else nn.Identity()
        # self.crf = CRF(self.num_classes, batch_first=True)
        self.post_init()

    def forward_features(self, x, attention_mask):
        for blk in self.mix_block:
            x = blk(x, attention_mask)
        for blk in self.block:
            x = blk(x, attention_mask)
        return x

    def forward(self, x,  # [bs, sql, hidden_size]
                attention_mask=None,  # [bs, sql, hidden_size]
                token_type_ids=None,  # [bs, sql, hidden_size]
                position_ids=None,
                # masked_lm_labels=None,  # [bs, sql]
                ):  # [bs, sql]
        batch_size, seq_length = x.shape
        # mask = copy.deepcopy(attention_mask)
        x = self.bert(
            input_ids=x,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )  # [bs, sql, hidden_size]
        x = self.embedding_hidden_mapping_in(x)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=x.device
            )
        extend_mask_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extend_mask_attention_mask = extend_mask_attention_mask.to(torch.float32)
        extend_mask_attention_mask = (1.0 - extend_mask_attention_mask) * torch.finfo(torch.float32).min
        sequence_output = self.forward_features(x, extend_mask_attention_mask)  # [B, N, C]
        pooler_output = self.pooler(sequence_output)
        # token_pred_logits = self.head(sequence_output)  # [B, N, vocab_size]
        torch.cuda.empty_cache()
        return (sequence_output, pooler_output)  # [bs, sql, num_class], [bs, num_class]
