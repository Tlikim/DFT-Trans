import torch.nn as nn
import torch
from .dft_trans import DFT_Trans
from torch.nn.init import trunc_normal_
import math
from transformers import AlbertPreTrainedModel, AlbertConfig
class BertForQuestionAnswering(AlbertPreTrainedModel):

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.model=DFT_Trans(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                start_positions=None,
                end_positions=None):
        """
        :param input_ids: [src_len,batch_size]
        :param attention_mask: [batch_size,src_len]
        :param token_type_ids: [src_len,batch_size]
        :param position_ids:
        :param start_positions: [batch_size]
        :param end_positions:  [batch_size]
        :return:
        """
        sequence_output, _ = self.model(
            x=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        # sequence_output: [batch_size, src_len, hidden_size]
        logits = self.qa_outputs(sequence_output)  # [batch_size, src_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        # [src_len,batch_size,1]  [src_len,batch_size,1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

