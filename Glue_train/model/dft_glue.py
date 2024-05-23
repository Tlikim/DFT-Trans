import torch.nn as nn
import torch
from .dft_trans import DFT_Trans
from torch.nn.init import trunc_normal_
import math
from torch.nn import CrossEntropyLoss
from transformers import AlbertPreTrainedModel, AlbertConfig, BertConfig, AutoModelForCausalLM, BertPreTrainedModel
class BertForSequenceClassification(BertPreTrainedModel):# AlbertPreTrainedModel
    def __init__(self, config: AlbertConfig): # AlbertConfig
        super().__init__(config)
        self.num_labels=config.num_labels
        self.model=DFT_Trans(config)
        # self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).bert
        # self.pooler = Pooler(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.model(x=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,)
        # sequence_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
            
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
