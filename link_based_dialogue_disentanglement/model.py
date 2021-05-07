import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, max_len, dropout = 0.1):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight[-x.shape[1]:].unsqueeze(0)
        x = x + weight
        return self.dropout(x)

class CrossEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.linear = nn.Linear(config.hidden_size, 1)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
        self.pos_enc = LearnedPositionEncoding(config.hidden_size, kwargs['max_num_contexts'])

    def forward(self, text_input_ids, text_input_masks, type_ids, labels=None):
        batch_size, neg, dim = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(-1, dim)
        text_input_masks = text_input_masks.reshape(-1, dim)
        type_ids = type_ids.reshape(-1, dim)
        text_vec = self.bert(text_input_ids, text_input_masks, type_ids)[0][:,0,:]  # [bs*neg,dim]
        text_vec = text_vec.reshape(batch_size, neg, -1)
        text_vec = self.pos_enc(text_vec)
        text_vec = self.transformer(text_vec)#[0]
        score = self.linear(text_vec).squeeze(-1)
        if labels is not None:
            loss = -(F.log_softmax(score, -1)*labels).sum(-1).mean()
            return loss
        else:
            return score.squeeze()
