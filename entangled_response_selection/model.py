import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, max_len, dropout = 0.1):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.unsqueeze(0)
        x = x + weight
        return self.dropout(x)

class Encoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.aux_weight = kwargs['aux_weight']
        self.max_num_contexts = kwargs['max_num_contexts']
        self.linear = nn.Linear(config.hidden_size, 1)
        self.linear2 = nn.Linear(config.hidden_size, 1)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
        self.pos_enc = LearnedPositionEncoding(config.hidden_size, self.max_num_contexts+1)

    def forward(self, text_input_ids, text_input_masks, text_input_segments, labels=None):
        batch_size, neg, seq_len, sent_len = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(-1, sent_len)
        text_input_masks = text_input_masks.reshape(-1, sent_len)
        text_input_segments = text_input_segments.reshape(-1, sent_len)
        text_vec = self.bert(text_input_ids, text_input_masks, text_input_segments)[0][:,0,:]  # [bs,dim]
        dim = text_vec.shape[-1]
        text_vec = text_vec.reshape(-1, seq_len, dim)
        text_vec = self.pos_enc(text_vec)
        text_vec = self.transformer(text_vec)#[0]
        score = self.linear(text_vec)
        # use a self attention to promote linking
        # mask debug
        '''
        mask = [0]*seq_len
        mask[-1] = float('-inf')
        mask = torch.tensor(mask).unsqueeze(-1).to(score.device)
        score = score + mask
        '''
        attn_weights = F.softmax(score, -2)
        score = self.linear2((attn_weights*text_vec).sum(1))
        score = score.view(batch_size, neg)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(score, labels.float())
            aux_loss = (attn_weights.squeeze()[:,-1]-(1-labels.reshape(-1)))**2
            aux_loss = aux_loss.mean()
            return loss + self.aux_weight*aux_loss
        else:
            return score, (torch.argmax(attn_weights.squeeze(), 1) == seq_len-1).float()
