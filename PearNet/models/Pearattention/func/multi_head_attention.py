import torch
from torch import nn
from models.Pearattention.func.pearson_attention import PearsonAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, node_size, n_heads, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.attentions = [PearsonAttention(feature_dim=feature_dim, node_size=node_size, dropout=dropout)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.fc = nn.Linear(feature_dim*n_heads, feature_dim, bias=False)

    def forward(self, x):
        multi_head = torch.cat([att(x) for att in self.attentions], dim=-1)
        multi_head = self.fc(multi_head)
        return multi_head






