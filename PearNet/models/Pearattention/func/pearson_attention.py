import torch
import torch.nn.functional as F
from torch import nn
from models.Pearattention.func.vif import mycorr
from torch.nn.parameter import Parameter

"""
Figure 1: Graph Attention Module
"""


class PearsonAttention(nn.Module):
    def __init__(self, feature_dim, node_size, dropout=0.5):
        super(PearsonAttention, self).__init__()
        self.feature_dim = feature_dim
        self.node_size = node_size
        self.dropout = nn.Dropout(dropout)
        self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.beta = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True)
        self.A = nn.Sequential(nn.Linear(self.node_size, self.node_size*4, bias=False),
                               nn.ReLU(),
                               nn.Linear(self.node_size*4, self.node_size, bias=False))
        self.relu = nn.ReLU()

    def forward(self, h):
        Wh = self.W_h(h)
        corr_p = mycorr(Wh)
        e = self.beta * torch.absolute(corr_p)
        "Adaptive Graph Structure Learning"
        adj = e + self.A(e)
        adj = self.relu(adj)
        zero_vec = -1e12 * torch.ones_like(e)
        "Construct the connections for the nodes whose values greater than 0."
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return h_prime






