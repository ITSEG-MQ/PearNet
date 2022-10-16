from torch import nn
from models.Pearattention.func.layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    LayerNorm + sublayer(Self-Attention/Dense) + Dropout + Residual
    """

    def __init__(self, feature_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
