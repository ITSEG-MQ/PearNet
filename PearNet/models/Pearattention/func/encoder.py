from torch import nn
from models.Pearattention.func.clones import clones
from models.Pearattention.func.layer_norm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, feature_dim, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.feature_dim = feature_dim
        self.norm = LayerNorm(feature_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


