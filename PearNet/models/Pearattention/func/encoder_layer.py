from torch import nn
from models.Pearattention.func.clones import clones
from models.Pearattention.func.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, feature_dim, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(feature_dim, dropout), 2)
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.sublayer[0](x, self.self_attn)
        return self.sublayer[1](x, self.feed_forward)


