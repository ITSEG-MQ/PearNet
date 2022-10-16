from torch import nn
from models.Pearattention.func.node_generator import NodeGenerator
from models.Pearattention.func.multi_head_attention import MultiHeadAttention
from models.Pearattention.func.feedforward import FeedForward
from models.Pearattention.func.encoder_layer import EncoderLayer
from models.Pearattention.func.encoder import Encoder
from models.Pearattention.func.classifier import Classifier
from models.Pearattention.func.model_frame import PearsonGATFrame
import copy

'''
PearNet: A Pearson Correlation-based Graph Attention Network for Sleep Stage Recognition
'''


class PearsonGAT(nn.Module):
    def __init__(self, feature_dim, num_channels, meta, node_size, n, n_heads, n_class, dropout):
        super(PearsonGAT, self).__init__()

        c = copy.deepcopy

        node_generator = NodeGenerator(num_inputs=feature_dim, num_channels=num_channels, meta=meta, dropout=dropout)
        attn = MultiHeadAttention(feature_dim=feature_dim, node_size=node_size,
                                  n_heads=n_heads, dropout=dropout)
        ff = FeedForward(feature_dim, dropout)
        self.model = PearsonGATFrame(
                node_generator,
                Encoder(EncoderLayer(feature_dim, c(attn), c(ff), dropout), feature_dim, n),
                Classifier(node_size, feature_dim, n_class)
        )

    def forward(self, x):
        res, vif = self.model(x)
        return res, vif



