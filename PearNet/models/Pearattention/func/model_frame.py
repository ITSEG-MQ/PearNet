from torch import nn
from models.Pearattention.func.vif import myvif


class PearsonGATFrame(nn.Module):
    def __init__(self, node_generator, encoder, classifier):
        super(PearsonGATFrame, self).__init__()
        self.node_generator = node_generator
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.node_generator(x)
        vif = myvif(x)
        return self.classifier(self.encoder(x)), vif
