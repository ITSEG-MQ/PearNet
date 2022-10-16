from torch import nn
import torch.nn.functional as F


"""
Figure 1 last module: Recognition
"""


class Classifier(nn.Module):
    def __init__(self, node_size, feature_dim, n_class):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(node_size*feature_dim, feature_dim*4),
            nn.ReLU(),
            nn.Linear(feature_dim*4, n_class)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)



