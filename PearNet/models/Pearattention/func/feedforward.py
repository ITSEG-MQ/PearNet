from torch import nn


class FeedForward(nn.Module):
    def __init__(self, feature_dim, dropout=0.5):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(feature_dim, feature_dim*4)
        self.w_2 = nn.Linear(feature_dim*4, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.ReLU()(self.w_1(x))))



