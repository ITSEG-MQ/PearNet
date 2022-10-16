import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

########################################################################################
"""
Figure 1: Node Generation
"""
########################################################################################


"""
Spatial Convolution
"""


class ResLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ResLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBasicBlock(nn.Module):
    "Residual Squeeze-and-Excitation(SE) Block"
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.reslayer = ResLayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.reslayer(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Node2Vec(nn.Module):
    "Figure 2: Spatial Convolutional Network"
    def __init__(self, afr_reduced_cnn_size, dropout):
        super(Node2Vec, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.dropout = nn.Dropout(dropout)
        self.inplanes = 128
        self.AFR = self._make_layer(ResBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features1(x)
        x = self.dropout(x)
        x = self.AFR(x)
        return x

########################################################################################


"""
Temporal Convolution
"""


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    "Figure 3: Dilated Causal Convolution Block"
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x))

########################################################################################


"""
Node Generation
"""


class NodeGenerator(nn.Module):
    "Spatial Convolution + Temporal Convolution"
    def __init__(self, num_inputs, num_channels, meta, kernel_size=2, dropout=0.2):
        super(NodeGenerator, self).__init__()
        self.num_inputs = num_inputs
        self.meta = meta  # meta represents the number of base segments
        self.model = []
        self.num_hidden = len(num_channels)
        "The length of num_channels represents the level of feature extraction"
        for i in range(self.num_hidden):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.node2vec = Node2Vec(30, dropout)
            self.model.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size, dropout=dropout))

        self.model_list = nn.ModuleList([*self.model])

    def forward(self, x):
        node_meta = []
        node_all = []
        x = x.view(x.size()[0], self.meta, -1)
        for idx in range(self.meta):
            data = x[:, idx, :]
            data = data.unsqueeze(1)
            out = self.node2vec(data)    # Spatial Convolution Features
            out = out.view(x.size()[0], 1, -1)
            node_meta.append(out)
        node_meta = torch.cat(node_meta, dim=1).permute(0, 2, 1)

        for model in self.model_list:
            output = model(node_meta)
            node_all.append(output)
        node_all = torch.cat(node_all, dim=-1)    # Temporal Convolution Features
        node = torch.cat((node_meta, node_all), dim=-1).permute(0, 2, 1).contiguous()    # Spatial-Temporal Graph Nodes
        node = node.unsqueeze(1)
        node = nn.BatchNorm2d(node.size()[1]).cuda()(node)
        return node


