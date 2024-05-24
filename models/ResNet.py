import torch.nn as nn
from .img_encoder import MMMLayer


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TSEncoder(nn.Module):
    def __init__(self, dim, n_classes):
        super(TSEncoder, self).__init__()
        self.embed_layer = nn.Sequential(nn.Conv2d(1, 16 * 2, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(16 * 2),
                                         nn.GELU())

        self.mamba = MMMLayer(input_dim=16 * 2, output_dim=16 * 2)

        self.embed_layer3 = nn.Sequential(nn.Conv2d(16 * 2, 16 * 4, kernel_size=[1, 8], padding='same'),
                                          nn.BatchNorm2d(16 * 4),
                                          nn.GELU())

        self.mamba3 = MMMLayer(input_dim=16 * 4, output_dim=16 * 4)

        self.embed_layer1 = nn.Sequential(nn.Conv2d(16 * 4, 16 * 2, kernel_size=[1, 8], padding='same'),
                                          nn.BatchNorm2d(16 * 2),
                                          nn.GELU())

        self.mamba1 = MMMLayer(input_dim=16 * 2, output_dim=16 * 2)

        self.embed_layer2 = nn.Sequential(nn.Conv2d(16 * 2, 16 * 2, kernel_size=[1, 1], padding='valid'),
                                          nn.BatchNorm2d(32),
                                          nn.GELU())

        self.mamba2 = MMMLayer(input_dim=32, output_dim=32)

        self.conv1 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gelu2 = nn.GELU()

        self.head = nn.Linear(8 * (dim // 4), n_classes)

    def forward(self, ts):
        ts = ts.unsqueeze(1)
        x = ts.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.mamba(x_src)
        x_src = self.embed_layer3(x_src)
        x_src = self.mamba3(x_src)
        x_src = self.embed_layer1(x_src)
        x_src = self.mamba1(x_src)
        encoded_ts = self.embed_layer2(x_src).squeeze(2)
        encoded_ts = self.mamba2(encoded_ts)
        x = self.gelu1(self.pool1(self.conv1(encoded_ts)))

        x = self.gelu2(self.pool2(self.conv2(x)))

        # 展平特征并应用全连接层
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x