import torch
import torch.nn as nn
from .img_encoder import MMMLayer


class TSEncoder(nn.Module):
    def __init__(self):
        super(TSEncoder, self).__init__()
        self.embed_layer = nn.Sequential(nn.Conv2d(1, 16 * 4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(16 * 4),
                                         nn.GELU())

        # self.mamba = MMMLayer(input_dim=16 * 4, output_dim=16 * 4)

        self.embed_layer1 = nn.Sequential(nn.Conv2d(16 * 4, 16 * 2, kernel_size=[1, 8], padding='same'),
                                          nn.BatchNorm2d(16 * 2),
                                          nn.GELU())

        # self.mamba1 = MMMLayer(input_dim=16 * 2, output_dim=16 * 2)

        self.embed_layer2 = nn.Sequential(nn.Conv2d(16 * 2, 16, kernel_size=[1, 1], padding='valid'),
                                          nn.BatchNorm2d(16),
                                          nn.GELU())

        # self.mamba2 = MMMLayer(input_dim=16, output_dim=16)

    def forward(self, ts):
        x = ts.unsqueeze(1)
        x_src = self.embed_layer(x)
        # x_src = self.mamba(x_src)
        x_src = self.embed_layer1(x_src)
        # x_src = self.mamba1(x_src)
        encoded_ts = self.embed_layer2(x_src).squeeze(2)
        # encoded_ts = self.mamba2(encoded_ts)
        return encoded_ts


if __name__ == '__main__':
    x = torch.randn(4, 1, 570).to('cuda')
    model = TSEncoder().to('cuda')
    y = model(x)
    print("LKA parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

