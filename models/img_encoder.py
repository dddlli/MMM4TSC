import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(input_dim, d_state, d_conv, expand)
        self.mamba2 = Mamba(input_dim, d_state, d_conv, expand)

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x_mamba = self.mamba(x_norm)
        x_mamba = self.mamba2(x_mamba)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MMMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

        self.mamba2 = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        alpha = torch.sigmoid(x_norm.mean(dim=-1, keepdim=True)) * self.skip_scale

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba2(self.mamba(x1)) + alpha * x1
        x_mamba2 = self.mamba2(self.mamba(x2)) + alpha * x2
        x_mamba3 = self.mamba2(self.mamba(x3)) + alpha * x3
        x_mamba4 = self.mamba2(self.mamba(x4)) + alpha * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class IMG_Encoder(nn.Module):

    def __init__(self, encoder_dim, input_channels=1, c_list=[8, 16, 24]):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        self.encoder3 = nn.Sequential(
            MMMLayer(input_dim=c_list[1], output_dim=c_list[2])
        )

        self.encoder4 = nn.Sequential(
            MMMLayer(input_dim=c_list[2], output_dim=c_list[1])
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[1])

        self.fc_layer = nn.Linear(64, encoder_dim)

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))

        # 首先，将张量展平成一维
        flattened_x = torch.flatten(out, start_dim=2)

        # 计算展平后张量的总长度
        total_length = flattened_x.size(2)
        batch_size = flattened_x.size(0)

        # 然后，将展平后的张量 reshape 成你想要的形状
        desired_shape = (batch_size, 16, total_length)

        reshaped_x = flattened_x.view(desired_shape)

        x = self.fc_layer(reshaped_x)

        return x


if __name__ == '__main__':
    x = torch.randn(4, 1, 128, 128).to('cuda')

    model = IMG_Encoder(encoder_dim=345)

    print("LKA parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model = model.to('cuda')

    out = model(x)

    print(out.shape)
