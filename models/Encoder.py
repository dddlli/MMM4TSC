import torch
import torch.nn as nn
import torch.nn.modules.transformer as trans

from torch.nn.modules.normalization import LayerNorm


class Encoder(nn.Module):
    def __init__(self, input_size: int = 0, output_size: int = 0,
                 nhead: int = 1, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        if input_size == 0 or output_size == 0:
            raise ValueError("Requiring input_size and output_size.")
        super().__init__()
        encoder_layer = trans.TransformerEncoderLayer(input_size, nhead, dim_feedforward, dropout)
        encoder_norm = LayerNorm(input_size, eps=1e-5)
        self.encoder = trans.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.ffn = nn.Linear(input_size, output_size, bias=True)

        self.input_size = input_size

    def forward(self, input):
        if input.size(-1) != self.input_size:
            raise RuntimeError("the feature number of input must be equal to input_size")
        input = self.encoder(input)
        return self.ffn(input)


if __name__ == '__main__':
    x = torch.randn(4, 1, 570).to('cuda')
    model = Encoder(input_size=570, output_size=4).to('cuda')
    y = model(x)
    print("LKA parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
