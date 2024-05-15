import torch
from torch import nn, Tensor
# from zeta.nn.modules.mlp import MLP
# from zeta.nn.modules.rms_norm import RMSNorm
# from zeta.nn.modules.visual_expert import VisualExpert

from .ts_encoder import TSEncoder
from .img_encoder import IMG_Encoder, MMMLayer
from .kan import KANLinear

import matplotlib.pyplot as plt
import seaborn as sns


class TSMultiModalMamba(nn.Module):
    """
    TSMultiModalMamba is a PyTorch module that combines text and image embeddings using a multimodal fusion approach.

    Args:
        dim (int): The dimension of the embeddings.
        depth (int): The depth of the Mamba block.
        dropout (float): The dropout rate.
        heads (int): The number of attention heads.
        d_state (int): The dimension of the state in the Mamba block.
        image_size (int): The size of the input image.
        patch_size (int): The size of the image patches.
        encoder_dim (int): The dimension of the encoder embeddings.
        encoder_depth (int): The depth of the encoder.
        encoder_heads (int): The number of attention heads in the encoder.
        fusion_method (str): The multimodal fusion method to use. Can be one of ["mlp", "concat", "add"].

    Examples:
    x = torch.randn(1, 16, 64)
    y = torch.randn(1, 3, 64, 64)
    model = MultiModalMambaBlock(
        dim = 64,
        depth = 5,
        dropout = 0.1,
        heads = 4,
        d_state = 16,
        image_size = 64,
        patch_size = 16,
        encoder_dim = 64,
        encoder_depth = 5,
        encoder_heads = 4
    )
    out = model(x, y)
    print(out.shape)

    """

    def __init__(
            self,
            dim: int,
            depth: int,
            dropout: float,
            heads: int,
            d_state: int,
            image_size: int,
            patch_size: int,
            encoder_depth: int,
            encoder_heads: int,
            n_classes: int,
            fusion_method: str = "mlp",
            *args,
            **kwargs,
    ):
        super(TSMultiModalMamba, self).__init__()
        self.dim = dim
        self.depth = depth
        self.dropout = dropout
        self.heads = heads
        self.d_state = d_state
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.fusion_method = fusion_method

        self.ts_encoder = TSEncoder()
        self.img_encoder = IMG_Encoder(encoder_dim=dim)

        # VisualExpert
        # self.visual_expert = VisualExpert(
        #     dim, self.hidden_dim, dropout, heads
        # )

        # MLP
        # self.mlp = MLP(
        #     dim, dim, expansion_factor=4, depth=1, norm=True
        # )

        self.mamba1 = nn.Sequential(MMMLayer(input_dim=32, output_dim=32))

        self.conv1 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.silu1 = nn.SiLU()

        self.mamba2 = nn.Sequential(MMMLayer(input_dim=16, output_dim=16))

        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.silu2 = nn.SiLU()

        # self.head = nn.Linear(8 * (dim // 4), n_classes)

        self.head = KANLinear(8 * (dim // 4), n_classes)


    def forward(self, ts: Tensor, img: Tensor) -> Tensor:
        ts = ts.unsqueeze(1)

        encoded_ts = self.ts_encoder(ts)
        encoded_img = self.img_encoder(img)

        # if self.fusion_method == "mlp":
        #     fusion_layer = self.mlp(encoded_img)
        #     fused = fusion_layer

        if self.fusion_method == "concat":
            fused = torch.concat([encoded_ts, encoded_img], dim=1)

        # if self.fusion_method == "add":
        #     fused = encoded_img + encoded_ts
        #
        # if self.fusion_method == "visual_expert":
        #     concat = torch.cat([encoded_ts, encoded_img], dim=1)
        #     fused = self.visual_expert(concat)

        x = self.mamba1(fused)

        x = self.silu1(self.pool1(self.conv1(x)))

        x = self.mamba2(x)

        x = self.silu2(self.pool2(self.conv2(x)))

        # 展平特征并应用全连接层
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x

    def check_fusion_method(self):
        print("""[mlp] [visualexpert] [projection] [concat] [add] """)
        print(f"""Current fusion method: {self.fusion_method}""")
