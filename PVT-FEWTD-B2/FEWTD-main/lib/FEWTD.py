import torch
import torch.nn as nn
import torch.nn.functional as F
from FSA import FSA
from IWT_DWT import DWT_2D, IDWT_2D
from MDWA import MDWA


class FEWTD(nn.Module):
    """
    Frequency-Enhanced Wavelet Transformer Decoder (FEWTD)
    Paper: Frequency-enhanced wavelet transformer based decoder for medical image segmentation
    """

    def __init__(self, in_channels, wavelet_type='haar', levels=2):
        super(FEWTD, self).__init__()
        self.levels = levels
        self.dwt = DWT_2D(wavelet_type)
        self.idwt = IDWT_2D(wavelet_type)

        # 对应论文中的 FSA 模块
        self.fsa = FSA(in_channels, in_channels)

        # 对应论文中的 MDWA 模块
        self.mdwa_l1 = MDWA(in_channels)
        self.mdwa_l2 = MDWA(in_channels)

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape  # 记录原始输入的 H 和 W (例如 7x7)

        # 1. 第一级分解 (Level 1)
        x_l1 = self.dwt(x)
        ll1, lh1, hl1, hh1 = x_l1.split(C, 1)
        ll1, lh1, hl1, hh1 = self.mdwa_l1(ll1, lh1, hl1, hh1)

        # 2. 第二级分解 (Level 2)
        x_l2 = self.dwt(ll1)
        ll2, lh2, hl2, hh2 = x_l2.split(C, 1)
        ll2, lh2, hl2, hh2 = self.mdwa_l2(ll2, lh2, hl2, hh2)

        # 3. 逆变换与融合 (Reconstruction)
        f2 = self.idwt(torch.cat([ll2, lh2, hl2, hh2], dim=1))


        if f2.shape[2:] != ll1.shape[2:]:
            f2 = f2[:, :, :ll1.size(2), :ll1.size(3)]
        ll1 = ll1 + f2
        f1 = self.idwt(torch.cat([ll1, lh1, hl1, hh1], dim=1))

        # 4. FSA
        f1 = self.fsa(f1)

        if f1.shape[2:] != (H, W):
            f1 = f1[:, :, :H, :W]

        # 5. 残差连接与平滑
        out = self.smooth_conv(identity) + f1
        return out