import torch
import torch.nn as nn

class ThreeFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(ThreeFeatureFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        return self.fusion(x)