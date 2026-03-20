import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from lib.FEWTD import *
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.up(x)



class FEWTD_Decoder(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(FEWTD_Decoder,self).__init__()

        self.up1 = up_conv(512,320)
        self.up2 = up_conv(320,128)
        self.up3 = up_conv(128, 64)

        self.DB_conv1 = DoubleConv(in_channels=320,out_channels=320)
        self.DB_conv2 = DoubleConv(in_channels=128, out_channels=128)
        self.DB_conv3 = DoubleConv(in_channels=64, out_channels=64)

        self.fewtd1 = FEWTD(128,"HAAR")
        self.fewtd2 = FEWTD(64,"HAAR")
        self.fewtd3 = FEWTD(320, "HAAR")
        self.fewtd4 = FEWTD(512, "HAAR")

    def forward(self, x, skips):

        d4 = self.fewtd4(x)
        d3 = self.up1(d4)

        d3 = self.DB_conv1(d3 + skips[0])
        d3 = self.fewtd3(d3)

        d2 = self.up2(d3)
        d2 = self.DB_conv2(d2+skips[1])
        d2 = self.fewtd1(d2)

        d1 = self.up3(d2)
        d1 = self.DB_conv3(d1+skips[2])
        d1 = self.fewtd2(d1)

        return [d4, d3, d2, d1]
