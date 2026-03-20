import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fusion(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fusion = FeatureFusion(in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        y = F.adaptive_avg_pool2d(x, 1).view(b, c, 1, 1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y


class FSA(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FSA, self).__init__()
        self.groups = groups
        # 创建一个1x1卷积层，输入通道数和输出通道数都翻倍是因为要同时处理实部和虚部
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=self.groups,
                                          bias=False)
        # 批归一化层，通道数是输出通道数的两倍（实部+虚部）
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # geLU激活函数
        self.gelu = torch.nn.GELU()
        self.se_real = SEBlock(in_channels)
        self.se_imag = SEBlock(in_channels)
        self.sigma = nn.Parameter(torch.tensor(0.2))
    def forward(self, x):
        batch, c, h, w = x.size()  # 获取输入张量的尺寸

        # 进行二维实数傅里叶变换，得到复数频谱
        # rfft2表示实数输入的快速傅里叶变换，norm='ortho'表示正交归一化
        ffted = torch.fft.rfft2(x, norm='ortho')

        # 将复数频谱的实部和虚部分离，并在最后添加一个维度
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)  # 实部
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)  # 虚部

        # 在最后一个维度上拼接实部和虚部
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        # 调整维度顺序，使实部和虚部的维度在通道维度之后
        # 从[batch, c, h, w/2+1, 2]变为[batch, c, 2, h, w/2+1]
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()

        # 重塑张量维度，将通道和实虚部维度合并
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ret_ffted = ffted
        # 通过1x1卷积层处理频域特征
        ffted = self.conv_layer(ffted)
        # 经过批归一化和ReLU激活
        ffted = self.gelu(self.bn(ffted))
        ffted = ret_ffted + ffted
        # 重新调整维度，准备进行逆傅里叶变换
        # 将处理后的特征重新分离为实部和虚部
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        real = ffted[..., 0]  # 获取最后一维的第一个分量
        imag = ffted[..., 1]  # 获取最后一维的第二个分量
        real = self.se_real(real)
        imag = self.se_imag(imag)
        real = torch.unsqueeze(real,dim=-1)
        imag = torch.unsqueeze(imag, dim=-1)
        ffted = torch.cat((real, imag), dim=-1)
        # 将实部和虚部转换为复数形式
        ffted = torch.view_as_complex(ffted)

        # 进行二维逆傅里叶变换，将频域信号转换回空域
        # s=(h, w)指定输出大小，norm='ortho'保持正交归一化
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        output = x + output*self.sigma
        return output

# 主程序入口
if __name__ == '__main__':
    block =FSA(64,64).cuda()  # 实例化模型
    input = torch.load('x1.pt').cuda()

    print(input.shape)  # 输出张量的形状

    output = block(input)

    print(output.size())  # 打印输出张量尺寸