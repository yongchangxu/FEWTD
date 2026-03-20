import pywt
import torch
from torch.autograd import Function
import torch.nn as nn

# 定义一个DWT功能类，继承自Function
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保证输入张量 x 在内存中是连续存储的
        x = x.contiguous()

        # 对输入张量进行 padding，确保 H 和 W 是偶数
        H, W = x.shape[2], x.shape[3]
        pad_h, pad_w = 0, 0
        if H % 2 != 0:
            pad_h = 1
        if W % 2 != 0:
            pad_w = 1
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # Padding
        ctx.pad = (pad_h, pad_w)

        # 保存后向传播需要的参数
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape  # 保存前向传播张量的形状

        # 获取输入张量的通道数
        dim = x.shape[1]

        # 对 x 进行二维卷积操作，得到低频和高频分量
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)

        # 将四个分量按通道维度拼接起来
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算 x 的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的权重
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape

            # 根据保存的形状信息重塑 dx
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)

            # 将四个小波滤波器沿零维度拼接，并重复 C 次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)

            # 使用转置卷积进行上采样
            dx = torch.nn.functional.conv_transpose2d(
                dx,
                filters,
                stride=2,
                groups=C
            )

            # **修正输出形状**
            # 在反卷积操作中，output_padding 可能导致多余的像素，因此需要裁剪
            pad_h, pad_w = ctx.pad
            dx = dx[:, :, :H - pad_h, :W - pad_w]  # 去掉多余的 padding

        # 返回 dx 以及其他参数的 None
        return dx, None, None, None, None


# 定义一个二维离散小波变换模块，继承自nn.Module
class DWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建分解低通和高通滤波器的Tensor
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # 计算二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 注册缓冲区变量来存储滤波器
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # 确保滤波器的数据类型为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用DWT_Function的forward方法
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个IDWT功能类，继承自Function
class IDWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, filters):
        # 保存后向传播需要的参数
        ctx.save_for_backward(filters)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 根据保存的形状信息调整x的形状
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 计算通道数
        C = x.shape[1]
        # 重塑x
        x = x.reshape(B, -1, H, W)
        # 重复滤波器C次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回上采样的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的滤波器
            filters = ctx.saved_tensors
            filters = filters[0]
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            # 分解滤波器
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 对dx进行二维卷积操作，得到低频和高频分量
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 将四个分量按通道维度拼接起来
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回dx以及其余不需要梯度的参数
        return dx, None

# 定义一个二维逆离散小波变换模块，继承自nn.Module
class IDWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建重构低通和高通滤波器的Tensor
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # 为滤波器添加额外的维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        # 将四个小波滤波器沿零维度拼接
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册缓冲区变量来存储滤波器
        self.register_buffer('filters', filters)
        # 确保滤波器的数据类型为float32
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用IDWT_Function的forward方法
        return IDWT_Function.apply(x, self.filters)