import torch.nn as nn
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
增强版 ICAFusion 网络结构
在原始 Generator 基础上，可选地集成以下四个模块：
  1. HFP   (High Frequency Perception)          — 高频感知，增强边缘纹理
  2. LEGM  (Learnable Efficient Global Module)  — 频域MLP，增强全局频域信息
  3. MGDC  (Multi-scale Grouped Dilated Conv)   — 多尺度分组空洞卷积
  4. FFCM  (Fused Fourier Convolution Mixer)    — 融合傅里叶卷积混合器

使用方式：
  G = Generator_Enhanced(use_hfp=True, use_legm=True, use_mgdc=True, use_ffcm=True)

可根据消融实验需要，自由开关任意组合。
"""


# ============================================================================
# 原始模块（保持与 Models.py 完全一致）
# ============================================================================

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.PReLU = nn.PReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_relu is True:
            out = self.PReLU(out)
        return out


class ConvLayer_dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer_dis, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv2d(x)
        if self.use_relu is True:
            out = self.LeakyReLU(out)
        return out


class Inter_Att(torch.nn.Module):
    def __init__(self, channels):
        super(Inter_Att, self).__init__()
        self.sigmod = nn.Sigmoid()
        self.ca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, 1)

    def forward(self, ir, vis):
        w_ir_avg = self.ca_avg(ir)
        w_ir_max = self.ca_max(ir)
        w_ir = torch.cat([w_ir_avg, w_ir_max], dim=1)
        w_ir = self.conv1(w_ir)
        w_ir_f = self.sigmod(w_ir)

        w_vis_avg = self.ca_avg(vis)
        w_vis_max = self.ca_max(vis)
        w_vis = torch.cat([w_vis_avg, w_vis_max], dim=1)
        w_vis = self.conv1(w_vis)
        w_vis_f = self.sigmod(w_vis)

        EPSILON = 1e-10
        mask_ir = torch.exp(w_ir_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        mask_vis = torch.exp(w_vis_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        out_ir = mask_ir * ir
        out_vis = mask_vis * vis

        avgout_ir = torch.mean(out_ir, dim=1, keepdim=True)
        maxout_ir, _ = torch.max(out_ir, dim=1, keepdim=True)
        x_ir = torch.cat([avgout_ir, maxout_ir], dim=1)
        x1_ir = self.conv(x_ir)
        x2_ir = self.sigmod(x1_ir)

        avgout_vis = torch.mean(out_vis, dim=1, keepdim=True)
        maxout_vis, _ = torch.max(out_vis, dim=1, keepdim=True)
        x_vis = torch.cat([avgout_vis, maxout_vis], dim=1)
        x1_vis = self.conv(x_vis)
        x2_vis = self.sigmod(x1_vis)

        mask_ir_sa = torch.exp(x2_ir) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)
        mask_vis_sa = torch.exp(x2_vis) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)

        output_ir = mask_ir_sa * out_ir
        output_vis = mask_vis_sa * out_vis
        output = torch.cat([output_ir, output_vis], dim=1)
        return output


class Comp_Att(torch.nn.Module):
    def __init__(self, channels):
        super(Comp_Att, self).__init__()
        self.ca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.sigmod = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, 1)

    def forward(self, x):
        w_avg = self.ca_avg(x)
        w_max = self.ca_max(x)
        w = torch.cat([w_avg, w_max], dim=1)
        w = self.conv1(w)
        w_f = self.sigmod(w)
        output_ca = x * w_f

        avgout = torch.mean(output_ca, dim=1, keepdim=True)
        maxout, _ = torch.max(output_ca, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        x2 = self.sigmod(x1)
        output = output_ca * x2
        return output


class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, shape, x):
        x = self.up(x)
        shape = shape.size()
        shape_x = x.size()
        left = right = top = bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 == 0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)
        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 == 0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)
        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x


# ============================================================================
# 新增模块 1：HFP (High Frequency Perception)
# 论文：HS-FPN (AAAI 2025)
# 说明：使用 FFT 实现高通滤波（替代原版的 DCT，无需额外安装 torch_dct）
# ============================================================================

class HFPMask(nn.Module):
    """
    基于 FFT 的高通滤波掩码。
    ratio: 低频抑制比例，如 (0.25, 0.25) 表示保留中心 25% 以外的频率成分。
    """
    def __init__(self, ratio=(0.25, 0.25)):
        super(HFPMask, self).__init__()
        self.ratio = ratio
        # 不使用 register_buffer，避免保存到 state_dict
        self._cached_mask = None
        self._cached_shape = None

    def _get_mask(self, h, w, device):
        mask = torch.ones((h, w), dtype=torch.float32, device=device)
        h0 = int(h * self.ratio[0])
        w0 = int(w * self.ratio[1])
        # 抑制低频（中心区域）
        mask[:h0, :w0] = 0.0
        return mask

    def forward(self, x):
        _, c, h, w = x.shape
        if self._cached_shape != (h, w) or self._cached_mask is None:
            self._cached_shape = (h, w)
            self._cached_mask = self._get_mask(h, w, x.device)
        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        # 构建与 x_fft 同 shape 的掩码 (B,C,H,W//2+1)
        _, _, fh, fw = x_fft.shape
        mask_fft = torch.ones((fh, fw), dtype=torch.float32, device=x.device)
        fh0 = int(fh * self.ratio[0])
        fw0 = int(fw * self.ratio[1])
        mask_fft[:fh0, :fw0] = 0.0
        mask_fft = mask_fft.view(1, 1, fh, fw)
        # 应用掩码
        x_fft = x_fft * mask_fft
        # IFFT
        x_out = torch.fft.irfft2(x_fft, s=(h, w), norm='ortho')
        return x_out
    """
    基于 FFT 的高通滤波掩码。
    ratio: 低频抑制比例，如 (0.25, 0.25) 表示保留中心 25% 以外的频率成分。
    """
    def __init__(self, ratio=(0.25, 0.25)):
        super(HFPMask, self).__init__()
        self.ratio = ratio
        self.register_buffer('mask', None)

    def _get_mask(self, h, w, device):
        mask = torch.ones((h, w), dtype=torch.float32, device=device)
        h0 = int(h * self.ratio[0])
        w0 = int(w * self.ratio[1])
        # 抑制低频（中心区域）
        mask[:h0, :w0] = 0.0
        return mask

    def forward(self, x):
        _, c, h, w = x.shape
        if self.mask is None or self.mask.shape != (h, w):
            self.mask = self._get_mask(h, w, x.device)
        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        # 构建与 x_fft 同 shape 的掩码 (B,C,H,W//2+1)
        _, _, fh, fw = x_fft.shape
        mask_fft = torch.ones((fh, fw), dtype=torch.float32, device=x.device)
        fh0 = int(fh * self.ratio[0])
        fw0 = int(fw * self.ratio[1])
        mask_fft[:fh0, :fw0] = 0.0
        mask_fft = mask_fft.view(1, 1, fh, fw)
        # 应用掩码
        x_fft = x_fft * mask_fft
        # IFFT
        x_out = torch.fft.irfft2(x_fft, s=(h, w), norm='ortho')
        return x_out


class HFP(nn.Module):
    """
    High Frequency Perception 模块
    包含：空间高频路径 + 通道高频路径 + 输出融合
    """
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8)):
        super(HFP, self).__init__()
        # 空间路径：高通滤波 -> 生成空间注意力图
        self.spatial_hpf = HFPMask(ratio=ratio)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 通道路径：对高通滤波后的特征做通道注意力
        self.channel_hpf = HFPMask(ratio=ratio)
        self.channel_pool_h = patch[0]
        self.channel_pool_w = patch[1]
        self.channel_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=max(1, in_channels // 8), bias=False),
            nn.ReLU(inplace=True)
        )
        self.channel_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=max(1, in_channels // 8), bias=False),
            nn.Sigmoid()
        )
        # 输出融合
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, in_channels // 16), in_channels)
        )

    def forward(self, x):
        # 空间路径
        hpf_spatial = self.spatial_hpf(x)           # 高通滤波特征
        spatial_att = self.spatial_conv(hpf_spatial)  # (B,1,H,W)
        spatial_out = x * spatial_att

        # 通道路径
        hpf_channel = self.channel_hpf(x)             # 高通滤波特征
        # 自适应池化到 patch 尺寸后求和
        amaxp = nn.functional.adaptive_max_pool2d(hpf_channel, output_size=(self.channel_pool_h, self.channel_pool_w))
        aavgp = nn.functional.adaptive_avg_pool2d(hpf_channel, output_size=(self.channel_pool_h, self.channel_pool_w))
        n, c, _, _ = amaxp.size()
        amaxp = torch.sum(nn.functional.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(nn.functional.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)
        channel = self.channel_conv1(amaxp) + self.channel_conv1(aavgp)
        channel_att = self.channel_conv2(channel)     # (B,C,1,1)
        channel_out = x * channel_att

        # 融合
        out = self.out(spatial_out + channel_out)
        return out


# ============================================================================
# 新增模块 2：LEGM (Learnable Efficient Global Module)
# 论文：DarkIR (CVPR 2025)
# 说明：频域 MLP，对特征做 FFT -> 幅度处理 -> IFFT，增强全局频域感知
# ============================================================================

class LayerNorm2d(nn.Module):
    """通道维 LayerNorm，适配 2D 特征图"""
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class FreMLP(nn.Module):
    """频域 MLP：在 FFT 后的幅度谱上做 1x1 卷积处理"""
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class LEGM(nn.Module):
    """
    Learnable Efficient Global Module
    对输入特征做 LayerNorm -> 频域 MLP -> 残差缩放
    """
    def __init__(self, channels):
        super(LEGM, self).__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(nc=channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        A = inp
        x = self.norm(inp)
        x_freq = self.freq(x)
        x = A * x_freq
        x = A + x * self.gamma
        return x


# ============================================================================
# 新增模块 3：MGDC (Multi-scale Grouped Dilated Convolution)
# 论文：BHViT (CVPR 2025)
# 说明：并行的多尺度空洞卷积分支，捕获不同感受野的特征
# ============================================================================

class MGDC(nn.Module):
    """
    Multi-scale Grouped Dilated Convolution
    三个并行分支：dilation=1, 3, 5，使用分组卷积降低参数量
    修复：padding 与 dilation 匹配，保持输出尺寸不变
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, groups=None):
        super(MGDC, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if groups is None:
            groups = max(1, in_channels // 8)

        # 分支1：小感受野 (dilation=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, groups=groups, bias=False),
            nn.PReLU()
        )
        # 分支2：中感受野 (dilation=3, padding=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=3, dilation=3, groups=groups, bias=False),
            nn.PReLU()
        )
        # 分支3：大感受野 (dilation=5, padding=5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=5, dilation=5, groups=groups, bias=False),
            nn.PReLU()
        )
        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        out = self.fuse(x_cat)
        return out


# ============================================================================
# 新增模块 4：FFCM (Fused Fourier Convolution Mixer)
# 论文：FADformer (ECCV 2024)
# 说明：融合局部多尺度卷积 + 全局 FFT 卷积 + 通道注意力
# ============================================================================

class FourierUnit(nn.Module):
    """傅里叶单元：FFT -> 1x1 卷积 -> IFFT"""
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output


class FreqFusion(nn.Module):
    """频域融合模块"""
    def __init__(self, dim):
        super(FreqFusion, self).__init__()
        self.dim = dim
        self.conv_init_1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(dim * 2, dim * 2)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))
        return x


class FFCM(nn.Module):
    """
    Fused Fourier Convolution Mixer
    输入输出通道数相同 (dim -> dim)
    """
    def __init__(self, dim):
        super(FFCM, self).__init__()
        self.dim = dim
        self.mixer_global = FreqFusion(dim=dim)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2,
                      groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=5 // 2,
                      groups=dim, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_global = self.mixer_global(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_global)
        x = self.ca(x) * x
        return x


# ============================================================================
# 增强版生成器 Generator_Enhanced
# ============================================================================

class Generator_Enhanced(nn.Module):
    """
    增强版生成器，支持可选地集成 HFP / LEGM / MGDC / FFCM 模块。

    参数说明：
      use_hfp  : 是否在编码器深层特征后添加 HFP 高频感知模块
      use_legm : 是否在解码器中添加 LEGM 频域增强模块
      use_mgdc : 是否用 MGDC 替换部分编码器卷积（conv3/conv4/conv5）
      use_ffcm : 是否在交互注意力后添加 FFCM 全局频域混合
    """
    def __init__(self, use_hfp=False, use_legm=False, use_mgdc=False, use_ffcm=False):
        super(Generator_Enhanced, self).__init__()
        self.use_hfp = use_hfp
        self.use_legm = use_legm
        self.use_mgdc = use_mgdc
        self.use_ffcm = use_ffcm

        self.encoder_pad = nn.ReflectionPad2d([1, 1, 1, 1])
        kernel_size_2 = 3

        # ---------- 编码器 ----------
        self.conv1 = ConvLayer(2, 16, kernel_size=3, stride=1, use_relu=True)
        self.conv2 = ConvLayer(1, 16, kernel_size=3, stride=1, use_relu=True)

        if self.use_mgdc:
            # MGDC 替换 conv3/conv4/conv5，增强多尺度特征提取
            self.conv3 = MGDC(16, 16)
            self.conv4 = MGDC(16, 16)
            self.conv5 = MGDC(16, 32)  # 注意：MGDC 不改变空间尺寸，需额外下采样
            self.down5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv6 = ConvLayer(32, 64, kernel_size=3, stride=2, use_relu=True)
        else:
            self.conv3 = ConvLayer(16, 16, kernel_size=3, stride=1, use_relu=True)
            self.conv4 = ConvLayer(16, 16, kernel_size=3, stride=1, use_relu=True)
            self.conv5 = ConvLayer(16, 32, kernel_size=3, stride=2, use_relu=True)
            self.conv6 = ConvLayer(32, 64, kernel_size=3, stride=2, use_relu=True)

        # 交互注意力后的降维卷积
        self.inter_conv_1 = ConvLayer(64, 32, kernel_size=3, stride=2, use_relu=True)
        self.inter_conv_2 = ConvLayer(128, 64, kernel_size=3, stride=2, use_relu=True)

        # 交互注意力
        self.Inter_Att1 = Inter_Att(32)
        self.Inter_Att2 = Inter_Att(64)
        self.Inter_Att3 = Inter_Att(128)

        # 补偿注意力
        self.Comp_Att1 = Comp_Att(16)
        self.Comp_Att2 = Comp_Att(32)
        self.Comp_Att3 = Comp_Att(64)

        # ---------- 可选模块：FFCM 增强交互注意力输出 ----------
        if self.use_ffcm:
            self.ffcm_128 = FFCM(dim=128)   # 用于 inter_att_3 输出 (256通道，拆分为128+128)
            # 注意：inter_att_3 输出是 256 通道（两个128拼接），需要适配
            self.ffcm_reduce = nn.Conv2d(256, 128, 1)

        # ---------- 可选模块：HFP 增强深层特征 ----------
        if self.use_hfp:
            self.hfp_ir = HFP(64, ratio=(0.25, 0.25))
            self.hfp_vis = HFP(64, ratio=(0.25, 0.25))

        # ---------- 解码器 ----------
        self.UP1 = UpsampleReshape()
        self.UP2 = UpsampleReshape()

        self.conv7 = ConvLayer(384, 384, 3, stride=1, use_relu=True)
        self.conv8 = ConvLayer(384, 128, kernel_size_2, stride=1, use_relu=True)
        self.conv9 = ConvLayer(192, 192, 3, stride=1, use_relu=True)
        self.conv10 = ConvLayer(192, 64, kernel_size_2, stride=1, use_relu=True)
        self.conv11 = ConvLayer(96, 32, kernel_size_2, stride=1, use_relu=True)
        self.conv12 = ConvLayer(32, 1, kernel_size_2, stride=1, use_relu=False)

        # ---------- 可选模块：LEGM 增强解码器 ----------
        if self.use_legm:
            self.legm_384 = LEGM(384)
            self.legm_192 = LEGM(192)
            self.legm_96 = LEGM(96)

        self.tanh = nn.Tanh()

    def forward(self, input_ir, input_vis):
        # ==================== 编码器 ====================
        concat_path = torch.cat([input_ir, input_vis], dim=1)
        concat_1 = self.conv1(concat_path)

        ir_1 = self.conv2(input_ir)
        vis_1 = self.conv2(input_vis)

        ir_2 = self.conv3(ir_1)
        vis_2 = self.conv3(vis_1)
        concat_2 = self.conv4(concat_1)

        if self.use_mgdc:
            ir_3 = self.conv5(ir_2)
            ir_3 = self.down5(ir_3)
            vis_3 = self.conv5(vis_2)
            vis_3 = self.down5(vis_3)
        else:
            ir_3 = self.conv5(ir_2)
            vis_3 = self.conv5(vis_2)

        ir_4 = self.conv6(ir_3)
        vis_4 = self.conv6(vis_3)

        # ---------- 第一级交互注意力 ----------
        inter_ir_1 = torch.cat([concat_2, ir_2], dim=1)
        inter_vis_1 = torch.cat([concat_2, vis_2], dim=1)
        inter_att_1 = self.Inter_Att1(inter_ir_1, inter_vis_1)
        inter_out_1 = self.inter_conv_1(inter_att_1)

        # ---------- 第二级交互注意力 ----------
        inter_ir_2 = torch.cat([inter_out_1, ir_3], dim=1)
        inter_vis_2 = torch.cat([inter_out_1, vis_3], dim=1)
        inter_att_2 = self.Inter_Att2(inter_ir_2, inter_vis_2)
        inter_out_2 = self.inter_conv_2(inter_att_2)

        # ---------- 第三级交互注意力 ----------
        inter_ir_3 = torch.cat([inter_out_2, ir_4], dim=1)
        inter_vis_3 = torch.cat([inter_out_2, vis_4], dim=1)
        inter_att_3 = self.Inter_Att3(inter_ir_3, inter_vis_3)
        inter_out_3 = inter_att_3

        # 可选：FFCM 增强第三级交互输出
        if self.use_ffcm:
            # inter_out_3 是 256 通道，先用 1x1 卷积降到 128，再过 FFCM
            inter_out_3_tmp = self.ffcm_reduce(inter_out_3)
            inter_out_3 = self.ffcm_128(inter_out_3_tmp)
            # FFCM 输出 128 通道，需要恢复为 256 以兼容后续拼接
            inter_out_3 = torch.cat([inter_out_3, inter_out_3], dim=1)

        # ==================== 解码器 ====================
        # 补偿注意力
        ir_att_4 = self.Comp_Att3(ir_4)
        vis_att_4 = self.Comp_Att3(vis_4)

        # 可选：HFP 增强深层补偿特征
        if self.use_hfp:
            ir_att_4 = self.hfp_ir(ir_att_4)
            vis_att_4 = self.hfp_vis(vis_att_4)

        encoder_out = torch.cat([inter_out_3, ir_att_4, vis_att_4], dim=1)
        encoder_out = self.UP1(inter_ir_2, encoder_out)
        encoder_out = self.conv7(encoder_out)

        # 可选：LEGM 增强 384 通道特征
        if self.use_legm:
            encoder_out = self.legm_384(encoder_out)

        de_1 = self.conv8(encoder_out)

        ir_att_3 = self.Comp_Att2(ir_3)
        vis_att_3 = self.Comp_Att2(vis_3)
        de_1_out = torch.cat([de_1, ir_att_3, vis_att_3], dim=1)
        de_1_out = self.UP2(inter_ir_1, de_1_out)
        de_1_out = self.conv9(de_1_out)

        # 可选：LEGM 增强 192 通道特征
        if self.use_legm:
            de_1_out = self.legm_192(de_1_out)

        de_2 = self.conv10(de_1_out)

        ir_att_2 = self.Comp_Att1(ir_2)
        vis_att_2 = self.Comp_Att1(vis_2)
        de_2_out = torch.cat([de_2, ir_att_2, vis_att_2], dim=1)

        # 可选：LEGM 增强 96 通道特征
        if self.use_legm:
            de_2_out = self.legm_96(de_2_out)

        de_3 = self.conv11(de_2_out)
        output = self.conv12(de_3)
        output = self.tanh(output)
        return output


# ============================================================================
# 判别器（与原始完全一致）
# ============================================================================

class D_IR(nn.Module):
    def __init__(self):
        super(D_IR, self).__init__()
        fliter = [1, 16, 32, 64, 128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).to(device)
        out = self.tanh(linear(out))
        return out.squeeze()


class D_VI(nn.Module):
    def __init__(self):
        super(D_VI, self).__init__()
        fliter = [1, 16, 32, 64, 128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).to(device)
        out = self.tanh(linear(out))
        return out.squeeze()
