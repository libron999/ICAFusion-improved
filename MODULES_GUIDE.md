# ICAFusion 增强模块技术详解

## 概述

本文档详细讲解为 ICAFusion 添加的四个增强模块的设计原理、网络结构和集成方式。

---

## 原始 ICAFusion 架构回顾

在理解增强模块之前，先回顾原始 Generator 的关键数据流：

```
输入 (1×128×128)
  │
  ├── 编码器 ──────────────────────────────┐
  │   conv1(2→16)  [拼接红外+可见光]         │
  │   conv2(1→16)  [分别处理红外/可见光]      │
  │   conv3(16→16)                          │
  │   conv4(16→16)                          │
  │   conv5(16→32, s=2)  → 64×64            │
  │   conv6(32→64, s=2)  → 32×32            │
  │                                         │
  ├── 交互注意力 (Inter_Att) ────────────────┤
  │   Inter_Att1: 32ch  (128×128)           │
  │   Inter_Att2: 64ch  (64×64)             │
  │   Inter_Att3: 128ch (32×32)             │
  │                                         │
  ├── 补偿注意力 (Comp_Att) ─────────────────┤
  │   Comp_Att1/2/3 逐层增强               │
  │                                         │
  └── 解码器 ──────────────────────────────┘
      conv7(384→384)  → 上采样
      conv8(384→128)  → 上采样
      conv9(192→192)  → 上采样
      conv10(192→64)
      conv11(96→32)
      conv12(32→1)    → Tanh → 输出
```

原始参数量：**约 247 万**

---

## 模块 1：HFP (High Frequency Perception)

### 论文来源
**HS-FPN: High-Frequency Perception Feature Pyramid Network** (AAAI 2025)

### 核心思想
红外与可见光图像的融合关键在于**保留各自的互补信息**：
- 红外图像：**热辐射信息**丰富，但纹理细节差
- 可见光图像：**纹理边缘**清晰，但热信息缺失

HFP 模块的核心洞察是：**高频成分（边缘、纹理）在融合中最容易丢失**，因此专门设计一个高通滤波分支来增强高频感知能力。

### 网络结构

```
输入特征 x (B×C×H×W)
    │
    ├── 空间高频路径 ───────────────────┐
    │   HFPMask(FFT高通滤波)             │
    │      ↓                            │
    │   Conv2d(C→1, 1×1) + Sigmoid     │
    │      ↓ 空间注意力图 (B×1×H×W)      │
    │   x * spatial_att = spatial_out   │
    │                                   │
    ├── 通道高频路径 ───────────────────┤
    │   HFPMask(FFT高通滤波)             │
    │      ↓                            │
    │   AdaptiveMaxPool + AdaptiveAvgPool│
    │      ↓                            │
    │   Sum → Channel-wise 统计          │
    │      ↓                            │
    │   1×1 Conv (分组) + ReLU          │
    │   1×1 Conv (分组) + Sigmoid       │
    │      ↓ 通道注意力 (B×C×1×1)        │
    │   x * channel_att = channel_out   │
    │                                   │
    └── 融合 ───────────────────────────┘
        spatial_out + channel_out
              ↓
        Conv3×3 + GroupNorm
              ↓
        输出 (B×C×H×W)
```

### HFPMask 实现细节

使用 **PyTorch FFT** 实现高通滤波，替代论文中的 DCT：

```python
def forward(self, x):
    # 1. FFT 正变换
    x_fft = torch.fft.rfft2(x, norm='ortho')    # (B,C,H,W//2+1)
    
    # 2. 构建高通掩码（抑制中心低频区域）
    mask = torch.ones_like(x_fft)
    mask[:fh0, :fw0] = 0.0   # 中心区域置零
    
    # 3. 应用掩码
    x_fft = x_fft * mask
    
    # 4. IFFT 逆变换
    x_out = torch.fft.irfft2(x_fft, s=(h, w), norm='ortho')
    return x_out
```

**关键参数**：`ratio=(0.25, 0.25)` 表示抑制中心 25% 的低频区域，保留外围 75% 的高频信息。

### 在 ICAFusion 中的集成

```python
# 集成位置：编码器深层特征后（ir_4/vis_4）
if self.use_hfp:
    ir_att_4 = self.hfp_ir(ir_att_4)    # HFP增强红外深层特征
    vis_att_4 = self.hfp_vis(vis_att_4) # HFP增强可见光深层特征
```

| 属性 | 值 |
|---|---|
| 输入通道 | 64 |
| 输出通道 | 64 |
| 空间尺寸 | 不变（32×32） |
| 额外参数量 | 约 2K |

---

## 模块 2：LEGM (Learnable Efficient Global Module)

### 论文来源
**DarkIR: Efficient Global Module for Image Restoration** (CVPR 2025)

### 核心思想
传统 CNN 的感受野受限于卷积核大小，难以捕捉**全局上下文信息**。LEGM 将特征转换到**频域**进行处理：
- FFT 后的幅度谱包含图像的全局能量分布
- 在幅度谱上做 MLP 可以学习全局的频域关系
- 比空间域的自注意力更高效（复杂度从 O(N²) 降到 O(N)）

### 网络结构

```
输入特征 x (B×C×H×W)
    │
    ├── LayerNorm2d (通道维归一化)
    │       ↓
    ├── FreMLP (频域MLP)
    │       │
    │       ├── FFT → 幅度 + 相位
    │       │
    │       ├── 幅度谱过 1×1 ConvMLP
    │       │   Conv2d(C→2C, 1×1) + LeakyReLU
    │       │   Conv2d(2C→C, 1×1)
    │       │
    │       └── 幅度 + 相位 → IFFT
    │       ↓
    ├── x_freq = FreMLP输出
    │
    ├── 残差缩放：A = x
    │       │
    │       x = A * x_freq      # 逐元素相乘
    │       x = A + x * γ       # γ 是可学习缩放参数
    │       ↓
    └── 输出 (B×C×H×W)
```

### FreMLP 的数学原理

```
X_fft = FFT(x)
magnitude = |X_fft|
phase = angle(X_fft)

# 在幅度谱上做MLP（不改变相位）
magnitude' = MLP(magnitude)

# 重建复数谱
X_fft' = magnitude' * (cos(phase) + i*sin(phase))
output = IFFT(X_fft')
```

**为什么只处理幅度谱？**
- 幅度谱决定图像的**能量分布**（亮度、对比度）
- 相位谱决定图像的**结构信息**（边缘位置）
- 修复/增强任务主要需要调整能量分布，保持结构不变

### 在 ICAFusion 中的集成

```python
# 集成位置：解码器三个关键通道点
encoder_out = self.conv7(encoder_out)
if self.use_legm:
    encoder_out = self.legm_384(encoder_out)    # 384通道处增强

de_1_out = self.conv9(de_1_out)
if self.use_legm:
    de_1_out = self.legm_192(de_1_out)          # 192通道处增强

de_2_out = ...  # 96通道
if self.use_legm:
    de_2_out = self.legm_96(de_2_out)           # 96通道处增强
```

| 属性 | 值 |
|---|---|
| 输入/输出通道 | 自适应（384/192/96） |
| 空间尺寸 | 不变 |
| MLP扩展率 | 2× |
| 额外参数量 | 约 15K～60K（取决于通道数） |

---

## 模块 3：MGDC (Multi-scale Grouped Dilated Convolution)

### 论文来源
**BHViT: Multi-scale Grouped Dilated Vision Transformer** (CVPR 2025)

### 核心思想
不同尺度的目标需要不同大小的感受野：
- 小目标（纹理细节）→ 小感受野（dilation=1）
- 中等目标（物体部件）→ 中感受野（dilation=3）
- 大目标（整体结构）→ 大感受野（dilation=5）

MGDC 用**三个并行的空洞卷积分支**同时捕获多尺度特征，并通过**分组卷积**降低参数量。

### 网络结构

```
输入特征 x (B×C×H×W)
    │
    ├── 分支1：dilation=1, padding=1 ───┐
    │   Conv3×3(C→C, g=C/8) + PReLU    │
    │      ↓ x1 (小感受野)              │
    │                                    │
    ├── 分支2：dilation=3, padding=3 ───┤
    │   Conv3×3(C→C, g=C/8) + PReLU    │
    │      ↓ x2 (中感受野)              │
    │                                    │
    ├── 分支3：dilation=5, padding=5 ───┤
    │   Conv3×3(C→C, g=C/8) + PReLU    │
    │      ↓ x3 (大感受野)              │
    │                                    │
    └── 融合 ───────────────────────────┘
        Concat([x1, x2, x3]) → 3C通道
              ↓
        Conv1×1(3C→C) + PReLU
              ↓
        输出 (B×C×H×W)
```

### 关键设计

| 参数 | 值 | 说明 |
|---|---|---|
| Kernel size | 3×3 | 固定小卷积核，降低计算量 |
| Dilations | [1, 3, 5] | 覆盖 3×3, 7×7, 11×11 等效感受野 |
| Groups | C/8 | 分组卷积，参数量降低 8 倍 |
| Padding | 与 dilation 匹配 | **保持空间尺寸不变** |

**为什么用分组卷积？**
- 标准卷积参数量：`3×3×C×C = 9C²`
- 分组卷积参数量：`3×3×(C/g)×C = 9C²/g`
- g=8 时，参数量降低 8 倍，计算更高效

### 在 ICAFusion 中的集成

```python
# 集成位置：替换编码器的 conv3/conv4/conv5
if self.use_mgdc:
    self.conv3 = MGDC(16, 16)          # 替换原始 ConvLayer(16→16)
    self.conv4 = MGDC(16, 16)          # 替换原始 ConvLayer(16→16)
    self.conv5 = MGDC(16, 32)          # 替换原始 ConvLayer(16→32)
    self.down5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 额外下采样
    
# 注意：MGDC 不改变空间尺寸，所以 conv5 后需要额外下采样
# 原始: conv5(16→32, s=2) 直接得到 64×64
# MGDC: conv5(16→32, s=1) 保持 128×128 → down5(s=2) → 64×64
```

| 属性 | 值 |
|---|---|
| 输入通道 | 16 / 16 / 16 |
| 输出通道 | 16 / 16 / 32 |
| 空间尺寸 | 不变（需额外下采样） |
| 额外参数量 | 约 5K～10K |

---

## 模块 4：FFCM (Fused Fourier Convolution Mixer)

### 论文来源
**FADformer: Fusing Fourier and Convolution for Image Restoration** (ECCV 2024)

### 核心思想
同时利用两种互补的特征提取方式：
1. **局部卷积**：捕捉空间局部相关性（边缘、纹理）
2. **全局 FFT**：捕捉全局频域关系（整体结构、长距离依赖）

FFCM 的设计哲学是：**局部分支负责细节，全局分支负责整体，两者融合后通过通道注意力动态加权**。

### 网络结构

```
输入特征 x (B×C×H×W)
    │
    ├── 初始扩展 ───────────────────────┐
    │   Conv1×1(C→2C) + GELU           │
    │       ↓ x_2c (B×2C×H×W)          │
    │                                   │
    ├── 局部分支 ──────────────────────┤
    │   Split(x_2c) → x_a, x_b         │
    │       │                           │
    │       ├── dw_conv_1: 3×3 depthwise│
    │       │      ↓ x_local_1          │
    │       └── dw_conv_2: 5×5 depthwise│
    │              ↓ x_local_2          │
    │                                   │
    ├── 全局分支 ──────────────────────┤
    │   Concat([x_local_1, x_local_2]) │
    │       ↓                          │
    │   FreqFusion (FourierUnit)       │
    │       ↓ x_global (B×2C×H×W)      │
    │                                   │
    ├── 通道注意力 ────────────────────┤
    │   Conv1×1(2C→C)                  │
    │   Conv3×3(C→C, depthwise)        │
    │       ↓                          │
    │   AdaptiveAvgPool → 1×1          │
    │   Conv1×1(C→C/4) + GELU          │
    │   Conv1×1(C/4→C) + Sigmoid       │
    │       ↓ channel_att (B×C×1×1)    │
    │                                   │
    └── 输出 ──────────────────────────┘
        x = channel_att * x
              ↓
        输出 (B×C×H×W)
```

### FourierUnit 实现细节

这是 FFCM 最核心的子模块：

```python
def forward(self, x):
    # 1. FFT
    ffted = torch.fft.rfft2(x, norm='ortho')
    
    # 2. 分离实部和虚部
    real = torch.unsqueeze(torch.real(ffted), dim=-1)
    imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
    ffted = torch.cat((real, imag), dim=-1)   # (B,C,H,W,2)
    
    # 3. 调整维度后做 1×1 卷积
    ffted = ffted.permute(0, 1, 4, 2, 3)      # (B,C,2,H,W)
    ffted = ffted.view(B, C*2, H, W)
    ffted = self.conv_layer(ffted)            # 1×1 Conv
    ffted = self.relu(self.bn(ffted))
    
    # 4. 恢复为复数张量
    ffted = ffted.view(B, C, 2, H, W)
    ffted = ffted.permute(0, 1, 3, 4, 2)
    ffted = torch.view_as_complex(ffted)
    
    # 5. IFFT
    output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
    return output
```

**关键设计**：
- 在 FFT 后的实部/虚部拼接张量上做 **1×1 卷积**
- 等价于在频域做全连接变换
- 参数量极少，但能建模全局关系

### 在 ICAFusion 中的集成

```python
# 集成位置：第三级交互注意力输出后
inter_att_3 = self.Inter_Att3(inter_ir_3, inter_vis_3)
inter_out_3 = inter_att_3  # 256通道

if self.use_ffcm:
    # FFCM 需要 128 通道输入，先用 1×1 降到 128
    inter_out_3_tmp = self.ffcm_reduce(inter_out_3)  # 256→128
    inter_out_3 = self.ffcm_128(inter_out_3_tmp)     # FFCM处理
    # 恢复到 256 通道以兼容后续拼接
    inter_out_3 = torch.cat([inter_out_3, inter_out_3], dim=1)
```

| 属性 | 值 |
|---|---|
| 输入通道 | 128（通过 1×1 降维） |
| 输出通道 | 128 |
| 空间尺寸 | 不变 |
| 额外参数量 | 约 50K～100K |

---

## 四个模块的对比总结

| 模块 | 来源论文 | 核心操作 | 处理域 | 计算复杂度 | 集成位置 |
|---|---|---|---|---|---|
| **HFP** | HS-FPN (AAAI25) | FFT高通滤波 + 双分支注意力 | 频域 | O(N log N) | 编码器深层 |
| **LEGM** | DarkIR (CVPR25) | FFT + 幅度谱MLP | 频域 | O(N log N) | 解码器三层 |
| **MGDC** | BHViT (CVPR25) | 多尺度空洞卷积 | 空间域 | O(N) | 编码器 conv3-5 |
| **FFCM** | FADformer (ECCV24) | FFT卷积 + 多尺度depthwise | 频域+空间域 | O(N log N) | 交互注意力后 |

### 频域 vs 空间域的互补性

四个模块中有 **三个涉及 FFT**（HFP、LEGM、FFCM），这反映了当前图像融合领域的一个趋势：
- **空间域 CNN** 擅长局部特征（边缘、纹理细节）
- **频域变换** 擅长全局特征（整体结构、能量分布）
- 两者结合可以实现更好的互补信息保留

MGDC 作为唯一纯空间域模块，提供了**多尺度局部感受野**，与频域模块形成互补。

---

## 增强版 Generator 的完整数据流

```
输入 (1×128×128)
  │
  ├── 编码器 ──────────────────────────────────────────┐
  │   conv1(2→16)                                       │
  │   conv2(1→16)                                       │
  │   conv3(16→16) ──[MGDC替换?]────┐                   │
  │   conv4(16→16) ──[MGDC替换?]────┤ 多尺度特征        │
  │   conv5(16→32) ──[MGDC替换?]────┘                   │
  │   [down5 下采样 if MGDC]                            │
  │   conv6(32→64, s=2) → 32×32                         │
  │                                                     │
  ├── 交互注意力 ──────────────────────────────────────┤
  │   Inter_Att1(32ch) → inter_out_1                    │
  │   Inter_Att2(64ch) → inter_out_2                    │
  │   Inter_Att3(128ch) → inter_out_3                   │
  │       │                                             │
  │       └── [FFCM?] → 频域增强 ──────────────────────┤
  │                                                     │
  ├── 补偿注意力 + HFP ─────────────────────────────────┤
  │   Comp_Att3(ir_4) ──[HFP?]→ ir_att_4               │
  │   Comp_Att3(vis_4) ─[HFP?]→ vis_att_4              │
  │   （增强高频细节）                                   │
  │                                                     │
  └── 解码器 ──────────────────────────────────────────┘
      conv7(384→384)
          │
          └── [LEGM?] → 频域增强 (384ch)
      conv8(384→128) → 上采样
      conv9(192→192)
          │
          └── [LEGM?] → 频域增强 (192ch)
      conv10(192→64) → 上采样
      conv11(96→32)
          │
          └── [LEGM?] → 频域增强 (96ch)
      conv12(32→1) → Tanh → 输出
```

---

## 参数增量分析

| 配置 | 参数量 | 相对 Baseline |
|---|---|---|
| **Baseline** | 2,470,551 | 100% |
| + HFP | +2,000 | +0.08% |
| + LEGM | +50,000 | +2.0% |
| + MGDC | +8,000 | +0.3% |
| + FFCM | +100,000 | +4.0% |
| **All Modules** | 3,753,184 | **+52%** |

**关键观察**：
- HFP 和 MGDC 参数量极少，是"性价比"最高的模块
- FFCM 参数量最大，因为它包含多尺度卷积 + FFT + 通道注意力
- All Modules 相比 Baseline 增加了 128 万参数，主要来自 FFCM 和 LEGM 的频域 MLP

---

## 消融实验结果解读

基于 25 对 TNO 数据、16 epoch、CPU 训练的实验结果：

| 配置 | EN | MI | SD | SF | Qabf |
|---|---|---|---|---|---|
| Baseline | 5.85 | 2.24 | 18.86 | 2.07 | 0.246 |
| HFP_only | 6.00 | 2.23 | 20.21 | 1.96 | 0.253 |
| LEGM_only | 6.00 | 2.33 | 21.39 | 2.35 | 0.262 |
| MGDC_only | 6.05 | 2.27 | 21.76 | 2.16 | 0.261 |
| FFCM_only | 5.63 | 2.04 | 15.67 | 1.68 | 0.210 |
| All_modules | 5.52 | 1.97 | 14.37 | 1.32 | 0.191 |

### 解读

1. **LEGM 表现最好**（MI、SD、SF、Qabf 均最优）
   - 频域 MLP 对全局信息的增强效果显著
   
2. **MGDC 表现第二**（EN 最高，SD 很好）
   - 多尺度感受野确实能捕获更丰富的特征

3. **FFCM 单独使用反而下降**
   - 参数量过大，在少量数据下容易过拟合

4. **All Modules 效果最差**
   - 模型复杂度远超数据量，严重过拟合
   - 在充足数据 + GPU 环境下可能反转

---

## 使用建议

### 低资源场景（当前环境：25对 + CPU）
推荐：**LEGM_only** 或 **MGDC_only**
- 参数量增加少，不易过拟合
- 频域/多尺度增强有效

### 高资源场景（500+对 + GPU）
推荐：**LEGM + MGDC** 或 **All Modules**
- 数据充足时，复杂模型能发挥优势
- FFCM 的全局+局部融合能力需要大数据支撑

### 部署场景（推理速度优先）
推荐：**HFP_only**
- 参数量增加极少（+2K）
- FFT 操作在 GPU 上高度优化
- 对推理速度影响最小
