from torch import nn
import torch


class g_content_loss(nn.Module):
    """
    增强版内容损失（仅 train_enhanced.py 使用，原始 loss.py 保持不变）。

    与原始版本的差异（方案B：损失再平衡）：
      原始: content = MSE(融合图, 红外) + L1(融合图, 可见光)
      现在: content = ir_weight * L1(融合图, 红外) + L1(融合图, 可见光)

    原因：MSE 是二阶损失，融合图接近目标后梯度指数级衰减；而可见光项是
    一阶 L1，梯度恒定。原组合下红外项拉力远弱于可见光项，训练被"复制可见光"
    主导，红外目标特征丢失。改为同为 L1 后两项梯度强度同量级，红外信息得以
    注入融合图。ir_weight 默认 2.0，若融合图偏亮/红外过重可下调至 1.0。

    返回 (content_loss, L1_loss, F_loss) 与原始版结构一致；
    注意日志中的 F_loss 含义变为"红外 L1 距离"，数值比原来的 MSE 大属正常。
    """
    def __init__(self, ir_weight=2.0):
        super(g_content_loss, self).__init__()
        self.L1_loss = torch.nn.L1Loss(reduction="mean")
        self.ir_weight = ir_weight

    def forward(self, img_ir, img_vi, img_fusion):
        # 红外保持项：原 MSE → L1（一阶梯度，与可见光项同量级）
        F_loss = self.L1_loss(img_fusion, img_ir)
        # 可见光保持项（与原始版相同）
        L1_loss = self.L1_loss(img_fusion, img_vi)
        content_loss = self.ir_weight * F_loss + L1_loss
        return content_loss, L1_loss, F_loss
