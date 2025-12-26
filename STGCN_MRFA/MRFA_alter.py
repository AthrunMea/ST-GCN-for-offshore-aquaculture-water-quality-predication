import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2508.09000
    论文题目：UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale（ICCV 2025）
    中文题目：UniConvNet：在保持渐近高斯分布的同时扩展卷积网络的有效感受野，适用于任意规模的卷积网络（ICCV 2025）
    讲解视频：https://www.bilibili.com/video/BV1hQHUztEpK/
    新增：门控机制前加入SE/ECA通道注意力机制
"""


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 缩放 γ
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 平移 β
        self.eps = eps
        self.channel_format = data_format
        if self.channel_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.channel_format == "channels_last":  # [N, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        elif self.channel_format == "channels_first":  # [N, C, H, W]
            chan_mean = x.mean(1, keepdim=True)  # μ_c
            chan_var = (x - chan_mean).pow(2).mean(1, keepdim=True)  # σ_c^2
            x_norm = (x - chan_mean) / torch.sqrt(chan_var + self.eps)  # 归一化
            x_out = self.gamma[:, None, None] * x_norm + self.beta[:, None, None]
            return x_out


# ------------------------ 通道注意力模块 ------------------------
class SEModule(nn.Module):
    """压缩-激励模块（Squeeze-and-Excitation）"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 空间维度压缩为1
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),  # 与原模块激活函数对齐
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()  # 输出通道权重（0~1）
        )

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.fc(weight)
        return x * weight  # 通道加权


class ECAModule(nn.Module):
    """高效通道注意力模块（Efficient Channel Attention）"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W] -> [B, C, 1]
        weight = self.avg_pool(x).squeeze(-1)
        # [B, C, 1] -> [B, 1, C] -> 1D卷积 -> [B, 1, C] -> [B, C, 1, 1]
        weight = self.conv(weight.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        weight = self.sigmoid(weight)
        return x * weight  # 通道加权


# ------------------------ 集成注意力的MRFA模块 ------------------------
class MRFAWithAttention(nn.Module):
    def __init__(self, dim, attention_type='se'):
        """
        Args:
            dim: 输入通道数（需被4整除）
            attention_type: 注意力类型，可选 'se' 或 'eca'
        """
        super().__init__()
        self.channels_total = dim
        assert attention_type in ['se', 'eca'], "attention_type must be 'se' or 'eca'"
        self.attention_type = attention_type

        # ------------------------ Stage 1（C/4 通道） ------------------------
        self.ln_stage1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # 上下文分支：1×1 + DWConv-7×7
        self.ctx_branch_s1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )

        # 门控分支前添加通道注意力
        if attention_type == 'se':
            self.attn_s1 = SEModule(channels=dim // 4)
        else:
            self.attn_s1 = ECAModule(channels=dim // 4)
        self.gate_branch_s1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.post_gate_s1 = nn.Conv2d(dim // 4, dim // 4, 1)

        self.prep_quarter2 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # ------------------------ Stage 2（C/2 通道） ------------------------
        self.ln_stage2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.ctx_branch_s2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        # 门控分支前添加通道注意力
        if attention_type == 'se':
            self.attn_s2 = SEModule(channels=dim // 2)
        else:
            self.attn_s2 = ECAModule(channels=dim // 2)
        self.gate_branch_s2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.post_gate_s2 = nn.Conv2d(dim // 2, dim // 2, 1)

        self.prep_quarter3 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.proj_ctx_to_q3 = nn.Conv2d(dim // 2, dim // 4, 1)

        # ------------------------ Stage 3（3C/4 通道） ------------------------
        self.ln_stage3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")

        self.ctx_branch_s3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        # 门控分支前添加通道注意力
        if attention_type == 'se':
            self.attn_s3 = SEModule(channels=dim * 3 // 4)
        else:
            self.attn_s3 = ECAModule(channels=dim * 3 // 4)
        self.gate_branch_s3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.post_gate_s3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)

        self.prep_quarter4 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.refine3x3_s3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.proj_ctx_to_q4 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)

    def forward(self, x):
        # 输入 [B, C, H, W]，C需被4整除
        x = self.ln_stage1(x)
        quarters = torch.split(x, self.channels_total // 4, dim=1)  # 4个 [B, C/4, H, W]

        # ==================== Stage 1 ====================
        ctx_feat = self.ctx_branch_s1(quarters[0])
        # 门控前添加通道注意力
        attn_feat_s1 = self.attn_s1(quarters[0])
        gated_feat = ctx_feat * self.gate_branch_s1(attn_feat_s1)  # 注意力加权后再门控
        gated_feat = self.post_gate_s1(gated_feat)

        s1_q2 = self.refine3x3_s1(self.prep_quarter2(quarters[1]))
        s1_q2 = s1_q2 + ctx_feat
        s1_out = torch.cat((s1_q2, gated_feat), dim=1)

        # ==================== Stage 2 ====================
        s1_out = self.ln_stage2(s1_out)
        ctx_feat = self.ctx_branch_s2(s1_out)
        # 门控前添加通道注意力
        attn_feat_s2 = self.attn_s2(s1_out)
        gated_feat = ctx_feat * self.gate_branch_s2(attn_feat_s2)
        gated_feat = self.post_gate_s2(gated_feat)

        s2_q3 = self.refine3x3_s2(self.prep_quarter3(quarters[2]))
        s2_q3 = s2_q3 + self.proj_ctx_to_q3(ctx_feat)
        s2_out = torch.cat((s2_q3, gated_feat), dim=1)

        # ==================== Stage 3 ====================
        s2_out = self.ln_stage3(s2_out)
        ctx_feat = self.ctx_branch_s3(s2_out)
        # 门控前添加通道注意力
        attn_feat_s3 = self.attn_s3(s2_out)
        gated_feat = ctx_feat * self.gate_branch_s3(attn_feat_s3)
        gated_feat = self.post_gate_s3(gated_feat)

        s3_q4 = self.refine3x3_s3(self.prep_quarter4(quarters[3]))
        s3_q4 = s3_q4 + self.proj_ctx_to_q4(ctx_feat)
        s3_out = torch.cat((s3_q4, gated_feat), dim=1)

        return s3_out


if __name__ == '__main__':
    # 测试SE注意力版本
    print("=== 测试SE注意力版本 ===")
    x = torch.randn(1, 32, 50, 50)
    model_se = MRFAWithAttention(dim=32, attention_type='se')
    y_se = model_se(x)
    print(f"输入形状: {x.shape}, SE版输出形状: {y_se.shape}")

    # 测试ECA注意力版本
    print("\n=== 测试ECA注意力版本 ===")
    model_eca = MRFAWithAttention(dim=32, attention_type='eca')
    y_eca = model_eca(x)
    print(f"输入形状: {x.shape}, ECA版输出形状: {y_eca.shape}")

    # 验证参数合理性
    total_params_se = sum(p.numel() for p in model_se.parameters())
    total_params_eca = sum(p.numel() for p in model_eca.parameters())
    print(f"\nSE版总参数量: {total_params_se:,}")
    print(f"ECA版总参数量: {total_params_eca:,}")