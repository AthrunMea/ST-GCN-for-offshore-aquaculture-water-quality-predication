import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian
from STGCN_MRFA.MRFA import MRFA  # 导入MRFA模块


# 新增ECA通道注意力模块
class ECA(nn.Module):
    """
    ECA模块：高效通道注意力机制
    参考论文：ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    """

    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 根据通道数自适应计算卷积核大小
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1  # 确保为奇数

        # 全局平均池化：保留批次和通道维度
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D卷积实现局部跨通道交互
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, N, T]（批次、通道、节点、时间步）
        b, c, _, _ = x.size()

        # 全局平均池化：[B, C, N, T] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        # 调整维度适应1D卷积：[B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D卷积捕获局部通道关系：[B, 1, C] -> [B, 1, C]
        y = self.conv(y)
        # 恢复维度并生成注意力权重：[B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        # 应用注意力权重到输入特征
        return x * y.expand_as(x)


class TimeBlock(nn.Module):
    # 保持原有实现不变
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        return out.permute(0, 2, 3, 1)


class STGCNBlock(nn.Module):
    """修改后的STGCN块，加入MRFA模块和ECA通道注意力"""

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        # 新增MRFA模块
        self.mrfa = MRFA(dim=out_channels)
        # 新增ECA通道注意力模块（在MRFA之后）
        self.eca = ECA(in_channels=out_channels)  # 输入通道数与MRFA输出一致
        # self.dropout = nn.Dropout(0.3) # 可选dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)  # 时间卷积

        # MRFA模块处理
        t_mrfa = t.permute(0, 3, 1, 2)  # 转换为 [B, C, N, T]
        t_mrfa = self.mrfa(t_mrfa)  # 多感受野聚合

        # ECA注意力机制处理
        t_eca = self.eca(t_mrfa)  # 应用通道注意力

        # 还原维度并进行图卷积
        t_eca = t_eca.permute(0, 2, 3, 1)  # 还原为 [B, N, T, C]
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t_eca.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t2 = self.dropout(t2) # 可选dropout

        t3 = self.temporal2(t2)  # 第二次时间卷积
        return self.batch_norm(t3)


class STGCN(nn.Module):
    """保持原有结构不变，使用修改后的STGCNBlock"""

    def __init__(self, num_nodes, pred_num, num_features, out_channels, spatial_channels,
                 num_timesteps_input, num_timesteps_output, adj_mat, device):
        super(STGCN, self).__init__()
        self.adj = torch.from_numpy(calculate_scaled_laplacian(
            sym_adj(adj_mat - np.min(adj_mat)))).to(device)

        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=out_channels,
            spatial_channels=spatial_channels,
            num_nodes=num_nodes
        )
        self.block2 = STGCNBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            spatial_channels=spatial_channels,
            num_nodes=num_nodes
        )

        self.last_temporal = TimeBlock(in_channels=out_channels, out_channels=out_channels)
        self.pred_num = pred_num
        self.fc1 = nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num) * 2,
                             num_timesteps_output)
        self.fc2 = nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num) * 2,
                             num_timesteps_output)
        self.fc3 = nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num) * 2,
                             num_timesteps_output)
        self.dropout = nn.Dropout(0.5)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num),
                               num_timesteps_output)

    def decoder(self, input1, input2, input3):
        out1 = torch.softmax(input2 * input3 / input1.size()[1], dim=1) * input1
        out = torch.cat([input1, out1], 1)
        self.dropout(out)
        return out

    def forward(self, X):
        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)

        x = out3.reshape((out3.shape[0], out3.shape[1], -1))
        x1 = x[:, 0:5, :]
        x2 = x[:, 5:10, :]
        x3 = x[:, 10:15, :]

        out_d1 = self.decoder(x1, x2, x3)
        out_d2 = self.decoder(x2, x3, x1)
        out_d3 = self.decoder(x3, x1, x2)

        out5 = self.fc1(out_d1.reshape(out_d1.shape[0], 1, -1))
        out6 = self.fc2(out_d2.reshape(out_d2.shape[0], 1, -1))
        out7 = self.fc3(out_d3.reshape(out_d3.shape[0], 1, -1))
        return torch.cat((out5, out6, out7), dim=1)