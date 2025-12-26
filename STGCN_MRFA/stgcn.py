import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian
from STGCN_MRFA.MRFA import MRFA  # 导入MRFA模块


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
    """修改后的STGCN块，加入MRFA模块"""

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        # 新增MRFA模块，用于捕捉多尺度空间特征
        self.mrfa = MRFA(dim=out_channels)  # 使用输出通道数作为MRFA的维度
        # self.dropout = nn.Dropout(0.3) #dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)  # temporal convolution

        # 应用MRFA模块处理时间卷积后的特征
        # 需要调整维度以适应MRFA的输入格式 [B, C, H, W]
        t_mrfa = t.permute(0, 3, 1, 2)  # 转换为 [B, C, N, T]
        t_mrfa = self.mrfa(t_mrfa)  # 多感受野聚合
        t_mrfa = t_mrfa.permute(0, 2, 3, 1)  # 还原为 [B, N, T, C]

        # 图卷积操作
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t_mrfa.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t2 = self.dropout(t2) #dropout

        t3 = self.temporal2(t2)  # 第二次时间卷积
        return self.batch_norm(t3)


class STGCN(nn.Module):
    """修改后的STGCN模型，保持整体结构不变"""

    def __init__(self, num_nodes, pred_num, num_features, out_channels, spatial_channels,
                 num_timesteps_input, num_timesteps_output, adj_mat, device):
        super(STGCN, self).__init__()
        self.adj = torch.from_numpy(calculate_scaled_laplacian(
            sym_adj(adj_mat - np.min(adj_mat)))).to(device)

        # 使用修改后的STGCNBlock
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
        # 保持解码器部分不变
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
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
        # out4 = self.fully(out3.reshape((out3.shape[0], self.pred_num, -1)))
        #
        # return out4
        return torch.cat((out5, out6, out7), dim=1)