import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian
from STGCN_FFT.DualDomainTimeBlock import DualDomainTimeBlock

class STGCNBlock(nn.Module):
    """
    Temporal -> GraphConv -> Temporal
    """

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes,
                 kernel_size=3, freq_kernel_size=5, detrend=True, fuse_mode="add"):
        super().__init__()

        # 把原 TimeBlock 换成 FFTGatedTimeBlock（输出形状不变）
        self.temporal1 = DualDomainTimeBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            freq_kernel_size=freq_kernel_size,
            detrend=detrend,
            fuse_mode=fuse_mode,
        )

        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))

        self.temporal2 = DualDomainTimeBlock(
            in_channels=spatial_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            freq_kernel_size=freq_kernel_size,
            detrend=detrend,
            fuse_mode=fuse_mode,
        )

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        X: (B, N, T, Cin)  (按你 TimeBlock 的约定)
        """
        t = self.temporal1(X)  # (B, N, T1, Cout)

        # 图卷积部分：保持你原实现 :contentReference[oaicite:3]{index=3}
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))  # (B, N, T1, spatial_channels)

        t3 = self.temporal2(t2)  # (B, N, T2, Cout)
        return self.batch_norm(t3)


class STGCN_FFT_JOINT(nn.Module):
    """
    STGCN (with optional FFT-Joint temporal blocks)
    """

    def __init__(
        self,
        num_nodes,
        pred_num,
        num_features,
        out_channels,
        spatial_channels,
        num_timesteps_input,
        num_timesteps_output,
        adj_mat,
        device,
        kernel_size=3,
        freq_kernel_size=5,
        detrend=True,
        fuse_mode='add'
    ):
        super().__init__()
        self.pred_num = pred_num

        self.adj = torch.from_numpy(
            calculate_scaled_laplacian(sym_adj(adj_mat - np.min(adj_mat)))
        ).to(device)

        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=out_channels,
            spatial_channels=spatial_channels,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            freq_kernel_size=freq_kernel_size,
            detrend=detrend,
            fuse_mode=fuse_mode,
        )

        self.block2 = STGCNBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            spatial_channels=spatial_channels,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            detrend=detrend,
        )

        self.last_temporal = DualDomainTimeBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            freq_kernel_size=5,
            detrend=True,
            fuse_mode="add",
        )

        self.fully = nn.Linear(
            (num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num),
            num_timesteps_output,
        ) #通道维度和时序卷积层个数相关，因为padding形式，每层时序卷积都会使时间步-2（kernel_size=3, step=1导致的）

    def forward_features(self, X):
        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        # out3: (B, N, T', C)
        return out3
    def forward(self, X):
        """
        最终进入 block 时保持 (B, N, T, C)。
        """
        X = X.permute(0, 2, 1, 3)  # -> (B, N, T, C)

        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], self.pred_num, -1)))
        return out4
