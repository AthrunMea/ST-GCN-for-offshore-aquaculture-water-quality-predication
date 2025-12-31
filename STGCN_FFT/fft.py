import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian


class FFTGatedTimeBlock(nn.Module):
    """
    TimeBlock + FFT magnitude gating.
    输入:  (B, N, T, Cin)
    输出:  (B, N, T_out, Cout)  (T_out = T - kernel_size + 1)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_fft_gate: bool = True,
        detrend: bool = True,
        gate_hidden: int = None,
        gate_dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.use_fft_gate = use_fft_gate
        self.detrend = detrend
        self.eps = eps

        # 原 TimeBlock 的三路卷积（保持一致）:contentReference[oaicite:1]{index=1}
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        # FFT 门控：把频域摘要(Cin) -> Cout，输出 gate 形状 (B, Cout, N, 1)
        if gate_hidden is None:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True),
            )
        else:
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels, gate_hidden, kernel_size=(1, 1), bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=gate_dropout),
                nn.Conv2d(gate_hidden, out_channels, kernel_size=(1, 1), bias=True),
            )

        self.gate_dropout = nn.Dropout(p=gate_dropout) if gate_dropout > 0 else nn.Identity()

    def _fft_gate(self, X_nchw: torch.Tensor, out_time: torch.Tensor) -> torch.Tensor:
        """
        X_nchw:  (B, Cin, N, T)
        out_time:(B, Cout, N, T_out)
        return:  gate (B, Cout, N, 1)  broadcast 到 T_out
        """
        # 可选：去均值（去趋势），让频谱更稳定
        if self.detrend:
            X_center = X_nchw - X_nchw.mean(dim=-1, keepdim=True)
        else:
            X_center = X_nchw

        # rFFT 沿时间维（最后一维）: (B, Cin, N, F)
        Xf = torch.fft.rfft(X_center, dim=-1)

        # 幅值 + log 压缩，避免动态范围太大
        mag = torch.abs(Xf).clamp_min(self.eps)
        mag = torch.log1p(mag)  # (B, Cin, N, F)

        # 频域摘要：对频点维做平均/池化 -> (B, Cin, N, 1)
        mag_pool = mag.mean(dim=-1, keepdim=True)

        # 生成 gate: (B, Cout, N, 1)
        gate = self.gate_net(mag_pool)
        gate = self.gate_dropout(gate)
        gate = torch.sigmoid(gate)

        return gate

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, T, Cin)
        """
        # (B, Cin, N, T)  :contentReference[oaicite:2]{index=2}
        X_nchw = X.permute(0, 3, 1, 2)

        # 原 TimeBlock 的时域计算（保持一致）
        temp = self.conv1(X_nchw) + torch.sigmoid(self.conv2(X_nchw))
        out_time = F.relu(temp + self.conv3(X_nchw))  # (B, Cout, N, T_out)

        if self.use_fft_gate:
            gate = self._fft_gate(X_nchw, out_time)    # (B, Cout, N, 1)
            # 频域门控调制：残差式放大/抑制（比直接乘 gate 更稳）
            out_time = out_time * (1.0 + gate)

        # (B, N, T_out, Cout)
        out = out_time.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Temporal -> GraphConv -> Temporal
    """

    def __init__(
        self,
        in_channels,
        spatial_channels,
        out_channels,
        num_nodes,
        kernel_size=3,
        use_fft_gate=True,
        detrend=True,
    ):
        super().__init__()
        # 把原 TimeBlock 换成 FFTGatedTimeBlock（输出形状不变）
        self.temporal1 = FFTGatedTimeBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_fft_gate=use_fft_gate,
            detrend=detrend,
        )

        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))

        self.temporal2 = FFTGatedTimeBlock(
            in_channels=spatial_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_fft_gate=use_fft_gate,
            detrend=detrend,
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


class STGCN_FFT(nn.Module):
    """
    STGCN (with optional FFT-gated temporal blocks)
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
        use_fft_gate=True,
        detrend=True,
        separate_third_block=True,  # 建议：第三层不要复用 block2 参数
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
            use_fft_gate=use_fft_gate,
            detrend=detrend,
        )

        self.block2 = STGCNBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            spatial_channels=spatial_channels,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            use_fft_gate=use_fft_gate,
            detrend=detrend,
        )

        # 你原来是 out5 = self.block2(out2, self.adj) 复用同一组参数 :contentReference[oaicite:4]{index=4}
        # 这里给你一个开关：默认用独立 block3（更合理，也方便做消融）
        self.block3 = None
        if separate_third_block:
            self.block3 = STGCNBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                spatial_channels=spatial_channels,
                num_nodes=num_nodes,
                kernel_size=kernel_size,
                use_fft_gate=use_fft_gate,
                detrend=detrend,
            )

        self.last_temporal = FFTGatedTimeBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_fft_gate=use_fft_gate,
            detrend=detrend,
        )

        self.fully = nn.Linear(
            (num_timesteps_input - 2 * 5) * out_channels * int(num_nodes / pred_num),
            num_timesteps_output,
        )

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
