import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):

        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):

        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, kernel_size=3):

        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels, kernel_size=kernel_size)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):

        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class STGCN(nn.Module):
    def __init__(self, num_nodes, pred_num, num_features, out_channels,spatial_channels,num_timesteps_input, #4 2 24 3
                 num_timesteps_output, adj_mat,device,kernel_size=3):

        super(STGCN, self).__init__()
        self.pred_num = pred_num
        self.adj = torch.from_numpy(calculate_scaled_laplacian(sym_adj(adj_mat-np.min(adj_mat)))).to(device)
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=out_channels,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes,kernel_size=kernel_size)
        self.block2 = STGCNBlock(in_channels=out_channels, out_channels=out_channels,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes,kernel_size=kernel_size)
        self.last_temporal = TimeBlock(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes/pred_num), num_timesteps_output)


    def forward_features(self, X):
        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        # out3: (B, N, T', C)
        return out3

    def forward(self, X):
        out4 = self.forward_features(X)
        out5 = self.fully(out4.reshape((out4.shape[0], self.pred_num, -1)))
        return out5


