import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import sym_adj, calculate_normalized_laplacian, calculate_scaled_laplacian


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
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
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        #64 24 4 2 (input)-> 64 4 8 64 (t)-> 4 64 8 64 jklm -> 64 4 8 64 (lfs) -> 64 4 8 16 (t2)
         -> 64 16 4 8 -> 64 64 4 3 -> 64 4 3 64(t3)
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, pred_num, num_features, out_channels,spatial_channels,num_timesteps_input, #4 2 24 3
                 num_timesteps_output, adj_mat,device,kernel_size=3):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        # self.adj = torch.from_numpy(sym_adj(adj_mat-np.min(adj_mat))).to(device)
        self.pred_num = pred_num
        self.adj = torch.from_numpy(calculate_scaled_laplacian(sym_adj(adj_mat-np.min(adj_mat)))).to(device)
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=out_channels,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes,kernel_size=kernel_size)
        self.block2 = STGCNBlock(in_channels=out_channels, out_channels=out_channels,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes,kernel_size=kernel_size)
        self.block3 = STGCNBlock(in_channels=out_channels, out_channels=out_channels,
                                 spatial_channels=spatial_channels, num_nodes=num_nodes, kernel_size=kernel_size)
        self.last_temporal = TimeBlock(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size)
        self.fully = nn.Linear((num_timesteps_input - 2 * 7) * out_channels * int(num_nodes/pred_num), num_timesteps_output)
        #nn.Linear((num_timesteps_input - 2 * 5) * out_channels * int(num_nodes/pred_num),
                     #          num_timesteps_output)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out5 = self.block3(out2, self.adj)
        out3 = self.last_temporal(out5)
        out4 = self.fully(out3.reshape((out3.shape[0], self.pred_num, -1)))
        return out4


