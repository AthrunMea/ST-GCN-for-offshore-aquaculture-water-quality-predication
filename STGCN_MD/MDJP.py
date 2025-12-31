# MDJP.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCN_MDJP(nn.Module):
    def __init__(self, trunk, input_step:int = 48, pred_step: int = 48, dropout: float = 0.1, gate_dropout: float = 0.0,n_filters:int = 32):
        super().__init__()
        self.trunk = trunk
        self.pred_step = pred_step
        self.dropout = dropout
        self.dropout = nn.Dropout(0.5)

        self.headU = None
        self.headM = None
        self.headD = None

        self.fc1 = nn.Linear((input_step - 2 * 5) * n_filters * int(5) * 2,
                             pred_step)
        self.fc2 = nn.Linear((input_step - 2 * 5) * n_filters * int(5) * 2,
                             pred_step)
        self.fc3 = nn.Linear((input_step - 2 * 5) * n_filters * int(5) * 2,
                             pred_step)


    def decoder(self,input1,input2,input3):
        score = (input2 * input3) / (input1.size(-1) + 1e-6)  # 用C缩放
        gate = torch.softmax(score, dim=-1)  # 对C做softmax
        out1 = gate * input1
        out = torch.cat([input1, out1], dim=-1)  # 在特征维拼接 -> (B,5,T',2C)
        self.dropout(out)
        return out
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # H: (B, 15, T', C)
        H = self.trunk.forward_features(X)
        x1 = H[:, 0:5, :, :]
        x2 = H[:, 5:10, :, :]
        x3 = H[:, 10:15, :, :]

        out_d1 = self.decoder(x1, x2, x3)
        out_d2 = self.decoder(x2, x3, x1)
        out_d3 = self.decoder(x3, x1, x2)

        out5 = self.fc1(out_d1.reshape(out_d1.shape[0], 1, -1))
        out6 = self.fc2(out_d2.reshape(out_d2.shape[0], 1, -1))
        out7 = self.fc3(out_d3.reshape(out_d3.shape[0], 1, -1))

        out8 = torch.cat((out5, out6, out7), dim=1)
        return out8
