import torch
import torch.nn as nn
import torch.nn.functional as F


class DualDomainTimeBlock(nn.Module):
    """
    Dual-branch block:
      - Time branch: original temporal conv (TCN-like)
      - Freq branch: rFFT -> learnable conv on frequency axis -> iFFT -> temporal conv (align length)
    Input : (B, N, T, Cin)
    Output: (B, N, T_out, Cout)   where T_out = T - kernel_size + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        freq_kernel_size: int = 5,
        detrend: bool = True,
        eps: float = 1e-8,
        fuse_mode: str = "add",     # "add" or "concat"
    ):
        super().__init__()
        assert fuse_mode in ("add", "concat")
        self.detrend = detrend
        self.eps = eps
        self.kernel_size = kernel_size
        self.fuse_mode = fuse_mode

        # -------- Time branch (keep your original style) --------
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.t_conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.t_conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        # -------- Freq branch (independent learning) --------
        # rfft gives complex -> we represent it as [real, imag] channels => 2*Cin
        # Learn on frequency axis with Conv2d kernel (1, freq_kernel_size)
        # Output 2*Cout channels -> back to complex for irfft
        pad_f = freq_kernel_size // 2
        self.f_conv = nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=2 * out_channels,
            kernel_size=(1, freq_kernel_size),
            padding=(0, pad_f),
            bias=True,
        )

        # After iFFT we are back in time length T (not shrunk).
        # Use a temporal conv to align to T_out (same shrink as time branch).
        self.f_time_align = nn.Conv2d(out_channels, out_channels, (1, kernel_size))

        # Fusion
        if fuse_mode == "add":
            # learnable scaling for freq branch (start small -> stable)
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.fuse_proj = None
        else:
            # concat then 1x1 projection back to Cout
            self.alpha = None
            self.fuse_proj = nn.Conv2d(2 * out_channels, out_channels, kernel_size=(1, 1))

    def _time_branch(self, X_nchw: torch.Tensor) -> torch.Tensor:
        # X_nchw: (B, Cin, N, T)
        temp = self.t_conv1(X_nchw) + torch.sigmoid(self.t_conv2(X_nchw))
        out_time = F.relu(temp + self.t_conv3(X_nchw))  # (B, Cout, N, T_out)
        return out_time

    def _freq_branch(self, X_nchw: torch.Tensor) -> torch.Tensor:
        # X_nchw: (B, Cin, N, T)
        if self.detrend:
            Xc = X_nchw - X_nchw.mean(dim=-1, keepdim=True)
        else:
            Xc = X_nchw

        # rFFT along time -> (B, Cin, N, F) complex
        Xf = torch.fft.rfft(Xc, dim=-1)

        # split to real/imag and stack as channels -> (B, 2*Cin, N, F)
        Xf_ri = torch.cat([Xf.real, Xf.imag], dim=1)

        # learnable transform in frequency domain -> (B, 2*Cout, N, F)
        Yf_ri = self.f_conv(Xf_ri)

        # back to complex -> (B, Cout, N, F)
        Cout = Yf_ri.shape[1] // 2
        Yf = torch.complex(Yf_ri[:, :Cout, :, :], Yf_ri[:, Cout:, :, :])

        # iFFT back to time length T -> (B, Cout, N, T)
        T = X_nchw.shape[-1]
        y_time = torch.fft.irfft(Yf, n=T, dim=-1)

        # align to T_out via temporal conv (shrink) -> (B, Cout, N, T_out)
        y_time = F.relu(self.f_time_align(y_time))
        return y_time

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, T, Cin) -> (B, Cin, N, T)
        X_nchw = X.permute(0, 3, 1, 2)

        out_time = self._time_branch(X_nchw)
        out_freq = self._freq_branch(X_nchw)

        if self.fuse_mode == "add":
            out = out_time + (self.alpha.tanh() * out_freq)  # tanh limits magnitude (stable)
        else:
            out = torch.cat([out_time, out_freq], dim=1)      # (B, 2*Cout, N, T_out)
            out = F.relu(self.fuse_proj(out))                # (B, Cout, N, T_out)

        # back to (B, N, T_out, Cout)
        return out.permute(0, 2, 3, 1)
