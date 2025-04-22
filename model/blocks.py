import torch
import torch.nn as nn


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.dwConv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
        self.ln = nn.LayerNorm(in_channels)
        self.pwConv4 = nn.Conv2d(in_channels=in_channels, out_channels=4*in_channels, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.pwConv = nn.Conv2d(in_channels=4*in_channels, out_channels=in_channels, kernel_size=1, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, 16, 65, 65]
        original = x

        residual = self.dwConv(x) # [b, 16, 65, 65]
        residual = residual.permute(0, 2, 3, 1) # [b, 65, 65, 16]
        residual = self.ln(residual)
        residual = residual.permute(0, 3, 1, 2) # [b, 16, 65, 65]
        residual = self.pwConv4(residual) # [b, 64, 65, 65]
        residual = self.gelu(residual)
        residual = self.pwConv(residual) # [b, 16, 65, 65]

        return original + residual