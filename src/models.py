import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

class DownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=64
    ):
        super(DownSampling, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)
    
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int=64,
        out_channels: int=64
    ):
        super(ResBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return shortcut + out
    
class ConcatT(nn.Module):
    def __init__(
        self,
        in_channels: int=64,
        out_channels: int=64,
        kernel_size: int=3,
        stride: int=1,
        padding: int=1
    ):
        super(ConcatT, self).__init__()

        self.model = nn.Conv2d(
            in_channels=in_channels+1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, t, x):
        t_tensor = torch.ones_like(x[:, :1, :, :]) * t
        tx_tensor = torch.concat([t_tensor, x], dim=1)

        return self.model(tx_tensor)
    
class ODEFunc(nn.Module):
    def __init__(
        self,
        n_channels: int=64
    ):
        super(ODEFunc, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatT(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_channels)
        self.conv2 = ConcatT(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=n_channels)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)

        return out
    
class ODEBlock(nn.Module):
    def __init__(
        self,
        odefunc: ODEFunc,
        tol: float,
        use_adjoint: bool
    ):
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc

        self.t = torch.tensor([0., 1.])
        self.tol = tol
        self.solver = odeint_adjoint if use_adjoint else odeint

    def forward(self, x):
        self.t = self.t.type_as(x)
        out = self.solver(
            func=self.odefunc,
            y0=x,
            t=self.t,
            rtol=self.tol,
            atol=self.tol
        )

        # out shape: 2 x Batch size x N channels x H x W
        return out[1]