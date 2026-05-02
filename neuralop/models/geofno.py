import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from .base_model import BaseModel


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class FNO2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        in_channels,
        out_channels,
        is_mesh=True,
        s1=40,
        s2=40,
        n_fourier_layers: int = 5,
    ):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains multiple layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. n Fourier layers of the integral operators u' = (W + K)(u).
            W defined by learned 1x1 conv; K defined by self.spectral_layers .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2
        assert n_fourier_layers >= 2, "Need at least input and output Fourier layers"
        self.n_fourier_layers = n_fourier_layers

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        spectral_layers = []
        spectral_layers.append(
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        )
        for _ in range(self.n_fourier_layers - 2):
            spectral_layers.append(
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            )
        spectral_layers.append(
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        )
        self.spectral_layers = nn.ModuleList(spectral_layers)

        self.pointwise_layers = nn.ModuleList(
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.n_fourier_layers - 2)
        )
        self.grid_convs = nn.ModuleList(
            nn.Conv2d(2, self.width, 1) for _ in range(self.n_fourier_layers - 1)
        )
        self.b_out = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def set_spatial_size(self, s1: int, s2: int):
        """Synchronize stored spatial sizes with runtime input."""
        self.s1, self.s2 = s1, s2
        if self.spectral_layers:
            first = self.spectral_layers[0]
            last = self.spectral_layers[-1]
            if hasattr(first, "s1"):
                first.s1, first.s2 = s1, s2
            if hasattr(last, "s1"):
                last.s1, last.s2 = s1, s2

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc = self.spectral_layers[0](u, x_in=x_in, iphi=iphi, code=code)
        uc = uc + self.grid_convs[0](grid)
        uc = F.gelu(uc)

        for idx in range(1, self.n_fourier_layers - 1):
            uc1 = self.spectral_layers[idx](uc)
            uc2 = self.pointwise_layers[idx - 1](uc)
            uc3 = self.grid_convs[idx](grid)
            uc = uc1 + uc2 + uc3
            uc = F.gelu(uc)

        u = self.spectral_layers[-1](uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b_out(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()
        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 2)
        self.center = torch.tensor([0.5,0.5], device="cuda").reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)

    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = F.gelu(xd)
        xd = self.fc2(xd)
        xd = F.gelu(xd)
        xd = self.fc3(xd)
        return x + x * xd
    
class GeoFNO2d(BaseModel):
    """
    Public API:
        Inputs:  x_img  (B, 1, H, W)   single-channel image / waveform map
        Optional: code  (B, C_code)    extra conditioning for IPHI (default C_code=42)

    Flow:
        1) Build coordinate grid in [0,1]^2, flattened to (B, N, 2), N=H*W
        2) Flatten image to (B, N, 1)
        3) Concatenate (u, x, y) -> (B, N, 3), feed FNO2d with x_in/x_out and IPHI
        4) FNO2d outputs (B, N, out_channels) -> reshape to (B, out_channels, H, W)
    """
    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        out_channels: int = 1,
        code_dim: int = 42,
        is_mesh: bool = True,
        s1: int = 40,
        s2: int = 40,
        n_fourier_layers: int = 5,
    ):
        super().__init__()
        # FNO expects 3 input channels: (u, x, y)
        self.in_channels = 3
        self.code_dim = code_dim

        # Geometry map (IPHI)
        self.iphi = IPHI(width=width)

        # Main FNO; s1/s2 defaults are synced to input H/W in forward
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=self.in_channels,
            out_channels=out_channels,
            is_mesh=is_mesh,
            s1=s1,
            s2=s2,
            n_fourier_layers=n_fourier_layers,
        )

    @torch.no_grad()
    def _make_grid(self, B: int, H: int, W: int, device, dtype=torch.float32):
        """Build a [0,1]^2 grid and flatten to (B, N, 2)."""
        xs = torch.linspace(0., 1., H, device=device, dtype=dtype)
        ys = torch.linspace(0., 1., W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="xy")   # gy:(W,H), gx:(W,H)
        # indexing="xy": dim0 is x (width W), dim1 is y (height H); transpose to (H,W) layout
        gx = gx.t()   # (H,W)
        gy = gy.t()   # (H,W)
        grid = torch.stack([gx, gy], dim=-1).view(1, H * W, 2).repeat(B, 1, 1)  # (B, N, 2)
        return grid

    def _sync_sizes(self, H: int, W: int):
        """Sync FNO2d / SpectralConv2d s1/s2 to current spatial size."""
        self.fno.set_spatial_size(H, W)

    def _ensure_iphi_buffers(self, device, dtype):
        """Align IPHI buffers (e.g. center/B) to current device/dtype."""
        if hasattr(self.iphi, "center"):
            self.iphi.center = self.iphi.center.to(device=device, dtype=dtype)
        if hasattr(self.iphi, "B"):
            self.iphi.B = self.iphi.B.to(device=device, dtype=dtype)

    def forward(self, x_img: torch.Tensor, code: torch.Tensor | None = None, **kwargs):
        """
        x_img: (B, 1, H, W)
        code : (B, code_dim) or None
        return: (B, out_channels, H, W)
        """
        assert x_img.dim() == 4 and x_img.size(1) == 1, "x_img must be single-channel with shape (B,1,H,W)"
        B, _, H, W = x_img.shape
        device = x_img.device
        f_dtype = x_img.dtype if x_img.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32

        # 1) Coordinates + flatten
        x_flat = self._make_grid(B, H, W, device=device, dtype=f_dtype)     # (B, N, 2)

        # 2) Flatten image
        u_flat = x_img.view(B, 1, H * W).transpose(1, 2).to(dtype=f_dtype)  # (B, N, 1)

        # 3) Concat (u, x, y) -> (B, N, 3)
        feat = torch.cat([u_flat, x_flat], dim=-1)                           # (B, N, 3)

        # 4) Sync FNO s1/s2
        self._sync_sizes(H, W)

        # 5) Align IPHI buffers
        self._ensure_iphi_buffers(device=device, dtype=f_dtype)

        # 6) Optional code
        if code is not None:
            assert code.dim() == 2 and code.size(0) == B, "code must have shape (B, code_dim)"
            code_in = code.to(device=device, dtype=f_dtype)
        else:
            code_in = None

        # 7) FNO with explicit x_in/x_out/iphi
        y = self.fno(
            u=feat,               # (B, N, 3)
            code=code_in,         # (B, code_dim) or None
            x_in=x_flat,          # (B, N, 2)
            x_out=x_flat,         # (B, N, 2)
            iphi=self.iphi,
        )                          # → (B, N, out_channels)

        # 8) Reshape to (B, out_channels, H, W)
        out = y.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out
