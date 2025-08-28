import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.to(dtype=x.dtype), eps=self.eps)


class SwiGLU(nn.Module):
    """SwiGLU MLP with optional depthwise 1D conv mixer.

    in -> Wg (h) and Wv (h); swish(g) * v -> optional DWConv -> Wo -> out
    Hidden size h defaults to 4/3 * dim, but we typically keep total params
    comparable to GELU-4d by setting h = 2/3 * 4d = 8/3 d if desired.
    We keep h = 4d // 2 = 2d here for simplicity and speed.
    """

    def __init__(self, dim: int, hidden_mult: float = 4.0, use_dwconv: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        hidden = int(dim * hidden_mult)
        # Gate and value projections
        self.wg = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.wv = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        # Optional depthwise conv on the token axis
        self.use_dwconv = use_dwconv
        if use_dwconv:
            # Depthwise conv over sequence: (B, hidden, T)
            self.dw = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False, dtype=dtype)
        # Output projection
        self.wo = nn.Linear(hidden, dim, bias=False, dtype=dtype)

        # Weight decay multipliers similar to original MLP
        self.wg.wd_mul = 2.0
        self.wv.wd_mul = 2.0
        self.wo.wd_mul = 2.0

        # Init
        nn.init.normal_(self.wg.weight, std=0.5 * (dim ** -0.5))
        nn.init.normal_(self.wv.weight, std=0.5 * (dim ** -0.5))
        nn.init.zeros_(self.wo.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        g = self.wg(x)
        v = self.wv(x)
        x = F.silu(g) * v
        if self.use_dwconv:
            # x: (B, T, H) -> (B, H, T)
            x = x.transpose(1, 2)
            x = self.dw(x)
            x = x.transpose(1, 2)
        x = self.wo(x)
        return x


class MLP(nn.Module):
    """Fallback MLP (ReLU^2 or GELU), kept for compatibility."""

    def __init__(self, dim: int, hidden_mult: float = 4.0, relu_squared: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        hidden = int(hidden_mult * dim)
        self.fc_w = nn.Linear(dim, hidden, bias=False, dtype=dtype)
        self.proj_w = nn.Linear(hidden, dim, bias=False, dtype=dtype)
        nn.init.normal_(self.fc_w.weight, std=0.5 * (dim ** -0.5))
        nn.init.zeros_(self.proj_w.weight)
        # match previous wd multipliers
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0
        self.relu_squared = relu_squared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        x = self.fc_w(x)
        if self.relu_squared:
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        x = self.proj_w(x)
        return x


class FiLM(nn.Module):
    """Simple FiLM conditioner producing (gamma, beta) from a conditioning vector.
    We accept sigma and optional block embedding that is already combined upstream.
    """

    def __init__(self, dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, dtype=dtype),
        )

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # c: (B, dim)
        uv = self.net(c)
        gamma, beta = uv.chunk(2, dim=-1)
        return gamma, beta

