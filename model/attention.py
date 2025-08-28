import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.layers import RMSNorm, SwiGLU, MLP as MLPFallback


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # standard RoPE frequency base
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (B, H, T, D)
        assert self.cos.size(0) >= x.size(-2)
        # Broadcast cos/sin over (B, H, T, D/2) to match x1/x2
        cos = self.cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
        sin = self.sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int, *,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 use_value_embeds: bool = True, attn_scale: float = 0.12,
                 qk_learned_scale: bool = False):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads"
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.head_dim = head_dim
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        # Separate Q, K, V, O projections to support GQA
        self.wq = nn.Linear(dim, q_dim, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(dim, kv_dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(dim, kv_dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(q_dim, dim, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            nn.init.normal_(self.wq.weight, std=0.5 * (dim ** -0.5))
            nn.init.normal_(self.wk.weight, std=0.5 * (dim ** -0.5))
            nn.init.normal_(self.wv.weight, std=0.5 * (dim ** -0.5))
            nn.init.zeros_(self.wo.weight)

        self.rotary = Rotary(head_dim, max_seq_len) if use_rotary else None
        self.use_qk_norm = use_qk_norm
        self.attn_scale = attn_scale
        self.qk_learned_scale = qk_learned_scale
        if qk_learned_scale:
            # per-head learned scales
            self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))
            self.k_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))

        # Optional value embeddings
        self.use_value_embeds = use_value_embeds
        if use_value_embeds:
            self.value_embeds = nn.Parameter(torch.randn(max_seq_len, q_dim).bfloat16() * 0.02)
            self.lambda_v = nn.Parameter(torch.tensor([0.9, 0.1], dtype=torch.bfloat16))

    def _project_qkv(self, x: torch.Tensor):
        q = self.wq(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        k = self.wk(x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim)
        return q, k, v

    def forward(self, x: torch.Tensor, *, block_mask=None) -> torch.Tensor:
        B, T, _ = x.size()
        x = x.to(torch.bfloat16)
        q, k, v = self._project_qkv(x)

        # QK normalization and optional learned temperature per head
        if self.use_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
        if self.qk_learned_scale:
            # Broadcast per-head learned scales over (B, T, H, D)
            q = q * self.q_scale.view(1, 1, self.num_heads, 1)

        if self.rotary is not None:
            q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
            k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        # Delay value-embed mix until after possible KV head expansion

        # Expand K,V heads to full Q heads for grouped-query attention
        if self.num_kv_heads != self.num_heads:
            # repeat each kv head group_size times along head dimension
            k = k.repeat_interleave(self.group_size, dim=2)
            v = v.repeat_interleave(self.group_size, dim=2)
        # Apply learned k scale after expansion so shapes match
        if self.qk_learned_scale:
            k = k * self.k_scale.view(1, 1, self.num_heads, 1)

        # Optional value embeddings after expansion
        if self.use_value_embeds:
            ve = self.value_embeds[:T].view(1, T, self.num_heads, self.head_dim)
            v = self.lambda_v[0] * v + self.lambda_v[1] * ve

        # Attention compute (FlexAttention if mask provided)
        if block_mask is not None:
            from torch.nn.attention.flex_attention import flex_attention
            if T < block_mask.shape[2]:
                adjusted_mask = block_mask._adjust(T, T)
            else:
                adjusted_mask = block_mask
            y = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=adjusted_mask,
                scale=self.attn_scale,
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=self.attn_scale,
                is_causal=True,
            ).transpose(1, 2)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.wo(y)
        return y


class TwoStreamBlock(nn.Module):
    """Two-stream block: self-attn on xt with cross-attn into x0; shared parameters.
    Keeps a single projection set and computes q from xt and k,v from cat(x0, xt).
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 attn_scale: float = 0.12, qk_learned_scale: bool = False):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.head_dim = head_dim
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        self.wq = nn.Linear(dim, q_dim, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(dim, kv_dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(dim, kv_dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(q_dim, dim, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            nn.init.normal_(self.wq.weight, std=0.5 * (dim ** -0.5))
            nn.init.normal_(self.wk.weight, std=0.5 * (dim ** -0.5))
            nn.init.normal_(self.wv.weight, std=0.5 * (dim ** -0.5))

        self.use_qk_norm = use_qk_norm
        self.rotary = Rotary(head_dim, max_seq_len * 2) if use_rotary else None
        self.attn_scale = attn_scale
        self.qk_learned_scale = qk_learned_scale
        if qk_learned_scale:
            self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))
            self.k_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))

    def forward(self, xt: torch.Tensor, x0: torch.Tensor, *, block_mask=None, x0_kv: Optional[tuple]=None, query_from_x0: bool = False) -> torch.Tensor:
        B, T, D = xt.shape
        xt = xt.to(torch.bfloat16)
        x0 = x0.to(torch.bfloat16)
        # Project q from xt; K,V from xt and (optionally cached) x0
        q = self.wq(x0 if query_from_x0 else xt)
        if x0_kv is None:
            kv_src = torch.cat([xt, x0], dim=1)
            k = self.wk(kv_src)
            v = self.wv(kv_src)
        else:
            k0, v0 = x0_kv
            assert k0.dim() == 4 and v0.dim() == 4, "x0_kv must be (k0,v0) with shapes (B,T,H_kv,D)"
            assert k0.size(0) == B and v0.size(0) == B, "x0_kv batch size mismatch"
            assert k0.size(1) == T and v0.size(1) == T, "x0_kv seq len mismatch vs xt/x0"
            assert k0.size(2) == self.n_kv_heads and v0.size(2) == self.n_kv_heads, "x0_kv H_kv mismatch"
            assert k0.size(3) == self.head_dim and v0.size(3) == self.head_dim, "x0_kv head_dim mismatch"
            # K,V for xt tokens
            kx = self.wk(xt)
            vx = self.wv(xt)
            # reshape and concat along sequence axis (xt first, then x0)
            kx = kx.view(B, T, self.n_kv_heads, self.head_dim)
            vx = vx.view(B, T, self.n_kv_heads, self.head_dim)
            k = torch.cat([kx, k0], dim=1)
            v = torch.cat([vx, v0], dim=1)
        # reshape
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, 2 * T, self.n_kv_heads, self.head_dim)
        v = v.view(B, 2 * T, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
        if self.qk_learned_scale:
            # Broadcast per-head learned scales over (B, T, H, D)
            q = q * self.q_scale.view(1, 1, self.n_heads, 1)

        if self.rotary is not None:
            q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
            k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

        # Expand kv heads to match q heads
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.group_size, dim=2)
            v = v.repeat_interleave(self.group_size, dim=2)
        if self.qk_learned_scale:
            k = k * self.k_scale.view(1, 1, self.n_heads, 1)

        # Use flex_attention over concatenated kv; block_mask should be the BD3LM one
        if block_mask is not None:
            from torch.nn.attention.flex_attention import flex_attention
            # block_mask should already reflect 2T length; adjust if needed
            adjusted = block_mask
            y = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=adjusted,
                scale=self.attn_scale,
            ).transpose(1, 2)
        else:
            # Attention to all keys (x0 then xt); no causal flag here
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=self.attn_scale,
                is_causal=False,
            ).transpose(1, 2)

        y = y.contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.wo(y)
        return y


class TwoStreamTransformerBlock(nn.Module):
    """Full pre-norm Transformer block for two-stream flow.
    Updates xt stream using cross-attn to k/v from xt||x0 and a feedforward.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 use_value_embeds: bool = False, attn_scale: float = 0.12,
                 qk_learned_scale: bool = False, use_swiglu: bool = False,
                 use_local_mixer: bool = False, residual_scale: float = 1.0,
                 use_prenorm: bool = True):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.res_scale = residual_scale
        self.attn = TwoStreamBlock(
            dim, n_heads, n_kv_heads, head_dim, max_seq_len,
            use_rotary=use_rotary, use_qk_norm=use_qk_norm,
            attn_scale=attn_scale, qk_learned_scale=qk_learned_scale
        )
        if use_prenorm:
            self.rms1 = RMSNorm(dim)
            self.rms2 = RMSNorm(dim)
        # AdaLN modulation for attention and MLP
        self.adaln = nn.Linear(dim, 6 * dim, bias=True, dtype=torch.bfloat16)
        self.mlp = SwiGLU(dim, hidden_mult=4.0, use_dwconv=use_local_mixer) if use_swiglu else MLPFallback(dim, hidden_mult=4.0, relu_squared=True)

    def forward(self, xt: torch.Tensor, x0: torch.Tensor, block_mask=None, x0_kv: Optional[tuple]=None, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(xt.size(0), xt.size(-1), device=xt.device, dtype=xt.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(cond).unsqueeze(1).chunk(6, dim=-1)
        h = self.rms1(xt) if self.use_prenorm else xt
        h = h * (1 + scale_msa) + shift_msa
        attn_out = self.attn(h, x0, block_mask=block_mask, x0_kv=x0_kv)
        xt = xt + self.res_scale * (gate_msa * attn_out)
        h2 = self.rms2(xt) if self.use_prenorm else xt
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(h2)
        xt = xt + self.res_scale * (gate_mlp * mlp_out)
        return xt

    @torch.no_grad()
    def project_x0_kv(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project x0 tokens to (k0, v0) before normalization/rotary.
        Returns tensors of shape (B, T, H_kv, D) in bf16.
        """
        B, T, D = x0.shape
        k0 = self.attn.wk(x0).view(B, T, self.attn.n_kv_heads, self.attn.head_dim)
        v0 = self.attn.wv(x0).view(B, T, self.attn.n_kv_heads, self.attn.head_dim)
        return k0.to(torch.bfloat16), v0.to(torch.bfloat16)
