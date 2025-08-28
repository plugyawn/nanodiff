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
    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq_len: int, *,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 use_value_embeds: bool = True, attn_scale: float = 0.12,
                 qk_learned_scale: bool = False):
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = head_dim
        hdim = n_heads * head_dim

        # Merged QKV weights
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim, dtype=torch.bfloat16))
        with torch.no_grad():
            self.qkvo_w[:3].normal_(std=0.5 * (dim ** -0.5))
            self.qkvo_w[3].zero_()

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
            self.value_embeds = nn.Parameter(torch.randn(max_seq_len, hdim).bfloat16() * 0.02)
            self.lambda_v = nn.Parameter(torch.tensor([0.9, 0.1], dtype=torch.bfloat16))

    def _project_qkv(self, x: torch.Tensor):
        qkv = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1))
        B, T, _ = qkv.shape
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).unbind(2)
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
            k = k * self.k_scale.view(1, 1, self.num_heads, 1)

        if self.rotary is not None:
            q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
            k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

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
        y = F.linear(y, self.qkvo_w[3])
        return y


class TwoStreamBlock(nn.Module):
    """Two-stream block: self-attn on xt with cross-attn into x0; shared parameters.
    Keeps a single projection set and computes q from xt and k,v from cat(x0, xt).
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq_len: int,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 attn_scale: float = 0.12, qk_learned_scale: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        hdim = n_heads * head_dim
        self.qkv_proj = nn.Parameter(torch.empty(3, hdim, dim, dtype=torch.bfloat16))
        self.out_proj = nn.Parameter(torch.zeros(dim, hdim, dtype=torch.bfloat16))
        with torch.no_grad():
            self.qkv_proj.normal_(std=0.5 * (dim ** -0.5))

        self.use_qk_norm = use_qk_norm
        self.rotary = Rotary(head_dim, max_seq_len * 2) if use_rotary else None
        self.attn_scale = attn_scale
        self.qk_learned_scale = qk_learned_scale
        if qk_learned_scale:
            self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))
            self.k_scale = nn.Parameter(torch.ones(n_heads, 1, 1, dtype=torch.bfloat16))

    def forward(self, xt: torch.Tensor, x0: torch.Tensor, *, block_mask=None) -> torch.Tensor:
        B, T, D = xt.shape
        xt = xt.to(torch.bfloat16)
        x0 = x0.to(torch.bfloat16)
        # Project q from xt; k,v from concat(x0, xt)
        q = F.linear(xt, self.qkv_proj[0])
        kv_src = torch.cat([xt, x0], dim=1)
        k = F.linear(kv_src, self.qkv_proj[1])
        v = F.linear(kv_src, self.qkv_proj[2])
        # reshape
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, 2 * T, self.n_heads, self.head_dim)
        v = v.view(B, 2 * T, self.n_heads, self.head_dim)

        if self.use_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
        if self.qk_learned_scale:
            # Broadcast per-head learned scales over (B, T, H, D)
            q = q * self.q_scale.view(1, 1, self.n_heads, 1)
            k = k * self.k_scale.view(1, 1, self.n_heads, 1)

        if self.rotary is not None:
            q = self.rotary(q.transpose(1, 2)).transpose(1, 2)
            k = self.rotary(k.transpose(1, 2)).transpose(1, 2)

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
        y = F.linear(y, self.out_proj)
        return y


class TwoStreamTransformerBlock(nn.Module):
    """Full pre-norm Transformer block for two-stream flow.
    Updates xt stream using cross-attn to k/v from xt||x0 and a feedforward.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, max_seq_len: int,
                 use_rotary: bool = True, use_qk_norm: bool = True,
                 use_value_embeds: bool = False, attn_scale: float = 0.12,
                 qk_learned_scale: bool = False, use_swiglu: bool = False,
                 use_local_mixer: bool = False, residual_scale: float = 1.0,
                 use_prenorm: bool = True):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.res_scale = residual_scale
        self.attn = TwoStreamBlock(
            dim, n_heads, head_dim, max_seq_len,
            use_rotary=use_rotary, use_qk_norm=use_qk_norm,
            attn_scale=attn_scale, qk_learned_scale=qk_learned_scale
        )
        if use_prenorm:
            self.rms1 = RMSNorm(dim)
            self.rms2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_mult=4.0, use_dwconv=use_local_mixer) if use_swiglu else MLPFallback(dim, hidden_mult=4.0, relu_squared=True)

    def forward(self, xt: torch.Tensor, x0: torch.Tensor, block_mask=None) -> torch.Tensor:
        h = self.rms1(xt) if self.use_prenorm else xt
        xt = xt + self.res_scale * self.attn(h, x0, block_mask=block_mask)
        h2 = self.rms2(xt) if self.use_prenorm else xt
        xt = xt + self.res_scale * self.mlp(h2)
        return xt
