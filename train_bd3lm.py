#!/usr/bin/env python3
"""
BD3LM Speedrun: Discrete Block Diffusion Language Model
Aligns training with official BD3LM (SUBS) and uses NanoGPT FineWeb data.
"""

import os
import sys
import time
import uuid
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, asdict
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
import numpy as np
from tqdm import tqdm
import wandb
import glob
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Decoding requires transformers
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    if torch.cuda.is_available():
        torch.empty(1, device="cuda", requires_grad=True).backward()  # Fix for some systems
except Exception:
    pass

# -----------------------------------------------------------------------------
# Configuration

@dataclass
class Config:
    # Model
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    vocab_size: int = 50304  # Original vocabulary
    max_seq_len: int = 1024
    
    # Block Diffusion (Discrete)
    block_size: int = 16
    diffusion_steps: int = 1000
    noise_schedule: str = "loglinear"  # align with official default
    sigma_min: float = 1e-4
    sigma_max: float = 20.0
    parameterization: str = "subs"  # subs | sedd | ar
    cross_attn: bool = True
    mdlm_loss_scale: bool = False
    antithetic_sampling: bool = False
    sampling_eps_min: float = 1e-3
    sampling_eps_max: float = 1.0
    eval_var_min: bool = False
    clip_search_widths: tuple = (0.25, 0.5, 0.75)
    first_hitting: bool = True
    
    # Training
    batch_size: int = 64  # global batch size (will be sharded across ranks)
    learning_rate: float = 0.02
    # AdamW often needs a much smaller LR; keep separate knob
    adamw_lr: float = 3e-4
    weight_decay: float = 0.01
    momentum: float = 0.95
    max_steps: int = 125_000
    warmup_steps: int = 256
    cooldown_frac: float = 0.45
    optimizer: str = "mixed"  # mixed | muon | adamw
    
    # Architecture
    use_flex_attention: bool = True
    use_rotary: bool = True
    use_qk_norm: bool = True
    use_relu_squared: bool = True
    use_value_embeds: bool = True
    use_unet_skips: bool = True
    softcap: float = 15.0
    compile_model: bool = False
    compile_zeropower: bool = False
    
    # System
    device: str = "cuda"
    use_wandb: bool = False
    project_name: str = "bd3lm-speedrun"
    run_name: str = None
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 1000
    samples_per_eval: int = 1
    sample_length: int = 1024
    top_p: float = 0.95
    
    # Data (reuse NanoGPT cached FineWeb .bin shards)
    train_files: str = "./data/fineweb_train_*.bin"
    align_to_bos: bool = True
    val_files: str = "./data/fineweb_val_*.bin"
    val_tokens: int = 10_485_760  # 10M tokens like NanoGPT defaults
    
    @property
    def vocab_size_with_mask(self):
        """Vocabulary size including MASK token"""
        return self.vocab_size + 1
    
    @property
    def mask_token_id(self):
        """ID for the MASK token"""
        return self.vocab_size

# -----------------------------------------------------------------------------
# Muon Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalization"""
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, compile_zeropower: bool = False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        # Group deterministically by (shape, dtype) in parameter order.
        # Using a set here would lead to nondeterministic order across ranks
        # and can misalign reduce_scatter inputs (dtype mismatch errors).
        params = list(params)
        grouped = {}
        for p in params:
            key = (tuple(p.shape), p.dtype)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(p)
        param_groups = [dict(params=v) for v in grouped.values()]
        super().__init__(param_groups, defaults)
        # Optionally compile Newton–Schulz for first-class performance; off by default to avoid long first-step stalls
        self._zeropower = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead") if compile_zeropower else zeropower_via_newtonschulz5
    
    @torch.no_grad()
    def step(self):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Distributed Muon with reduce_scatter / all_gather
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            reduce_scatter_futures = []
            all_gather_futures = []
            # phase 1: reduce_scatter grads in size groups
            for group in self.param_groups:
                params = group["params"]
                grad_pad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                # pad with zeros to multiples of world_size
                if len(grad_pad) % world_size != 0:
                    pad_num = world_size - (len(grad_pad) % world_size)
                    grad_pad += [torch.zeros_like(params[-1])] * pad_num
                for base in range(0, len(grad_pad), world_size):
                    buf = torch.empty_like(grad_pad[base + rank])
                    fut = dist.reduce_scatter(buf, grad_pad[base:base+world_size], op=dist.ReduceOp.AVG, async_op=True).get_future()
                    reduce_scatter_futures.append((fut, group, base, buf))
            # phase 2: local update on owned shards, then all_gather weights
            for fut, group, base, buf in reduce_scatter_futures:
                fut.wait()
                params = group["params"]
                momentum = group["momentum"]
                idx = base + rank
                if idx >= len(params):
                    continue
                p = params[idx]
                grad = buf
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                # Handle per-parameter effective LR scaling robustly for any tensor rank
                if p.ndim >= 2:
                    shape_scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                else:
                    shape_scale = 1.0
                eff_lr = group["lr"] * shape_scale * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                p.mul_(1 - eff_weight_decay)
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                # Apply Newton–Schulz orthogonalization; for vectors, lift to 2D
                if grad.ndim >= 2:
                    v = self._zeropower(grad.bfloat16(), 5)
                else:
                    v = self._zeropower(grad.unsqueeze(0).bfloat16(), 5).squeeze(0)
                # Apply update in the parameter's dtype
                v = v.to(dtype=p.dtype)
                p.add_(other=v, alpha=-eff_lr)
                # async gather to other ranks
                # gather into a temp tensor of same shape as p (no need to pad)
                if hasattr(dist, 'all_gather_into_tensor'):
                    # all_gather_into_tensor expects [world_size, *p.shape]
                    out = torch.empty((world_size, *p.shape), device=p.device, dtype=p.dtype)
                    fut2 = dist.all_gather_into_tensor(out, p, async_op=True).get_future()
                    all_gather_futures.append(fut2)
                else:
                    tmp = [torch.empty_like(p) for _ in range(world_size)]
                    fut2 = dist.all_gather(tmp, p, async_op=True).get_future()
                    all_gather_futures.append(fut2)
            if all_gather_futures:
                torch.futures.collect_all(all_gather_futures).wait()
            return
        # Fallback: single-process Muon
        for group in self.param_groups:
            params = group["params"]
            momentum = group["momentum"]
            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                # Handle per-parameter effective LR scaling robustly for any tensor rank
                if p.ndim >= 2:
                    shape_scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                else:
                    shape_scale = 1.0
                eff_lr = group["lr"] * shape_scale * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                p.mul_(1 - eff_weight_decay)
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                # Apply Newton–Schulz orthogonalization; for vectors, lift to 2D
                if grad.ndim >= 2:
                    v = self._zeropower(grad.bfloat16(), 5)
                else:
                    v = self._zeropower(grad.unsqueeze(0).bfloat16(), 5).squeeze(0)
                v = v.to(dtype=p.dtype)
                p.add_(other=v, alpha=-eff_lr)

# -----------------------------------------------------------------------------
# Model Components

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)
    
    def forward(self, x):
        assert self.cos.size(0) >= x.size(-3)
        cos = self.cos[None, :x.size(-3), None, :]
        sin = self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

def create_block_diff_mask(seq_len: int, block_size: int, device=None):
    """Creates block diffusion mask for FlexAttention"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def block_mask_fn(b, h, q_idx, kv_idx):
        block_q = q_idx // block_size
        block_kv = kv_idx // block_size
        block_diagonal = (block_q == block_kv)
        block_causal = (block_q > block_kv)
        return block_diagonal | block_causal
    
    return create_block_mask(block_mask_fn, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

def create_bd3lm_mask(n_tokens: int, block_size: int, device=None):
    """BD3LM 3-part mask over concatenated xt||x0 of total length 2*n.
    Implements M_BD | M_OBC | M_BC as in the reference implementation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = n_tokens
    def block_mask_fn(b, h, q_idx, kv_idx):
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)
        block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = ((block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0))
        block_causal = ((block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1))
        return block_diagonal | offset_block_causal | block_causal
    total_len = 2 * n
    return create_block_mask(block_mask_fn, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len, device=device)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        hdim = config.n_heads * config.head_dim
        
        # Merged QKV weights
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, config.dim, dtype=torch.bfloat16))
        self.qkvo_w.data[:3].normal_(std=0.5 * (config.dim ** -0.5))
        self.qkvo_w.data[3].zero_()
        
        # Cross-attn concatenates xt||x0, so effective max seq length can be 2x
        effective_max_seq_len = config.max_seq_len * (2 if config.cross_attn else 1)
        self.rotary = Rotary(config.head_dim, effective_max_seq_len) if config.use_rotary else None
        self.attn_scale = 0.12
        
        # Value embeddings
        if config.use_value_embeds:
            self.value_embeds = nn.Parameter(
                torch.randn(effective_max_seq_len, hdim).bfloat16() * 0.02
            )
            self.lambda_v = nn.Parameter(torch.tensor([0.9, 0.1]))
    
    def forward(self, x, block_mask=None):
        B, T, C = x.size()
        x = x.to(torch.bfloat16)
        
        # QKV projection
        qkv = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1))
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).unbind(2)
        
        # QK normalization
        if self.config.use_qk_norm:
            q, k = norm(q), norm(k)
        
        # Rotary embeddings
        if self.rotary:
            q = self.rotary(q)
            k = self.rotary(k)
        
        # Value embeddings
        if self.config.use_value_embeds:
            ve = self.value_embeds[:T].view(1, T, self.num_heads, self.head_dim)
            v = self.lambda_v[0] * v + self.lambda_v[1] * ve
        
        # Attention
        if self.config.use_flex_attention and block_mask is not None:
            if T < block_mask.shape[2]:
                adjusted_mask = block_mask._adjust(T, T)
            else:
                adjusted_mask = block_mask
            
            y = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=adjusted_mask,
                scale=self.attn_scale
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=self.attn_scale,
                is_causal=True
            ).transpose(1, 2)
        
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hdim = 4 * config.dim
        self.fc_w = nn.Parameter(torch.empty(hdim, config.dim, dtype=torch.bfloat16))
        self.proj_w = nn.Parameter(torch.zeros(config.dim, hdim, dtype=torch.bfloat16))
        self.fc_w.data.normal_(std=0.5 * (config.dim ** -0.5))
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0
        self.use_relu_squared = config.use_relu_squared
    
    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = F.linear(x, self.fc_w)
        if self.use_relu_squared:
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        x = F.linear(x, self.proj_w)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config) if layer_idx != 7 else None
        self.mlp = MLP(config)
    
    def forward(self, x, x0, block_mask):
        if self.attn is not None:
            x = x + self.attn(x, block_mask)
        
        if self.config.use_unet_skips and x0 is not None:
            if self.layer_idx < self.config.n_layers // 2:
                x = x + 0.1 * x0
        
        x = x + self.mlp(x)
        return x

class BlockDiffusionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Include MASK token in vocabulary
        vocab_size = config.vocab_size_with_mask
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, config.dim)
        # Cross-attn doubles sequence length (xt||x0), expand pos embeddings accordingly
        pos_len = config.max_seq_len * (2 if config.cross_attn else 1)
        self.pos_emb = nn.Embedding(pos_len, config.dim)
        
        # Time embedding for diffusion (condition on sigma)
        self.time_emb = nn.Sequential(
            nn.Linear(1, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()  # zero-init head
        
        # Block mask for attention
        if config.use_flex_attention:
            # Build masks on the same device as the module to avoid cross-device issues in DDP
            module_device = torch.device(config.device if isinstance(config.device, str) else config.device)
            self.block_mask = create_block_diff_mask(config.max_seq_len, config.block_size, device=module_device)
            # prebuild BD3LM mask for xt||x0 (total length = 2 * max_seq_len)
            self.bd3lm_mask = create_bd3lm_mask(config.max_seq_len, config.block_size, device=module_device)
        else:
            self.block_mask = None
            self.bd3lm_mask = None

        # Learnable gating for UNet-like skips
        assert config.n_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(config.n_layers // 2, dtype=torch.bfloat16))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x, sigma=None, use_bd3lm_mask: bool = False):
        B, T = x.shape
        device = x.device
        # Guard against positional index overflow
        assert T <= self.pos_emb.num_embeddings, (
            f"Sequence length {T} exceeds positional embeddings {self.pos_emb.num_embeddings}. "
            f"Consider enabling cross-attn expansion or increasing max_seq_len."
        )
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = (tok_emb + pos_emb).to(torch.bfloat16)
        
        # Sigma conditioning embedding
        if sigma is not None:
            # Map per-example sigma (B,1) to a time embedding (B,1,dim)
            # and broadcast across the sequence length T.
            s = sigma.float().unsqueeze(-1)  # (B,1,1)
            t_emb = self.time_emb(s)         # (B,1,dim)
            x = x + t_emb                    # broadcasts to (B,T,dim)
        
        x0 = x if self.config.use_unet_skips else None
        
        # Transformer blocks
        # Select appropriate mask
        if self.config.use_flex_attention:
            attn_mask = self.bd3lm_mask if use_bd3lm_mask else self.block_mask
            # Ensure the attention mask matches the current sequence length T.
            # Torch's BlockMask._adjust is unreliable across versions; rebuild when sizes differ.
            T_eff = x.size(1)
            if attn_mask is not None:
                mask_len = attn_mask.shape[2]
                if T_eff != mask_len:
                    if use_bd3lm_mask:
                        # x is xt||x0, so n_tokens is half the effective length
                        n_tok = max(1, T_eff // 2)
                        attn_mask = create_bd3lm_mask(n_tok, self.config.block_size, device=device)
                    else:
                        attn_mask = create_block_diff_mask(T_eff, self.config.block_size, device=device)
        else:
            attn_mask = None

        # U-Net style skip connections with gating
        n = len(self.blocks) // 2
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x, x0, attn_mask)
            if i < n:
                skips.append(x)
            else:
                x = x + self.skip_weights[i - n] * skips.pop()
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if self.config.softcap > 0:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        
        return logits

# -----------------------------------------------------------------------------
# Discrete Block Diffusion Trainer

class BlockDiffusionTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.mask_token_id = config.mask_token_id
        
        # Setup noise schedule (align with official BD3LM)
        self.noise = self._get_noise(config.noise_schedule)
        
        # Optimizers
        self.optimizer_muon = None
        self.optimizer_adam = None

        if config.optimizer == "mixed":
            # Muon for 2D params (excluding embeddings/head), AdamW for others
            muon_params = []
            adam_params = []
            for name, p in model.named_parameters():
                if (p.ndim >= 2) and (not name.startswith('token_emb')) and (not name.startswith('lm_head')):
                    muon_params.append(p)
                else:
                    adam_params.append(p)
            if len(muon_params):
                self.optimizer_muon = Muon(
                    muon_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    momentum=config.momentum,
                    compile_zeropower=config.compile_zeropower,
                )
            if len(adam_params):
                self.optimizer_adam = torch.optim.AdamW(
                    # AdamW needs a much smaller LR than Muon
                    adam_params, lr=config.adamw_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay
                )
        elif config.optimizer == "muon":
            self.optimizer_muon = Muon(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                compile_zeropower=config.compile_zeropower,
            )
        elif config.optimizer == "adamw":
            self.optimizer_adam = torch.optim.AdamW(
                model.parameters(), lr=config.adamw_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Metrics
        self.step = 0
        self.start_time = time.time()
        
        # WandB init handled in main() on rank 0
        
        # Checkpoint dir (set in main)
        self.ckpt_dir: Optional[Path] = None
        
    # ---------------------- Noise schedules (official) ----------------------
    class _Noise:
        def __call__(self, t):
            return self.compute_loss_scaling_and_move_chance(t)

    class _LogLinearNoise(_Noise):
        def __init__(self, eps=1e-3, sigma_max=20.0):
            self.eps = eps
            self.sigma_max = torch.tensor(sigma_max)
        def rate_noise(self, t):
            return (1 - self.eps) / (1 - (1 - self.eps) * t)
        def total_noise(self, t):
            return -torch.log1p(-(1 - self.eps) * t)
        def compute_loss_scaling_and_move_chance(self, t):
            # Return loss scaling and move chance p in [eps, 1]
            loss_scaling = - 1.0 / torch.clamp(t, min=self.eps)
            return loss_scaling, t

    class _CosineNoise(_Noise):
        def __init__(self, eps=1e-3):
            self.eps = eps
            self.sigma_max = None
        def compute_loss_scaling_and_move_chance(self, t):
            cos = - (1 - self.eps) * torch.cos(t * math.pi / 2)
            sin = - (1 - self.eps) * torch.sin(t * math.pi / 2)
            move_chance = cos + 1
            loss_scaling = sin / (move_chance + self.eps) * math.pi / 2
            return loss_scaling, move_chance

    def _get_noise(self, kind: str):
        if kind == "loglinear":
            return self._LogLinearNoise(eps=1e-3, sigma_max=self.config.sigma_max)
        elif kind == "cosine":
            return self._CosineNoise(eps=1e-3)
        else:
            raise ValueError(f"Unknown noise schedule: {kind}")

    @staticmethod
    def _sigma_from_p(p, sigma_max: Optional[float]):
        sigma = -torch.log1p(-p)
        if sigma_max is not None:
            sigma = torch.clamp_max(sigma, torch.as_tensor(sigma_max, device=sigma.device))
        return sigma
    
    def _sample_t(self, batch_size, seqlen, device):
        # per-block uniform sampling (with optional antithetic)
        bs = self.config.block_size
        assert seqlen % bs == 0
        num_blocks = seqlen // bs
        t_blocks = torch.rand(batch_size, num_blocks, device=device)
        if self.config.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device) / (batch_size * num_blocks)
            offset = offset.view(batch_size, num_blocks)
            t_blocks = (t_blocks / (batch_size * num_blocks) + offset) % 1
        t = t_blocks.repeat_interleave(bs, dim=1)  # (B, T)
        return t

    def _subs_parameterization(self, logits, xt):
        # logits: (B,T,V) unnormalized; xt: (B,T)
        B, T, V = logits.shape
        logits[:, :, self.mask_token_id] = -1e30  # never predict MASK
        # normalize to log probs
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # set unmasked tokens to prob 1 at their index, -inf elsewhere
        unmasked = (xt != self.mask_token_id)
        if unmasked.any():
            logits[unmasked] = -1e30
            logits[unmasked, xt[unmasked]] = 0.0
        return logits

    def _score_entropy(self, log_score, sigma, xt, x0):
        """SEDD loss term per token following reference implementation."""
        masked = xt == self.mask_token_id
        if not masked.any():
            return torch.zeros_like(xt, dtype=torch.float32)
        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1.0 / torch.clamp(expsig_minus_1[masked], min=1e-6)
        words = x0[masked]
        neg_term = q_ratio * torch.gather(log_score[masked], -1, words[..., None]).squeeze(-1)
        score = log_score[masked].exp()
        if self.mask_token_id == self.config.vocab_size_with_mask - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_token_id].sum(dim=-1) + score[:, self.mask_token_id + 1 :].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)
        entropy = torch.zeros_like(xt, dtype=torch.float32)
        entropy[masked] = pos_term - neg_term + const
        return entropy

    def _resample_bounds(self, x, xt, move, p, bs, eps_min, eps_max):
        B, L = x.shape
        num_blocks = L // bs
        xt_blocks = xt.view(B, num_blocks, bs)
        # resample until masked fraction per block in [eps_min, eps_max]
        while True:
            frac = (xt_blocks == self.mask_token_id).float().mean(dim=-1)
            bad_low = frac < eps_min
            bad_high = frac > eps_max
            bad = bad_low | bad_high
            if not bad.any():
                break
            regen_idx = bad.repeat_interleave(bs, dim=-1)
            move[regen_idx] = (torch.rand_like(move) <= p)[regen_idx]
            xt = torch.where(move, self.mask_token_id, x)
            xt_blocks = xt.view(B, num_blocks, bs)
        return xt

    def corrupt_tokens(self, x, p, eps_min=None, eps_max=None):
        """Apply discrete masking q(x_t|x_0) with per-token mask prob p in [0,1]."""
        B, L = x.shape
        device = x.device
        move = torch.rand(B, L, device=device) <= p
        # don't mask BOS at position 0
        move[:, 0] = False
        xt = torch.where(move, torch.full_like(x, self.mask_token_id), x)
        if self.config.block_size > 1 and self.config.sampling_eps_min > 1e-3 and (eps_min is not None and eps_max is not None):
            xt = self._resample_bounds(x, xt, move, p, self.config.block_size, eps_min, eps_max)
        return xt
    
    def compute_loss(self, x):
        """Compute BD3LM objective according to parameterization."""
        B, T = x.shape
        device = x.device

        if self.config.parameterization == 'ar':
            # next-token prediction
            inputs = x[:, :-1]
            targets = x[:, 1:]
            logits = self.model(inputs, sigma=None, use_bd3lm_mask=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='mean')
            return loss

        # sample per-block times and get noise schedule terms
        t = self._sample_t(B, T, device)  # (B,T)
        loss_scale, move_chance = self.noise(t)
        # compute sigma (per-example mean over tokens)
        sigma = self._sigma_from_p(move_chance.mean(dim=1, keepdim=True), getattr(self.noise, 'sigma_max', None))

        # optional MDLM loss-scale override
        if self.config.mdlm_loss_scale:
            # follow reference: sigma=total_noise(t), dsigma=rate_noise(t)
            sigma = self.noise.total_noise(t.mean(dim=1, keepdim=True))
            dsigma = self.noise.rate_noise(t.mean(dim=1, keepdim=True))
            move_chance = 1 - torch.exp(-sigma)
            loss_scale = - (dsigma / torch.expm1(sigma))

        # corrupt tokens
        xt = self.corrupt_tokens(x, move_chance, self.config.sampling_eps_min, self.config.sampling_eps_max)

        # model forward: if cross_attn, feed xt||x0 and use 3-part mask
        if self.config.cross_attn:
            x_input = torch.cat([xt, x], dim=1)
            logits = self.model(x_input, sigma, use_bd3lm_mask=True)
            # decode from xt queries (first T positions), per paper
            logits = logits[:, :T]
        else:
            logits = self.model(xt, sigma)

        if self.config.parameterization == 'subs':
            log_probs = self._subs_parameterization(logits, xt)
            log_p_theta = torch.gather(log_probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
            masked = (xt == self.mask_token_id)
            # ensure there is at least one masked token per batch
            num_masked = int(masked.sum().item())
            assert num_masked > 0, "No masked tokens in batch; adjust noise schedule"
            loss_mat = loss_scale * log_p_theta * masked.float()
            loss = loss_mat.sum() / num_masked
            return loss
        elif self.config.parameterization == 'sedd':
            # compute dsigma as in reference
            # use dsigma = -loss_scale * expm1(sigma)
            dsigma = - loss_scale * torch.expm1(sigma)
            log_score = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            loss_mat = self._score_entropy(log_score, sigma, xt, x)
            loss = (dsigma.squeeze(-1)[:, None] * loss_mat).sum() / (B * T)
            return loss
        else:
            raise ValueError(f"Unknown parameterization: {self.config.parameterization}")
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        if self.optimizer_muon is not None:
            self.optimizer_muon.zero_grad(set_to_none=True)
        if self.optimizer_adam is not None:
            self.optimizer_adam.zero_grad(set_to_none=True)

        # Simple LR schedule: linear warmup, then hold, cosine cooldown
        def _lr_scale(step, max_steps, warmup_steps, cooldown_frac):
            if max_steps <= 0:
                return 1.0
            cooldown_start = int(max_steps * (1.0 - cooldown_frac))
            if step < warmup_steps:
                return max(1e-3, step / max(1, warmup_steps))
            if step < cooldown_start:
                return 1.0
            # cosine to zero across the cooldown window
            denom = max(1, max_steps - cooldown_start)
            t = (step - cooldown_start) / denom
            return 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))

        scale = _lr_scale(self.step, self.config.max_steps, self.config.warmup_steps, self.config.cooldown_frac)
        # Apply scheduled LRs per optimizer family
        if self.optimizer_muon is not None:
            for g in self.optimizer_muon.param_groups:
                base = self.config.learning_rate
                g["lr"] = base * scale
        if self.optimizer_adam is not None:
            for g in self.optimizer_adam.param_groups:
                base = self.config.adamw_lr
                g["lr"] = base * scale

        loss = self.compute_loss(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # step optimizers
        if self.optimizer_muon is not None and len(list(self.optimizer_muon.param_groups[0]['params'])):
            self.optimizer_muon.step()
        if self.optimizer_adam is not None and len(list(self.optimizer_adam.param_groups[0]['params'])):
            self.optimizer_adam.step()
        self.step += 1
        
        return loss.item()

    # ------------------------------ Checkpointing ------------------------------
    def save_checkpoint(self, model, rank: int = 0):
        if rank != 0 or self.ckpt_dir is None:
            return
        try:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            state = {
                'model_state': module.state_dict(),
                'config': asdict(self.config),
                'step': self.step,
                'timestamp': datetime.utcnow().isoformat()
            }
            ckpt_path = self.ckpt_dir / f"ckpt_step{self.step:06d}.pt"
            torch.save(state, ckpt_path)
        except Exception as e:
            print(f"[warn] Failed to save checkpoint at step {self.step}: {e}")

    # ------------------------------- Sampling ---------------------------------
    @torch.no_grad()
    def _decode_tokens(self, ids: torch.Tensor) -> List[str]:
        # ids: (B, T)
        tok = AutoTokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        return tok.batch_decode(ids.tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def sample_ar(self, model, length: int, num_samples: int = 1, top_p: float = 0.95) -> List[str]:
        device = self.device
        model.eval()
        B = num_samples
        # Start with BOS token id if present (GPT-2 uses 50256)
        bos_id = 50256
        x = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(length - 1):
            logits = model(x, sigma=None, use_bd3lm_mask=False)
            next_logits = logits[:, -1]  # (B, V)
            # nucleus sampling
            probs = next_logits.float().softmax(dim=-1)
            mask_id = self.mask_token_id
            # disallow MASK and ids beyond GPT-2 vocab for decoding
            probs[:, mask_id] = 0.0
            decode_limit = 50257  # GPT-2 vocab size
            if probs.size(-1) > decode_limit:
                probs[:, decode_limit:] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                # ensure at least one token
                mask[..., 0] = True
                probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                probs = probs / probs.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs, num_samples=1)
                next_ids = sorted_idx.gather(-1, idx)
            else:
                next_ids = torch.argmax(probs, dim=-1, keepdim=True)
            x = torch.cat([x, next_ids], dim=1)
        return self._decode_tokens(x[:, :length])

    @torch.no_grad()
    def sample_bd3lm_blockwise(self, model, length: int, num_samples: int = 1, top_p: float = 0.95) -> List[str]:
        """Blockwise sampler with annealed per-block schedule, decoding xt.
        Uses first-hitting style: commit one token per step within a block,
        with t annealed from high to low noise; disallows MASK and ids >= GPT-2 vocab.
        """
        device = self.device
        model.eval()
        bs = self.config.block_size
        B = num_samples
        bos_id = 50256
        mask_id = self.mask_token_id
        T = length
        # initialize x0 guess
        x0 = torch.full((B, T), mask_id, dtype=torch.long, device=device)
        x0[:, 0] = bos_id
        # current noisy xt matches current guess
        xt = x0.clone()
        num_blocks = math.ceil(T / bs)
        # annealed schedule per block: L' steps from p_max to p_min
        p_max = 0.95
        p_min = 0.05
        schedule = torch.linspace(p_max, p_min, steps=bs, device=device)
        decode_limit = 50257
        for b in range(num_blocks):
            start = b * bs
            end = min(T, start + bs)
            step_idx = 0
            # iteratively fill this block one token at a time
            while True:
                masked = (x0[:, start:end] == mask_id)
                if not masked.any():
                    break
                # noise level for this step
                p = schedule[min(step_idx, schedule.numel() - 1)].expand(B, 1)
                sigma = self._sigma_from_p(p, getattr(self.noise, 'sigma_max', None))
                # build input as xt||x0 as in training when cross_attn
                x_in = torch.cat([xt, x0], dim=1)
                logits = model(x_in, sigma, use_bd3lm_mask=True)
                logits = logits[:, :T]  # decode xt half
                block_logits = logits[:, start:end]
                probs = block_logits.float().softmax(dim=-1)
                # Disallow MASK and ids beyond GPT-2 vocab
                probs[..., mask_id] = 0.0
                if probs.size(-1) > decode_limit:
                    probs[..., decode_limit:] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                # nucleus within the current block
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    nmask = cdf <= top_p
                    nmask[..., 0] = True
                    filt = torch.where(nmask, sorted_probs, torch.zeros_like(sorted_probs))
                    filt = filt / filt.sum(dim=-1, keepdim=True)
                    # sample proposals for all positions
                    idx = torch.multinomial(filt.view(-1, filt.size(-1)), num_samples=1).view(B, end - start, 1)
                    sampled = sorted_idx.gather(-1, idx).squeeze(-1)
                else:
                    sampled = torch.argmax(probs, dim=-1)
                # choose one masked position per batch by confidence (max prob)
                maxp, argmax_tok = probs.max(dim=-1)
                masked_scores = maxp.masked_fill(~masked, -1.0)
                pos_idx = masked_scores.argmax(dim=-1)
                # gather sampled token for that position and update
                arange_b = torch.arange(B, device=device)
                x0[arange_b, start + pos_idx] = sampled[arange_b, pos_idx]
                xt[arange_b, start + pos_idx] = x0[arange_b, start + pos_idx]
                step_idx += 1
        return self._decode_tokens(x0)

    @torch.no_grad()
    def generate_and_save_samples(self, model, out_dir: Path, rank: int = 0):
        if rank != 0:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        num = self.config.samples_per_eval
        length = min(self.config.sample_length, self.config.max_seq_len)
        top_p = self.config.top_p
        if self.config.parameterization == 'ar':
            texts = self.sample_ar(model, length=length, num_samples=num, top_p=top_p)
            # write to AR samples tex
            tex_path = out_dir / 'ar_samps.tex'
            caption = f"Sample from an AR model with length $L={length}$."
        else:
            texts = self.sample_bd3lm_blockwise(model, length=length, num_samples=num, top_p=top_p)
            tex_path = out_dir / 'sar_samps.tex'
            caption = "Sample from \\algo{} for block size $L'=%d$ of length $L=%d$." % (self.config.block_size, length)
        # Only keep the first sample for LaTeX snippet
        sample_text = texts[0] if texts else ""
        # format latex (avoid f-strings to prevent brace conflicts)
        latex = (
            "\\begin{figure}[H]\n"
            "    \\centering\n"
            "    \\begin{tabular}{c}\n\n"
            "        \\fbox{\\begin{minipage}{0.9\\textwidth}}\n"
            "            \\tiny\n"
            f"            {sample_text}\n"
            "        \\end{minipage}} \\\n\n"
            "        \\\n\n"
            "    \\end{tabular}\n"
            f"    \\caption{{{caption}}}\n"
            "    \\label{generated-samples}\n"
            "\\end{figure}\n"
        )
        try:
            tex_path.write_text(latex)
        except Exception as e:
            print(f"[warn] Failed to write samples to {tex_path}: {e}")

# -----------------------------------------------------------------------------
# Data: NanoGPT cached FineWeb .bin reader (distributed)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class FineWebBinSampler:
    """Simple sampler over NanoGPT cached FineWeb .bin shards producing (B,T) int64 batches."""
    def __init__(self, filename_pattern: str, seq_len: int, local_batch_size: int, device: torch.device, align_to_bos: bool = True, show_progress: bool = False):
        self.files: List[Path] = [Path(f) for f in sorted(glob.glob(filename_pattern))]
        assert len(self.files) > 0, f"No data files matched: {filename_pattern}"
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.device = device
        self.align_to_bos = align_to_bos
        self.file_iter = iter(self.files)
        self.tokens = None
        self.boundaries = None
        # Shard loading progress on rank 0 only
        self._pbar = tqdm(total=len(self.files), desc="Loading data shards", disable=not show_progress, leave=False)
        self._loaded = 0
        self._load_next_file()

    def _load_next_file(self):
        try:
            self.tokens = _load_data_shard(next(self.file_iter))
            if self._pbar is not None:
                self._loaded += 1
                self._pbar.update(1)
        except StopIteration:
            # restart from beginning
            self.file_iter = iter(self.files)
            self.tokens = _load_data_shard(next(self.file_iter))
            # reset progress bar on wrap-around
            if self._pbar is not None:
                self._pbar.reset(total=len(self.files))
                self._loaded = 1
                self._pbar.update(1)
        # precompute BOS boundaries if requested
        if self.align_to_bos:
            bpos = torch.nonzero(self.tokens == 50256, as_tuple=False).squeeze(-1)
            valid = bpos <= (len(self.tokens) - self.seq_len - 1)
            self.boundaries = bpos[valid]
        else:
            self.boundaries = None

    def next_batch(self) -> torch.Tensor:
        if self.align_to_bos and (self.boundaries is None or len(self.boundaries) == 0):
            # no valid boundaries in this shard, load next
            self._load_next_file()
        # choose starts
        if self.align_to_bos and len(self.boundaries) >= self.local_batch_size:
            idx = torch.randint(0, len(self.boundaries), (self.local_batch_size,))
            starts = self.boundaries[idx]
        else:
            max_start = len(self.tokens) - self.seq_len - 1
            if max_start <= 0:
                self._load_next_file()
                return self.next_batch()
            starts = torch.randint(0, max_start, (self.local_batch_size,))
        # assemble batch
        batch_list = [self.tokens[s.item(): s.item() + self.seq_len] for s in starts]
        batch = torch.stack(batch_list)  # uint16 on pinned CPU
        return batch.to(device=self.device, dtype=torch.int64, non_blocking=True)

def build_sampler(filename_pattern: str, batch_size: int, seq_len: int, device: torch.device, align_to_bos: bool):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert batch_size % world_size == 0, "batch_size must be divisible by world_size"
    local_bs = batch_size // world_size
    # Show shard progress on rank 0 only
    rank = dist.get_rank() if dist.is_initialized() else 0
    show_progress = (rank == 0)
    return FineWebBinSampler(filename_pattern, seq_len, local_bs, device, align_to_bos, show_progress=show_progress)

@torch.no_grad()
def evaluate_val_loss(trainer: BlockDiffusionTrainer, config: Config, sampler: FineWebBinSampler, rank: int, world_size: int):
    # Number of local steps so that total across ranks ~= val_tokens
    # Each step across all ranks processes (global) batch_size * seq_len tokens.
    tokens_per_step_global = config.batch_size * config.max_seq_len
    steps = max(1, math.ceil(config.val_tokens / tokens_per_step_global))
    total_loss = torch.tensor(0.0, device=trainer.device)
    total_tokens = torch.tensor(0.0, device=trainer.device)
    pbar = tqdm(total=steps, desc="Eval", disable=(rank != 0), leave=False)
    for _ in range(steps):
        batch = sampler.next_batch()
        loss = trainer.compute_loss(batch)
        toks = torch.tensor(batch.numel(), device=trainer.device, dtype=torch.float32)
        total_loss += loss * toks
        total_tokens += toks
        pbar.update(1)
    pbar.close()
    # reduce across ranks
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_tokens).item()
    return avg_loss

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description='BD3LM Speedrun Training')
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--parameterization', type=str, default=None, choices=['subs','sedd','ar'])
    parser.add_argument('--no_cross_attn', action='store_true')
    parser.add_argument('--no_flex_attn', action='store_true')
    parser.add_argument('--device', type=str, default=None, help='Override device (e.g., cpu or cuda:X)')
    parser.add_argument('--train_files', type=str, default=None, help='Glob for NanoGPT FineWeb .bin shards')
    parser.add_argument('--no_align_bos', action='store_true', help='Do not align batches to BOS tokens')
    parser.add_argument('--val_files', type=str, default=None, help='Glob for NanoGPT FineWeb val .bin shards')
    parser.add_argument('--val_tokens', type=int, default=None, help='Number of tokens to evaluate per validation run')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--optimizer', type=str, default=None, choices=['mixed','muon','adamw'], help='Select optimizer: mixed (default), muon, or adamw')
    parser.add_argument('--adamw_lr', type=float, default=None, help='Override AdamW base LR (default 3e-4)')
    parser.add_argument('--compile-zeropower', action='store_true', help='Compile Muon Newton–Schulz kernel (first step may be slow)')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for logs/checkpoints')
    parser.add_argument('--save_interval', type=int, default=None, help='Override save interval steps')
    parser.add_argument('--samples_per_eval', type=int, default=None, help='How many samples to generate at eval time')
    parser.add_argument('--sample_length', type=int, default=None, help='Sample length for generation')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling top-p')
    # WandB
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default=None, help='W&B project name (defaults to config.project_name)')
    args = parser.parse_args()
    
    config = Config(
        block_size=args.block_size,
        batch_size=args.batch_size
    )
    if args.optimizer is not None:
        config.optimizer = args.optimizer
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.compile_zeropower:
        config.compile_zeropower = True
    if args.train_files is not None:
        config.train_files = args.train_files
    if args.no_align_bos:
        config.align_to_bos = False
    if args.parameterization is not None:
        config.parameterization = args.parameterization
    if args.no_cross_attn:
        config.cross_attn = False
    if args.no_flex_attn:
        config.use_flex_attention = False
    if args.device is not None:
        config.device = args.device
    if args.val_files is not None:
        config.val_files = args.val_files
    if args.val_tokens is not None:
        config.val_tokens = args.val_tokens
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.samples_per_eval is not None:
        config.samples_per_eval = args.samples_per_eval
    if args.sample_length is not None:
        config.sample_length = args.sample_length
    if args.top_p is not None:
        config.top_p = args.top_p
    # WandB toggles and project
    if args.wandb:
        config.use_wandb = True
    # Allow overriding project name from CLI or environment
    if args.project_name is not None:
        config.project_name = args.project_name
    else:
        # Respect WANDB_PROJECT if provided via environment
        env_proj = os.environ.get('WANDB_PROJECT')
        if env_proj:
            config.project_name = env_proj
    
    # Setup distributed
    if torch.cuda.device_count() > 1 and not args.test and (config.device.startswith('cuda')):
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        config.device = f"cuda:{rank}"
    else:
        rank = 0
        world_size = 1
        # fallback to cpu in test mode when CUDA is unavailable
        if (args.test or args.device is None) and not torch.cuda.is_available():
            config.device = "cpu"
    
    if rank == 0:
        print("BD3LM Speedrun - Discrete Block Diffusion")
        print("=" * 50)
        print(f"Model: {config.n_layers} layers, {config.dim} dim")
        print(f"Vocabulary: {config.vocab_size} + MASK = {config.vocab_size_with_mask}")
        print(f"Block size: {config.block_size}")
        # batch_size is global; show both for clarity
        local_bs = config.batch_size // max(1, world_size)
        print(f"Global batch: {config.batch_size} (local {local_bs} x {world_size})")
        print(f"Optimizer: {config.optimizer}")
    
    # Create model
    # Force block_size=1 for AR
    if config.parameterization == 'ar':
        config.block_size = 1
        config.cross_attn = False
    model = BlockDiffusionLM(config).to(config.device)
    if config.compile_model and not args.test:
        try:
            model = torch.compile(model)
        except Exception as e:
            if rank == 0:
                print(f"torch.compile failed: {e}")
    
    if world_size > 1 and not args.test:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False, static_graph=True
        )
    
    # Init W&B only on rank 0
    if rank == 0 and config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=(os.environ.get('WANDB_NAME') or config.run_name or f"bd3lm-bs{config.block_size}-{uuid.uuid4().hex[:8]}"),
            config=asdict(config)
        )

    trainer = BlockDiffusionTrainer(model, config)
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params/1e6:.1f}M")
        print()
    
    # Training loop
    if args.test:
        # Quick test
        for i in range(5):
            # random batch sanity check
            batch = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device=config.device)
            loss = trainer.train_step(batch)
            print(f"Step {i+1}: Loss = {loss:.4f}")
        print("\n✅ Training working correctly!")
    else:
        # Full training: use NanoGPT cached FineWeb data
        # Prepare output dirs on rank 0
        samples_dir = Path(__file__).parent / 'samples'
        ckpt_dir = Path(__file__).parent / 'checkpoints' / (config.run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        if rank == 0:
            samples_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.ckpt_dir = ckpt_dir
        if rank == 0:
            print(f"Using data: {config.train_files}")
        train_sampler = build_sampler(
            filename_pattern=config.train_files,
            batch_size=config.batch_size,
            seq_len=config.max_seq_len,
            device=torch.device(config.device),
            align_to_bos=config.align_to_bos,
        )
        val_sampler = build_sampler(
            filename_pattern=config.val_files,
            batch_size=config.batch_size,
            seq_len=config.max_seq_len,
            device=torch.device(config.device),
            align_to_bos=True,
        )
        running_loss = 0
        log_time = time.time()
        
        # Rank-0 progress bar across training steps
        train_pbar = tqdm(total=config.max_steps, initial=trainer.step, desc="Train", disable=(rank != 0))
        while trainer.step < config.max_steps:
            # Use local shard per rank; DDP handles gradient sync automatically.
            batch = train_sampler.next_batch()
            loss = trainer.train_step(batch)
            running_loss += loss
            if rank == 0:
                train_pbar.update(1)
            
            if trainer.step % config.log_interval == 0 and rank == 0:
                avg_loss = running_loss / config.log_interval
                elapsed = time.time() - log_time
                # config.batch_size is global already; don't multiply by world_size again
                tokens_per_sec = (config.batch_size * config.max_seq_len * config.log_interval) / elapsed
                
                print(f"Step {trainer.step}: Loss = {avg_loss:.4f}, Tokens/s = {tokens_per_sec:.0f}")
                try:
                    train_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "tok/s": f"{tokens_per_sec:.0f}"})
                except Exception:
                    pass
                
                if config.use_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "tokens_per_second": tokens_per_sec,
                        "step": trainer.step
                    })
                
                running_loss = 0
                log_time = time.time()

            if trainer.step % config.eval_interval == 0:
                val_loss = evaluate_val_loss(trainer, config, val_sampler, rank, world_size)
                if rank == 0:
                    print(f"Eval step {trainer.step}: val_loss_subs = {val_loss:.4f}")
                    if config.use_wandb:
                        wandb.log({
                            "val_loss_subs": val_loss,
                            "step": trainer.step
                        })
                    # Generate and write sample rollouts
                    trainer.generate_and_save_samples(model, out_dir=samples_dir, rank=rank)
                # Save periodic checkpoints
                if trainer.step % config.save_interval == 0:
                    trainer.save_checkpoint(model, rank=rank)
        # Final checkpoint
        trainer.save_checkpoint(model, rank=rank)
        if rank == 0:
            train_pbar.close()
    
    if rank == 0 and config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
