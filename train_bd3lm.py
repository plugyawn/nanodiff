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
# Speed-friendly math defaults on Ampere/Hopper
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prefer TF32/fast paths where applicable
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
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
    antithetic_sampling: bool = True
    sampling_eps_min: float = 1e-3
    sampling_eps_max: float = 1.0
    eval_var_min: bool = True
    clip_search_widths: tuple = (0.25, 0.5, 0.75)
    first_hitting: bool = True
    
    # Training
    batch_size: int = 64  # global batch size (will be sharded across ranks)
    learning_rate: float = 0.02
    # AdamW often needs a much smaller LR; keep separate knob
    adamw_lr: float = 3e-4
    weight_decay: float = 0.0
    momentum: float = 0.95
    max_steps: int = 125_000
    warmup_steps: int = 2500
    cooldown_frac: float = 0.45
    optimizer: str = "mixed"  # mixed | muon | adamw
    
    # Architecture
    use_flex_attention: bool = True
    use_rotary: bool = True
    use_qk_norm: bool = True
    use_relu_squared: bool = True
    use_swiglu: bool = False
    use_local_mixer: bool = False
    use_value_embeds: bool = True
    use_unet_skips: bool = True
    softcap: float = 15.0
    use_prenorm: bool = True
    residual_scale: float = 1.0
    qk_learned_scale: bool = False
    tie_weights: bool = False
    use_film: bool = False
    use_two_stream: bool = False
    sedd_mix_frac: float = 0.0
    compile_model: bool = False
    compile_zeropower: bool = False
    activation_checkpoint: bool = False
    # EMA for eval stability
    use_ema_for_eval: bool = True
    ema_decay: float = 0.999
    
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
    # Eval behavior
    ce_only: bool = False  # if True, skip BD3LM loss eval and only compute CE
    
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
        if compile_zeropower:
            try:
                _rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            except Exception:
                _rank = 0
            if _rank == 0:
                print("[compile] Compiling zeropower kernel (Newton–Schulz)...")
                sys.stdout.flush()
                _t0 = time.time()
            self._zeropower = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")
            if _rank == 0:
                print(f"[compile] Zeropower compiled in {time.time()-_t0:.1f}s")
        else:
            self._zeropower = zeropower_via_newtonschulz5
    
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

from model.utils import rms_norm as norm
from model.utils import create_block_diff_mask, create_bd3lm_mask, create_bd3lm_xt_queries_mask
from model.layers import RMSNorm, SwiGLU, MLP as MLPFallback, FiLM
from model.attention import CausalSelfAttention, TwoStreamTransformerBlock

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'use_swiglu', False):
            self.impl = SwiGLU(config.dim, hidden_mult=4.0, use_dwconv=getattr(config, 'use_local_mixer', False))
        else:
            self.impl = MLPFallback(config.dim, hidden_mult=4.0, relu_squared=config.use_relu_squared)

    def forward(self, x):
        return self.impl(x)

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # attention
        attn_max_len = config.max_seq_len * (2 if config.cross_attn else 1)
        self.attn = CausalSelfAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            max_seq_len=attn_max_len,
            use_rotary=config.use_rotary,
            use_qk_norm=config.use_qk_norm,
            use_value_embeds=config.use_value_embeds,
            attn_scale=0.12,
            qk_learned_scale=getattr(config, 'qk_learned_scale', False),
        ) if layer_idx != 7 else None
        self.mlp = MLP(config)
        # Pre-norms
        self.use_prenorm = getattr(config, 'use_prenorm', True)
        if self.use_prenorm:
            self.rms1 = RMSNorm(config.dim)
            self.rms2 = RMSNorm(config.dim)
        # Residual scale (stabilize deep residuals)
        self.res_scale = getattr(config, 'residual_scale', 1.0)
    
    def forward(self, x, x0, block_mask):
        if self.attn is not None:
            h = self.rms1(x) if self.use_prenorm else x
            x = x + self.res_scale * self.attn(h, block_mask)
        
        if self.config.use_unet_skips and x0 is not None:
            if self.layer_idx < self.config.n_layers // 2:
                x = x + 0.1 * x0
        
        h2 = self.rms2(x) if self.use_prenorm else x
        x = x + self.res_scale * self.mlp(h2)
        return x

class BlockDiffusionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Include MASK token in vocabulary
        vocab_size = config.vocab_size_with_mask
        
        # Embeddings (use bf16 to reduce cast/memory overhead)
        self.token_emb = nn.Embedding(vocab_size, config.dim, dtype=torch.bfloat16)
        # Cross-attn doubles sequence length (xt||x0), expand pos embeddings accordingly
        pos_len = config.max_seq_len * (2 if config.cross_attn else 1)
        self.pos_emb = nn.Embedding(pos_len, config.dim, dtype=torch.bfloat16)
        
        # Time embedding for diffusion (condition on sigma)
        self.time_emb = nn.Sequential(
            nn.Linear(1, config.dim, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim, dtype=torch.bfloat16)
        )
        
        # Transformer blocks
        if getattr(config, 'use_two_stream', False):
            attn_len = config.max_seq_len * (2 if config.cross_attn else 1)
            self.ts_blocks = nn.ModuleList([
                TwoStreamTransformerBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    head_dim=config.head_dim,
                    max_seq_len=attn_len,
                    use_rotary=config.use_rotary,
                    use_qk_norm=config.use_qk_norm,
                    use_value_embeds=False,
                    attn_scale=0.12,
                    qk_learned_scale=getattr(config, 'qk_learned_scale', False),
                    use_swiglu=getattr(config, 'use_swiglu', False),
                    use_local_mixer=getattr(config, 'use_local_mixer', False),
                    residual_scale=getattr(config, 'residual_scale', 1.0),
                    use_prenorm=getattr(config, 'use_prenorm', True),
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(config, i) for i in range(config.n_layers)
            ])
        
        # Output
        # Keep normalization and projection dtypes consistent with bf16 activations
        self.ln_f = nn.LayerNorm(config.dim, dtype=torch.bfloat16)
        # Project in bf16 to match preceding activations and avoid dtype mismatch
        self.lm_head = nn.Linear(config.dim, vocab_size, bias=False, dtype=torch.bfloat16)
        if getattr(config, 'tie_weights', False):
            # Tie with token embeddings
            self.lm_head.weight = self.token_emb.weight
        else:
            with torch.no_grad():
                self.lm_head.weight.zero_()  # zero-init head
        
        # Block mask for attention
        if config.use_flex_attention:
            # Build masks on the same device as the module to avoid cross-device issues in DDP
            module_device = torch.device(config.device if isinstance(config.device, str) else config.device)
            self.block_mask = create_block_diff_mask(config.max_seq_len, config.block_size, device=module_device)
            # prebuild BD3LM mask for xt||x0 (total length = 2 * max_seq_len)
            self.bd3lm_mask = create_bd3lm_mask(config.max_seq_len, config.block_size, device=module_device)
            # two-stream: queries=xt (len n), kv=xt||x0 (len 2n)
            self.bd3lm_ts_mask = create_bd3lm_xt_queries_mask(config.max_seq_len, config.block_size, device=module_device)
            # simple LRU cache for masks by (kind, n)
            self._mask_cache = {
                'block': {},      # key: T
                'bd3lm': {},      # key: n (xt tokens)
                'bd3lm_ts': {},   # key: n (xt tokens)
            }
        else:
            self.block_mask = None
            self.bd3lm_mask = None
            self.bd3lm_ts_mask = None
            self._mask_cache = None

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
        
        # Sigma conditioning: per-token additive on xt (when provided),
        # fallback to sequence-wise FiLM or additive for scalar sigma.
        film = None
        if sigma is not None:
            # Support sigma as (B,), (B,1), (B,T), or (B,2T)
            sig = sigma
            if isinstance(sig, torch.Tensor) and sig.dim() == 1:
                sig = sig.unsqueeze(1)
            # If per-token sigma was provided, prefer additive time embedding per position.
            if isinstance(sig, torch.Tensor) and sig.dim() == 2 and sig.size(1) > 1:
                # Shape (B, L) where L == T (no-CA) or 2T (CA). Embed to (B,L,D)
                s = sig.to(x.dtype).unsqueeze(-1)
                t_full = self.time_emb(s).to(x.dtype)
                if use_bd3lm_mask:
                    # Apply only to xt half (first n tokens)
                    n_tok = t_full.size(1) // 2
                    x[:, :n_tok, :] = x[:, :n_tok, :] + t_full[:, :n_tok, :]
                else:
                    # Non-CA path: inputs are xt only, apply everywhere
                    x = x + t_full
            else:
                # Scalar sigma per example: allow FiLM or additive sequence-wise
                s = sig.to(x.dtype).unsqueeze(-1)  # (B,1,1)
                t_emb = self.time_emb(s).to(x.dtype)  # (B,1,dim)
                if getattr(self.config, 'use_film', False):
                    # Map (B,1,dim) -> (B,dim)
                    t_vec = t_emb.squeeze(1)
                    if not hasattr(self, 'film'):
                        from model.layers import FiLM as _FiLM
                        self.film = _FiLM(self.config.dim)
                    gamma, beta = self.film(t_vec)
                    film = (gamma.to(x.dtype).unsqueeze(1), beta.to(x.dtype).unsqueeze(1))
                else:
                    # Always add sequence-wise embedding in scalar-sigma case
                    x = x + t_emb
        
        x0 = x if self.config.use_unet_skips else None
        
        # Transformer blocks
        if getattr(self.config, 'use_two_stream', False):
            # Two-stream path supports both BD3LM (xt||x0) and single-stream CE eval.
            if use_bd3lm_mask:
                # Input is xt||x0; split evenly
                total_T = x.size(1)
                n_tok = total_T // 2
                xt = x[:, :n_tok, :]
                x0e = x[:, n_tok:, :]
                # Fetch/cache a matching two-stream mask for current length
                if self.config.use_flex_attention:
                    cache = self._mask_cache['bd3lm_ts']
                    ts_mask = cache.get(n_tok)
                    if ts_mask is None:
                        ts_mask = create_bd3lm_xt_queries_mask(n_tok, self.config.block_size, device=device)
                        cache[n_tok] = ts_mask
                else:
                    ts_mask = None
            else:
                # Single-stream inputs: reuse two-stream blocks by setting xt=x and x0=x
                # and using a two-stream mask that yields block-causal behavior.
                xt = x
                x0e = x
                if self.config.use_flex_attention:
                    n_tok = x.size(1)
                    cache = self._mask_cache['bd3lm_ts']
                    ts_mask = cache.get(n_tok)
                    if ts_mask is None:
                        ts_mask = create_bd3lm_xt_queries_mask(n_tok, self.config.block_size, device=device)
                        cache[n_tok] = ts_mask
                else:
                    ts_mask = None

            # U-Net style skip connections with gating on xt stream
            n_half = len(self.ts_blocks) // 2
            skips = []
            for i, block in enumerate(self.ts_blocks):
                # Apply FiLM to residual stream pre-block if enabled
                if film is not None and getattr(self.config, 'use_film', False):
                    gamma, beta = film
                    xt = gamma * xt + beta
                if getattr(self.config, 'activation_checkpoint', False):
                    import torch.utils.checkpoint as _ckpt
                    def _fn(_xt):
                        return block(_xt, x0e, block_mask=ts_mask)
                    xt = _ckpt.checkpoint(_fn, xt)
                else:
                    xt = block(xt, x0e, block_mask=ts_mask)
                if i < n_half:
                    skips.append(xt)
                else:
                    xt = xt + self.skip_weights[i - n_half] * skips.pop()
            x = xt
        else:
            # Single-stream path (original)
            if self.config.use_flex_attention:
                T_eff = x.size(1)
                if use_bd3lm_mask:
                    n_tok = max(1, T_eff // 2)
                    # cache BD3LM 2n-length mask keyed by n_tok
                    cache = self._mask_cache['bd3lm']
                    attn_mask = cache.get(n_tok)
                    if attn_mask is None:
                        attn_mask = create_bd3lm_mask(n_tok, self.config.block_size, device=device)
                        cache[n_tok] = attn_mask
                else:
                    # cache block mask keyed by T_eff
                    cache = self._mask_cache['block']
                    attn_mask = cache.get(T_eff)
                    if attn_mask is None:
                        attn_mask = create_block_diff_mask(T_eff, self.config.block_size, device=device)
                        cache[T_eff] = attn_mask
            else:
                attn_mask = None

            n = len(self.blocks) // 2
            skips = []
            for i, block in enumerate(self.blocks):
                if film is not None and getattr(self.config, 'use_film', False):
                    gamma, beta = film
                    x = gamma * x + beta
                if getattr(self.config, 'activation_checkpoint', False):
                    import torch.utils.checkpoint as _ckpt
                    def _fn(_x):
                        return block(_x, x0, attn_mask)
                    x = _ckpt.checkpoint(_fn, x)
                else:
                    x = block(x, x0, attn_mask)
                if i < n:
                    skips.append(x)
                else:
                    x = x + self.skip_weights[i - n] * skips.pop()
        
        # Output
        x = self.ln_f(x)
        # Some PyTorch builds may upcast LayerNorm internally; enforce bf16 before linear
        x = x.to(torch.bfloat16)
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
        # Lazily created decode tokenizer to avoid per-eval reloading cost
        self._decode_tok = None
        
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
            # Prefer fused AdamW when available (CUDA builds) for faster optimizer steps
            adamw_kwargs = dict(lr=config.adamw_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
            try:
                self.optimizer_adam = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)  # type: ignore[call-arg]
            except TypeError:
                # Fallback to non-fused on older PyTorch builds
                self.optimizer_adam = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Metrics
        self.step = 0
        self.start_time = time.time()
        
        # WandB init handled in main() on rank 0
        
        # Checkpoint dir (set in main)
        self.ckpt_dir: Optional[Path] = None

        # Exponential Moving Average of parameters (for eval)
        self.ema = None
        if getattr(config, 'use_ema_for_eval', False):
            self.ema = _ModelEMA(model, decay=getattr(config, 'ema_decay', 0.999))
        
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
            # rand_like on a bool tensor is invalid; sample floats then compare
            move[regen_idx] = (torch.rand(move.shape, device=move.device) <= p)[regen_idx]
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
    
    def compute_loss(self, x, eps_min: Optional[float] = None, eps_max: Optional[float] = None):
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
        # compute sigma per-token for conditioning, and per-example for loss scale terms
        sigma_tok = self._sigma_from_p(move_chance, getattr(self.noise, 'sigma_max', None))  # (B,T)
        sigma = self._sigma_from_p(move_chance.mean(dim=1, keepdim=True), getattr(self.noise, 'sigma_max', None))  # (B,1)

        # optional MDLM loss-scale override
        if self.config.mdlm_loss_scale:
            # follow reference: sigma=total_noise(t), dsigma=rate_noise(t)
            sigma = self.noise.total_noise(t.mean(dim=1, keepdim=True))
            dsigma = self.noise.rate_noise(t.mean(dim=1, keepdim=True))
            move_chance = 1 - torch.exp(-sigma)
            loss_scale = - (dsigma / torch.expm1(sigma))

        # corrupt tokens (allow temporary override for schedule clipping)
        use_eps_min = self.config.sampling_eps_min if eps_min is None else eps_min
        use_eps_max = self.config.sampling_eps_max if eps_max is None else eps_max
        xt = self.corrupt_tokens(x, move_chance, use_eps_min, use_eps_max)

        # model forward: if cross_attn, feed xt||x0 and use 3-part mask
        if self.config.cross_attn:
            x_input = torch.cat([xt, x], dim=1)
            # Provide per-token sigma on xt positions only (zeros for x0)
            sigma_in = torch.cat([sigma_tok, torch.zeros_like(sigma_tok)], dim=1)
            logits = self.model(x_input, sigma_in, use_bd3lm_mask=True)
            # decode from xt queries; two-stream path returns only xt logits
            if not getattr(self.config, 'use_two_stream', False):
                logits = logits[:, :T]
        else:
            # Non-CA path: inputs are xt only; provide per-token sigma
            logits = self.model(xt, sigma_tok)

        if self.config.parameterization == 'subs':
            # SUBS primary objective; optionally mix in SEDD for calibration
            log_probs = self._subs_parameterization(logits, xt)
            log_p_theta = torch.gather(log_probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
            masked = (xt == self.mask_token_id)
            num_masked = int(masked.sum().item())
            assert num_masked > 0, "No masked tokens in batch; adjust noise schedule"
            loss_mat = loss_scale * log_p_theta * masked.float()
            loss_subs = loss_mat.sum() / num_masked

            mix = float(getattr(self.config, 'sedd_mix_frac', 0.0) or 0.0)
            if mix > 0.0:
                dsigma = - loss_scale * torch.expm1(sigma)
                log_score = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                loss_sedd_mat = self._score_entropy(log_score, sigma, xt, x)
                loss_sedd = (dsigma.squeeze(-1)[:, None] * loss_sedd_mat).sum() / (B * T)
                return (1.0 - mix) * loss_subs + mix * loss_sedd
            else:
                return loss_subs
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
        # Update EMA after the step
        if self.ema is not None:
            self.ema.update(self.model)
        self.step += 1
        
        return loss.item()

    # ------------------------------ Checkpointing ------------------------------
    def save_checkpoint(self, model, rank: int = 0):
        if rank != 0 or self.ckpt_dir is None:
            return
        try:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            # If compiled, unwrap to the original module for a clean, portable state_dict
            base_module = getattr(module, "_orig_mod", module)
            state = {
                'model_state': base_module.state_dict(),
                'config': asdict(self.config),
                'step': self.step,
                'timestamp': datetime.utcnow().isoformat(),
                # Optimizer states for seamless resume
                'optimizer_muon': (self.optimizer_muon.state_dict() if self.optimizer_muon is not None else None),
                'optimizer_adam': (self.optimizer_adam.state_dict() if self.optimizer_adam is not None else None),
                # RNG states to minimize loss of training continuity
                'torch_rng_state': torch.random.get_rng_state(),
                'cuda_rng_state': (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
            }
            ckpt_path = self.ckpt_dir / f"ckpt_step{self.step:06d}.pt"
            torch.save(state, ckpt_path)
        except Exception as e:
            print(f"[warn] Failed to save checkpoint at step {self.step}: {e}")

    # ------------------------------- Sampling ---------------------------------
    @torch.no_grad()
    def _decode_tokens(self, ids: torch.Tensor) -> List[str]:
        # ids: (B, T)
        if self._decode_tok is None:
            tok = AutoTokenizer.from_pretrained('gpt2')
            tok.pad_token = tok.eos_token
            self._decode_tok = tok
        return self._decode_tok.batch_decode(ids.tolist(), skip_special_tokens=True)

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
                if not getattr(self.config, 'use_two_stream', False):
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
        was_training = model.training
        model.eval()
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
        finally:
            if was_training:
                model.train()

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
        batch = torch.stack(batch_list)  # uint16 CPU tensor (may not be pinned yet)
        # Ensure pinned memory for asynchronous H2D copies when using CUDA
        if torch.device(self.device).type == 'cuda':
            batch = batch.pin_memory()
        return batch.to(device=self.device, dtype=torch.int64, non_blocking=True)

    def next_batch_pinned(self) -> torch.Tensor:
        """Return a CPU batch (uint16) pinned in memory for async H2D copy.
        The caller is responsible for moving to device/dtype.
        """
        if self.align_to_bos and (self.boundaries is None or len(self.boundaries) == 0):
            self._load_next_file()
        if self.align_to_bos and len(self.boundaries) >= self.local_batch_size:
            idx = torch.randint(0, len(self.boundaries), (self.local_batch_size,))
            starts = self.boundaries[idx]
        else:
            max_start = len(self.tokens) - self.seq_len - 1
            if max_start <= 0:
                self._load_next_file()
                return self.next_batch_pinned()
            starts = torch.randint(0, max_start, (self.local_batch_size,))
        batch_list = [self.tokens[s.item(): s.item() + self.seq_len] for s in starts]
        batch = torch.stack(batch_list)  # uint16 CPU
        return batch.pin_memory()

def build_sampler(filename_pattern: str, batch_size: int, seq_len: int, device: torch.device, align_to_bos: bool):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert batch_size % world_size == 0, "batch_size must be divisible by world_size"
    local_bs = batch_size // world_size
    # Show shard progress on rank 0 only
    rank = dist.get_rank() if dist.is_initialized() else 0
    show_progress = (rank == 0)
    return FineWebBinSampler(filename_pattern, seq_len, local_bs, device, align_to_bos, show_progress=show_progress)

class PrefetchingSampler:
    """Wrap a sampler to overlap CPU batch prep and H2D copies using a
    dedicated CUDA stream. Falls back to direct path on CPU.
    """
    def __init__(self, base: FineWebBinSampler, device: torch.device):
        self.base = base
        self.device = torch.device(device)
        self._prefetched = None
        if self.device.type == 'cuda':
            self._stream = torch.cuda.Stream(device=self.device)
        else:
            self._stream = None
        # Prime first batch
        self._enqueue()

    def _enqueue(self):
        if self._stream is None:
            # CPU path: just prepare synchronously as int64 on CPU
            self._prefetched = self.base.next_batch()
            return
        with torch.cuda.stream(self._stream):
            try:
                cpu_batch = self.base.next_batch_pinned()
            except AttributeError:
                # Fallback for older base impls; schedule copy within this stream
                self._prefetched = self.base.next_batch()
                return
            self._prefetched = cpu_batch.to(device=self.device, dtype=torch.int64, non_blocking=True)

    def next_batch(self) -> torch.Tensor:
        if self._stream is not None:
            # Make current stream wait for the prefetch stream
            torch.cuda.current_stream(device=self.device).wait_stream(self._stream)
            out = self._prefetched
            # Immediately start preparing the next batch
            self._enqueue()
            return out
        # CPU path
        out = self._prefetched
        self._enqueue()
        return out

@torch.no_grad()
def evaluate_val_loss(trainer: BlockDiffusionTrainer, config: Config, sampler: FineWebBinSampler, rank: int, world_size: int):
    """Evaluate validation losses.

    Returns a dict with:
    - 'bd3lm': the current training objective (SUBS/SEDD) averaged over tokens
    - 'ce': NanoGPT-style mean next-token cross-entropy on FineWeb

    If eval_var_min is enabled, SUBS/SEDD loss uses variance-min clipping sweep
    to choose eps_min; CE is computed once per batch and averaged.
    """
    tokens_per_step_global = config.batch_size * config.max_seq_len
    steps = max(1, math.ceil(config.val_tokens / tokens_per_step_global))

    # Always use inference mode to avoid autograd overhead and torch.compile graph churn
    # Preserve/restore model training mode around eval
    model = trainer.model
    # Optionally swap in EMA weights for evaluation
    ema_ctx = getattr(trainer, 'ema', None)
    if ema_ctx is not None and getattr(config, 'use_ema_for_eval', False):
        ema_ctx.swap_in(model)
    was_training = model.training
    model.eval()
    # Use bfloat16 autocast for faster matmuls if CUDA is available
    # torch.cuda.amp.autocast is deprecated; use torch.amp.autocast('cuda', ...)
    amp_ctx = torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16)

    # Fast path: CE-only evaluation (user cares about CE)
    if getattr(config, 'ce_only', False):
        total_ce = torch.tensor(0.0, device=trainer.device)
        total_ce_tokens = torch.tensor(0.0, device=trainer.device)
        pbar = tqdm(total=steps, desc="Eval(CE)", disable=(rank != 0), leave=False)
        with amp_ctx, torch.inference_mode():
            for _ in range(steps):
                batch = sampler.next_batch()
                # Keep sequence length equal to training to avoid recompiles; slice logits for CE
                logits = trainer.model(batch, sigma=None, use_bd3lm_mask=False)
                ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction='mean')
                total_ce += ce * batch[:, 1:].numel()
                total_ce_tokens += torch.tensor(float(batch[:, 1:].numel()), device=trainer.device)
                pbar.update(1)
        pbar.close()
        if world_size > 1:
            dist.all_reduce(total_ce, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        # restore training state
        if ema_ctx is not None and getattr(config, 'use_ema_for_eval', False):
            ema_ctx.swap_out(model)
        if was_training:
            model.train()
        return {
            'bd3lm': None,
            'ce': (total_ce / torch.clamp_min(total_ce_tokens, 1.0)).item(),
        }

    if not config.eval_var_min:
        total_loss = torch.tensor(0.0, device=trainer.device)
        total_tokens = torch.tensor(0.0, device=trainer.device)
        total_ce = torch.tensor(0.0, device=trainer.device)
        total_ce_tokens = torch.tensor(0.0, device=trainer.device)
        pbar = tqdm(total=steps, desc="Eval", disable=(rank != 0), leave=False)
        with amp_ctx, torch.inference_mode():
            for _ in range(steps):
                batch = sampler.next_batch()
                loss = trainer.compute_loss(batch)
                toks = torch.tensor(batch.numel(), device=trainer.device, dtype=torch.float32)
                total_loss += loss * toks
                total_tokens += toks
                # NanoGPT-style CE on the same batch (keep input length equal to training)
                logits = trainer.model(batch, sigma=None, use_bd3lm_mask=False)
                ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction='mean')
                total_ce += ce * (batch.size(0) * (batch.size(1) - 1))
                total_ce_tokens += torch.tensor(float(batch.size(0) * (batch.size(1) - 1)), device=trainer.device)
                pbar.update(1)
        pbar.close()
        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ce, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        # restore training state
        if ema_ctx is not None and getattr(config, 'use_ema_for_eval', False):
            ema_ctx.swap_out(model)
        if was_training:
            model.train()
        return {
            'bd3lm': (total_loss / total_tokens).item(),
            'ce': (total_ce / total_ce_tokens).item(),
        }

    # var_min-enabled path
    candidates = list(config.clip_search_widths) if len(config.clip_search_widths) else [0.25, 0.5, 0.75]
    # losses_by_c[c] stores list of per-batch average losses at clipping (c, 1.0)
    losses_by_c = {c: [] for c in candidates}
    pbar = tqdm(total=steps, desc="Eval(var_min)", disable=(rank != 0), leave=False)
    ce_accum = torch.tensor(0.0, device=trainer.device)
    ce_tokens = torch.tensor(0.0, device=trainer.device)
    with amp_ctx, torch.inference_mode():
        for _ in range(steps):
            batch = sampler.next_batch()
            for c in candidates:
                loss_c = trainer.compute_loss(batch, eps_min=float(c), eps_max=1.0)
                # store python float to keep small and all-reduce later
                losses_by_c[c].append(float(loss_c))
            # Also accumulate NanoGPT-style CE per batch once (keep input length equal to training)
            logits = trainer.model(batch, sigma=None, use_bd3lm_mask=False)
            ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction='mean')
            ce_accum += ce * (batch.size(0) * (batch.size(1) - 1))
            ce_tokens += torch.tensor(float(batch.size(0) * (batch.size(1) - 1)), device=trainer.device)
            pbar.update(1)
    pbar.close()

    # Stack into tensors and all-reduce means/variances across ranks
    best_c = None
    best_var = None
    best_mean = None
    for c in candidates:
        local_losses = torch.tensor(losses_by_c[c], device=trainer.device, dtype=torch.float32)
        if world_size > 1:
            # compute mean and mean of squares across ranks for variance
            local_sum = local_losses.sum()
            local_sq_sum = (local_losses ** 2).sum()
            local_count = torch.tensor(float(local_losses.numel()), device=trainer.device)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            mean = (local_sum / local_count)
            var = (local_sq_sum / local_count) - mean ** 2
        else:
            mean = local_losses.mean() if local_losses.numel() > 0 else torch.tensor(float('inf'), device=trainer.device)
            var = local_losses.var(unbiased=False) if local_losses.numel() > 1 else torch.tensor(float('inf'), device=trainer.device)

        # choose by minimal variance; break ties by mean
        if best_var is None or var.item() < best_var or (math.isclose(var.item(), best_var, rel_tol=1e-3) and mean.item() < best_mean):
            best_var = var.item()
            best_mean = mean.item()
            best_c = c

    # adopt the chosen clipping interval for training
    if best_c is not None:
        # only rank 0 prints decision
        if rank == 0:
            print(f"[var_min] chosen eps_min={best_c:.2f}, var={best_var:.6g}, mean={best_mean:.6g}")
        config.sampling_eps_min = float(best_c)
        # keep max at 1.0
        config.sampling_eps_max = 1.0

    # Return the average loss under the chosen clipping
    chosen_losses = torch.tensor(losses_by_c[best_c], device=trainer.device, dtype=torch.float32)
    if world_size > 1:
        chosen_sum = chosen_losses.sum()
        chosen_count = torch.tensor(float(chosen_losses.numel()), device=trainer.device)
        dist.all_reduce(chosen_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(chosen_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(ce_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ce_tokens, op=dist.ReduceOp.SUM)
        # restore training state
        if ema_ctx is not None and getattr(config, 'use_ema_for_eval', False):
            ema_ctx.swap_out(model)
        if was_training:
            model.train()
        return {
            'bd3lm': (chosen_sum / chosen_count).item(),
            'ce': (ce_accum / ce_tokens).item(),
        }
    # restore training state
    if ema_ctx is not None and getattr(config, 'use_ema_for_eval', False):
        ema_ctx.swap_out(model)
    if was_training:
        model.train()
    return {
        'bd3lm': (chosen_losses.mean() if chosen_losses.numel() > 0 else torch.tensor(float('inf'), device=trainer.device)).item(),
        'ce': (ce_accum / torch.clamp_min(ce_tokens, 1.0)).item(),
    }

# ------------------------------- EMA Helper ---------------------------------

class _ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = None
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def swap_in(self, model: nn.Module):
        # Save current params and load EMA params
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def swap_out(self, model: nn.Module):
        if self.backup is None:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[name])
        self.backup = None

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
    parser.add_argument('--eval_interval', type=int, default=None, help='Override eval interval steps')
    parser.add_argument('--samples_per_eval', type=int, default=None, help='How many samples to generate at eval time')
    parser.add_argument('--sample_length', type=int, default=None, help='Sample length for generation')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling top-p')
    parser.add_argument('--eval_ce_only', action='store_true', help='Eval only CE (skip BD3LM loss/var-min)')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model forward pass')
    parser.add_argument('--ddp_fp16_compress', action='store_true', help='Compress gradients to FP16 during DDP all-reduce')
    parser.add_argument('--activation_checkpoint', action='store_true', help='Enable activation checkpointing in transformer blocks')
    parser.add_argument('--max_steps', type=int, default=None, help='Override total training steps')
    parser.add_argument('--train_tokens', type=int, default=None, help='Target total training tokens; derives max_steps = ceil(train_tokens / (global_batch * seq_len))')
    # Architecture knobs
    parser.add_argument('--use_swiglu', action='store_true', help='Use SwiGLU MLP')
    parser.add_argument('--use_local_mixer', action='store_true', help='Add depthwise conv mixer in MLP')
    parser.add_argument('--tie_weights', action='store_true', help='Tie lm_head to token_emb')
    parser.add_argument('--use_film', action='store_true', help='Use FiLM conditioning from sigma in SUBS')
    parser.add_argument('--qk_learned_scale', action='store_true', help='Learned per-head scale on Q/K')
    parser.add_argument('--use_two_stream', action='store_true', help='Use two-stream attention for xt queries into xt||x0 keys')
    parser.add_argument('--sedd_mix_frac', type=float, default=None, help='Blend SEDD loss into SUBS with this fraction')
    parser.add_argument('--residual_scale', type=float, default=None, help='Residual branch scale factor')
    parser.add_argument('--no_prenorm', action='store_true', help='Disable pre-norm in transformer blocks')
    # Resume
    parser.add_argument('--resume_dir', type=str, default=None, help='Resume from latest ckpt in this directory')
    parser.add_argument('--resume_path', type=str, default=None, help='Resume from this exact ckpt path')
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
    if args.compile:
        config.compile_model = True
    if args.activation_checkpoint:
        config.activation_checkpoint = True
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
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.samples_per_eval is not None:
        config.samples_per_eval = args.samples_per_eval
    if args.sample_length is not None:
        config.sample_length = args.sample_length
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.eval_ce_only:
        config.ce_only = True
    # Arch flags
    if args.use_swiglu:
        config.use_swiglu = True
    if args.use_local_mixer:
        config.use_local_mixer = True
    if args.tie_weights:
        config.tie_weights = True
    if args.use_film:
        config.use_film = True
    if args.qk_learned_scale:
        config.qk_learned_scale = True
    if args.use_two_stream:
        config.use_two_stream = True
    if args.sedd_mix_frac is not None:
        config.sedd_mix_frac = float(args.sedd_mix_frac)
    if args.residual_scale is not None:
        config.residual_scale = float(args.residual_scale)
    if args.no_prenorm:
        config.use_prenorm = False
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
    
    # Resolve resume checkpoint path if requested (rank will be set next)
    resume_ckpt_path = None
    if args.resume_path:
        from pathlib import Path as _Path
        p = _Path(args.resume_path)
        if not p.exists():
            raise FileNotFoundError(f"--resume_path not found: {p}")
        resume_ckpt_path = p
    elif args.resume_dir:
        from pathlib import Path as _Path
        d = _Path(args.resume_dir)
        if not d.exists():
            raise FileNotFoundError(f"--resume_dir not found: {d}")
        cand = sorted(d.glob('ckpt_step*.pt'))
        if not cand:
            raise FileNotFoundError(f"No ckpt_step*.pt under --resume_dir: {d}")
        resume_ckpt_path = cand[-1]

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
        print(f"Max steps: {config.max_steps}  |  Tokens/step: {config.batch_size * config.max_seq_len}")
        print(f"Optimizer: {config.optimizer}")
    
    # Create model (and optionally override config from checkpoint when resuming)
    ckpt_state = None
    if resume_ckpt_path is not None:
        # Load saved config to guarantee architectural match
        import torch as _torch
        ckpt_state = _torch.load(resume_ckpt_path, map_location='cpu')
        if 'config' not in ckpt_state:
            raise KeyError(f"Checkpoint missing 'config': {resume_ckpt_path}")
        saved_cfg = ckpt_state['config']
        # Keep runtime device, wandb toggles, and optional run_name override
        cli_cfg = config  # preserve CLI/runtime choices made above
        keep_device = cli_cfg.device
        keep_use_wandb = cli_cfg.use_wandb
        keep_project = cli_cfg.project_name
        keep_run_name = (args.run_name if args.run_name is not None else saved_cfg.get('run_name'))
        config = Config(**saved_cfg)
        # Restore selected runtime overrides (safe to change across resume)
        config.device = keep_device
        config.use_wandb = keep_use_wandb
        if keep_project is not None:
            config.project_name = keep_project
        if keep_run_name is not None:
            config.run_name = keep_run_name
        # Apply additional runtime overrides from CLI/runtime config
        for key in [
            'batch_size', 'eval_interval', 'save_interval', 'samples_per_eval', 'sample_length', 'top_p',
            'optimizer', 'adamw_lr', 'learning_rate', 'weight_decay', 'momentum',
            'compile_model', 'compile_zeropower', 'train_files', 'val_files', 'val_tokens', 'align_to_bos'
        ]:
            if hasattr(cli_cfg, key):
                setattr(config, key, getattr(cli_cfg, key))

    # Allow overriding training length post resume merge
    if args.max_steps is not None:
        config.max_steps = int(args.max_steps)
    elif args.train_tokens is not None:
        # Derive steps from target token budget and final (possibly resumed) batch size
        tokens_per_step = int(config.batch_size) * int(config.max_seq_len)
        # Guard against zero or negative
        tokens_per_step = max(1, tokens_per_step)
        config.max_steps = max(1, math.ceil(int(args.train_tokens) / tokens_per_step))

    # Create model
    # Force block_size=1 for AR
    if config.parameterization == 'ar':
        config.block_size = 1
        config.cross_attn = False
    # Create base model first (uncompiled) so we can load checkpoints reliably
    model = BlockDiffusionLM(config).to(config.device)
    
    # If resuming, load model weights BEFORE compilation/DDP wrapping to avoid OptimizedModule key prefixes
    if resume_ckpt_path is not None:
        load_state = ckpt_state['model_state']
        # Accept checkpoints saved from a compiled model (keys prefixed with _orig_mod.)
        if any(k.startswith('_orig_mod.') for k in load_state.keys()):
            load_state = { (k[10:] if k.startswith('_orig_mod.') else k): v for k, v in load_state.items() }
        try:
            msg = model.load_state_dict(load_state, strict=True)
        except RuntimeError as e:
            # Fallback: attempt non-strict to surface partial mismatches without failing
            if rank == 0:
                print(f"[warn] strict load failed ({e}); retrying non-strict load")
            msg = model.load_state_dict(load_state, strict=False)
        # carry step forward
        trainer_step_from_ckpt = int(ckpt_state.get('step', 0))
    else:
        trainer_step_from_ckpt = 0

    # Optional compilation happens after successful load
    # Workaround: some inductor pointwise kernels (e.g., fused div/mul/tanh over large tensors)
    # request a Triton XBLOCK larger than the default 4096, triggering
    # "increase TRITON_MAX_BLOCK['X'] to ..." assertions during torch.compile/inductor.
    # Bump the runtime limit conservatively to 8192 before compiling the model.
    try:
        import torch._inductor.runtime.hints as _ind_hints  # type: ignore
        if isinstance(_ind_hints.TRITON_MAX_BLOCK, dict):
            _ind_hints.TRITON_MAX_BLOCK["X"] = max(_ind_hints.TRITON_MAX_BLOCK.get("X", 4096), 8192)
        # Also update the reference in triton_heuristics (same dict object in-process, but be explicit)
        import torch._inductor.runtime.triton_heuristics as _ind_th  # type: ignore
        if isinstance(getattr(_ind_th, 'TRITON_MAX_BLOCK', None), dict):
            _ind_th.TRITON_MAX_BLOCK["X"] = max(_ind_th.TRITON_MAX_BLOCK.get("X", 4096), 8192)
    except Exception:
        pass

    # Ensure Triton kernels compile in-process so the above override is visible
    # (Inductor normally uses subprocess workers; setting threads=1 disables that path.)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    try:
        import torch._inductor.config as _ind_cfg  # type: ignore
        _ind_cfg.compile_threads = 1
    except Exception:
        pass
    if config.compile_model and not args.test:
        try:
            if rank == 0:
                print("[compile] Compiling model with torch.compile... this can take a while")
                sys.stdout.flush()
                _t0 = time.time()
            model = torch.compile(model)
            if rank == 0:
                print(f"[compile] Done in {time.time()-_t0:.1f}s")
        except Exception as e:
            if rank == 0:
                print(f"torch.compile failed: {e}")

    if world_size > 1 and not args.test:
        # Broadcast buffers rarely needed here; disabling saves time. Enable gradient view for fewer copies.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=False,
            static_graph=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        # Optional FP16 gradient compression hook to reduce all-reduce bandwidth
        if args.ddp_fp16_compress:
            try:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                model.register_comm_hook(state=None, hook=fp16_compress_hook)
            except Exception as e:
                if rank == 0:
                    print(f"[warn] Failed to enable DDP FP16 compression hook: {e}")
    
    # Init W&B only on rank 0
    if rank == 0 and config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=(os.environ.get('WANDB_NAME') or config.run_name or f"bd3lm-bs{config.block_size}-{uuid.uuid4().hex[:8]}"),
            config=asdict(config)
        )

    trainer = BlockDiffusionTrainer(model, config)
    # If resuming, set trainer step and optionally load optimizer state
    if resume_ckpt_path is not None:
        # carry step forward
        trainer.step = trainer_step_from_ckpt
        # Optimizer states (optional)
        try:
            opt_mu = ckpt_state.get('optimizer_muon')
            if opt_mu is not None and trainer.optimizer_muon is not None:
                trainer.optimizer_muon.load_state_dict(opt_mu)
        except Exception as e:
            if rank == 0:
                print(f"[warn] Could not load Muon optimizer state: {e}")
        try:
            opt_adam = ckpt_state.get('optimizer_adam')
            if opt_adam is not None and trainer.optimizer_adam is not None:
                trainer.optimizer_adam.load_state_dict(opt_adam)
        except Exception as e:
            if rank == 0:
                print(f"[warn] Could not load AdamW optimizer state: {e}")
        # RNG states to maintain continuity best-effort
        try:
            if 'torch_rng_state' in ckpt_state:
                torch.random.set_rng_state(ckpt_state['torch_rng_state'])
            if 'cuda_rng_state' in ckpt_state and ckpt_state['cuda_rng_state'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt_state['cuda_rng_state'])
        except Exception as e:
            if rank == 0:
                print(f"[warn] Could not restore RNG states: {e}")

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
        if resume_ckpt_path is not None:
            # Continue writing into the same checkpoint directory as the loaded CKPT
            ckpt_dir = resume_ckpt_path.parent
            # If no run_name provided, align it to directory name for logging
            if config.run_name is None:
                config.run_name = ckpt_dir.name
        else:
            ckpt_dir = Path(__file__).parent / 'checkpoints' / (config.run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        if rank == 0:
            samples_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.ckpt_dir = ckpt_dir
        if rank == 0:
            print(f"Using data: {config.train_files}")
        base_train_sampler = build_sampler(
            filename_pattern=config.train_files,
            batch_size=config.batch_size,
            seq_len=config.max_seq_len,
            device=torch.device(config.device),
            align_to_bos=config.align_to_bos,
        )
        base_val_sampler = build_sampler(
            filename_pattern=config.val_files,
            batch_size=config.batch_size,
            seq_len=config.max_seq_len,
            device=torch.device(config.device),
            align_to_bos=True,
        )
        # Wrap with CUDA prefetch to overlap H2D copies with compute
        if torch.device(config.device).type == 'cuda':
            train_sampler = PrefetchingSampler(base_train_sampler, torch.device(config.device))
            val_sampler = PrefetchingSampler(base_val_sampler, torch.device(config.device))
        else:
            train_sampler = base_train_sampler
            val_sampler = base_val_sampler
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

            if (config.eval_interval > 0) and (config.val_tokens > 0) and (trainer.step % config.eval_interval == 0):
                val_metrics = evaluate_val_loss(trainer, config, val_sampler, rank, world_size)
                if rank == 0:
                    ce_val = val_metrics.get('ce', None)
                    bd3lm_val = val_metrics.get('bd3lm', None)
                    val_ppl = math.exp(ce_val) if (ce_val is not None and not math.isinf(ce_val)) else float('inf')
                    ce_str = f"{ce_val:.4f}" if (ce_val is not None and not math.isinf(ce_val)) else "N/A"
                    bd3lm_str = f"{bd3lm_val:.4f}" if (bd3lm_val is not None and not math.isinf(bd3lm_val)) else "N/A"
                    print(f"Eval step {trainer.step}: val_loss_subs = {bd3lm_str} | val_loss_ce = {ce_str} | val_ppl = {val_ppl:.2f}")
                    if config.use_wandb:
                        wandb.log({
                            "val_loss_subs": val_metrics['bd3lm'],
                            "val_loss_ce": val_metrics['ce'],
                            "val_ppl": val_ppl,
                            "step": trainer.step
                        })
                    # Generate and write sample rollouts (optional)
                    if config.samples_per_eval and config.samples_per_eval > 0:
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
