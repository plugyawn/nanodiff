import math
import torch
from torch.nn.attention.flex_attention import create_block_mask


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


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

    return create_block_mask(
        block_mask_fn, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )


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
        block_q = torch.where(
            x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
        )
        block_kv = torch.where(
            x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
        )

        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (
            (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
        )
        block_causal = (
            (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
        )
        return block_diagonal | offset_block_causal | block_causal

    total_len = 2 * n
    return create_block_mask(
        block_mask_fn,
        B=1,
        H=1,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )


def create_bd3lm_xt_queries_mask(n_tokens: int, block_size: int, device=None):
    """Mask for two-stream: queries are xt (length n), keys/values are xt||x0 (length 2n).
    Allow xt -> xt only within the same block (block diagonal) and xt -> x0 for
    previous blocks only (strict block causality toward x0). Disallow xt -> x0
    of the same block and any future blocks.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = n_tokens

    def block_mask_fn(b, h, q_idx, kv_idx):
        # q_idx in [0, n), kv_idx in [0, 2n)
        q_block = q_idx // block_size
        is_kv_x0 = (kv_idx >= n)
        kv_block = torch.where(is_kv_x0 == 1, (kv_idx - n) // block_size, kv_idx // block_size)

        same_block_xt = (is_kv_x0 == 0) & (kv_block == q_block)
        prev_block_x0 = (is_kv_x0 == 1) & (kv_block < q_block)
        return same_block_xt | prev_block_x0

    return create_block_mask(
        block_mask_fn,
        B=1,
        H=1,
        Q_LEN=n,
        KV_LEN=2 * n,
        device=device,
    )


def create_bd3lm_x0_queries_mask(n_tokens: int, block_size: int, device=None):
    """Mask for two-stream: queries are x0 (length n), keys/values are xt||x0 (length 2n).
    Allow x0 to attend only to x0 in block-causal fashion (previous or same block).
    Disallow attending to xt entirely.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = n_tokens

    def block_mask_fn(b, h, q_idx, kv_idx):
        # q_idx in [0, n) for x0 positions; kv_idx in [0, 2n)
        is_kv_x0 = (kv_idx >= n)
        q_block = q_idx // block_size
        kv_block = torch.where(is_kv_x0 == 1, (kv_idx - n) // block_size, kv_idx // block_size)
        # Only x0->x0, block causal (<=)
        allow = (is_kv_x0 == 1) & (kv_block <= q_block)
        return allow

    return create_block_mask(
        block_mask_fn,
        B=1,
        H=1,
        Q_LEN=n,
        KV_LEN=2 * n,
        device=device,
    )
