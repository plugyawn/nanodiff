#!/usr/bin/env python3
"""
Evaluate a BD3LM Speedrun checkpoint and sample a few sentences.

Usage examples:

  # Evaluate and sample from the latest ckpt in a run dir
  python eval_and_sample.py \
    --ckpt_dir checkpoints/bd3lm-speedrun-bs16-20250827-031731 \
    --num_samples 3 --length 160 --top_p 0.95

  # Or point directly to a ckpt file
  python eval_and_sample.py --ckpt_path checkpoints/.../ckpt_step001000.pt

Outputs:
  - Prints validation loss (same objective as training: SUBS/SEDD/AR)
  - Prints N sampled texts to stdout
"""

import argparse
import sys
from pathlib import Path
import torch

# Reuse training code
from train_bd3lm import (
    Config,
    BlockDiffusionLM,
    BlockDiffusionTrainer,
    build_sampler,
    evaluate_val_loss,
)


def find_latest_ckpt(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    cand = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not cand:
        raise FileNotFoundError(f"No ckpt_step*.pt files found under {ckpt_dir}")
    return cand[-1]


def load_model_from_ckpt(ckpt_path: Path, device: str | None = None):
    state = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = state.get("config")
    if not cfg_dict:
        raise KeyError("Checkpoint missing 'config' field")
    cfg = Config(**cfg_dict)
    # Place model on requested or best-available device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    model = BlockDiffusionLM(cfg).to(cfg.device)
    msg = model.load_state_dict(state["model_state"], strict=True)
    return cfg, model, state


def main():
    ap = argparse.ArgumentParser(description="Evaluate and sample from a BD3LM checkpoint")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing ckpt_step*.pt")
    ap.add_argument("--ckpt_path", type=str, default=None, help="Path to a specific ckpt .pt file")
    ap.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda:0")
    ap.add_argument("--num_samples", type=int, default=3, help="How many texts to sample")
    ap.add_argument("--length", type=int, default=160, help="Sample length (tokens)")
    ap.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    args = ap.parse_args()

    if not args.ckpt_dir and not args.ckpt_path:
        ap.error("Provide either --ckpt_dir or --ckpt_path")

    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = find_latest_ckpt(Path(args.ckpt_dir))

    print(f"Loading checkpoint: {ckpt_path}")
    cfg, model, state = load_model_from_ckpt(ckpt_path, device=args.device)

    # Build a small val sampler and compute val loss in the same objective as training
    try:
        sampler = build_sampler(
            filename_pattern=cfg.val_files,
            batch_size=cfg.batch_size,
            seq_len=cfg.max_seq_len,
            device=torch.device(cfg.device),
            align_to_bos=cfg.align_to_bos,
        )
        trainer = BlockDiffusionTrainer(model, cfg)
        val_loss = evaluate_val_loss(trainer, cfg, sampler, rank=0, world_size=1)
        print(f"Validation loss ({cfg.parameterization}): {val_loss:.6f}")
    except Exception as e:
        print(f"[warn] Validation failed: {e}")

    # Sample a few texts
    trainer = BlockDiffusionTrainer(model, cfg)
    if cfg.parameterization == "ar":
        texts = trainer.sample_ar(model, length=min(args.length, cfg.max_seq_len), num_samples=args.num_samples, top_p=args.top_p)
    else:
        texts = trainer.sample_bd3lm_blockwise(model, length=min(args.length, cfg.max_seq_len), num_samples=args.num_samples, top_p=args.top_p)
    print("\n=== Samples ===")
    for i, t in enumerate(texts, 1):
        print(f"[{i}] {t}\n")


if __name__ == "__main__":
    main()
