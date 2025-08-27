# BD3LM Speedrun: Discrete Block Diffusion Language Model

A correct implementation of Block Diffusion Language Models (BD3LM) with NanoGPT speedrun optimizations.

## Key Features

### ✅ Correct Discrete Diffusion
- **MASK tokens**: Proper discrete corruption with vocabulary + 1 for MASK
- **Block-wise masking**: Masks entire blocks based on diffusion timestep
- **Loss on masked only**: Computes loss only on corrupted positions
- **Discrete noise schedule**: Cosine schedule for masking probabilities

### 🚀 NanoGPT Speedrun Optimizations
- **Muon optimizer** with Newton-Schulz orthogonalization
- **Modern architecture**: RoPE, QK-Norm, ReLU²
- **Value embeddings** with U-Net skip connections
- **FlexAttention** with block-aware masking
- **Softcap logits** (15.0)

## Quick Start

```bash
# Test the implementation
python train_bd3lm.py --test

# Run training on 8 GPUs with block size 16
bash run.sh 16

# Single GPU training
python train_bd3lm.py --block_size 16 --batch_size 8
```

## Model Details

- **Parameters**: 131.2M (GPT-2 scale)
- **Architecture**: 12 layers, 768 dim, 12 heads
- **Vocabulary**: 50,304 + 1 MASK token = 50,305
- **Block size**: Configurable (default 16)
- **Diffusion steps**: 1000

## Training

The model uses discrete diffusion:
- Tokens are corrupted by replacing with MASK token (not Gaussian noise)
- Loss is computed only on masked positions
- Generation happens block-by-block autoregressively
