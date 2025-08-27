#!/bin/bash
# Setup script for BD3LM Speedrun

echo "BD3LM Speedrun Setup"
echo "===================="

# Check PyTorch version
python -c "import torch; assert torch.__version__ >= '2.5.0', f'PyTorch {torch.__version__} < 2.5.0 (required for FlexAttention)'"
if [ $? -eq 0 ]; then
    echo "✓ PyTorch version OK"
else
    echo "❌ Please upgrade PyTorch: pip install torch>=2.5.0"
    exit 1
fi

# Check FlexAttention availability
python -c "from torch.nn.attention.flex_attention import create_block_mask, flex_attention"
if [ $? -eq 0 ]; then
    echo "✓ FlexAttention available"
else
    echo "❌ FlexAttention not available. Installing PyTorch nightly..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
fi

# Install other requirements
echo "Installing requirements..."
pip install wandb tiktoken datasets transformers tqdm numpy

# Create necessary directories
mkdir -p checkpoints logs data_cache sample_logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start training:"
echo "  Single GPU:  python train_speedrun.py --block_size 16"
echo "  Multi-GPU:   bash run.sh 16"
echo ""
echo "To test the model:"
echo "  python test_model.py"