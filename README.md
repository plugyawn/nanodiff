`nanodiff` is a small, experimental improvement over classical Block Diffusion (BD3LM, from Vladimir Kuleyshov's group; ICLR 2025 Oral), written in the style of Keller Jordan's modded-NanoGPT. 
Block diffusion is a new, semi-autoregressive method of text generation that combines the parallelization allowed by discrete diffusion models (used in Llada), and the benefits of an autoregressive backend (KV Cache, etc) from autoregressive transformers.

THe current repo, started with `run.sh <block_size>` trains over a subset of FineWeb and logs results to `wandb` for tracking. 

The code has been vetted to run on an 8xH200 system. It is set to around ~150M parameters (close to GPT-2). Set `SAMPLES_PER_EVAL=N` where `N > 1` to trigger sample generation during training. Turn it off for faster training. 

Current training token throughput on an 8xH200 system is over 1M tokens/second, which seems to be about right for a GPT-2 style architecture. 
Grouped Query Attention, EMA, gradient accumulation and activation checkpointing are supported as of now, with further plans for parallel decoding (where I have huge hopes) and better handling of the $x_0 | x_t$ streams, where there seems to be FLOPs to be better utilized still. 


