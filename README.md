An small improvement over classical Block Diffusion (BD3LM, from Vladimir Kuleyshov's group; ICLR 2025 Oral), written in the style of NanoGPT. 

The code has been vetted to run on an 8xH200 system. It is set to around ~150M parameters (close to GPT-2). Set `SAMPLES_PER_EVAL=N` where `N > 1` to trigger sample generation during training. Turn it off for faster training. 
