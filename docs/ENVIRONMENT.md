# Environment

## Training dependencies

Use Python 3.10 or 3.11 for the pinned dependencies.
Install the dependencies in [requirements.txt](../requirements.txt) in an isolated
environment with a CUDA-enabled PyTorch build compatible with the GPU and driver.
Training and validation require CUDA.

Use Linux with the multiprocessing start method set to `fork`; the DataLoader
workers require inherited runtime and tokenizer state. Windows, `spawn` and
`forkserver` are not supported by this entry. The entry uses the environment's
start method, with 56 training workers and 32 validation workers by default.
Set `-train_workers` and `-val_workers` to non-negative counts, including 0.
