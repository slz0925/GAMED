# Environment

## Training dependencies

[requirements.txt](../requirements.txt) records the available experiment-package
constraints: torch 2.2.2, torchvision 0.17.2, NumPy 1.26.4, transformers 4.39.3,
timm 0.9.16, positional-encodings 6.0.3, pytorch-warmup 0.1.1,
OpenCV 4.9.0.80, Pillow 10.3.0, pandas 2.2.1, openpyxl 3.1.2 and tqdm 4.66.2.
These are recorded package versions, not a resolved environment lock. The exact
Python, CUDA and driver versions used for training are not yet documented.

Use the successful experiment environment when available. Otherwise install in
an isolated environment with a compatible PyTorch/CUDA build. Model construction
and forward use CUDA. Record the OS, Python, GPU, driver, CUDA build, packages
and model revisions alongside each experiment; retain the same numerical stack
when comparing outputs.

The supported research path is Linux/CUDA with `fork` DataLoader workers.
Workers inherit the parent's initialised `torch` and English tokenizer bindings.
With `spawn` or `forkserver`, a worker imports the entry without establishing
those globals, so the collate function can fail. Windows and these start methods
are not supported by the current entry. Check the environment's actual start
method; Linux alone does not guarantee `fork`. The entry does not set a start
method; training uses 56 workers and validation uses 32.

## Source-test interpreters

The source, filesystem, dispatch and publication checks are verified on
**CPython 3.12.14 and 3.13.5**. These results cover lightweight source tests only, not model execution or
training-package compatibility on Python 3.13.

NumPy and Pillow are needed for the image-decoding tests. The Python 3.13
source-test environment used NumPy 2.2.6 and Pillow 11.3.0.

Pixel tests use real NumPy/Pillow decoding and channel arithmetic. Dataset
channel-order tests use a tensor adapter and a same-size resize stub; they do
not validate PyTorch tensors or OpenCV interpolation.

Tests use the system's TMPDIR/TEMP/TMP directory. Test commands and their scope
are described in [REPRODUCE.md](REPRODUCE.md).
