# Running GAMED

All four Shell scripts locate the repository root and use the Python launcher.
They inherit CUDA visibility; `PYTHON` may select the Python executable.

| Script | Purpose |
|---|---|
| [train_yang.sh](train_yang.sh) | Train GAMED on Yang and validate after each epoch |
| [validate_yang.sh](validate_yang.sh) | Validate a trained GAMED checkpoint on Yang |
| [train_fakeddit.sh](train_fakeddit.sh) | Train GAMED on Fakeddit and validate after each epoch |
| [validate_fakeddit.sh](validate_fakeddit.sh) | Validate a trained GAMED checkpoint on Fakeddit |

```bash
bash scripts/train_yang.sh -output_file outputs
bash scripts/train_fakeddit.sh -output_file outputs
bash scripts/validate_yang.sh -checkpoint /path/to/yang_checkpoint.pkl -output_file outputs
bash scripts/validate_fakeddit.sh -checkpoint /path/to/fakeddit_checkpoint.pkl -output_file outputs
```

The `/path/to/...` values above are placeholders: replace them with your existing
trained state-dict files. Trained GAMED weights are not included.
For validation, pass `-checkpoint` with the path to your trained state dict.
Both validation scripts resolve it through the current resource configuration and
require a file before starting model initialisation. `--help` exits without a
checkpoint or model initialisation. Relative paths are based on the
repository root. Quote paths containing spaces. See
[CHECKPOINTS.md](../docs/CHECKPOINTS.md) for the required initialisation assets.

Append optional entry arguments such as `-batch_size`, `-epochs`, `-device` and
`-output_file` to any script. Unspecified options use the
[entry defaults](../docs/REPRODUCE.md). The scripts fix the dataset, model and
training/validation mode; use `run_gamed.py` directly for other selections.
Training scripts use `-val 0`; validation scripts use `-val 1`. Each fixes both
dataset selectors to its named dataset and selects `gamed`. User arguments cannot
override these selectors. Validation uses the validation dataset, still constructs
the training loader, and exports features/results and saves a state dict; prepare
both training and validation resources. See [REPRODUCE.md](../docs/REPRODUCE.md)
for output paths and evaluation behaviour. `-output_file` sets the common output
root for checkpoints, features and predictions; each invocation creates a new run
directory. Use `bash scripts/validate_yang.sh --help` (or any of the other scripts)
to list entry options without starting a run.

To select a configuration, set `GAMED_CONFIG` before invoking a Shell script.
For a dry-run plan, use `scripts/run.py --dry-run run_gamed.py` with the required
model and dataset selectors. Launcher options `--config` and `--dry-run` belong
before `run_gamed.py`; they are not forwarded entry options.
