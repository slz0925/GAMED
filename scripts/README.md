# Running GAMED

Both Shell scripts locate the repository root and use the Python launcher.
They inherit CUDA visibility; `PYTHON` may select the Python executable.

| Script | Purpose |
|---|---|
| [train_yang.sh](train_yang.sh) | Train GAMED on Yang and validate after each epoch |
| [validate_fakeddit.sh](validate_fakeddit.sh) | Validate a trained GAMED checkpoint on Fakeddit |

```bash
bash scripts/train_yang.sh
bash scripts/validate_fakeddit.sh --help
```

For validation, pass `-checkpoint` with the path to your trained state dict.
The script resolves it through the current resource configuration and requires
a file before starting model initialisation. Relative paths are based on the
repository root. Quote paths containing spaces. See
[CHECKPOINTS.md](../docs/CHECKPOINTS.md) for the required initialisation assets.

Append optional entry arguments such as `-batch_size`, `-epochs`, `-device` and
`-output_file` to either script. Unspecified options use the
[entry defaults](../docs/REPRODUCE.md). The scripts fix the dataset, model and
training/validation mode; use `run_gamed.py` directly for other selections.
Validation also exports features and saves a state dict.

To select a configuration, set `GAMED_CONFIG` before invoking a Shell script.
For a dry-run plan, use `scripts/run.py --dry-run run_gamed.py` with the required
model and dataset selectors. Launcher options `--config` and `--dry-run` belong
before `run_gamed.py`; they are not forwarded entry options.
