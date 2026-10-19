# Training, validation and reproducibility

The training and validation entry is [run_gamed.py](../run_gamed.py); `scripts/run.py`
provides optional configuration and a dry-run plan. Both set the project root as
the working directory. The CLI explicitly requires `-network_arch` (`gamed` or
`gamed_without_veto`) and `-train_dataset/-test_dataset` (`fakeddit` or `yang`).
Names outside these choices are rejected.

The [Shell entrypoints](../scripts/README.md) provide `train_yang.sh`,
`validate_yang.sh`, `train_fakeddit.sh` and `validate_fakeddit.sh`. Each selects
`gamed` and fixes both dataset selectors to its named dataset; training uses
`-val 0` and validation uses `-val 1`. Both validation scripts require a mapped
checkpoint file before model initialisation, except when requesting `--help`.
Other entry arguments are forwarded unchanged.

| Option | Default / effect |
|---|---|
| `-val` | 0 trains then validates each epoch; nonzero skips training and limits the loop to one validation pass |
| `-batch_size`, `-epochs` | 16, 100 when not supplied; the Shell scripts use these entry defaults unless overridden |
| `-checkpoint` | Empty; otherwise loads a state dict after model initialisation |
| `-output_file` | `checkpoint/`; root for all checkpoints, feature files and predictions, organised by dataset pair, model, run and epoch |
| `-finetune` | 0; enables alternate learning rates for faster parameter groups |
| `-device` | CUDA device 0; respects inherited CUDA visibility |
| `-is_filter`, `-duplicate_fake_times`, `-is_sample_positive`, `-not_on_12` | Accepted for command compatibility and omitted from help; they do not change supported training or validation |

Seed 25, cuDNN deterministic=False/benchmark=True, training workers 56 and
validation workers 32 are configured. The current loader requires the Linux
`fork` environment described in [ENVIRONMENT.md](ENVIRONMENT.md).
The entry uses AdamW with three name-based groups: image_encoder/text_encoder at
1e-5, ip_backbone/irrelevant_tensor at 1e-4, and remaining parameters at 5e-5.
With `-finetune` enabled, all three
use 1e-5. Each uses betas (0.9, 0.999) and weight decay 0.01.
Encoder optimiser updates start at zero-based epoch 6;
the schedules use CosineAnnealingLR and UntunedLinearWarmup.
The cosine period is `int(len(train_loader) * epochs * 1.1)`, constructed before
validation mode reduces the epoch loop to one pass.
The loss is mixture BCE plus the mean of IS/T/IP BCE, with a zero
ambiguity term in these dataset settings.
For `gamed`, the mixture BCE uses the logit returned by Veto Voting in both
training and validation. Selected expert values retain gradients; see
[MODEL_MAP.md](MODEL_MAP.md) for the voting rules and their training effects.

## Actual validation and output protocol

`-test_dataset` selects the validation dataset, not a separate test split.
Validation still constructs the training dataset/loader and initialises the model,
losses and optimisers. It requires the corresponding training resources even
when `-val 1`. Evaluation requests `return_features=True` and writes five
feature/label arrays and `predictions.xlsx` into the corresponding epoch directory.

`-output_file` accepts a relative or absolute output root, including paths with
spaces when quoted. Relative roots are resolved from the repository root. Resource
read mappings do not redirect this output root. Each invocation reserves a new run
directory using a UTC timestamp and a system-generated UUID, without consuming the
Python, NumPy or PyTorch random streams:

```text
<output_file>/
└── train_yang__val_fakeddit/
    └── gamed/
        └── train_<UTC timestamp>_<UUID>/
            ├── checkpoints/
            │   ├── epoch_001.pkl
            │   └── epoch_002.pkl
            ├── epoch_001/
            │   ├── mix_features.txt
            │   ├── is_features.txt
            │   ├── t_features.txt
            │   ├── ip_features.txt
            │   ├── mm_aux_features.txt
            │   └── predictions.xlsx
            └── epoch_002/
```

The dataset pair and model directory reflect the actual CLI selectors. Validation
runs use a `validate_` prefix and produce `epoch_001`; that number identifies the
current validation pass, not the training epoch of the loaded checkpoint. Directory
creation rejects reuse of an existing run or epoch directory. Each epoch's
checkpoint and exports share the same one-based epoch number. A custom output root
inside the repository is covered by the `train_*__val_*/` Git ignore rule.

Each TXT contains a label followed by 64 features, in the original sample order.
`mix_features.txt` holds the final mixture features; `is_features.txt`,
`t_features.txt` and `ip_features.txt` hold the IS/T/IP lite features before AdaIN.
`mm_aux_features.txt` holds `mm_aux_trim(mm_consistency_feature)`, not the main
768-dimensional MM representation. Direct callers of `evaluate()` must pass an
existing epoch directory through the required `output_directory` keyword.
The Excel `image_no` field remains zero for every record and `GT_path` is not
exported, so that field cannot identify source images.

The save loop writes a state dict after every validation pass, including in
validation mode, under the run's `checkpoints/` directory. The filename is
`epoch_<number>.pkl`; dataset pair, model variant and run identity are recorded by
its parent directories. Existing checkpoint files can still be loaded by path,
regardless of their filenames.

Evaluation scans 100 thresholds from 0.45 to 0.549 and reports the highest
validation accuracy. The log prints the threshold index rather than the
corresponding threshold value. F1 uses `max(1, P+R)` in its denominator, which differs
from standard F1 when P+R is below 1; it agrees when P+R is at least 1.
For a batch containing one sample, `squeeze()` produces a scalar prediction and
the caught IndexError skips its metrics and Excel record, although its features
and label have already been collected. This includes a final singleton batch.
Returned `val_loss` is the last processed batch's loss, not a dataset average.
See [MODEL_MAP.md](MODEL_MAP.md) for the veto output convention.

## Tests

Run the source and publication checks from the repository root:

```bash
python -B -m unittest discover -s tests -v
python -B tools/check_release.py
```

Tests cover source computations, image I/O, resource routing and entry dispatch.
Bash integration tests run when Bash is available. With PyTorch installed,
Veto tests execute the actual voting method on CPU tensors, including gradients
and the loss/evaluation interface. Checkpoint tests cover key conversion,
strict loading and optimiser grouping using small synthetic modules. Tensor
tests are skipped when PyTorch is absent.
These tests do not initialise the full networks or load trained checkpoints.
Numerical evaluation requires the model
environment and resources described in [ENVIRONMENT.md](ENVIRONMENT.md),
[DATASETS.md](DATASETS.md) and [CHECKPOINTS.md](CHECKPOINTS.md).

The English tokeniser and MAE/timm dependencies load before research seeding and
model construction. Entry import, help and dry-run do not load research assets.

## Log and spreadsheet fields

Training logs use `Mix_loss`, `IS_loss`, `T_loss`, `IP_loss`, `MM_mean_prob`,
`Irr_mean` and `Train_acc`. `MM_mean_prob` is the mean sigmoid probability of
the MM auxiliary prediction, not a loss. `Irr_mean` is the mean of the learned
`irrelevant_tensor`. The optional ambiguity branch reports `MM_ambiguity_loss`.
Validation reports `Mix_loss`, `IS_loss`, `T_loss` and `IP_loss`.

The spreadsheet stores `final_feature`, `is_feature`, `t_feature`, `ip_feature`
and `mm_feature` in the same order as the model's returned feature tuple.
`mm_feature` is the 64-dimensional MM auxiliary feature. `y_pred` is the Mix
probability; `y_pred_is`, `y_pred_t`, `y_pred_ip` and `y_pred_mm` are branch
probabilities. Consumers of the earlier `image_feature`, `text_feature`,
`vgg_feature`, `y_pred_img`, `y_pred_text` and `y_pred_vgg` columns should use
the corresponding IS, T and IP column names. Excel columns and values are
independent of the run directory and export filenames.
