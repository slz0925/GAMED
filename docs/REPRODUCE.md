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
| `-train_workers`, `-val_workers` | 56, 32; non-negative DataLoader worker counts, including 0 |
| `-checkpoint` | Empty; otherwise loads a state dict after model initialisation |
| `-output_file` | `checkpoint/`; root for all checkpoints, feature files and predictions, organised by dataset pair, model, run and epoch |
| `-finetune` | 0; enables alternate learning rates for faster parameter groups |
| `-device` | CUDA device 0; respects inherited CUDA visibility |

Seed 25 and cuDNN deterministic=False/benchmark=True are configured.
The entry uses AdamW with three name-based groups: image_encoder/text_encoder at
1e-5, ip_backbone/irrelevant_tensor at 1e-4, and remaining parameters at 5e-5.
With `-finetune` enabled, all three
use 1e-5. Each uses betas (0.9, 0.999) and weight decay 0.01.
Encoder optimiser updates start at zero-based epoch 6;
the schedules use CosineAnnealingLR and UntunedLinearWarmup.
The cosine period is `int(len(train_loader) * epochs * 1.1)`; optimisers and
schedulers are constructed only for training.
The loss is mixture BCE plus the mean of IS/T/IP BCE.
For `gamed`, the mixture BCE uses the logit returned by Veto Voting in both
training and validation. Selected expert values retain gradients; see
[MODEL_MAP.md](MODEL_MAP.md) for the voting rules and their training effects.

## Validation and outputs

`-test_dataset` selects the validation dataset, not a separate test split.
Validation initialises the selected validation dataset, model and loss. It does
not read training tables or construct training optimisers and schedulers.
Evaluation requests `return_features=True` and writes five
feature/label arrays and `predictions.xlsx` into the corresponding epoch directory.
`predictions.jsonl` stores the complete text, labels, probabilities and features,
with one JSON object per sample in the same order.

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
            │   ├── predictions.xlsx
            │   └── predictions.jsonl
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
The Excel `image_no` field is a zero-based export row number, aligned with the
feature files. `GT_path` is not exported, so this number is not an image filename.

The save loop writes a state dict after every validation pass, including in
validation mode, under the run's `checkpoints/` directory. The filename is
`epoch_<number>.pkl`; dataset pair, model variant and run identity are recorded by
its parent directories. Existing checkpoint files can still be loaded by path,
regardless of their filenames.

Evaluation scans 100 thresholds from 0.45 to 0.549 and reports the highest
validation accuracy and its threshold value. Ties use the first maximum.
Metrics include every sample, including singleton
batches. F1 is `2*TP/(2*TP+FP+FN)`. Undefined precision, recall and F1 values are
reported as zero, independently for each class. The per-class accuracy fields
report recall. Returned `val_loss` is the sample-mean mixture loss over the whole
validation loader. Evaluation supports scalar mean- or sum-reduced binary loss;
the entry uses mean-reduced `BCEWithLogitsLoss`. Empty loaders and incorrectly
shaped binary logits are rejected explicitly.
See [MODEL_MAP.md](MODEL_MAP.md) for the veto output convention.

## Log and spreadsheet fields

Training logs use `Mix_loss`, `IS_loss`, `T_loss`, `IP_loss`, `MM_mean_prob`,
`Irr_mean` and `Train_acc`. `MM_mean_prob` is the mean sigmoid probability of
the MM auxiliary prediction, not a loss. `Irr_mean` is the mean of the learned
`irrelevant_tensor`. Validation reports `Mix_loss`, `IS_loss`, `T_loss` and `IP_loss`.
Epoch loss and accuracy are sample-weighted; progress values use each batch's
mean loss and accuracy. Training predicts Fake at probability >= 0.5.

The spreadsheet stores `final_feature`, `is_feature`, `t_feature`, `ip_feature`
and `mm_feature` in the same order as the model's returned feature tuple.
`mm_feature` is the 64-dimensional MM auxiliary feature. `y_pred` is the Mix
probability; `y_pred_is`, `y_pred_t`, `y_pred_ip` and `y_pred_mm` are branch
probabilities. Excel's `text_preview` escapes control characters that Excel cannot
store and marks shortened text with a suffix and `text_truncated=True`. Complete
original text is in the JSONL `text` field. Both exports use the same `image_no`,
labels, probabilities and features.
