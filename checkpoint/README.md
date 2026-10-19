# Checkpoint resources and local outputs

Place `mae_pretrain_vit_base.pth` here or under `checkpoint/` in the configured
external resource directory. Pass trained GAMED state-dict files with `-checkpoint`;
existing local files are read from their specified locations.

`-output_file` defaults to `checkpoint/` and controls all run outputs. Saves use
`<output_file>/train_<train_dataset>__val_<validation_dataset>/<model>/<run_id>/`:
state dicts go in `checkpoints/epoch_<number>.pkl`, and features and predictions
go in `epoch_<number>/`. These directories are created as needed and ignored by Git.
See [CHECKPOINTS.md](../docs/CHECKPOINTS.md) for resources and
[REPRODUCE.md](../docs/REPRODUCE.md) for the complete output layout.
