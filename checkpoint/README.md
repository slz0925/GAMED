# Checkpoint resources and local outputs

Initialisation weights and trained checkpoints are loaded from the configured
resource directory. The default save destination is `checkpoint/<training_dataset>/`;
an existing local checkpoint is read from its specified location.
All payloads/results are ignored by Git. Only this note and directory placeholders
are distributed. See [CHECKPOINTS.md](../docs/CHECKPOINTS.md) and
[REPRODUCE.md](../docs/REPRODUCE.md).
