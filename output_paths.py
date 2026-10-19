"""Create isolated output directories without using research random generators."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_run_directory(output_root, train_dataset, validation_dataset, model_variant, validation=False):
    """Reserve a new run directory; an existing run is never reused."""
    if train_dataset not in ('yang', 'fakeddit') or validation_dataset not in ('yang', 'fakeddit'):
        raise ValueError('Unsupported output dataset')
    if model_variant not in ('gamed', 'gamed_without_veto'):
        raise ValueError('Unsupported output model variant')
    mode = 'validate' if validation else 'train'
    run_id = mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex
    parent = Path(output_root).resolve() / ('train_' + train_dataset + '__val_' + validation_dataset) / model_variant
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / run_id
    directory.mkdir()
    return directory


def create_epoch_directory(run_directory, epoch):
    """Reserve one epoch's exports, rejecting accidental reuse."""
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Output epoch must be a positive integer')
    directory = Path(run_directory) / f'epoch_{epoch:03d}'
    directory.mkdir()
    return directory
