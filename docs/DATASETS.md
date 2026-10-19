# Data and resource configuration

Arrange the data and initialisation assets as follows:

```text
<resource_root>/
  dataset/
    Fakeddit/
      multimodal_train_cleaned.tsv
      multimodal_validate_cleaned.tsv
      images/images/<id>.jpg
    Yang/
      yang_train.tsv
      yang_validate.tsv
      images/<image_name>
  checkpoint/
    mae_pretrain_vit_base.pth
  ernie/models--nghuyong--ernie-2.0-base-en/
```

Copy [paths.example.json](../config/paths.example.json) to the ignored
`config/local.json`; set `resource_root` to that external directory. Relative
configured roots are based on this repository. Explicit `prefix_map` values use
longest matching path prefixes.

Use `GAMED_CONFIG` to select a configuration file, or the launcher's `--config`.
`GAMED_RESOURCE_ROOT` overrides the JSON resource root. The example includes only
`resource_root` and `prefix_map`; no output configuration is needed.
Existing `groupshare_root`, `bert_root`, `cache_root` and `output_root` mappings
and their `GAMED_*` environment overrides remain supported. The `FND_*` aliases
are also accepted; `GAMED_*` takes precedence.

`resource_root` maps relative dataset/checkpoint/ERNIE reads and recognised
configured absolute resource paths. Existing local checkpoint files take
precedence over external resources. `output_root` maps destinations explicitly
passed to the write-path resolver; the training entry instead uses `-output_file`
for all run outputs. The launcher does not create directories from `output_root`.
Read routing does not write or extract resources. See [REPRODUCE.md](REPRODUCE.md) for outputs.

## Loading rules

Fakeddit requires `id`, `hasImage`, `clean_title`, `2_way_label`, `3_way_label`
and `6_way_label`. Existing code maps the joint condition 2-way=1, 3-way=0 and
6-way=0 to real (0), and other cases to fake (1); rows without a matching image
are skipped. Yang requires `image_name`, `type`, `title` and `text`; real maps to
0 and the text remains title+text without an added separator. Both loaders group
fake records before real records, preserving order within each group. Although
`remove_duplicate=True` is the default, its lookup dictionary is never populated,
so duplicate records remain. Train/validate files are used by the entry.

Fakeddit checks image basenames, then constructs `<id>.jpg`; the resource collection
must supply those JPEG filenames. Yang uses its `image_name` values. PIL decoding
produces float32 values divided by 255, discards channels beyond the first three
and expands greyscale to HWC. After channel conversion and resizing, the loaders
reverse three-channel RGB to BGR and return CHW tensors, without ImageNet mean/std
normalisation. The two-channel conversion allocates uint8 output, quantising the
normalised inputs before resizing.
The supported loaders resize images without cropping; they do not apply
rotation, flip or scale augmentation.

## Obtaining and preparing data

`data/` contains loading code. Actual tables and images belong in the external
`dataset/` tree above; no empty data directory is required in the repository.
Prepare resources in a separate directory and keep the original downloads.
The loaders read extracted images, not image ZIPs. Both training and validation
require the train and validation tables for the selected datasets.

### Fakeddit

The [official Fakeddit project](https://github.com/entitize/Fakeddit) provides
the v2.0 text/metadata and image download instructions. Use its multimodal data
as the source. Its image downloader saves files as `<id>.jpg`; place the images
in `dataset/Fakeddit/images/images/` under `resource_root`.

This loader requires `multimodal_train_cleaned.tsv` and
`multimodal_validate_cleaned.tsv`, identified in the
[resource manifest](../resources/manifest.json). They are derived tables, not
the upstream TSVs under new filenames. The release does not include a verified
raw-to-cleaned conversion: the exact row-selection rules and the correspondence
to the paper's test split still need to be supplied with the prepared tables.
Obtain these prepared files and their matching images before running GAMED;
renaming the original downloads does not reproduce the same inputs.

### Yang

The [TI-CNN paper](https://arxiv.org/pdf/1806.00749), Section IV-A, links to the
authors' [all_data.csv](https://drive.google.com/file/d/0B3e3qZpPtccsMFo5bk9Ib3VCc2c/view?usp=sharing&resourcekey=0-_eqAfKOCKbuE-xFFCmEzyg).
This is the raw table, not the prepared GAMED splits or a complete local image
collection. It includes image URLs; the GAMED tables also require `image_name`
and matching files in `dataset/Yang/images/`.

Obtain `yang_train.tsv` and `yang_validate.tsv` matching the resource manifest,
and place them as shown above. The exact image selection and split recipe is
not distributed here. URL basenames and a CSV-to-TSV conversion alone do not
establish the GAMED splits. The tables identified by the manifest contain 4,655
training and 1,165 validation rows, with no shared IDs but 117 shared image filenames; the paper
reports 4,655/582/583 train/validation/test. Their validation/test correspondence
must be confirmed before comparing results with the paper.

The entry uses train/validate files; its validation inputs are not established
as the paper's independent test sets. See [CHECKPOINTS.md](CHECKPOINTS.md) for
initialisation assets and [THIRD_PARTY.md](THIRD_PARTY.md) for resource terms.
