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
`FND_CONFIG` and `FND_RESOURCE_ROOT` are also accepted; `GAMED_*` takes precedence.

`resource_root` maps repository-relative or repository-absolute dataset,
checkpoint and ERNIE reads. Existing local resource files take precedence.
Use explicit `prefix_map` entries for other locations and `-output_file` for
run outputs. Read routing does not write or extract resources.
See [REPRODUCE.md](REPRODUCE.md) for outputs.

## Loading rules

Fakeddit requires `id`, `hasImage`, `clean_title`, `2_way_label`, `3_way_label`
and `6_way_label`. The loader maps the joint condition 2-way=1, 3-way=0 and
6-way=0 to real (0), and other cases to fake (1); rows without a matching image
are skipped. Yang requires `image_name`, `type`, `title` and `text`; real maps to
0 and the text remains title+text without an added separator. Both loaders group
fake records before real records, preserving order within each group. Each news
record is retained, including records that share an image and label.
Train/validate files are used by the entry.
Dataset items contain `(text, image, label, image_path)`,
with a float32 CHW image and the actual image path. The collate function tokenises
the batch once and preserves its order.

Fakeddit checks image basenames, then constructs `<id>.jpg`; the resource collection
must supply those filenames. Yang uses its `image_name` values. PIL decodes
palette, CMYK and binary images to RGB before float32 conversion and division
by 255. RGB colours are retained; greyscale is repeated across three channels,
and alpha is ignored for RGBA and LA images. High-bit-depth integer and floating
images (`I`, `I;16` and `F` modes) are rejected because their numeric ranges are
not defined by this 8-bit input pipeline. After resizing, the loaders reverse
RGB to BGR and return float32 CHW tensors in [0, 1], without ImageNet mean/std
normalisation.
The supported loaders resize images without cropping; they do not apply
rotation, flip or scale augmentation.

## Obtaining and preparing data

`data/` contains loading code. Actual tables and images belong in the external
`dataset/` tree above; no empty data directory is required in the repository.
Prepare resources in a separate directory and keep the original downloads.
The loaders read extracted images, not image ZIPs. Training requires its training
table and the selected validation table.
Validation-only runs require only the selected validation table and images.

### Fakeddit

The [official Fakeddit project](https://github.com/entitize/Fakeddit) provides
the v2.0 text/metadata and image download instructions. Use its multimodal data
as the source. Its image downloader saves files as `<id>.jpg`; place the images
in `dataset/Fakeddit/images/images/` under `resource_root`.

This loader requires `multimodal_train_cleaned.tsv` and
`multimodal_validate_cleaned.tsv`. These are derived tables, not
the upstream TSVs under new filenames. Obtain these prepared files and their
matching images before running GAMED. Preparation scripts for the derived
tables are not included; renaming the original downloads does not prepare them.

### Yang

The [TI-CNN paper](https://arxiv.org/pdf/1806.00749), Section IV-A, links to the
authors' [all_data.csv](https://drive.google.com/file/d/0B3e3qZpPtccsMFo5bk9Ib3VCc2c/view?usp=sharing&resourcekey=0-_eqAfKOCKbuE-xFFCmEzyg).
This is the raw table, not the prepared GAMED splits or a complete local image
collection. It includes image URLs; the GAMED tables also require `image_name`
and matching files in `dataset/Yang/images/`.

Obtain `yang_train.tsv` and `yang_validate.tsv`, and place them with their
matching images as shown above. These prepared tables
contain 4,655 training and 1,165 validation rows. Their preparation scripts are
not included; converting `all_data.csv` to TSV alone does not prepare them.

See [CHECKPOINTS.md](CHECKPOINTS.md) for
initialisation assets and [THIRD_PARTY.md](THIRD_PARTY.md) for resource terms.
