# Checkpoints and initialisation

Prepare the following assets under the external `resource_root` configured in
[DATASETS.md](DATASETS.md). Both training and validation load them before any
trained GAMED state dict.

## MAE ViT-Base

Download the ViT-Base **pre-trained** checkpoint from the
[Meta MAE project](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints):
[mae_pretrain_vit_base.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).
Place it at `checkpoint/mae_pretrain_vit_base.pth` under `resource_root`.
The constructor reads its `model` state dict. The ImageNet fine-tuned checkpoint
is a different asset.

## English ERNIE 2.0 Base

Obtain the model and tokenizer files from
[nghuyong/ernie-2.0-base-en](https://huggingface.co/nghuyong/ernie-2.0-base-en).
This is a PyTorch conversion of the official PaddlePaddle ERNIE model.
Place the following [files](https://huggingface.co/nghuyong/ernie-2.0-base-en/tree/main)
directly in `ernie/models--nghuyong--ernie-2.0-base-en/` under `resource_root`:

```text
config.json
pytorch_model.bin
vocab.txt
tokenizer_config.json
special_tokens_map.json
```

`AutoModel` and `AutoTokenizer` use the same directory resolver. Explicit
`prefix_map` entries take precedence over `resource_root`. Without a mapping,
the repository-relative directory is used.
If using a Hugging Face cache,
copy the complete snapshot contents into this location, including the actual
weight file; a cache container containing only `blobs/`, `refs/` and `snapshots/`
is not the expected layout. Chinese ERNIE1 is not required.

Alternative exports must preserve model and tokeniser compatibility. Preserve
the applicable component licences when obtaining or redistributing weights.

## Trained GAMED weights

MAE and ERNIE weights initialise components; they are not a trained GAMED model.
This release does not provide a download for trained GAMED weights. Use a state
dict from your selected model variant and pass its path with `-checkpoint`.
Both [validation scripts](../scripts/README.md), `validate_yang.sh` and
`validate_fakeddit.sh`, require this argument and check that the mapped file exists
before model initialisation. `--help` needs no checkpoint. The Python entry retains
its optional checkpoint interface.

Model construction loads MAE and English ERNIE before the GAMED state dict.
The IP branch constructs IRNv2 without separately loading ImageNet weights;
see [MODEL_MAP.md](MODEL_MAP.md).

The two variants share a constructor structure but have different forward paths.
Record the model selector, data version and command with each result. The loader
accepts current keys and automatically maps compatible earlier module names,
including an optional `module.` prefix. Both model variants also support this
mapping through `load_state_dict`. Loading is strict by default: missing keys,
unexpected keys, conflicting aliases and shape mismatches raise an error.
Explicit `strict=False` permits partial loading with a warning listing missing
and unexpected keys; collisions and shape mismatches still fail. When loading
older state dicts, only the eight IS/T auxiliary gate tensors with their expected
shapes are discarded. Saved state dicts use the registered module names.
This repository saves state dicts, not whole-model Python pickle objects.
Weights are deserialised on CPU and copied into the model's current device.
For direct model use, move the complete model and inputs to the same device with
`model.to(device)`.
Checkpoints are written to the current run's `checkpoints/epoch_<number>.pkl`;
the parent directories identify both datasets, the model and the run. Loading uses
the supplied path and state-dict keys, not the checkpoint filename.
See [REPRODUCE.md](REPRODUCE.md) for the output
layout and validation behaviour.
