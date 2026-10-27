# Third-party components and licences

GAMED's own code uses the root [MIT licence](../LICENSE). Third-party components
retain their copyright notices and licence terms.

The corresponding licence texts remain in `third_party/licenses/`. In particular,
MAE remains subject to CC BY-NC 4.0, including its attribution and noncommercial
conditions. Preserve the applicable notices when redistributing these components.

| Component | Source and licence information |
|---|---|
| MAE and positional embeddings (`models_mae.py`, `util/pos_embed.py`) | Adapted from [Meta MAE](https://github.com/facebookresearch/mae) for unmasked encoder features and NumPy dtype compatibility. [CC BY-NC 4.0](../third_party/licenses/MAE-CC-BY-NC-4.0.txt); copyright notices and the attribution and noncommercial conditions are retained. |
| Inception-ResNet-v2 (`CNN_architectures/pytorch_InceptionResNetV2.py`) | [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch), [BSD 3-Clause](../third_party/licenses/Cadene-BSD-3-Clause.txt). The BayarConv additions are separate from the upstream backbone implementation. |
| Progress utility (`utils.Progbar`) | Based on the [Keras Progbar utility](https://github.com/keras-team/keras/blob/2.2.4/keras/utils/generic_utils.py). The applicable Keras copyright and [MIT licence notice](../third_party/licenses/Keras-MIT.txt) are included for this component. |
| Image utilities (`data/util.py`) | Adapted from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/00bbaf87163956fd00c22db73d051b9e27bcd563/codes/data/util.py), [Apache 2.0](../third_party/licenses/BasicSR-Apache-2.0.txt), with PIL decoding, RGB expansion and resource-path changes. |

Libraries, datasets, images, pretrained weights and checkpoints distributed
separately remain subject to their respective terms. This source release does
not include those data or model assets.
