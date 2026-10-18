# Third-party components and licences

The root [MIT licence](../LICENSE) covers GAMED's own code that the copyright
holder is entitled to license under MIT. Third-party components retain their
original copyright notices and licence terms, which are not changed by the
root MIT licence.

The corresponding licence texts remain in `third_party/licenses/`. In particular,
MAE remains subject to CC BY-NC 4.0, including its attribution and noncommercial
conditions. Preserve the applicable notices when redistributing these components.

| Component | Source and licence information |
|---|---|
| MAE and positional embeddings (`models_mae.py`, `util/pos_embed.py`) | [Meta MAE](https://github.com/facebookresearch/mae), [CC BY-NC 4.0](../third_party/licenses/MAE-CC-BY-NC-4.0.txt). The original copyright headers are retained. The noncommercial and attribution conditions apply; local additions, including `forward_ying`, may carry additional authorship requirements. |
| Inception-ResNet-v2 (`CNN_architectures/pytorch_InceptionResNetV2.py`) | [Cadene pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch), [BSD 3-Clause](../third_party/licenses/Cadene-BSD-3-Clause.txt). The BayarConv additions are separate from the upstream backbone implementation. |
| Torchvision-related code | [Torchvision](https://github.com/pytorch/vision), [BSD 3-Clause](../third_party/licenses/Torchvision-BSD-3-Clause.txt), for any portions derived from that project. |
| Progress utility (`utils.Progbar`) | Based on the [Keras Progbar utility](https://github.com/keras-team/keras/blob/2.2.4/keras/utils/generic_utils.py). The applicable Keras copyright and [MIT licence notice](../third_party/licenses/Keras-MIT.txt) are included for this component. |

Libraries, datasets, images, pretrained weights and checkpoints distributed
separately remain subject to their respective terms. This source release does
not include those data or model assets.
