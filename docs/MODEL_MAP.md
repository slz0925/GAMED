# Method and implementation

Reference: Shen et al., GAMED, WSDM 2025, pp. 586–595,
[DOI](https://doi.org/10.1145/3701551.3703541), Figures 1–3 and Table 3.

| Public model | Definition | Effective behaviour |
|---|---|---|
| `gamed` / `GAMED` | [models/gamed.py](../models/gamed.py) | Active veto decision path |
| `gamed_without_veto` / `GAMEDWithoutVeto` | [models/gamed_without_veto.py](../models/gamed_without_veto.py) | Independent implementation without active veto |

| Paper role | Registered names / implementation |
|---|---|
| Image Pattern (IP) | `ip_backbone` is InceptionResNetV2 with `use_SRM=True`, whose enabled filter is BayarConv2D; `ip_trim`, `ip_classifier` |
| Image Semantic (IS) | MAE-Base `image_encoder.forward_ying`; `is_attention`, `is_experts`, `is_gate`, `is_trim`, `is_classifier` |
| Text (T), knowledge enhancement | English ERNIE2 `text_encoder`; text attention, experts, gate and classifier. No separate online knowledge-graph service is implemented |
| Fusion (MM) | Concatenated image/text tokens; `mm_attention`, `mm_experts`, `mm_gate/mm_aux_gate`, `shared_mm_feature/mm_consistency_feature`, `mm_aux_trim/mm_aux_classifier` |
| Irrelevant representation | Learned `irrelevant_tensor`, adjusted through `irrelevant_adain_mu/sigma` |
| Final mixture | IS/T/MM/IP/irrelevant stack, final attention/experts, `mix_trim/mix_classifier`; distinct from early MM |
| MMoE-Pro | TokenAttention MLPs, raw-score expert gates without softmax, weighted timm.Block expert loops |
| AdaIN | AdaIN class and `is_adain_*`, `t_adain_*`, `ip_adain_*`, `irrelevant_adain_*`; IS/T/IP/irrelevant adjusted, MM carried directly |
| Consistency-related path | `mm_consistency_feature → mm_aux_trim → mm_aux_classifier → 1-sigmoid →` irrelevant adjustment |

Checkpoint keys use these registered module names. The loader accepts compatible
checkpoint names as described in [CHECKPOINTS.md](CHECKPOINTS.md).
The two variants have matching constructor structure; state-dict keys alone do
not identify the selected veto behaviour. Record the model selector with each
checkpoint. `return_features=True` exposes the feature interfaces described in
[REPRODUCE.md](REPRODUCE.md).

## Implementation details

### Features and fusion

TokenAttention aggregates tokens with raw MLP scores. Expert gates also use
unnormalised scores. MM concatenates the image and text encoder tokens, with
`mm_gate` and `mm_aux_gate` weighting the shared MM experts for fusion and the
auxiliary prediction respectively. AdaIN computes
`sigma * (normalised_x + mu)` for the IS, T, IP and irrelevant representations;
the MM representation enters the final mixture directly.

The auxiliary probability is detached before conditioning the irrelevant
representation. In `gamed_without_veto`, the configured loss does not reach
`mm_aux_gate`, `mm_aux_trim` or `mm_aux_classifier`. In `gamed`, Veto Voting can connect
these modules to the mixture loss when the MM prediction is selected. The main
MM branch provides a gradient path to the shared experts, and the mapping MLPs
are trainable.
See [REPRODUCE.md](REPRODUCE.md) for the loss and encoder update schedule,
[DATASETS.md](DATASETS.md) for image processing, and
[CHECKPOINTS.md](CHECKPOINTS.md) for initialisation weights.

### Veto outputs

`gamed` combines Veto 1 replacement with Veto 2 averaging.
Let `p_i` be an expert's sigmoid probability and `p` the current mixture
probability, initially `sigmoid(mix_classifier(...))`:

- Veto 1: if `p_i > high` and `p_i > p`, replace `p` with `p_i`.
- Veto 2: if `p_i < low` and the expert belongs to the strict majority class,
  replace `p` with `(p + max_j p_j) / 2`.
- Values equal to either threshold do not trigger that rule.

The default thresholds are `low=0.05` and `high=0.95`. IP, IS, T and MM
vote using their initial probabilities (`p_i >= 0.5` means Fake); the mixture
does not vote. A 2:2 tie disables Veto 2. The maximum is taken over all four
experts, without excluding a class or including the mixture. Experts are
processed in IP/IS/T/MM order; every qualifying expert updates the current
mixture once.

The model returns **logits** in both training and evaluation. Replacement uses
the selected expert logit; averaging uses a stable, equivalent logit expression,
without probability clipping. If no rule triggers, the mixture logit is retained
exactly. `BCEWithLogitsLoss` therefore receives logits, and the evaluation sigmoid
recovers the voted probability. `gamed_without_veto` keeps its unmodified output.

Voting participates in the training loss. Hard comparisons do not carry
gradients, but selected values do: replacement routes the mixture loss to the
selected expert; averaging connects the current mixture and maximum expert.
Tied maxima route gradients to the first expert in IP/IS/T/MM order.
Veto adds no parameters or checkpoint keys.

## Ablations

Use the two model selectors for experiments with and without active veto.
Branch outputs and exported features are available for further analysis;
additional ablation training recipes are not included in this repository.
