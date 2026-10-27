# Method and implementation

Reference: Shen et al., GAMED, WSDM 2025, pp. 586–595,
[DOI](https://doi.org/10.1145/3701551.3703541).

| Public model | Definition | Effective behaviour |
|---|---|---|
| `gamed` / `GAMED` | [models/gamed.py](../models/gamed.py) | Active veto decision path |
| `gamed_without_veto` / `GAMEDWithoutVeto` | [models/gamed_without_veto.py](../models/gamed_without_veto.py) | Independent implementation without active veto |

| Component | Registered names / implementation |
|---|---|
| Image Pattern (IP) | `ip_backbone` is InceptionResNetV2 with `use_SRM=True`, whose enabled filter is BayarConv2D; `ip_trim`, `ip_classifier` |
| Image Semantic (IS) | MAE-Base `image_encoder.forward_ying`; `is_attention`, `is_experts`, `is_gate`, `is_trim`, `is_classifier` |
| Text (T), knowledge enhancement | English ERNIE2 `text_encoder`; text attention, experts, gate and classifier |
| Fusion (MM) | Concatenated image/text tokens; `mm_attention`, `mm_experts`, `mm_gate/mm_aux_gate`, `shared_mm_feature/mm_consistency_feature`, `mm_aux_trim/mm_aux_classifier` |
| Irrelevant representation | Learned `irrelevant_tensor`, adjusted through `irrelevant_adain_mu/sigma` |
| Final mixture | IS/T/MM/IP/irrelevant stack, final attention/experts, `mix_trim/mix_classifier`; distinct from early MM |
| MMoE-Pro | TokenAttention MLPs, raw-score expert gates without softmax, weighted timm.Block expert loops |
| AdaIN | AdaIN class and `is_adain_*`, `t_adain_*`, `ip_adain_*`, `irrelevant_adain_*`; IS/T/IP/irrelevant adjusted, MM carried directly |
| Consistency-related path | `mm_consistency_feature → mm_aux_trim → mm_aux_classifier → 1-sigmoid →` irrelevant adjustment |

Checkpoint keys use these registered module names. The loader accepts compatible
checkpoint names as described in [CHECKPOINTS.md](CHECKPOINTS.md).
Both model constructors use a single logit for binary classification.
Direct model calls take `input_ids`, `attention_mask`, `token_type_ids` and
`image`; optional forward controls are keyword-only.
`get_pretrain_features` accepts the same inputs and returns the image
and text encoder tokens; `image_feature` and `text_feature` can reuse these tokens
in a subsequent forward call.
The two variants have matching constructor structure; state-dict keys alone do
not identify the selected veto behaviour. Record the model selector with each
checkpoint. `return_features=True` exposes the feature interfaces described in
[REPRODUCE.md](REPRODUCE.md).

## Implementation details

### Features and fusion

TokenAttention applies softmax across the token dimension to the MLP scores,
then takes a weighted sum. Expert gates use unnormalised, signed scores.
MM concatenates the image and text encoder tokens, with
`mm_gate` and `mm_aux_gate` weighting the shared MM experts for fusion and the
auxiliary prediction respectively. AdaIN computes
`sigma * normalised_x + mu` for the IS, T, IP and irrelevant representations.
Normalisation uses the population feature variance plus `1e-5`; the learned
scale remains signed. Half-precision feature statistics and attention reductions
accumulate in float32. The MM representation enters the final mixture directly.

Bayar filters have centre weight -1 and off-centre weights summing to 1.
The weights are projected before convolution; near-zero off-centre sums use a
uniform off-centre filter. The mask and centre tensors follow module device and
dtype changes without adding checkpoint keys.

The MM auxiliary prediction conditions the irrelevant representation and learns
through the mixture loss in both variants. IS/T/IP predictions are detached
before their conditioning MLPs. In `gamed`, Veto Voting determines which prediction
paths receive the mixture gradient. No separate consistency-label loss is used.
See [REPRODUCE.md](REPRODUCE.md) for the loss and encoder update schedule,
[DATASETS.md](DATASETS.md) for image processing, and
[CHECKPOINTS.md](CHECKPOINTS.md) for initialisation weights.

### Veto outputs

The Veto Voting mechanism combines confidence-based expert selection with
majority-aware arbitration.

For a binary logit `O`, let `p = sigmoid(O)` and class confidence
`C = max(p, 1-p)`. A probability `p >= 0.5` predicts Fake; otherwise it predicts
Real. The default class-confidence thresholds are `low=0.55` and `high=0.95`,
configurable through `GAMED.veto_voting(low_threshold=..., high_threshold=...)`.
Valid thresholds satisfy `0.5 < low < high < 1`.

- **Veto 1:** an expert must have `C_i > high` and `C_i > C_mix`. Select the
  highest-confidence expert and return its original logit. Opposing classes
  tied at the highest confidence retain the original mixture. Successful
  replacement takes precedence over Veto 2.
- **Veto 2:** IP, IS, T and MM vote using their original predictions; the mixture
  does not vote. At least three votes must agree, and every member of that
  majority must have `C_i < low`. The mixture must disagree with the majority
  and have `C_mix < high`. A minority expert must have `C_i > low`; select the
  most confident minority expert and update once:
  `p_new = (p_mix + p_selected) / 2`. With four voters, a strict majority leaves
  at most one minority expert. A 2:2 tie, no minority, or insufficient minority
  confidence leaves the mixture unchanged.

Both rules treat Real and Fake symmetrically. Equality does not satisfy a
strict expert threshold; a mixture with `C_mix >= high` is protected from
Veto 2. Expert ordering does not affect the result.

The model returns **logits** in both training and evaluation. Replacement uses
the selected expert logit; averaging uses a stable, equivalent logit expression,
without probability clipping. Float16 and bfloat16 inputs use float32 for
comparisons and averaging, with the output returned in the input dtype.
If no rule triggers, the mixture logit is retained exactly.
`BCEWithLogitsLoss` receives logits, and the evaluation sigmoid recovers the
voted probability. `gamed_without_veto` returns the mixture logit without voting.

Voting participates in the training loss. Hard comparisons do not carry
gradients, but selected values do: replacement routes the mixture loss to the
selected expert; averaging connects the original mixture and selected minority
expert. Same-class tied maxima split the replacement gradient equally;
opposing tied maxima retain the mixture gradient. Veto adds no parameters or
checkpoint keys.

## Ablations

Use the two model selectors for experiments with and without active veto.
Branch outputs and exported features are available for further analysis;
additional ablation training recipes are not included in this repository.
