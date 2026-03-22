# Adapter Tuning Ideas For FRF Distilled Data

## Goal

Investigate whether the FRF-distilled synthetic dataset can be used to train adapters that move performance closer to full fine-tuning, while preserving the exact behavior of the current default linear-probe pipeline.

## Non-Negotiable Constraint

The existing implementation must remain behaviorally unchanged unless adapter training is explicitly enabled.

Concretely:

- Running the same linear probing experiment before and after the code change should produce the same result.
- New code must be opt-in only.
- Existing default paths, default hyperparameters, and default result files should keep their current meaning.

## Current Design Decision

We added a new adapter branch as an optional evaluation path after FRF synthetic set generation.

- Default behavior remains the current `frozen backbone + linear classifier`.
- Adapter tuning is only activated when the new adapter flag is enabled.
- Linear probe outputs remain separate from adapter outputs.

Current output semantics:

- `metrics.json`: linear-probe result only
- `linear_probe.pth`: linear-probe classifier checkpoint only
- `adapter_metrics.json`: adapter result, only when adapter evaluation is enabled
- `adapter_probe.pth`: adapter + classifier checkpoint payload, only when adapter evaluation is enabled

## Why Use Internal Adapters Instead Of Feature-Only Adapters

If the objective is to get as close as possible to full fine-tuning, internal adapters are a better first choice than attaching a small MLP after the frozen final feature.

Reasoning:

- Feature-only adapters are closer to a stronger linear probe.
- Internal adapters can modify intermediate representations and are therefore more expressive.
- This makes them more suitable when the goal is to approximate the behavior of full fine-tuning under a parameter-efficient budget.

## Backbone Scope

We only need to support the four main backbones:

- `dinov2_vitb`
- `eva02_vitb`
- `clip_vitb`
- `mocov3_resnet50`

Other backbones are out of scope for now.

## Placement Strategy Discussion

### Main question

Should adapters be inserted only in later layers, or across the full network?

### Practical conclusion

If the goal is to maximize similarity to full fine-tuning, the most relevant configurations are:

- adapters in the later half of the network
- adapters in all layers/stages

Observed/common intuition:

- Later layers are usually more task-specific.
- Earlier layers are usually more generic.
- Therefore, tuning later layers often gives a strong parameter-efficiency tradeoff.
- However, full-layer insertion may provide a higher ceiling when the objective is to get closer to full FT.

### Recommendation

For each backbone, compare at least:

- later-half insertion
- full insertion

Suggested interpretations:

- later-half insertion: strong efficiency/performance balance
- full insertion: stronger candidate for approaching full FT

## Adapter Capacity Discussion

### Main question

Should adapter capacity be made as large as possible?

### Practical conclusion

Not always. Larger adapters can move closer to full FT, but can also overfit, especially because the downstream training set here is distilled and compact.

Useful working rule:

- too small: likely underpowered
- medium: often the best practical region
- very large: may help, but with higher instability/overfitting risk

### Recommendation

Use these as the first search points:

- `adapter_reduction = 16`
- `adapter_reduction = 8`

Interpretation:

- `16`: safer starting point
- `8`: higher-capacity setting for pushing toward full FT

## Classifier Training Decision

We agreed that adapter tuning should train:

- adapter parameters
- classifier head

The backbone base weights stay frozen.

This is preferred over training adapter-only while keeping the classifier fixed.

## Layer-Wise Capacity Strategy

### Question

Should later layers receive larger adapters while earlier layers receive smaller adapters?

### Discussion outcome

This idea is supported by common PEFT intuition and literature trends:

- later layers are more task-specific
- they often benefit more from adaptation

However, this is not necessarily a must-have first step here.

In this FRF setting, the dominant factor may still be the quality and coverage of the distilled data rather than fine-grained layer-wise capacity allocation.

### Recommendation

Do not make layer-wise adapter sizing a first-priority change.

Instead, first compare:

- later-half insertion with uniform capacity
- full insertion with uniform capacity

If adapter tuning shows promise but leaves room for improvement, then add layer-wise capacity as a second-stage optimization.

## Training Objective Discussion

### Baseline recommendation

Start with plain cross-entropy:

- `loss = CE`

Why:

- simplest and cleanest baseline
- easiest to interpret
- least invasive integration

### Recommended second-stage improvement

If CE-only adapter tuning is unstable or overfits, the next most natural addition is feature anchoring:

- `loss = CE + lambda * ||z_adapter - z_frozen||^2`

Why feature anchoring is attractive here:

- it does not require an external teacher
- it fits naturally with a frozen-backbone setup
- it may reduce overfitting on small distilled datasets
- it can behave like a constrained form of fine-tuning

### Recommendation

Stage 1:

- use pure CE

Stage 2 if needed:

- add feature-anchor regularization

## Current Experimental Priority

The recommended priority order is:

1. Train `adapter + classifier`
2. Compare later-half insertion vs full insertion
3. Compare `adapter_reduction = 16` vs `8`
4. Keep the first version on pure CE
5. Add feature anchoring only if needed
6. Leave layer-wise capacity as a later refinement

## Suggested First-Round Sweep

For ViT-style backbones (`dinov2_vitb`, `eva02_vitb`, `clip_vitb`):

- later-half insertion + reduction 16
- later-half insertion + reduction 8
- full insertion + reduction 16
- full insertion + reduction 8

For `mocov3_resnet50`:

- `layer3 + layer4` equivalent setting
- full stage insertion
- reduction 16 and 8 as the first two capacity levels

## Open Follow-Up

If adapter tuning improves over linear probing but still does not get close enough to full FT, the next candidate improvements are:

- feature-anchor regularization
- layer-wise adapter capacity

## Loss Improvement Follow-Up

After the first adapter sweep, the gain over linear probing was positive but small. That suggests the remaining bottleneck may not be adapter placement alone.

The next low-risk loss change to try is augmentation-view consistency, still with the backbone frozen and the adapter branch opt-in only.

Recommended additions:

- feature consistency between two independently augmented views of the same synthetic image
- logit consistency between two independently augmented views

Rationale:

- does not require an external teacher
- fits very small synthetic datasets better than relying on CE alone
- encourages the adapter to preserve task-relevant invariance instead of memorizing a single augmented view

Practical form:

- `loss = CE(view1) + CE(view2)`
- `+ lambda_feat * (1 - cosine(z1, z2))`
- `+ lambda_logit * symKL(logits1, logits2)`

These terms should remain disabled by default so the original adapter CE path is unchanged unless explicitly enabled.
- backbone-specific default placement/capacity presets
