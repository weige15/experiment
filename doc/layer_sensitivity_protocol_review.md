## Review

### Correct
- The protocol explicitly requires **32 independent single-layer interventions**, 8 smoke and 32 formal requests per dataset, isolated train-only allocation selection, and directly measured joint configurations—not renamed historical group results (`doc/layer_sensitivity_protocol.md:14–19,32–36`).
- The reusable KL implementation has correct causal alignment and computes `KL(reference || candidate)` on identical continuation tokens, with FP32 log probabilities and FP64 accumulation (`quantization_sensitivity/metrics.py:5–23,42–57`). Existing analytic tests address direction and alignment (`tests/test_metrics.py:10–28`); I did not execute them.
- Weight-level RTN/QDQ, pinned model/tokenizer revisions, deterministic decoding, and split-isolated sampling fit the proposed method (`quantization_sensitivity/quantization.py:27–56`; `model.py:17–38,57–75`; `data.py:59–77`).
- The protocol appropriately distinguishes the request-KL heuristic from a mathematical accuracy upper bound, and separately defines a finite-candidate label oracle (`doc/layer_sensitivity_protocol.md:33–35`).

### Findings

**P1 — GSM8K scoring accepts incomplete numeric prefixes and selects the first rather than final answer marker.**  
Location: `quantization_sensitivity/data.py:89–97`; reused under `doc/layer_sensitivity_protocol.md:21`.

The unanchored `re.search` accepts a numeric prefix without validating the remainder. Static counterexamples:
- `#### 1/2` parses as `1`, potentially falsely matching gold `1`.
- `#### 12\nCorrection: #### 13` parses as `12`, despite the corrected final answer.

`tests/test_data.py:6–11` covers ordinary marked integers/decimals but neither case. This can contaminate both single-layer accuracy changes and allocation comparisons.

**Smallest fix:** freeze a policy for multiple markers and unsupported numeric expressions; select the final marker or reject ambiguity, and reject partial matches of fractions/scientific notation unless explicitly supported. Add these regression cases before formal scoring. Preserve raw answers so scoring corrections remain auditable without repeating model inference.

**P2 — Outcome-dependent escalation lacks a defined decision rule and uncertainty treatment.**  
Location: `doc/layer_sensitivity_protocol.md:27,39`.

The protocol proposes percentile bootstrap intervals and expanding from 32 to 64 requests per dataset when W4 is “too weak to resolve structure,” but does not define that trigger or distinguish fixed-sample inference from outcome-dependent stopping. Consequently, a later nominal 95% interval could be presented without disclosing that its sample size depended on earlier outcomes.

**Smallest fix:** specify the diagnostic trigger before inspecting formal results. Preserve and report the original 96-request analysis separately; label outcome-triggered extensions exploratory with nominal, non-sequentially-adjusted intervals, unless a sequential procedure is explicitly implemented.

### Residual risks / implementation checks
- **Reuse only weight-level quantization primitives.** Existing `group_parameters`/`quantized_group` deliberately operate on four layers and 28 matrices; their metadata also declares `layers_per_group=4` (`quantization_sensitivity/quantization.py:7–12,59–87`). This is not a defect in the historical code. The new path must independently assert seven exact matrix names and exclude all other 31 layers.
- **State the random-baseline CI scope.** Request bootstrap intervals for three frozen random configurations are conditional on those particular configurations, not uncertainty over uniformly random layer sets (`doc/layer_sensitivity_protocol.md:27,35`). Report seed-to-seed results separately and avoid broader random-strategy superiority claims.
- **Freeze the association specification.** “Residualized normalized profile” does not yet specify normalization or response variables (`doc/layer_sensitivity_protocol.md:28`). Record these choices before analyzing formal outcomes; distinguish magnitude from ranking/shape and retain the stated noncausal interpretation.
- **Do not overclaim stability on short answers.** Full-trace halves can contain stop delimiters rather than answer content. The protocol’s content-only analysis and explicit insufficient-length counts are essential, especially for ARC/BoolQ (`doc/layer_sensitivity_protocol.md:26`; `quantization_sensitivity/data.py:25–28`).
- New measurement code needs its own source hashes: the reusable metadata function hashes only `quantization_sensitivity/*.py` (`quantization_sensitivity/model.py:114–115`).

### Merge verdict
**OK with notes for the protocol direction; correct P1 before final scoring and freeze the escalation clarification before formal outcome inspection.** This is not an implementation approval or completion finding.