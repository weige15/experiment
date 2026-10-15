## Review

### Must-fix reporting findings

- **P1 — Qualify the R² uncertainty presentation.** `results/layer_sensitivity/report.md:13` presents R² **0.23018 [0.23464, 0.30471]** and dataset partial R² **0.024503 [0.025160, 0.064933]** as confidence intervals. The implementation refits and scores each bootstrap sample **in sample**, then takes percentiles (`layer_sensitivity/statistics.py:121–132,188–205`). These ranges include resampling/refitting optimism; they are not uncertainty for heldout explanatory performance. A percentile interval excluding its point estimate is possible—not itself a coding error—but this displacement needs explicit explanation. **Smallest correction:** retain the preregistered outputs, label these as nominal in-sample bootstrap ranges with unassessed coverage/bias, and avoid interpreting positive partial-R² bounds as evidence of independent explanatory effects. Any optimism-corrected or heldout analysis should be separately labeled supplementary, not silently substituted.

- **P1 — Reconcile the completion audit with completed evidence.** `doc/layer_sensitivity_completion_audit.md:3,14,42,52` still states that formal measurements, allocations and reporting are missing. The inspected validation artifacts and `results/layer_sensitivity/analysis/metadata.json:26–28` instead record **96 requests, 3,072 individual-layer measurements and 480 joint measurements**. Preserve the initial audit historically, but append a dated final reconciliation with actual command ledgers and remaining scientific limitations. This is stale documentation, not evidence that inference must be repeated.

### Correct scientific interpretation

- **Bootstrap unit and pairing are appropriate.** Requests are resampled within datasets; layer/configuration outcomes remain paired; duplicate copies of an original request are excluded from different-request pairs (`layer_sensitivity/statistics.py:90–100,145–205`). The conservative paired intervals correctly combine stratified gain/loss bounds by a union bound (`statistics.py:15–40`). Inference remains conditional on the frozen selection and three random sets.

- **The accuracy comparison does not establish benefit.** Request-KL versus fixed is **−4.17 pp [−11.46, +3.13]**, with **5 gains and 9 losses**. Its conservative interval is wider: **[−28.71, +21.78] pp** (`analysis/conservative_paired_accuracy_intervals.csv:3`). Report wording at `report.md:11` appropriately calls this inconclusive. It is neither equivalence nor proof that request-adaptive allocation cannot help.

- **The two oracles are correctly distinguished.** The KL-information heuristic is not an accuracy bound (`report.md:29`). The finite label oracle achieves **78/96 = 81.25%** versus the heuristic’s **64/96**, but uses labels to choose among five measured configurations (`analysis/statistics.csv:103,111`). It bounds those candidates, not every possible eight-layer allocation, and demonstrates no deployable selector.

- **Post-model covariates are appropriately identified** (`report.md:52`). Output length and BF16 correctness support descriptive associations only; residual variation does not establish request-specific structure independent of difficulty or all length-related mechanisms.

### Limitations and useful corrections

- **P2 — Surface the nonlinear fit’s fragility.** `analysis/metadata.json:129–131` reports rank **14** for the 15-column nonlinear design and maximum leverage approximately **1**. Thus at least one observation is effectively interpolated. Add these numerical warnings beside the nonlinear diagnostic; its lower residual RMS is not evidence of better generalization.

- **P2 — Put the stability population in the headline.** The **0.8383** content-half correlation and **135 replicated reversals** concern only **32 GSM8K requests**, not all 96. `report.md:9` should say this immediately, although the eligibility table later clarifies it. Content profiles exist for all 96, but ARC/BoolQ have no eligible content halves. Moreover, BoolQ content-only and first-token summaries are identical (`analysis/statistics.csv:236–247`), so they are not independent replications. These secondary diagnostics address delimiter dependence, not temporal or cross-wording stability.

### Escalation decision

**Keeping `required: false` is defensible; an additional cohort is not essential for the bounded conclusions.** The frozen criteria (`doc/layer_sensitivity_protocol.md:47`) are comfortably cleared:

- Mean KL **0.001009** already implies maximum KL exceeds the **1e−6** numerical floor.
- **462** material reversed pairs exceed the required two.
- Eligible content-stability lower bound **0.8134 > 0**.
- Heterogeneity lower bound **0.3968 > 0.01**.

The protocol explicitly permits GSM8K-only segment-stability conclusions and inconclusive allocation utility. More data would be needed for broader stability claims or a precise utility estimate—not to obtain a favorable result. Any extension should retain the original cohort separately and use a frozen, precision-driven exploratory plan.

**Merge verdict: BLOCK final scientific sign-off pending the two P1 reporting corrections; no additional GPU study required.**