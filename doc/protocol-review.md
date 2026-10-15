## Review

**Correct:** The base-model formatting is appropriate: explicit plain-text few-shot Q/A, one BOS, and no Instruct-only control tokens (`m1_data.py:17`, `m1_model.py:30–46`). The protocol correctly labels W4 as Q/DQ, separates teacher-forced KL from actual answers, clusters bootstrap samples by source request, and limits oracle claims (`doc/protocol.md:9,19,29–30`). Later-stage implementation being absent is not a defect.

### Prioritized findings

1. **P1 — Interaction test can detect overall request sensitivity rather than group preference.**  
   `doc/protocol.md:28` removes dataset/length effects separately per group, but not request-wide sensitivity. For example, \(KL_{rg}=a_r b_g\) produces reproducible residual association despite identical normalized group profiles and rankings across requests.  
   **Minimal remedy:** Predeclare an interaction statistic on request-centered profiles, also test normalized profiles, and require noise-resolved replicated rank reversals or material cross-wording selection benefit before claiming request-dependent group choice. Raw heterogeneity alone is insufficient.

2. **P1 — Baseline “PASS” does not establish finite inference outputs.**  
   `m1_model.py:78–82` checks only the first eight prompt tokens. Generation has no finite-logit check, yet `scripts/baseline.py:28` prints PASS. This falls short of `doc/protocol.md:17`.  
   **Minimal remedy:** Check finite logits throughout prompt/decode inference, preferably without retaining full-vocabulary outputs; fail explicitly on nonfinite values.

3. **P1 — Duplicate exclusion differs from the locked protocol.**  
   `doc/protocol.md:12` promises normalized-question deduplication. `m1_data.py:52–56,72–75` instead hashes complete content. Identical ARC questions with reordered options, or BoolQ questions with different passages, can cross roles.  
   **Minimal remedy:** Store and enforce a normalized-question hash across roles; retain the content hash for auditing. This is a reachable leakage path, not evidence that sampled leakage occurred.

4. **P1 — KL alignment and numerical-noise gates need executable definitions before scaling.**  
   `doc/protocol.md:18–19` specifies correct KL semantics, but reference-versus-reference alone does not establish causal-token alignment or independent-run repeatability.  
   **Minimal remedy:** Add a tiny causal-shift test: with prompt length \(P\), continuation token \(t\) uses logits at \(P+t-1\), for zero-based \(t\). Include the first generated token and predeclare EOS/delimiter inclusion. Independently repeat BF16 and one W4 pass using the measurement path; compare cached decoding with teacher-forced replay and restored BF16. Make near-tie decisions respect the observed numerical floor, not just the fixed threshold at line 25.

### Remaining interpretation safeguards

- **Quantization:** Before calling the intervention validated, test zero-scale blocks, rounding boundaries, all seven projections, and exact restoration (`doc/protocol.md:9,18`). Findings concern this RTN Q/DQ quantizer and **one quantized four-layer group**, not full-model W4 or optimized production quantizers.
- **Statistics/oracles:** With 32 requests per dataset, treat 28 pairwise reversal summaries as descriptive unless multiplicity is addressed. Bootstrap intervals condition on the sampled demonstrations/calibration unless those are also resampled. Preserve the existing distinction between evaluation oracles and deployable performance; KL-oracle superiority is partly definitional, not evidence of improved correctness.
- **Prompt validity:** Smoke-test formatting, parsing, and truncation on all three datasets before evaluation. Do not require correct baseline answers as a gate or tune prompts against formal labels.

**Merge verdict: OK with notes for the baseline-only stage; address P1 gates before formal scientific sign-off.** No files edited or workloads run.
