## Review

### Correct
- **Source-cluster bootstrap and re-selection are implemented correctly:** requests are resampled within dataset, retaining both wordings/all groups; evaluation-best fixed choices are recomputed inside each replicate (`m1_analysis.py:60–63,92–103,152–163`). Tests explicitly cover pairing and re-selection (`tests/test_analysis.py:66–83`).
- **Comparators are metric-matched and budget-matched:** calibration freezes KL-best and accuracy-best groups; evaluation-fixed and request-level choices are explicitly oracles. BF16 is not an oracle action (`scripts/select_fixed.py:13–16,35–40`; `m1_analysis.py:92–103,123–131`).
- **The primary interaction diagnostic removes a pure multiplicative request-magnitude/global-layer profile**, with a dedicated synthetic test (`m1_analysis.py:166–179`; `tests/test_analysis.py:26–37`).
- **Boundary robustness is substantive:** masks use BF16 IDs, are shared across groups, and exclude special/standalone-whitespace tokens. Primary, content and first-token results remain separate. Completed pilot coverage includes all 24 sources; content-segment analysis correctly reports no eligible ARC/BoolQ sources.
- Pairwise reversals remain descriptive. Holm correction covers all 16 controlled diagnostics (`scripts/analyze_m1.py:124–137`).

### Finding: P1 — Residual permutation is not an established conditional-null test
**Location:** `m1_analysis.py:174–178,194–219`; `scripts/analyze_m1.py:134`.

Continuous-length regressions are fitted once, then their residual rows are shuffled within coarse request-length quartiles. Those residuals are generally **not exchangeable** within these bins.

Even with correctly specified linear means and independent homoskedastic errors, regression residual covariance is proportional to \(M=I-H\). Permuting rows changes it to \(PMP^\top\), generally different from \(M\) because input/output lengths vary within bins. The current permutation therefore lacks an exact conditional-randomization justification. Holm adjustment cannot repair uncalibrated constituent p-values.

The tests demonstrate detection of a strong interaction and rejection of a deterministic multiplicative confound—not null rejection-rate calibration (`tests/test_analysis.py:26–50`). I have **not** established the numerical size or direction of miscalibration.

**Smallest safe resolution:** explicitly label these values exploratory/approximate and do not use them as confirmatory significance evidence. If confirmatory inference is required, document the conditional-null assumptions, implement an appropriate nuisance-aware procedure, and check its null calibration with CPU simulations before inspecting formal results. Merely increasing permutations is insufficient.

### Finding: P2 — Content masking drops mixed content/whitespace tokens at an undelimited ending
**Location:** `m1_analysis.py:16–24`; `tests/test_analysis.py:15–23`.

Without a delimiter, `end = len(text.rstrip())` can fall **inside** the last token. The prefix test then discards that entire token, although it is neither special nor whitespace-only.

A direct constructed example is a final token decoding to `".\n"`: its decoded length exceeds the stripped boundary, so the punctuation token is excluded. For a single such token, content becomes empty. This follows directly from the implementation; it was not observed as missing-source coverage in the completed pilot.

**Smallest fix:** use `len(text)` when no delimiter exists; the existing token-level `.strip()` filter already removes standalone whitespace. Add a mixed-content/trailing-whitespace token test. Delimiter-crossing tokens require a separately documented whole-token convention.

### Limitations—not established defects or current conclusion overclaims
- “Beyond length” means beyond the **specified log-linear controls**, not arbitrary length effects. Dataset-specific slopes and nonlinear length effects remain possible.
- Normalization does not eliminate every request-wide nuisance: \(KL_{rg}=a_r+b_g\) retains normalized-profile variation despite identical group ordering. Consequently residual association alone cannot establish request-dependent group choice. The protocol’s additional reversal/materiality gates are important.
- Bootstrap intervals condition on demonstrations/calibration and use only 32 independent sources per dataset. Oracle gains and selected-best intervals are not deployable performance guarantees.
- For any eventual quality-improvement claim, use a directly bootstrapped **paired difference**. Current paired accuracy differences cover selected comparisons, but not every strategy-versus-BF16 or cross-wording-versus-fixed comparison (`m1_analysis.py:109–138`).
- The completed pilot’s smallest Holm-adjusted p-value is approximately **0.0796**, not below 0.05. Positive bootstrap correlation intervals should not be substituted for the permutation decision.
- Numerical-floor arguments currently default to zero in analysis. Exact replay assertions support zero for checked passes (`scripts/sensitivity.py:38–55`); retain that supporting evidence rather than implying universal noise characterization.

**Merge verdict: OK with notes for continued measurement; confirmatory significance sign-off requires resolving or explicitly downgrading P1.** No reason identified to stop or rerun formal GPU inference. No formal answers/results inspected, files edited, or workloads run.
