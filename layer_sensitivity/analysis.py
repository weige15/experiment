"""Rebuild all tables, figures and report from validated real individual-layer records."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from quantization_sensitivity.analysis import content_indices, control_design as cubic_design, profile_residuals
from quantization_sensitivity.data import save_json
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.runner import file_hash, read_json
from . import DEFAULT_RESULTS
from .runner import source_hashes
from .statistics import (bootstrap_summary, conservative_paired_accuracy_interval, control_design,
                         stratified_binomial_interval, row_spearman, shape_normalize, top_mask)
from .validate import check_coverage, validate_gates

STRATEGIES = ["fixed", "random_0", "random_1", "random_2", "request_oracle"]


def matrices_from_records(root):
    run = root / "runs/evaluation"
    frame = pd.read_csv(run / "measurements.csv", keep_default_na=False)
    baseline = pd.read_csv(run / "baselines.csv", keep_default_na=False).sort_values(["dataset", "request_id"])
    ids = baseline.request_id.tolist()
    check_coverage(frame, ids)
    def wide(column):
        return frame.pivot(index="request_id", columns="layer", values=column).reindex(index=ids, columns=range(32)).to_numpy()
    values, correct = wide("kl_mean"), wide("correct").astype(float)
    allocation = pd.read_csv(root / "precision_allocation_check/measurements.csv", keep_default_na=False)
    def allocation_wide(column):
        return allocation.pivot(index="request_id", columns="strategy", values=column).reindex(index=ids, columns=STRATEGIES).to_numpy(dtype=float)
    joint_kl, additive_kl = allocation_wide("kl_mean"), allocation_wide("sum_individual_layer_kl")
    ratio = np.divide(joint_kl, additive_kl, out=np.full_like(joint_kl, np.nan), where=additive_kl > 1e-12)
    return frame, baseline, values, correct, allocation, allocation_wide("correct"), joint_kl, ratio


def segments(frame, baseline, tokenizer):
    ids = baseline.request_id.tolist()
    index = frame.set_index(["request_id", "layer"])
    n = len(ids)
    parts = {name: np.full((n, 32), np.nan) for name in ["full_first", "full_second", "content_first", "content_second", "content", "first_token"]}
    rows, metrics = [], {"full_halves": np.full(n, np.nan), "content_halves": np.full(n, np.nan)}
    for i, base in enumerate(baseline.itertuples()):
        tokens = json.loads(base.generated_ids)
        mask = content_indices(tokenizer, tokens)
        trace = np.array([json.loads(index.loc[(base.request_id, layer), "token_kl"]) for layer in range(32)])
        assert trace.shape == (32, len(tokens))
        parts["first_token"][i] = trace[:, 0]
        if mask:
            parts["content"][i] = trace[:, mask].mean(axis=1)
        for label, selected in [("full", trace), ("content", trace[:, mask])]:
            eligible = selected.shape[1] >= 4
            row = dict(request_id=base.request_id, dataset=base.dataset, segment=label,
                       tokens=selected.shape[1], eligible=eligible, spearman=np.nan,
                       top4_overlap_fraction=np.nan, top8_overlap_fraction=np.nan)
            if eligible:
                midpoint = selected.shape[1] // 2
                first, second = selected[:, :midpoint].mean(axis=1), selected[:, midpoint:].mean(axis=1)
                parts[f"{label}_first"][i], parts[f"{label}_second"][i] = first, second
                rho = float(row_spearman(first[None, :], second[None, :])[0])
                metrics[f"{label}_halves"][i] = rho
                row["spearman"] = rho
                for k in [4, 8]:
                    row[f"top{k}_overlap_fraction"] = float((top_mask(first[None], k) * top_mask(second[None], k)).sum() / k)
            rows.append(row)
    return pd.DataFrame(rows), parts, metrics


def reversal_table(values, baseline, parts):
    rows = []
    datasets = baseline.dataset.to_numpy()
    for scope, selected in [("all", np.ones(len(values), dtype=bool))] + [(d, datasets == d) for d in sorted(set(datasets))]:
        for i in range(32):
            for j in range(i + 1, 32):
                delta = values[selected, i] - values[selected, j]
                positive, negative = int((delta > 1e-6).sum()), int((delta < -1e-6).sum())
                a = parts["content_first"][selected, i] - parts["content_first"][selected, j]
                b = parts["content_second"][selected, i] - parts["content_second"][selected, j]
                replicated_positive = int(((a > 1e-6) & (b > 1e-6)).sum())
                replicated_negative = int(((a < -1e-6) & (b < -1e-6)).sum())
                rows.append(dict(scope=scope, layer_a=i, layer_b=j, positive_requests=positive,
                    negative_requests=negative, material_reversal=positive >= 2 and negative >= 2,
                    content_replicated_positive=replicated_positive, content_replicated_negative=replicated_negative,
                    content_replicated_reversal=replicated_positive >= 2 and replicated_negative >= 2,
                    content_eligible_requests=int((np.isfinite(a) & np.isfinite(b)).sum())))
    return pd.DataFrame(rows)


def figures(root, baseline, values, pairwise, statistics, allocation_correct):
    out = root / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    def save(fig, name):
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=240)
        plt.close(fig)
    labels = baseline.request_id.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(10, len(labels) * .115)), layout="constrained")
    image = axes[0].imshow(np.log10(np.maximum(values, 1e-12)), aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=axes[0], label="log10 mean KL (nats/token)", shrink=.6)
    from scipy.stats import rankdata
    image = axes[1].imshow(rankdata(-values, axis=1), aspect="auto", cmap="viridis_r", vmin=1, vmax=32)
    fig.colorbar(image, ax=axes[1], label="Sensitivity rank (1 = highest)", shrink=.6)
    for ax in axes:
        ax.set_xlabel("Individually quantized Transformer layer")
        ax.set_xticks(range(32))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=5)
    axes[0].set_title("All formal requests — absolute W4 Q/DQ sensitivity")
    axes[1].set_title("Same requests — within-request layer ranks")
    save(fig, "request_layer_heatmap")
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.boxplot([values[:, i] for i in range(32)], tick_labels=range(32), showfliers=True)
    ax.set(yscale="log", xlabel="Individually quantized layer", ylabel="Mean KL (nats/token)",
           title="Sensitivity distributions — every formal request included")
    save(fig, "layer_distributions")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ["spearman", "top4_overlap_fraction", "top8_overlap_fraction"],
                               ["Layer-rank Spearman", "Top-4 overlap / 4", "Top-8 overlap / 8"]):
        image = ax.imshow(pairwise[metric], vmin=-1 if metric == "spearman" else 0, vmax=1,
                          cmap="coolwarm" if metric == "spearman" else "viridis")
        ax.set(title=title, xlabel="Request index (heatmap order)", ylabel="Request index")
        fig.colorbar(image, ax=ax, shrink=.7)
    save(fig, "request_pair_rank_overlap")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    names = ["BF16", *STRATEGIES, "candidate_label_oracle"]
    table = statistics[statistics.scope == "all"].set_index("metric")
    metric_names = ["BF16_accuracy", *[f"allocation_{s}_accuracy" for s in STRATEGIES], "allocation_candidate_label_oracle_accuracy"]
    estimate = np.array([table.loc[m, "estimate"] for m in metric_names]) * 100
    low = np.array([table.loc[m, "ci_low"] for m in metric_names]) * 100
    high = np.array([table.loc[m, "ci_high"] for m in metric_names]) * 100
    colors = ["#777777", "#0072B2", "#56B4E9", "#56B4E9", "#56B4E9", "#D55E00", "#CC79A7"]
    ax.bar(np.arange(len(names)), estimate, color=colors)
    ax.errorbar(np.arange(len(names)), estimate, yerr=np.maximum(0, np.array([estimate - low, high - estimate])),
                fmt="none", color="black", capsize=3)
    ax.set_xticks(np.arange(len(names)), names, rotation=20, ha="right")
    ax.set(ylabel="Actual answer accuracy (%)", ylim=(0, 100), title="8 BF16 layers / 24 jointly W4 layers; request-bootstrap 95% CI")
    save(fig, "precision_allocation_accuracy")


def markdown_table(frame):
    # No optional tabulate dependency is required to reconstruct the report.
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(map(cell, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def write_report(root, statistics, baseline, frame, allocation, reversals, segment_table, escalation, iterations):
    stats = statistics.set_index(["scope", "metric"])
    def ci(metric, scope="all", scale=1):
        row = stats.loc[(scope, metric)]
        if not np.isfinite(row.estimate):
            return "not estimable"
        return f"{row.estimate * scale:.5g} [{row.ci_low * scale:.5g}, {row.ci_high * scale:.5g}]"
    meta = read_json(root / "runs/evaluation/attempt_0.json")
    selection = read_json(root / "precision_allocation_selection.json")
    accuracy_rows = []
    for scope in ["all", *sorted(baseline.dataset.unique())]:
        count = len(baseline) if scope == "all" else int((baseline.dataset == scope).sum())
        accuracy_rows.append(dict(dataset=scope, requests=count, BF16_percent=ci("BF16_accuracy", scope, 100),
            fixed_percent=ci("allocation_fixed_accuracy", scope, 100),
            request_KL_oracle_percent=ci("allocation_request_oracle_accuracy", scope, 100),
            random_mean_percent=ci("allocation_random_mean_accuracy", scope, 100)))
    material = int(reversals.loc[reversals.scope == "all", "material_reversal"].sum())
    replicated = int(reversals.loc[reversals.scope == "all", "content_replicated_reversal"].sum())
    eligible = segment_table.groupby(["dataset", "segment"]).eligible.sum().reset_index()
    failure_counts = pd.concat([baseline.assign(configuration="BF16"), frame.assign(configuration="single_layer_W4"),
                                allocation.assign(configuration="joint_W4")]).groupby("configuration").agg(
                                    answers=("correct", "size"), incorrect=("correct", lambda s: int((~s).sum())),
                                    parse_failures=("parse_failure", "sum"), truncated=("truncated", "sum")).reset_index()
    sections = [f"# Llama 3.1 8B individual-layer quantization sensitivity\n\n"
        f"Generated from validated **real model inference**: {len(baseline)} formal requests × 32 individual layers = {len(frame)} single-layer W4 measurements. "
        f"Additionally {len(allocation)} directly generated joint-allocation answers. No historical four-layer-group measurement is counted here.\n\n"
        f"Analysis status: {'targeted follow-up required by predeclared W4 diagnostic' if escalation['required'] else 'primary W4 structure diagnostics resolved; scientific conclusions remain bounded below'}. "
        "A generated report or passing validator is not an automatic completion declaration.\n",
        "## Research questions and evidence\n\n"
        f"1. **Request-dependent sensitivity and stability.** Different-request mean layer-rank Spearman: {ci('different_request_spearman')}; "
        f"rank heterogeneity (1 − correlation): {ci('rank_heterogeneity')}. Content-half mean within-request Spearman: {ci('content_halves_mean_spearman')}. "
        f"{material}/496 layer pairs reverse by >1e-6 nats/token with at least two requests on each side; {replicated}/496 have both directions replicated across content halves. "
        "Reversals are descriptive, not 496 independent hypothesis tests. Stability claims apply only to eligible token traces, not every dataset.\n\n"
        f"2. **Higher-precision layer choice and actual utility.** Different-request Top-4 overlap fraction: {ci('different_request_top4_overlap_fraction')}; "
        f"Top-8: {ci('different_request_top8_overlap_fraction')}. At the directly measured 8-layer BF16 budget, request-KL oracle minus train-selected fixed "
        f"accuracy: **{ci('allocation_request_oracle_delta_vs_fixed', scale=100)} percentage points**; versus the three-set random mean: "
        f"{ci('allocation_request_oracle_delta_vs_random_mean', scale=100)} pp. A CI spanning zero is inconclusive, not evidence of deployable improvement.\n\n"
        f"3. **Type, length, difficulty proxies.** Descriptive standardized-profile regression R²: {ci('shape_R2', 'controls')}. Partial R² for dataset: "
        f"{ci('shape_partial_R2_dataset', 'controls')}, prompt length: {ci('shape_partial_R2_prompt_length', 'controls')}, BF16 output length: "
        f"{ci('shape_partial_R2_output_length', 'controls')}, BF16 correctness proxy: {ci('shape_partial_R2_difficulty_proxy', 'controls')}. "
        "These are in-sample associations, not causal effects or independent difficulty labels. All layer coefficients, pointwise intervals, residuals, "
        "dataset-specific nonlinear-control diagnostics and length-quartile results are retained.\n",
        "## Method and data\n\n"
        f"Model: `{MODEL_ID}` **base**, exact revision `{REVISION}`. Tokenizer is from the same snapshot. "
        "The base checkpoint has no official chat template: the explicit fixed Q/A template in [chat_template.jinja](chat_template.jinja) is used consistently. "
        "Three train demonstrations per dataset; greedy deterministic decoding, no sampling, one beam, BF16 activation and KV cache. GSM8K max 256 new tokens, "
        "ARC/BoolQ 16; stop at EOS or the next Q delimiter. No sample is dropped for wrong answers, parser failures or truncation.\n\n"
        "Every intervention changes exactly the seven attention/MLP linear weights of ONE of layers 0–31; the other 31 layers, embeddings, norms and head remain BF16. "
        "Symmetric RTN groupwise W4: group size 128 contiguous input features within output rows, FP32 absmax/7 scale, zero blocks scale 1, zero point 0, "
        "round-to-even codes [-7,7], no clipping, actual packed-nibble round-trip, dequantized BF16 forward. **Quantization sensitivity simulation only**: "
        "no W4 kernel, speedup or memory-saving claim. Quantization MSE/code digests, exact parameter isolation/restoration and actual precision probes are saved.\n\n"
        "`KL(P_BF16 || P_W4)` uses the same fixed BF16 greedy continuation for every layer of a request, identical full-sequence teacher-forcing path and causal shift. "
        "Full vocabulary log probabilities in FP32, sum in FP64; mean per target token in nats. Stop tokens remain in the primary full trace. Secondary content-only "
        "metrics exclude special/whitespace/next-question tokens. Numerical replay discrepancy in the required gates is zero. Independently generated answers, not "
        "teacher-forced labels, determine accuracy.\n\n"
        "Pinned GSM8K test, ARC-Challenge test, BoolQ validation samples: **8 smoke + 32 formal each**; train-only selection **8 each** and demonstrations **3 each**. "
        "Dataset revisions, source parquet SHA256, sampling seed 20261015 and excluded duplicates are in [data_manifest.json](data_manifest.json). "
        "[requests.jsonl](requests.jsonl) records every source ID, raw question, split and gold; baseline JSON/CSV records the entire rendered prompt and token IDs. "
        "Smoke and formal are disjoint; all roles are deduplicated by normalized question. No quantizer calibration data is needed.\n\n"
        "Scoring is frozen: GSM8K final `####` marker, entire first line must be an integer/decimal (valid thousands commas and optional terminal punctuation); "
        "unsupported fractions/exponents/units are parse failures. ARC requires a leading valid option label, BoolQ a leading yes/no word. "
        "Parsers are unit tested and raw generated texts/token IDs are retained for independent examination.\n",
        "## Direct precision allocation check\n\n"
        f"Fixed high-precision layers: **{selection['fixed_high_precision_layers']}**, selected by highest train-only mean KL and frozen at `{selection['frozen_utc']}` "
        "before formal inference. Every strategy retains exactly 8 BF16 layers and jointly quantizes the other 24. "
        "The `request_oracle` uses that evaluation request's full individual-layer KL vector: an expensive oracle-information benchmark, **not a learned/deployable router, "
        "and not a guaranteed accuracy upper bound**. The separately reported `candidate_label_oracle` chooses a correct answer among the five *actually measured* joint "
        "configurations and is only a finite-candidate, label-using upper bound.\n\n"
        + markdown_table(pd.DataFrame(accuracy_rows)) + "\n\n"
        + "Random sets (conditional request-bootstrap inference, not uncertainty over all possible random sets):\n\n"
        + markdown_table(pd.DataFrame([dict(strategy=s, BF16_layers=str(selection['random_high_precision_layers'][s]),
                                           accuracy_percent=ci(f'allocation_{s}_accuracy', scale=100)) for s in STRATEGIES[1:4]]))
        + "\n\nJoint KL is directly measured too: [precision_allocation_check/measurements.csv](precision_allocation_check/measurements.csv) includes "
        "joint KL and the sum of constituent individual-layer KLs. Their ratio measures nonadditivity, not a prediction of accuracy. "
        f"Request-oracle mean joint/additive ratio: {ci('allocation_request_oracle_joint_to_additive_KL_ratio')}.\n",
        "## Statistical uncertainty and all-request diagnostics\n\n"
        f"{iterations} paired, dataset-stratified request bootstrap replicates, seed 20261015, percentile 95% intervals. All 32 layer rows and all 5 configurations "
        "for one source remain paired; neither layer rows nor the 4,560 request pairs are treated as independent observations. Repeated bootstrap copies of the same "
        "source are excluded from different-request pairs. Random sets and train-selected fixed set are conditioned upon; selection-data uncertainty is not included. "
        "Per-layer/control intervals are pointwise and descriptive, not simultaneous tests. Bootstrap accuracy intervals can collapse when no discordance is observed; "
        "this does not establish zero population risk. [Exact binomial accuracy intervals](analysis/exact_accuracy_intervals.csv) and "
        "[conservative paired difference intervals](analysis/conservative_paired_accuracy_intervals.csv) are therefore also reported. The latter combine two "
        "Clopper–Pearson intervals for gain/loss probabilities by a union bound. Within one dataset these are two 97.5% intervals; for the combined "
        "three-dataset result, Bonferroni allocation across all six component probabilities and weighted aggregation respects the fixed dataset strata. "
        "This is conservative and does not assume independent paired answers or a common accuracy across datasets.\n\n"
        "Control response: per-request centered/unit-SD 32-layer profile; intercept, dataset indicators (ARC reference), standardized log prompt/output tokens and BF16 correctness. "
        "Output length/correctness are post-model variables; adjustments do not establish causality. A second descriptive fit uses dataset-specific cubic log prompt and linear "
        "log output. Model rank, leverage, residuals and all coefficients are saved. No preferred control model is chosen from favorable results.\n\n"
        + markdown_table(eligible) + "\n\n"
        "Insufficient content traces are explicitly ineligible, not assigned perfect stability. Full-trace halves may include delimiters, so they cannot alone establish "
        "within-answer stability for ARC/BoolQ. See [analysis/segment_stability.csv](analysis/segment_stability.csv) and "
        "[analysis/length_quartiles.csv](analysis/length_quartiles.csv).\n",
        "## Figures (all formal requests)\n\n"
        "![All requests and layers](figures/request_layer_heatmap.png)\n\n"
        "![Layer distributions](figures/layer_distributions.png)\n\n"
        "![Request pair ranks and overlap](figures/request_pair_rank_overlap.png)\n\n"
        "![Direct allocation accuracy](figures/precision_allocation_accuracy.png)\n\n"
        "Each figure also has a vector PDF in `figures/`. Row order is dataset/source ID, not selected by sensitivity.\n",
        "## Failures, null cases and limitations\n\n" + markdown_table(failure_counts) + "\n\n"
        "[analysis/answers.csv](analysis/answers.csv) includes every BF16, single-layer and joint answer; "
        "[analysis/answer_failures.csv](analysis/answer_failures.csv) includes all incorrect or unparseable/truncated answers, "
        "not a curated subset. [analysis/request_summary.csv](analysis/request_summary.csv) flags requests whose correctness never changes and those where allocation "
        "oracle does not beat fixed. Both gains and losses are retained.\n\n"
        "Limits: one base checkpoint/revision/template, one wording per source, small fixed benchmark subsets, greedy decoding, fixed output caps, possible pretrained "
        "benchmark contamination, no independent annotated difficulty, no real integer kernels. The deterministic sample IDs were also used in the historical grouped study; "
        "this is a newly specified individual-layer measurement, not an unseen-to-investigator dataset replication. No source was selected or removed based on favorable layer results. "
        "A sensitivity rank is about probability distributions, not necessarily "
        "correctness. Combined quantization has interactions and the oracle-information heuristic is not guaranteed optimal. Wide intervals and null results must not "
        "be rephrased as proven gains or absence of effects. No clustering, router or production batching is implemented.\n\n"
        f"Predeclared escalation decision: `{escalation}`. If triggered, first audit quantization/KL/parser, then extend the deterministic prefix to 64/dataset and inspect "
        "all length strata, then only if needed consider W3 with a recorded addendum. Keep original W4 results and mark outcome-triggered extensions exploratory with "
        "nominal, non-sequentially-adjusted intervals.\n\n"
        "Next verifiable question: does the observed individual-layer ordering persist on a larger disjoint, length-stratified heldout prefix, and do the directly "
        "measured joint-allocation accuracy differences narrow enough to distinguish benefit from loss? This report does not propose a production router.\n",
        "## Reproduction and evidence\n\n"
        f"Hardware recorded in [runs/evaluation/attempt_0.json](runs/evaluation/attempt_0.json): `{meta['gpu']}`, CUDA `{meta['cuda_runtime']}`, "
        f"PyTorch `{meta['versions']['torch']}`, Transformers `{meta['versions']['transformers']}`; BF16 single-request execution. "
        "Use a BF16-capable ~24 GiB GPU and ≥40 GiB available host RAM for the two CPU allocation weight copies, plus model cache/results storage.\n\n"
        "```bash\nbash scripts/install.sh\n.venv/bin/python -m pytest -q tests\n"
        "LAYER_GPU=<allocated_gpu> bash scripts/run_layer_sensitivity.sh smoke\n"
        "LAYER_GPU=<allocated_gpu> bash scripts/run_layer_sensitivity.sh full\n"
        "bash scripts/run_layer_sensitivity.sh report\n```\n\n"
        "Use `LAYER_OUT=results/layer_sensitivity_reproduction` for a fresh independent campaign. The default full command resumes only source/input-identical "
        "inference and revalidates reused records. The report command is CPU-only and preserves previous derived output. Actual commands, start/end timestamps, "
        "CUDA visibility, logs and exit codes are in [steps/](steps/). Do not infer execution success merely from the above command examples.\n\n"
        "Canonical outputs: [runs/evaluation/measurements.csv](runs/evaluation/measurements.csv), matching Parquet, "
        "[32-value vectors](runs/evaluation/sensitivity_vectors.csv), per-request JSON and BF16 reference logits retained locally; "
        "[analysis/statistics.csv](analysis/statistics.csv), [all layer-pair reversals](analysis/ranking_reversals.csv), "
        "[request pairs](analysis/request_pairs.csv), [analysis/metadata.json](analysis/metadata.json). "
        "The original prompt-to-artifact completion audit is [../../doc/layer_sensitivity_completion_audit.md](../../doc/layer_sensitivity_completion_audit.md).\n"]
    (root / "report.md").write_text("\n".join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    root = Path(args.root)
    validate_gates(root)
    replay_discrepancy = max(read_json(root / "gates" / stage / "result.json")["self_kl_max"]
                             for stage in ["single_layer", "single_request"])
    for name, path in [("evaluation", "runs/evaluation/measurements.csv"), ("precision_allocation", "precision_allocation_check/measurements.csv")]:
        validation = read_json(root / f"{name}_validation.json")
        assert validation["status"] == "verified" and validation["measurements_sha256"] == file_hash(root / path)
    frame, baseline, values, correct, allocation, allocation_correct, joint_kl, ratio = matrices_from_records(root)
    assert len(baseline) >= 96 and (baseline.groupby("dataset").size() >= 32).all()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    segment_table, parts, segment_metrics = segments(frame, baseline, tokenizer)
    segment_table.to_csv(out / "segment_stability.csv", index=False)
    for name, matrix in parts.items():
        pd.DataFrame(matrix, index=baseline.request_id, columns=[f"layer_{i}" for i in range(32)]).to_csv(out / f"{name}_vectors.csv")
    datasets = baseline.dataset.to_numpy()
    x, names = control_design(datasets, baseline.input_tokens, baseline.generated_tokens, baseline.correct)
    results, pairwise = bootstrap_summary(values, correct, baseline.correct.to_numpy(dtype=float), datasets,
        segment_metrics, allocation_correct, joint_kl, ratio, x, names, iterations=args.bootstrap)
    statistics = pd.DataFrame(results)
    statistics.to_csv(out / "statistics.csv", index=False)
    exact_rows, paired_rows = [], []
    for scope in ["all", *sorted(set(datasets))]:
        selected = np.ones(len(baseline), dtype=bool) if scope == "all" else datasets == scope
        outcomes = dict(BF16=baseline.correct.to_numpy(dtype=bool)[selected],
                        **{strategy: allocation_correct[selected, j].astype(bool) for j, strategy in enumerate(STRATEGIES)})
        for name, outcome in outcomes.items():
            lo, hi = stratified_binomial_interval(outcome, datasets[selected])
            exact_rows.append(dict(scope=scope, configuration=name, requests=len(outcome), correct=int(outcome.sum()),
                                   accuracy=float(outcome.mean()), ci_low=lo, ci_high=hi,
                                   method="dataset-stratified Clopper-Pearson / union-bound 95%"))
        for comparator in ["BF16", "fixed", "random_0", "random_1", "random_2"]:
            paired_rows.append(dict(scope=scope, comparison=f"request_oracle_minus_{comparator}",
                **conservative_paired_accuracy_interval(outcomes["request_oracle"], outcomes[comparator], datasets[selected])))
    pd.DataFrame(exact_rows).to_csv(out / "exact_accuracy_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "conservative_paired_accuracy_intervals.csv", index=False)
    reversals = reversal_table(values, baseline, parts)
    reversals.to_csv(out / "ranking_reversals.csv", index=False)
    pair_rows = []
    for i in range(len(baseline)):
        for j in range(i + 1, len(baseline)):
            pair_rows.append(dict(request_a=baseline.iloc[i].request_id, request_b=baseline.iloc[j].request_id,
                dataset_a=datasets[i], dataset_b=datasets[j], **{name: float(matrix[i, j]) for name, matrix in pairwise.items()}))
    pd.DataFrame(pair_rows).to_csv(out / "request_pairs.csv", index=False)
    residual = shape_normalize(values) - x @ np.linalg.lstsq(x, shape_normalize(values), rcond=None)[0]
    pd.DataFrame(residual, index=baseline.request_id, columns=[f"layer_{i}" for i in range(32)]).to_csv(out / "controlled_profile_residuals.csv")
    nonlinear = profile_residuals(values, datasets, baseline.input_tokens.to_numpy(), baseline.generated_tokens.to_numpy(), normalize="shape", nonlinear=True)
    pd.DataFrame(nonlinear, index=baseline.request_id, columns=[f"layer_{i}" for i in range(32)]).to_csv(out / "nonlinear_controlled_profile_residuals.csv")
    nonlinear_x = cubic_design(datasets, baseline.input_tokens.to_numpy(), baseline.generated_tokens.to_numpy(), nonlinear=True)
    leverage = np.einsum("ij,ji->i", x, np.linalg.pinv(x))
    summary = baseline[["request_id", "dataset", "input_tokens", "request_tokens", "generated_tokens", "correct", "parse_failure", "truncated"]].copy()
    summary["mean_KL"] = values.mean(axis=1)
    summary["KL_range"] = np.ptp(values, axis=1)
    summary["any_correctness_change"] = (correct != baseline.correct.to_numpy()[:, None]).any(axis=1)
    summary["oracle_allocation_delta_vs_fixed"] = allocation_correct[:, 4] - allocation_correct[:, 0]
    summary["control_leverage"] = leverage
    summary["nonlinear_control_leverage"] = np.einsum("ij,ji->i", nonlinear_x, np.linalg.pinv(nonlinear_x))
    for k in [4, 8]:
        summary[f"top{k}_layers"] = [json.dumps(np.flatnonzero(row).tolist()) for row in top_mask(values, k)]
    summary["within_dataset_prompt_length_quartile"] = summary.groupby("dataset").input_tokens.transform(
        lambda series: pd.qcut(series.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int))
    summary.to_csv(out / "request_summary.csv", index=False)
    summary.groupby(["dataset", "within_dataset_prompt_length_quartile"]).agg(
        requests=("request_id", "size"), min_prompt_tokens=("input_tokens", "min"), max_prompt_tokens=("input_tokens", "max"),
        mean_KL=("mean_KL", "mean"), mean_KL_range=("KL_range", "mean"), BF16_accuracy=("correct", "mean"),
        any_correctness_change_fraction=("any_correctness_change", "mean"), allocation_delta_vs_fixed=("oracle_allocation_delta_vs_fixed", "mean")
        ).to_csv(out / "length_quartiles.csv")
    answers = pd.concat([baseline.assign(configuration="BF16"), frame.assign(configuration=frame.layer.map(lambda i: f"W4_layer_{i}")),
                         allocation.assign(configuration=allocation.strategy)], ignore_index=True)
    answers.to_csv(out / "answers.csv", index=False)
    answers[(~answers.correct) | answers.parse_failure | answers.truncated].to_csv(out / "answer_failures.csv", index=False)
    indexed = statistics.set_index(["scope", "metric"])
    content_interval = indexed.loc[("all", "content_halves_mean_spearman")]
    heterogeneity = indexed.loc[("all", "rank_heterogeneity")]
    reasons = []
    if values.max() <= max(1e-6, 10 * replay_discrepancy):
        reasons.append("KL not above 10x replay discrepancy with absolute 1e-6 floor")
    if int(reversals[reversals.scope == "all"].material_reversal.sum()) < 2:
        reasons.append("fewer than two material reversed layer pairs")
    if not np.isfinite(content_interval.ci_low) or content_interval.ci_low <= 0:
        reasons.append("content segment rank stability not resolved")
    if not np.isfinite(heterogeneity.ci_low) or heterogeneity.ci_low <= .01:
        reasons.append("rank heterogeneity lower CI <= .01")
    escalation = dict(required=bool(reasons), reasons=reasons, numerical_replay_discrepancy=replay_discrepancy,
                      preserve_original_96=True, extensions_are_exploratory=True)
    save_json(out / "escalation.json", escalation)
    save_json(out / "metadata.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), bootstrap=args.bootstrap,
        source_sha256={name: file_hash(name) for name in ["layer_sensitivity/analysis.py", "layer_sensitivity/statistics.py"]},
        measurement_source_sha256=source_hashes(), input_sha256={path: file_hash(root / path) for path in [
            "runs/evaluation/measurements.csv", "runs/evaluation/baselines.csv", "precision_allocation_check/measurements.csv",
            "precision_allocation_selection.json", "requests.jsonl"]},
        requests=len(baseline), single_layer_measurements=len(frame), joint_measurements=len(allocation),
        design_rank=int(np.linalg.matrix_rank(x)), max_leverage=float(leverage.max()),
        nonlinear_design_rank=int(np.linalg.matrix_rank(nonlinear_x)),
        nonlinear_max_leverage=float(summary.nonlinear_control_leverage.max()),
        nonlinear_residual_RMS=float(np.sqrt(np.mean(nonlinear ** 2))),
        confidence_scope="paired request bootstrap; conditional on frozen train selection and three random layer sets; pointwise intervals"))
    figures(root, baseline, values, pairwise, statistics, allocation_correct)
    write_report(root, statistics, baseline, frame, allocation, reversals, segment_table, escalation, args.bootstrap)
    print(f"Report rebuilt from {len(frame)} individual-layer rows and {len(allocation)} joint rows; escalation={escalation}", flush=True)


if __name__ == "__main__":
    main()
