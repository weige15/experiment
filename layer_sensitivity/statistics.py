"""Request-level descriptive statistics and paired, dataset-stratified bootstrap."""
import numpy as np
from scipy.stats import beta, rankdata


def exact_binomial_interval(successes, n, alpha=.05):
    """Clopper–Pearson interval; unlike percentile bootstrap, nonzero width at 0/n."""
    if not 0 <= successes <= n or n < 1:
        raise ValueError("valid success count and positive n required")
    lo = 0. if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1. if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lo, hi


def stratified_binomial_interval(outcomes, datasets, alpha=.05):
    """Conservative weighted interval, not an iid binomial assumption across datasets."""
    outcomes, datasets = np.asarray(outcomes, dtype=bool), np.asarray(datasets)
    if outcomes.ndim != 1 or outcomes.shape != datasets.shape or not len(outcomes):
        raise ValueError("matched nonempty outcomes and strata required")
    strata = np.unique(datasets)
    low, high = 0., 0.
    for dataset in strata:
        values = outcomes[datasets == dataset]
        lo, hi = exact_binomial_interval(int(values.sum()), len(values), alpha / len(strata))
        weight = len(values) / len(outcomes)
        low, high = low + weight * lo, high + weight * hi
    return low, high


def conservative_paired_accuracy_interval(a, b, datasets=None):
    """95% union-bound gain-minus-loss interval, respecting fixed dataset strata."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("matched nonempty correctness arrays required")
    datasets = np.zeros(len(a), dtype=int) if datasets is None else np.asarray(datasets)
    gain, loss = int((a & ~b).sum()), int((~a & b).sum())
    gain_lo, gain_hi = stratified_binomial_interval(a & ~b, datasets, alpha=.025)
    loss_lo, loss_hi = stratified_binomial_interval(~a & b, datasets, alpha=.025)
    return dict(gain_requests=gain, loss_requests=loss, estimate=float((a.astype(float) - b).mean()),
                ci_low=gain_lo - loss_hi, ci_high=gain_hi - loss_lo)


def mean_finite(values, weights=None):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if weights is None:
        return float(values[valid].mean()) if valid.any() else np.nan
    weights = np.asarray(weights, dtype=float)
    denominator = weights[valid].sum()
    return float((values[valid] * weights[valid]).sum() / denominator) if denominator else np.nan


def top_mask(values, k):
    order = np.argsort(-np.asarray(values), axis=1, kind="stable")[:, :k]
    mask = np.zeros_like(values, dtype=np.int64)
    np.put_along_axis(mask, order, 1, axis=1)
    return mask


def row_spearman(a, b):
    a, b = rankdata(a, axis=-1), rankdata(b, axis=-1)
    a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(axis=-1), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)


def pair_matrices(values):
    ranks = rankdata(values, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(ranks, axis=1)
    denom = norm[:, None] * norm[None, :]
    rho = np.divide(ranks @ ranks.T, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    matrices = dict(spearman=rho)
    for k in [4, 8]:
        masks = top_mask(values, k)
        overlap = masks @ masks.T
        matrices[f"top{k}_overlap_fraction"] = overlap / k
        matrices[f"top{k}_jaccard"] = overlap / (2 * k - overlap)
    # Average material ordering disagreement over all 496 layer pairs.
    i, j = np.triu_indices(values.shape[1], k=1)
    delta = values[:, i] - values[:, j]
    positive, negative = (delta > 1e-6).astype(float), (delta < -1e-6).astype(float)
    matrices["ranking_reversal_rate"] = (positive @ negative.T + negative @ positive.T) / len(i)
    valid = np.isfinite(values).all(axis=1)
    for matrix in matrices.values():
        matrix[~(valid[:, None] & valid[None, :])] = np.nan
    return matrices


def pair_average(matrix, counts):
    """Distinct ORIGINAL requests only; bootstrap duplicates never become self-pairs."""
    weights = np.asarray(counts)[:, None] * np.asarray(counts)[None, :]
    weights = np.triu(weights, k=1)
    return mean_finite(matrix, weights)


def stratified_indices(datasets, rng):
    datasets = np.asarray(datasets)
    return np.concatenate([rng.choice(indices, len(indices), replace=True) for dataset in np.unique(datasets)
                           if len(indices := np.flatnonzero(datasets == dataset))])


def shape_normalize(values):
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def control_design(datasets, input_tokens, output_tokens, bf16_correct):
    datasets = np.asarray(datasets)
    def standardize(x):
        x = np.log(np.asarray(x, dtype=float))
        return (x - x.mean()) / max(x.std(), 1e-12)
    names = ["intercept", "dataset_boolq_vs_arc", "dataset_gsm8k_vs_arc",
             "log_prompt_tokens_z", "log_output_tokens_z", "BF16_correct_difficulty_proxy"]
    x = np.column_stack([np.ones(len(datasets)), datasets == "boolq", datasets == "gsm8k",
                         standardize(input_tokens), standardize(output_tokens), bf16_correct]).astype(float)
    return x, names


def cubic_prompt_output_design(datasets, input_tokens, output_tokens):
    """Protocol diagnostic: separate dataset-specific cubic log-length terms."""
    datasets = np.asarray(datasets)
    columns, names = [], []
    for dataset in np.unique(datasets):
        mask = datasets == dataset
        columns.append(mask.astype(float))
        names.append(f"{dataset}_intercept")
        for label, lengths in [("prompt", input_tokens), ("output", output_tokens)]:
            values = np.log(np.asarray(lengths, dtype=float)[mask])
            z = np.zeros(len(datasets))
            z[mask] = (values - values.mean()) / max(values.std(), 1e-12)
            for power in [1, 2, 3]:
                columns.append(z ** power)
                names.append(f"{dataset}_log_{label}_z_power_{power}")
    return np.column_stack(columns), names


def regression_statistics(x, values, names):
    shape = shape_normalize(values)
    coefficients = np.linalg.lstsq(x, shape, rcond=None)[0]
    residual = shape - x @ coefficients
    rss = float((residual ** 2).sum())
    tss = float(((shape - shape.mean(axis=0)) ** 2).sum())
    result = dict(shape_R2=1 - rss / tss if tss > 1e-18 else np.nan,
                  residual_shape_RMS=float(np.sqrt(np.mean(residual ** 2))))
    for name, removed in [("dataset", [1, 2]), ("prompt_length", [3]), ("output_length", [4]), ("difficulty_proxy", [5])]:
        reduced = np.delete(x, removed, axis=1)
        reduced_rss = float(((shape - reduced @ np.linalg.lstsq(reduced, shape, rcond=None)[0]) ** 2).sum())
        result[f"shape_partial_R2_{name}"] = (reduced_rss - rss) / reduced_rss if reduced_rss > 1e-18 else np.nan
    for j, name in enumerate(names):
        for layer in range(values.shape[1]):
            result[f"shape_coefficient_{name}_layer_{layer}"] = float(coefficients[j, layer])
    log_mean = np.log(np.maximum(values.mean(axis=1), 1e-12))
    scalar = np.linalg.lstsq(x, log_mean, rcond=None)[0]
    for name, coef in zip(names, scalar):
        result[f"log_mean_KL_coefficient_{name}"] = float(coef)
    return result


def bootstrap_summary(values, correct, baseline, datasets, segment_rho, allocation_correct,
                      allocation_kl, joint_additive_ratio, x, names, iterations=2000, seed=20261015,
                      secondary_values=None):
    """All rows of a source stay paired across layers, strategies and segments."""
    n = len(values)
    matrices = pair_matrices(values)
    secondary_pairs = {name: pair_matrices(matrix) for name, matrix in (secondary_values or {}).items()}
    scopes = {"all": np.ones(n, dtype=bool), **{d: np.asarray(datasets) == d for d in sorted(set(datasets))}}

    def calculate(indices):
        counts = np.bincount(indices, minlength=n)
        output = {}
        for scope, mask in scopes.items():
            selected = indices[mask[indices]]
            if not len(selected):
                continue
            weights = counts * mask
            result = dict(mean_KL=float(values[selected].mean()), BF16_accuracy=float(baseline[selected].mean()),
                mean_single_layer_accuracy=float(correct[selected].mean()),
                fraction_any_layer_correctness_change=float((correct[selected] != baseline[selected, None]).any(axis=1).mean()),
                mean_request_KL_range=float(np.ptp(values[selected], axis=1).mean()))
            for metric, matrix in matrices.items():
                result[f"different_request_{metric}"] = pair_average(matrix, weights)
            result["rank_heterogeneity"] = 1 - result["different_request_spearman"]
            for variant, pairs in secondary_pairs.items():
                for metric, matrix in pairs.items():
                    result[f"secondary_{variant}_different_request_{metric}"] = pair_average(matrix, weights)
            for segment, rho in segment_rho.items():
                result[f"{segment}_mean_spearman"] = mean_finite(rho[selected])
            for layer in range(32):
                result[f"layer_{layer}_mean_KL"] = float(values[selected, layer].mean())
                result[f"layer_{layer}_accuracy"] = float(correct[selected, layer].mean())
            for j, strategy in enumerate(["fixed", "random_0", "random_1", "random_2", "request_oracle"]):
                result[f"allocation_{strategy}_accuracy"] = float(allocation_correct[selected, j].mean())
                result[f"allocation_{strategy}_KL"] = float(allocation_kl[selected, j].mean())
                result[f"allocation_{strategy}_joint_to_additive_KL_ratio"] = mean_finite(joint_additive_ratio[selected, j])
            random = allocation_correct[selected, 1:4].mean(axis=1)
            oracle, fixed = allocation_correct[selected, 4], allocation_correct[selected, 0]
            result["allocation_random_mean_accuracy"] = float(random.mean())
            result["allocation_request_oracle_delta_vs_fixed"] = float((oracle - fixed).mean())
            result["allocation_request_oracle_delta_vs_random_mean"] = float((oracle - random).mean())
            result["allocation_fixed_delta_vs_random_mean"] = float((fixed - random).mean())
            result["allocation_candidate_label_oracle_accuracy"] = float(allocation_correct[selected].max(axis=1).mean())
            result["allocation_request_oracle_delta_vs_BF16"] = float((oracle - baseline[selected]).mean())
            output.update({f"{scope}/{name}": value for name, value in result.items()})
        output.update({f"controls/{name}": value for name, value in regression_statistics(x[indices], values[indices], names).items()})
        return output

    point = calculate(np.arange(n))
    draws = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = stratified_indices(datasets, rng)
        result = calculate(indices)
        for name in point:
            draws[name].append(result.get(name, np.nan))
    rows = []
    for name, value in point.items():
        finite = np.asarray(draws[name])[np.isfinite(draws[name])]
        lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else [np.nan, np.nan]
        scope, metric = name.split("/", 1)
        interpretation = ("nominal in-sample refit range; bias/coverage unassessed; not heldout-performance CI"
                          if scope == "controls" and (metric == "shape_R2" or metric.startswith("shape_partial_R2_"))
                          else "pointwise paired dataset-stratified request-bootstrap percentile interval")
        rows.append(dict(scope=scope, metric=metric, estimate=value, ci_low=float(lo), ci_high=float(hi),
                         interval_interpretation=interpretation,
                         valid_bootstraps=len(finite), bootstrap_replicates=iterations))
    return rows, matrices
