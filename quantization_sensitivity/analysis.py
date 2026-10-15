"""Offline analysis of measured records; never used to generate experiment values."""
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import SEED


def content_indices(tokenizer, generated_ids):
    """Keep whole non-special, non-whitespace tokens before the next-question delimiter."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    delimiter = re.search(r"\n\s*Q:", text)
    end = delimiter.start() if delimiter else len(text)
    boundary = 0
    for n in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:n], skip_special_tokens=True)
        if text.startswith(prefix) and len(prefix) <= end:
            boundary = n
    return [i for i, token in enumerate(generated_ids[:boundary])
            if token not in tokenizer.all_special_ids
            and tokenizer.decode([token], skip_special_tokens=True).strip()]


def add_content_metrics(frame, tokenizer):
    frame = frame.copy()
    masks = {}
    content_means, first_halves, second_halves, counts = [], [], [], []
    for row in frame.itertuples():
        key = (row.request_id, row.wording)
        ids = json.loads(row.bf16_generated_ids)
        values = np.array(json.loads(row.token_kl))
        if key not in masks:
            masks[key] = content_indices(tokenizer, ids)
        selected = values[masks[key]]
        n, mid = len(selected), len(selected) // 2
        counts.append(n)
        content_means.append(float(selected.mean()) if n else np.nan)
        first_halves.append(float(selected[:mid].mean()) if n >= 4 else np.nan)
        second_halves.append(float(selected[mid:].mean()) if n >= 4 else np.nan)
    frame["content_tokens"] = counts
    frame["kl_content_mean"] = content_means
    frame["kl_content_first_half"] = first_halves
    frame["kl_content_second_half"] = second_halves
    return frame


def cube(frame, column):
    """Source request is the sampling unit; wording and group stay paired."""
    if frame.duplicated(["request_id", "wording", "group"]).any():
        raise ValueError("duplicate measurement")
    wide = frame.pivot(index="request_id", columns=["wording", "group"], values=column)
    columns = pd.MultiIndex.from_product([["a", "b"], range(8)], names=["wording", "group"])
    wide = wide.reindex(columns=columns)
    return wide.index.tolist(), wide.to_numpy().reshape(-1, 2, 8)


def resample_sources(datasets, rng):
    datasets = np.asarray(datasets)
    return np.concatenate([rng.choice(idx, size=len(idx), replace=True)
                           for d in np.unique(datasets) if len(idx := np.flatnonzero(datasets == d))])


def row_spearman(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra, rb = ra - ra.mean(axis=-1, keepdims=True), rb - rb.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(ra, axis=-1) * np.linalg.norm(rb, axis=-1)
    return np.divide((ra * rb).sum(axis=-1), denom, out=np.full(denom.shape, np.nan), where=denom > 0)


def pair_signs(values, noise_floor=0):
    i, j = np.triu_indices(values.shape[-1], k=1)
    delta = values[..., i] - values[..., j]
    threshold = np.maximum(max(1e-5, 5 * noise_floor), .01 * (values[..., i] + values[..., j]) / 2)
    return np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0))


def reversal_summary(values, noise_floor=0):
    signs = pair_signs(values, noise_floor)
    positive, negative = (signs == 1).all(axis=1).sum(axis=0), (signs == -1).all(axis=1).sum(axis=0)
    n = len(values)
    return dict(replicated_reversal_pairs=float(((positive > 0) & (negative > 0)).sum()),
        replicated_reversal_rate=float((2 * positive * negative / (n * (n - 1))).mean()) if n > 1 else 0.)


def selected(values, choices):
    return np.take_along_axis(values, choices[..., None], axis=-1)[..., 0]


def strategy_choices(kl, correct, fixed_kl, fixed_accuracy):
    shape = kl.shape[:-1]
    choices = {
        "calibration_KL_fixed": np.full(shape, fixed_kl),
        "calibration_accuracy_fixed": np.full(shape, fixed_accuracy),
        "evaluation_KL_best_fixed_ORACLE": np.full(shape, kl.mean(axis=(0, 1)).argmin()),
        "evaluation_accuracy_best_fixed_ORACLE": np.full(shape, correct.mean(axis=(0, 1)).argmax()),
        "per_request_KL_ORACLE": kl.argmin(axis=-1),
        "per_request_label_ORACLE": correct.argmax(axis=-1),
    }
    if kl.shape[1] == 2:
        choices["cross_wording_KL_ORACLE_diagnostic"] = kl[:, ::-1].argmin(axis=-1)
    return choices


def basic_statistics(kl, correct, bf16_correct, fixed_kl, fixed_accuracy):
    correct = np.asarray(correct, dtype=float)
    out = {"BF16_accuracy": float(bf16_correct.mean()), "mean_KL": float(kl.mean()),
           "request_group_KL_range": float(np.ptp(kl, axis=-1).mean()),
           "fraction_correctness_varies": float((np.ptp(correct, axis=-1) > 0).mean())}
    choices = strategy_choices(kl, correct, fixed_kl, fixed_accuracy)
    for name, choice in choices.items():
        out[name + "_KL"] = float(selected(kl, choice).mean())
        chosen_correct = selected(correct, choice)
        out[name + "_accuracy"] = float(chosen_correct.mean())
        out[name + "_accuracy_delta_vs_BF16"] = float((chosen_correct - bf16_correct).mean())
        out[name + "_accuracy_delta_vs_calibration_KL_fixed"] = float((chosen_correct - correct[..., fixed_kl]).mean())
        out[name + "_accuracy_delta_vs_accuracy_best_fixed"] = float((chosen_correct - selected(
            correct, choices["evaluation_accuracy_best_fixed_ORACLE"])).mean())
    # Fractional allocation for exact ties, rather than assigning every tie to g0.
    maxima, minima = kl == kl.max(axis=-1, keepdims=True), kl == kl.min(axis=-1, keepdims=True)
    for group in range(8):
        out[f"g{group}_KL"] = float(kl[..., group].mean())
        out[f"g{group}_accuracy"] = float(correct[..., group].mean())
        out[f"g{group}_most_sensitive_fraction"] = float((maxima[..., group] / maxima.sum(-1)).mean())
        out[f"g{group}_least_sensitive_fraction"] = float((minima[..., group] / minima.sum(-1)).mean())
    fixed = out["evaluation_KL_best_fixed_ORACLE_KL"]
    out["KL_oracle_gain_vs_KL_best_fixed"] = fixed - out["per_request_KL_ORACLE_KL"]
    out["KL_oracle_relative_gain_vs_KL_best_fixed"] = out["KL_oracle_gain_vs_KL_best_fixed"] / fixed if fixed else 0.
    out["label_oracle_accuracy_gain_vs_accuracy_best_fixed"] = (
        out["per_request_label_ORACLE_accuracy"] - out["evaluation_accuracy_best_fixed_ORACLE_accuracy"])
    out["KL_oracle_accuracy_delta_vs_calibration_KL_fixed"] = (
        out["per_request_KL_ORACLE_accuracy"] - out["calibration_KL_fixed_accuracy"])
    out["KL_oracle_accuracy_delta_vs_accuracy_best_fixed"] = (
        out["per_request_KL_ORACLE_accuracy"] - out["evaluation_accuracy_best_fixed_ORACLE_accuracy"])
    if kl.shape[1] == 2:
        rho = row_spearman(kl[:, 0], kl[:, 1])
        out["wording_mean_spearman"] = float(rho[np.isfinite(rho)].mean()) if np.isfinite(rho).any() else np.nan
        out["wording_top_group_agreement"] = float((kl[:, 0].argmax(-1) == kl[:, 1].argmax(-1)).mean())
        out["wording_bottom_group_agreement"] = float((kl[:, 0].argmin(-1) == kl[:, 1].argmin(-1)).mean())
        out["cross_wording_KL_gain_vs_KL_best_fixed"] = fixed - out["cross_wording_KL_ORACLE_diagnostic_KL"]
        out["cross_wording_KL_relative_gain_vs_KL_best_fixed"] = out["cross_wording_KL_gain_vs_KL_best_fixed"] / fixed if fixed else 0.
        out.update(reversal_summary(kl))
    return out


def percentile_interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(ci_low=None, ci_high=None, valid_bootstraps=0)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return dict(ci_low=float(lo), ci_high=float(hi), valid_bootstraps=len(finite))


def bootstrap_basic(kl, correct, bf16_correct, datasets, fixed_kl, fixed_accuracy, iterations=2000):
    point = basic_statistics(kl, correct, bf16_correct, fixed_kl, fixed_accuracy)
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in point}
    for _ in range(iterations):
        idx = resample_sources(datasets, rng)
        # Evaluation-best fixed choices are re-selected INSIDE every bootstrap replicate.
        result = basic_statistics(kl[idx], correct[idx], bf16_correct[idx], fixed_kl, fixed_accuracy)
        for name, value in result.items():
            draws[name].append(value)
    return {name: dict(estimate=value if np.isfinite(value) else None, **percentile_interval(draws[name]))
            for name, value in point.items()}


def control_design(datasets, input_lengths, output_lengths=None, nonlinear=False):
    datasets = np.asarray(datasets)
    log_input = np.log(np.asarray(input_lengths, dtype=float))
    if not nonlinear:
        design = [np.asarray(datasets == d, dtype=float) for d in np.unique(datasets)]
        design.append(log_input)
        if output_lengths is not None:
            design.append(np.log(np.asarray(output_lengths, dtype=float)))
        return np.column_stack(design)
    design = []
    for dataset in np.unique(datasets):
        mask = datasets == dataset
        z = (log_input - log_input[mask].mean()) / max(log_input[mask].std(), 1e-12)
        design.extend([mask.astype(float)] + [mask * z ** power for power in [1, 2, 3]])
        if output_lengths is not None:
            output = np.log(np.asarray(output_lengths, dtype=float))
            z_out = (output - output[mask].mean()) / max(output[mask].std(), 1e-12)
            design.append(mask * z_out)
    return np.column_stack(design)


def profile_residuals(values, datasets, input_lengths, output_lengths=None, normalize=True, nonlinear=False):
    values = np.asarray(values, dtype=float)
    if normalize == "shape":
        values = values - values.mean(axis=-1, keepdims=True)
        scale = values.std(axis=-1, keepdims=True)
        values = np.divide(values, scale, out=np.zeros_like(values), where=scale > 1e-12)
    elif normalize:
        means = values.mean(axis=-1, keepdims=True)
        values = np.divide(values, means, out=np.ones_like(values), where=means > 1e-12)
    values = values - values.mean(axis=-1, keepdims=True)
    x = control_design(datasets, input_lengths, output_lengths, nonlinear)
    residual = values - x @ np.linalg.lstsq(x, values, rcond=None)[0]
    return residual - residual.mean(axis=-1, keepdims=True)


def residual_association(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    # A constant normalized group profile is a null even when magnitudes vary enormously.
    return float((a * b).sum() / denom) if denom > 1e-18 else np.nan


def controlled_association(kl, datasets, input_lengths, output_lengths=None, normalize=True, nonlinear=False):
    residuals = [profile_residuals(kl[:, w], datasets, input_lengths[:, w],
                 output_lengths[:, w] if output_lengths is not None else None, normalize, nonlinear) for w in range(2)]
    return residual_association(*residuals), residuals


def control_diagnostic(kl, datasets, input_lengths, request_lengths, output_lengths=None, iterations=2000):
    observed, (a, b) = controlled_association(kl, datasets, input_lengths, output_lengths)
    raw, _ = controlled_association(kl, datasets, input_lengths, output_lengths, normalize=False)
    rng = np.random.default_rng(SEED)
    strata = []
    datasets = np.asarray(datasets)
    for d in np.unique(datasets):
        idx = np.flatnonzero(datasets == d)
        ordered = idx[np.argsort(request_lengths[idx], kind="stable")]
        strata.extend([part for part in np.array_split(ordered, 4) if len(part)])
    null, draws = [], []
    for _ in range(iterations):
        permutation = np.arange(len(kl))
        for part in strata:
            permutation[part] = rng.permutation(part)
        null.append(residual_association(a, b[permutation]))
        idx = resample_sources(datasets, rng)
        stat, _ = controlled_association(kl[idx], datasets[idx], input_lengths[idx],
                                         output_lengths[idx] if output_lengths is not None else None)
        draws.append(stat)
    p = float((1 + np.sum(np.asarray(null) >= observed)) / (iterations + 1)) if np.isfinite(observed) else None
    return dict(normalized_residual_correlation=observed if np.isfinite(observed) else None,
        raw_centered_residual_correlation=raw if np.isfinite(raw) else None,
        normalized_residual_rms_a=float(np.sqrt(np.mean(a ** 2))),
        normalized_residual_rms_b=float(np.sqrt(np.mean(b ** 2))),
        permutation_p=p, permutation_null_mean=float(np.nanmean(null)) if np.isfinite(null).any() else None,
        permutations=iterations, strata_sizes=[len(s) for s in strata],
        **percentile_interval(draws))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    result = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    order = valid[np.argsort(np.asarray(p_values)[valid])]
    previous = 0.
    for i, idx in enumerate(order):
        previous = max(previous, min(1., p_values[idx] * (len(order) - i)))
        result[idx] = previous
    return result


def wild_shape_test(kl, datasets, input_lengths, output_lengths=None, iterations=2000,
                    bootstrap_iterations=None, seed=SEED):
    """Approximate nuisance-orthogonal HC2 source-cluster multiplier score test.

    All eight group contributions of a source share one Rademacher multiplier.
    Unlike raw-residual permutation, nuisance projection enters the score explicitly.
    Finite-sample validity is checked separately with synthetic null calibration.
    """
    datasets = np.asarray(datasets)
    association, (a, b) = controlled_association(kl, datasets, input_lengths, output_lengths,
                                                normalize="shape", nonlinear=True)
    x = control_design(datasets, input_lengths[:, 1],
                       output_lengths[:, 1] if output_lengths is not None else None, nonlinear=True)
    inverse = np.linalg.pinv(x)
    z = a - x @ (inverse @ a)
    leverage = np.clip(np.einsum("ij,ji->i", x, inverse), 0, 1 - 1e-9)
    contributions = np.einsum("ij,ij->i", z, b)
    hc2 = contributions / np.sqrt(1 - leverage)
    scale = np.linalg.norm(hc2)
    rng = np.random.default_rng(seed)
    if scale > 1e-12 and np.isfinite(association):
        observed = contributions.sum() / scale
        signs = rng.integers(0, 2, size=(iterations, len(kl)), dtype=np.int8) * 2 - 1
        null = (signs @ hc2) / scale
        p = float((1 + (null >= observed).sum()) / (iterations + 1))
    else:
        observed, p = None, None
    draws = []
    for _ in range(iterations if bootstrap_iterations is None else bootstrap_iterations):
        idx = resample_sources(datasets, rng)
        stat, _ = controlled_association(kl[idx], datasets[idx], input_lengths[idx],
                                         output_lengths[idx] if output_lengths is not None else None,
                                         normalize="shape", nonlinear=True)
        draws.append(stat)
    return dict(shape_residual_correlation=association if np.isfinite(association) else None,
        residual_rms_a=float(np.sqrt(np.mean(a ** 2))), residual_rms_b=float(np.sqrt(np.mean(b ** 2))),
        wild_score=observed, wild_p=p, wild_replicates=iterations,
        nuisance_design_rank=int(np.linalg.matrix_rank(x)), max_leverage=float(leverage.max()),
        method="approximate HC2 source-cluster Rademacher score; dataset-specific cubic log input length",
        **percentile_interval(draws))
