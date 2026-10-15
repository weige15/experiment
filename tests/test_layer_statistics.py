"""Synthetic method/unit checks only; these are never Llama measurements."""
import numpy as np
import pandas as pd
import pytest

from layer_sensitivity.statistics import (bootstrap_summary, control_design, pair_average,
    pair_matrices, row_spearman, shape_normalize, stratified_indices)


def test_accuracy_intervals_do_not_collapse_without_discordance():
    from layer_sensitivity.statistics import exact_binomial_interval, conservative_paired_accuracy_interval
    lo, hi = exact_binomial_interval(0, 96)
    assert lo == 0 and 0 < hi < .05
    lo, hi = exact_binomial_interval(96, 96)
    assert .95 < lo < 1 and hi == 1
    interval = conservative_paired_accuracy_interval(np.ones(96), np.ones(96))
    assert interval["estimate"] == 0 and interval["ci_low"] < 0 < interval["ci_high"]
    all_gain = conservative_paired_accuracy_interval(np.ones(96), np.zeros(96))
    assert all_gain["ci_low"] > .9 and all_gain["ci_high"] == 1
    from layer_sensitivity.statistics import stratified_binomial_interval
    strata = np.repeat(["arc", "boolq", "gsm8k"], 32)
    stratified = conservative_paired_accuracy_interval(np.ones(96), np.ones(96), strata)
    assert stratified["ci_low"] < interval["ci_low"] < interval["ci_high"] < stratified["ci_high"]
    assert stratified_binomial_interval(np.zeros(96), strata)[1] > exact_binomial_interval(0, 96)[1]
    with pytest.raises(ValueError):
        exact_binomial_interval(2, 1)


def test_exact_reverse_ranking_overlap_and_material_reversal():
    values = np.stack([np.arange(32), np.arange(32)[::-1]]).astype(float)
    pairs = pair_matrices(values)
    assert pairs["spearman"][0, 1] == pytest.approx(-1)
    assert pairs["top4_overlap_fraction"][0, 1] == 0
    assert pairs["top8_jaccard"][0, 1] == 0
    assert pairs["ranking_reversal_rate"][0, 1] == 1
    assert pairs["spearman"][0, 0] == pytest.approx(1)
    assert row_spearman(values[:1], values[1:])[0] == pytest.approx(-1)


def test_empty_secondary_content_is_not_an_artificial_topk_match():
    values = np.stack([np.arange(32), np.arange(32)[::-1], np.full(32, np.nan)]).astype(float)
    pairs = pair_matrices(values)
    for matrix in pairs.values():
        assert np.isnan(matrix[2]).all() and np.isnan(matrix[:, 2]).all()
    assert pair_average(pairs["spearman"], [1, 1, 1]) == pytest.approx(-1)
    assert pair_average(pairs["top4_overlap_fraction"], [1, 1, 1]) == 0


def test_bootstrap_duplicates_are_not_self_pairs():
    matrix = np.array([[1., .25], [.25, 1.]])
    assert pair_average(matrix, [3, 1]) == .25
    assert np.isnan(pair_average(matrix, [4, 0]))
    assert np.isnan(pair_average(np.full((2, 2), np.nan), [1, 1]))


def test_shape_normalization_removes_magnitude_and_offset():
    values = np.random.default_rng(2).random((12, 32))
    scale = np.arange(1, 13)[:, None]
    assert np.allclose(shape_normalize(values), shape_normalize(5 + values * scale))
    assert np.array_equal(shape_normalize(np.ones((2, 32))), np.zeros((2, 32)))
    assert np.isnan(row_spearman(np.ones((2, 32)), np.ones((2, 32)))).all()


def test_protocol_cubic_design_includes_output_powers_and_isolates_datasets():
    from layer_sensitivity.statistics import cubic_prompt_output_design
    rng = np.random.default_rng(8)
    datasets = np.repeat(["arc", "boolq", "gsm8k"], 12)
    prompt = rng.integers(20, 500, len(datasets))
    output = rng.integers(4, 200, len(datasets))
    x, names = cubic_prompt_output_design(datasets, prompt, output)
    assert x.shape == (36, 21)
    for dataset in set(datasets):
        mask = datasets == dataset
        log_output = np.log(output[mask])
        z = (log_output - log_output.mean()) / log_output.std()
        for power in [1, 2, 3]:
            column = x[:, names.index(f"{dataset}_log_output_z_power_{power}")]
            assert np.allclose(column[mask], z ** power)
            assert np.array_equal(column[~mask], np.zeros((~mask).sum()))
    target = x[:, names.index("gsm8k_log_output_z_power_3")]
    assert np.allclose(x @ np.linalg.lstsq(x, target, rcond=None)[0], target)
    constant, _ = cubic_prompt_output_design(datasets, prompt, np.ones(len(datasets)))
    assert np.isfinite(constant).all()


def test_dataset_strata_and_paired_allocation_uncertainty(tmp_path):
    rng = np.random.default_rng(3)
    datasets = np.repeat(["arc", "boolq", "gsm8k"], 4)
    values = rng.random((12, 32)) * .01
    correct = rng.integers(0, 2, size=(12, 32)).astype(float)
    baseline = np.ones(12)
    x, names = control_design(datasets, rng.integers(30, 400, 12), rng.integers(4, 30, 12), baseline)
    allocation = np.zeros((12, 5))
    allocation[:, 4] = 1
    segments = dict(full_halves=np.ones(12), content_halves=np.r_[np.full(8, np.nan), np.ones(4)])
    indices = stratified_indices(datasets, rng)
    assert all(sum(datasets[indices] == dataset) == 4 for dataset in set(datasets))
    rows, _ = bootstrap_summary(values, correct, baseline, datasets, segments,
        allocation, np.ones((12, 5)), np.ones((12, 5)), x, names, iterations=12,
        secondary_values={"content": values})
    table = pd.DataFrame(rows).set_index(["scope", "metric"])
    delta = table.loc[("all", "allocation_request_oracle_delta_vs_fixed")]
    assert delta.estimate == delta.ci_low == delta.ci_high == 1
    assert "in-sample refit range" in table.loc[("controls", "shape_R2"), "interval_interpretation"]
    assert "percentile interval" in delta.interval_interpretation
    assert table.loc[("all", "secondary_content_different_request_spearman"), "estimate"] == pytest.approx(
        table.loc[("all", "different_request_spearman"), "estimate"])
    assert table.loc[("gsm8k", "content_halves_mean_spearman"), "estimate"] == 1
    assert np.isnan(table.loc[("arc", "content_halves_mean_spearman"), "estimate"])
    assert table.loc[("arc", "content_halves_mean_spearman"), "valid_bootstraps"] == 0


def test_figures_are_reconstructible_without_optional_tabulate(tmp_path):
    from layer_sensitivity.analysis import ANALYSIS_SOURCES, figures, markdown_table
    assert "quantization_sensitivity/analysis.py" in ANALYSIS_SOURCES
    rng = np.random.default_rng(4)
    values = rng.random((12, 32)) * .01
    baseline = pd.DataFrame(dict(request_id=[f"synthetic_test_only_{i}" for i in range(12)]))
    metrics = ["BF16_accuracy", *[f"allocation_{name}_accuracy" for name in
               ["fixed", "random_0", "random_1", "random_2", "request_oracle"]], "allocation_candidate_label_oracle_accuracy"]
    statistics = pd.DataFrame([dict(scope="all", metric=m, estimate=.5, ci_low=.4, ci_high=.6) for m in metrics])
    figures(tmp_path, baseline, values, pair_matrices(values), statistics, np.zeros((12, 5)))
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 4
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert "a\\|b" in markdown_table(pd.DataFrame({"value": ["a|b"]}))
