import numpy as np
import pytest
from quantization_sensitivity.analysis import (basic_statistics, bootstrap_basic, content_indices, control_diagnostic,
                                              controlled_association, holm_adjust, profile_residuals, reversal_summary,
                                              row_spearman, strategy_choices, wild_shape_test)


class TinyTokenizer:
    all_special_ids = [9]
    vocabulary = {0: ' yes', 1: '\n\n', 2: 'Q', 3: ':', 4: 'reason', 5: 'ing', 6: ' ', 7: '.\n', 8: 'yes\n\nQ:', 9: ''}
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.vocabulary[i] for i in ids)


def test_content_mask_excludes_delimiter_and_eos_not_the_answer():
    tokenizer = TinyTokenizer()
    assert content_indices(tokenizer, [0, 1, 2, 3]) == [0]
    assert content_indices(tokenizer, [0, 9]) == [0]
    assert content_indices(tokenizer, [4, 5]) == [0, 1]
    assert content_indices(tokenizer, [4, 5, 6]) == [0, 1]
    assert content_indices(tokenizer, [9]) == []
    assert content_indices(tokenizer, [6, 0, 1, 2, 3]) == [1]
    assert content_indices(tokenizer, [4, 6, 5]) == [0, 2]
    assert content_indices(tokenizer, [7, 9]) == [0]  # mixed punctuation/whitespace token is content
    assert content_indices(tokenizer, [8]) == []  # indivisible token crossing delimiter is excluded


def test_global_magnitude_is_not_a_request_group_interaction():
    n = 32
    magnitude = np.geomspace(.001, 100, n)
    profile = np.arange(1, 9.)
    x = magnitude[:, None] * profile
    ds, length = np.repeat(['one', 'two'], n // 2), np.arange(100, 100 + n)
    residual = profile_residuals(x, ds, length)
    assert np.abs(residual).max() < 1e-10
    paired = np.stack([x, x * 2], axis=1)
    association, _ = controlled_association(paired, ds, np.column_stack([length, length]))
    assert np.isnan(association)
    assert reversal_summary(paired)['replicated_reversal_pairs'] == 0


def test_real_interaction_survives_controls_and_stratified_permutation():
    rng = np.random.default_rng(70)
    n = 64
    ds = np.repeat(['one', 'two'], n // 2)
    length = np.arange(100, 100 + n)
    x = rng.uniform(.1, 2, size=(n, 8))
    paired = np.stack([x, x * 3], axis=1)
    result = control_diagnostic(paired, ds, np.column_stack([length, length]), length, iterations=99)
    assert result['normalized_residual_correlation'] == pytest.approx(1)
    assert result['permutation_p'] <= .02
    assert result['valid_bootstraps'] == 99


def test_oracle_compares_to_metric_matched_best_fixed_and_cross_wording_can_lose():
    kl = np.array([[[1., 9.], [9., 1.]], [[1., 9.], [9., 1.]]])
    # Pad groups with uniformly bad choices; 8 alternatives must remain.
    kl = np.concatenate([kl, np.full((2, 2, 6), 20.)], axis=-1)
    correct = kl == 1
    choices = strategy_choices(kl, correct, 0, 0)
    assert choices['per_request_KL_ORACLE'].tolist() == [[0, 1], [0, 1]]
    stats = basic_statistics(kl, correct, np.ones((2, 2)), 0, 0)
    assert stats['KL_oracle_gain_vs_KL_best_fixed'] == 4
    assert stats['cross_wording_KL_gain_vs_KL_best_fixed'] == -4
    assert stats['label_oracle_accuracy_gain_vs_accuracy_best_fixed'] == .5


def test_bootstrap_resamples_sources_not_independent_wordings():
    kl = np.tile(np.arange(1., 9.), (12, 2, 1))
    correct = np.ones_like(kl, dtype=bool)
    # Every source has mean accuracy .5. Wrong independent-wording resampling gives a nonzero CI width.
    bf16 = np.tile([0., 1.], (12, 1))
    result = bootstrap_basic(kl, correct, bf16, np.repeat(['one', 'two'], 6), 0, 0, iterations=40)
    assert result['BF16_accuracy']['ci_low'] == result['BF16_accuracy']['ci_high'] == .5
    assert result['BF16_accuracy']['valid_bootstraps'] == 40


def test_best_fixed_is_reselected_not_frozen_in_resampling():
    kl = np.tile(np.arange(1., 9.), (2, 2, 1))
    correct = np.zeros_like(kl)
    correct[0, :, 0], correct[1, :, 1] = 1, 1
    point = basic_statistics(kl, correct, np.ones((2, 2)), 0, 0)
    resampled = basic_statistics(kl[[1, 1]], correct[[1, 1]], np.ones((2, 2)), 0, 0)
    assert point['evaluation_accuracy_best_fixed_ORACLE_accuracy'] == .5
    assert resampled['evaluation_accuracy_best_fixed_ORACLE_accuracy'] == 1


def test_reversals_must_replicate_in_both_wordings_and_ignore_near_ties():
    x = np.tile(np.arange(1., 9.), (2, 2, 1))
    x[1, :, :2] = [2, 1]
    assert reversal_summary(x)['replicated_reversal_pairs'] == 1
    x[1, 1, :2] = [1, 2]
    assert reversal_summary(x)['replicated_reversal_pairs'] == 0
    x = np.ones((2, 2, 8)) + np.arange(32).reshape(2, 2, 8) * 1e-7
    assert reversal_summary(x)['replicated_reversal_pairs'] == 0
    assert np.allclose(row_spearman(np.array([[1, 2, 3]]), np.array([[3, 2, 1]])), -1)


def test_holm_adjustment_is_monotonic_and_keeps_missing_values():
    adjusted = holm_adjust([.01, .04, .03, np.nan])
    assert np.allclose(adjusted[:3], [.03, .06, .06])
    assert np.isnan(adjusted[3])
    assert np.isnan(holm_adjust([None, np.nan])).all()


def test_shape_control_removes_additive_and_multiplicative_global_profiles():
    n = 32
    offset, scale = np.arange(n)[:, None] + 3, np.linspace(1, 4, n)[:, None]
    x = offset + scale * np.arange(1., 9.)
    paired = np.stack([x, x * 3 + 7], axis=1)
    length = np.column_stack([np.arange(100, 100+n)] * 2)
    result = wild_shape_test(paired, np.repeat('one', n), length, iterations=99, bootstrap_iterations=10)
    assert result['shape_residual_correlation'] is None
    assert result['wild_p'] is None
    assert result['residual_rms_a'] < 1e-10
    assert reversal_summary(paired)['replicated_reversal_pairs'] == 0


def test_wild_score_detects_strong_paired_shape_not_wording_independence():
    rng = np.random.default_rng(99)
    n = 64
    x = rng.uniform(.1, 2, size=(n, 8))
    paired = np.stack([x, x * 3 + .5], axis=1)
    lengths = np.column_stack([np.arange(100, 100+n)] * 2)
    result = wild_shape_test(paired, np.repeat(['one', 'two'], n // 2), lengths,
                             iterations=199, bootstrap_iterations=20)
    assert result['shape_residual_correlation'] == pytest.approx(1)
    assert result['wild_p'] <= .01
    assert result['valid_bootstraps'] == 20
