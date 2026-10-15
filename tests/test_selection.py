import numpy as np
import pytest
from scripts.select_fixed import choose_fixed


def test_fixed_selection_is_metric_matched_and_ties_are_deterministic():
    kl = np.array([[.1, .3, .5], [.1, .3, .5]])
    correct = np.array([[False, True, False], [False, True, False]])
    assert choose_fixed(kl, correct) == (0, 1)
    assert choose_fixed(np.ones((2, 8)), np.ones((2, 8), dtype=bool)) == (0, 0)
    with pytest.raises(ValueError):
        choose_fixed(np.array([[float('nan')]]), np.array([[True]]))
