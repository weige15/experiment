from types import SimpleNamespace
import math
import pytest
import torch

from quantization_sensitivity.metrics import continuation_logits, prediction_inputs, summarize_kl, token_kl
from quantization_sensitivity.model import RequireFiniteLogits


def test_causal_shift_includes_first_generated_token_and_excludes_last_input():
    prompt, continuation = torch.tensor([[10, 20, 30]]), [40, 50, 60]
    assert prediction_inputs(prompt, continuation).tolist() == [[10, 20, 30, 40, 50]]
    class ToyModel:
        def __call__(self, input_ids, logits_to_keep, **kwargs):
            return SimpleNamespace(logits=input_ids[:, -logits_to_keep:, None].float())
    assert continuation_logits(ToyModel(), prompt, continuation).flatten().tolist() == [30, 40, 50]
    assert continuation_logits(ToyModel(), prompt, [40]).flatten().tolist() == [30]


def test_analytic_kl_direction_and_identical_reference():
    p, q = torch.tensor([[.75, .25]]), torch.tensor([[.5, .5]])
    actual = token_kl(p.log(), q.log())[0]
    assert actual == pytest.approx(.75 * math.log(1.5) + .25 * math.log(.5), abs=1e-7)
    assert token_kl(p.log(), p.log()) == [0.0]
    assert token_kl(q.log(), p.log())[0] != pytest.approx(actual)


def test_kl_chunking_and_extreme_logits():
    p = torch.tensor([[1000., -1000.], [-1000., 1000.], [1., 2.]])
    q = torch.zeros_like(p)
    assert token_kl(p, q, chunk=1) == token_kl(p, q, chunk=16)
    assert token_kl(p, q)[0] == pytest.approx(math.log(2), abs=1e-6)
    with pytest.raises(FloatingPointError):
        token_kl(p * float('nan'), q)
    with pytest.raises(ValueError):
        token_kl(p, q[:1])


def test_short_segments_are_not_fabricated():
    assert summarize_kl([.1])["kl_first_half"] is None
    s = summarize_kl([1., 2., 3., 4., 5.])
    assert s["kl_first_half"] == 1.5 and s["kl_second_half"] == 4.
    assert s["kl_mean"] == 3.


def test_generation_finite_guard():
    guard = RequireFiniteLogits()
    x = torch.ones(1, 8)
    assert guard(None, x) is x
    with pytest.raises(FloatingPointError):
        guard(None, torch.tensor([[float('inf')]]))
