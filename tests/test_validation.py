import copy
import pytest
from scripts.validate_runs import check_measurement


def records():
    request = dict(request_id="test-only", dataset="boolq", labels=[], gold="yes")
    base = dict(request_id="test-only", generated_ids=[1, 2], input_tokens=3,
                reference_logits_sha256="unit-test-only", answer="yes", correct=True, gold="yes")
    row = dict(request_id="test-only", bf16_generated_ids=[1, 2], input_tokens=3,
               generated_tokens=2, generated_ids=[1, 2], bf16_answer="yes", bf16_correct=True,
               reference_logits_sha256="unit-test-only", token_kl=[.1, .3], kl_tokens=2,
               kl_mean=.2, answer="no", correct=False, parsed_answer="no", parse_failure=False, gold="yes")
    return row, base, request


def test_recomputes_validity_instead_of_trusting_record_flags():
    row, base, request = records()
    check_measurement(row, base, request)
    for field, wrong in [("correct", True), ("kl_mean", .9), ("token_kl", [.1, float("nan")]),
                         ("bf16_generated_ids", [2, 1]), ("reference_logits_sha256", "wrong")]:
        broken = copy.deepcopy(row)
        broken[field] = wrong
        with pytest.raises(AssertionError):
            check_measurement(broken, base, request)
