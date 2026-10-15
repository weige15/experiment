import pandas as pd
import m1_data
from m1_data import CHAT_TEMPLATE, messages_for, normalize, parse_answer, score_answer


def test_gsm8k_parser_requires_final_answer():
    assert parse_answer("we have 12 apples", "gsm8k") is None
    assert parse_answer("reasoning\n#### 1,200.00", "gsm8k") == "1.2E+3"
    r = dict(dataset="gsm8k", labels=[], gold="1200")
    assert score_answer("#### 1,200.00", r)["correct"]
    assert not score_answer("unfinished 1200", r)["correct"]


def test_arc_and_boolq_parser():
    assert parse_answer(" B.\n", "arc", ["A", "B"]) == "B"
    assert parse_answer("C", "arc", ["A", "B"]) is None
    assert parse_answer("The answer is B", "arc", ["A", "B"]) is None
    assert parse_answer("no. indeed", "boolq") == "no"
    assert parse_answer("yesterday", "boolq") is None
    assert parse_answer("YES", "boolq") == "yes"


def test_messages_do_not_contain_evaluation_answer():
    r = dict(dataset="gsm8k", text="question", gold="SECRET", demo_answer="SECRET")
    messages = messages_for(r, [], "a")
    assert "SECRET" not in str(messages)
    assert messages[-1]["role"] == "user"
    assert "bos_token" in CHAT_TEMPLATE


def test_duplicate_question_hash_ignores_passage_and_whitespace():
    a = normalize(dict(question="Is THIS true?", passage="a", answer=True), "boolq", "train", 0)
    b = normalize(dict(question="is  this true?", passage="b", answer=False), "boolq", "validation", 1)
    assert a["content_sha256"] != b["content_sha256"]
    assert a["question_sha256"] == b["question_sha256"]


def test_sampling_is_reproducible_and_split_isolated(monkeypatch):
    def split(dataset, name):
        return pd.DataFrame([dict(question=f"{name} question {i}", answer=f"#### {i}")
                             for i in range(100)])
    monkeypatch.setattr(m1_data, "load_split", split)
    first, excluded = m1_data.sample_dataset("gsm8k")
    second, _ = m1_data.sample_dataset("gsm8k")
    assert first == second and not excluded
    assert len({r["question_sha256"] for r in first}) == len(first) == 51
    assert sum(r["role"] == "evaluation" for r in first) == 32
    assert all(r["split"] == "train" for r in first if r["role"] in ["demo", "calibration"])
    assert all(r["split"] == "test" for r in first if r["role"] in ["smoke", "evaluation"])
