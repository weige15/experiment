from m1_data import CHAT_TEMPLATE, messages_for, parse_answer, score_answer


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
