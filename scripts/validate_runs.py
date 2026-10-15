"""Audit raw records against frozen samples, recompute KL/ranks/scores and token decoding."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.data import CHAT_TEMPLATE, SOURCES, messages_for, save_json, score_answer
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.quantization import QUANTIZATION
from quantization_sensitivity.runner import MEASUREMENT_SOURCES, file_hash, key, read_json


def check_measurement(row, base, request):
    assert row["request_id"] == base["request_id"] == request["request_id"]
    assert row["bf16_generated_ids"] == base["generated_ids"]
    assert row["reference_logits_sha256"] == base["reference_logits_sha256"]
    assert row["bf16_answer"] == base["answer"] and row["bf16_correct"] == base["correct"]
    assert row["gold"] == base["gold"] == request["gold"]
    assert row["input_tokens"] == base["input_tokens"]
    assert row["generated_tokens"] == len(row["generated_ids"])
    values = np.array(row["token_kl"])
    assert len(values) == len(base["generated_ids"]) == row["kl_tokens"]
    assert np.isfinite(values).all() and values.min() >= -1e-6
    assert np.isclose(values.mean(), row["kl_mean"], rtol=0, atol=1e-12)
    for field, expected in score_answer(row["answer"], request).items():
        assert row[field] == expected, (field, row["request_id"])
    assert row["bf16_correct"] == score_answer(base["answer"], request)["correct"]


def validate_run(path, all_requests, role, tokenizer):
    path = Path(path)
    requests = [r for r in all_requests if r["role"] == role]
    demos = {d: [r for r in all_requests if r["role"] == "demo" and r["dataset"] == d] for d in SOURCES}
    config, summary = read_json(path / "run_config.json"), read_json(path / "summary.json")
    assert config["role"] == role and summary["status"] == "complete"
    assert config["quantization"] == QUANTIZATION
    for attempt in path.glob("attempt_*.json"):
        meta = read_json(attempt)
        assert meta["model_revision"] == REVISION and meta["model_id"] == MODEL_ID
        assert meta["kv_cache_precision"] == "BF16"
    assert config["code_sha256"] == {f: file_hash(f) for f in MEASUREMENT_SOURCES}, (
        "measurement sources changed: use the original source snapshot to validate archived runs")
    csv = pd.read_csv(path / "measurements.csv", keep_default_na=False)
    base_csv = pd.read_csv(path / "baselines.csv", keep_default_na=False).set_index("request_id")
    assert len(base_csv) == len(requests) and base_csv.index.is_unique
    assert len(csv) == len(requests) * 8 == summary["measured_rows"]
    assert not csv.duplicated(["request_id", "group"]).any()
    expected_ranks = csv.groupby("request_id").kl_mean.rank(ascending=False, method="average")
    assert np.allclose(csv.sensitivity_rank, expected_ranks)
    index = csv.set_index(["request_id", "group"])
    group_sizes = []
    for g in range(8):
        assert read_json(path / "checks" / f"g{g}_restoration.json")["exact"]
        audit = read_json(path / "quantization" / f"g{g}.json")
        assert len(audit) == 28 and len({a["name"] for a in audit}) == 28
        suffixes = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]
        assert {a["name"] for a in audit} == {f"model.layers.{i}.{s}.weight"
            for i in range(g * 4, g * 4 + 4) for s in suffixes}
        assert all(a["changed_elements"] > 0 and a["mse"] > 0 for a in audit)
        group_sizes.append(sum(a["numel"] for a in audit))
    assert set(group_sizes) == {872415232}
    for r in requests:
        base = read_json(path / "baselines" / f"{key(r)}.json")
        prompt = tokenizer.apply_chat_template(messages_for(r, demos[r["dataset"]], config["wording"]),
                                               tokenize=False, add_generation_prompt=True)
        assert base["prompt"] == prompt
        assert tokenizer.encode(prompt, add_special_tokens=False) == base["input_ids"]
        assert len(base["input_ids"]) == base["input_tokens"]
        assert base_csv.loc[r["request_id"], "answer"] == base["answer"]
        assert bool(base_csv.loc[r["request_id"], "correct"]) == base["correct"]
        assert json.loads(base_csv.loc[r["request_id"], "input_ids"]) == base["input_ids"]
        assert tokenizer.decode(base["generated_ids"], skip_special_tokens=True) == base["answer"]
        assert len(base["generated_ids"]) == base["generated_tokens"]
        assert base["wording"] == config["wording"] and base["role"] == role
        for field, value in score_answer(base["answer"], r).items():
            assert base[field] == value
        for g in range(8):
            row = read_json(path / f"g{g}" / f"{key(r)}.json")
            check_measurement(row, base, r)
            assert row["group"] == g and row["role"] == role and row["wording"] == config["wording"]
            assert tokenizer.decode(row["generated_ids"], skip_special_tokens=True) == row["answer"]
            entry = index.loc[(r["request_id"], g)]
            assert np.isclose(entry.kl_mean, row["kl_mean"], rtol=0, atol=1e-12)
            assert entry.answer == row["answer"] and bool(entry.correct) == row["correct"]
            assert json.loads(entry.token_kl) == row["token_kl"]
    for d in SOURCES:
        sentinel = next(r for r in requests if r["dataset"] == d)
        check = read_json(path / "checks" / f"{key(sentinel)}_baseline.json")
        assert check["independent_replay_exact"] and check["self_kl_max"] == 0
        assert check["precision"]["kv_cache_dtypes"] == ["torch.bfloat16"]
        assert check["precision"]["linear_input_output_dtypes"] == [["torch.bfloat16", "torch.bfloat16"]]
    return dict(path=str(path), role=role, wording=config["wording"], requests=len(requests),
                measured_rows=len(csv), counts=dict(Counter(r["dataset"] for r in requests)), verified=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--role", required=True, choices=["smoke", "calibration", "evaluation"])
    p.add_argument("--requests", default=f"{DEFAULT_RESULTS}/requests.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.chat_template = CHAT_TEMPLATE
    results = []
    for path in args.runs:
        config = read_json(Path(path) / "run_config.json")
        assert config["requests_sha256"] == file_hash(args.requests)
        result = validate_run(path, rows, args.role, tokenizer)
        assert all(n >= (32 if args.role == "evaluation" else 8) for n in result["counts"].values())
        results.append(result)
        print("PASS", json.dumps(result), flush=True)
    assert sorted(r["wording"] for r in results) == ["a", "b"], "both predefined wordings required"
    for g in range(8):
        audits = [read_json(Path(r["path"]) / "quantization" / f"g{g}.json") for r in results]
        assert {a["name"]: a["packed_scale_sha256"] for a in audits[0]} == {
            a["name"]: a["packed_scale_sha256"] for a in audits[1]}, "quantized weights differ between wordings"
    save_json(args.out, dict(status="verified", runs=results))


if __name__ == "__main__":
    main()
