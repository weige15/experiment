"""Validate individual-layer raw evidence; grouped historical results cannot pass."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from quantization_sensitivity.data import CHAT_TEMPLATE, SOURCES, messages_for, save_json, score_answer
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.runner import file_hash, key, read_json
from scripts.validate_runs import check_measurement
from . import DEFAULT_RESULTS
from .quantization import LAYER_QUANTIZATION, SUFFIXES
from .runner import configuration, source_hashes


def check_coverage(frame, request_ids):
    expected = {(request_id, layer) for request_id in request_ids for layer in range(32)}
    assert len(frame) == len(expected), "requires every request × all 32 individual layers"
    assert not frame.duplicated(["request_id", "layer"]).any()
    assert set(zip(frame.request_id, frame.layer)) == expected, "wrong layer/request coverage"
    assert np.isfinite(frame.kl_mean).all() and (frame.kl_mean >= -1e-6).all()


def check_audit(audit, layer):
    assert len(audit) == 7 and {a["name"] for a in audit} == {
        f"model.layers.{layer}.{suffix}.weight" for suffix in SUFFIXES}
    assert all(a["changed_elements"] > 0 and a["mse"] > 0 and np.isfinite(a["mse"]) for a in audit)
    assert sum(a["numel"] for a in audit) == 218103808


def check_precision(check):
    assert check["kv_cache_dtypes"] == ["torch.bfloat16"]
    assert check["linear_input_output_dtypes"] == [["torch.bfloat16", "torch.bfloat16"]]


def validate_run(root, role, tokenizer):
    root = Path(root)
    path = root / "runs" / role
    all_requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    assert len({r["request_id"] for r in all_requests}) == len(all_requests)
    for dataset in SOURCES:
        subset = [r for r in all_requests if r["dataset"] == dataset]
        assert len({r["question_sha256"] for r in subset}) == len(subset)
        assert all(r["split"] == "train" for r in subset if r["role"] in ("demo", "calibration"))
        assert all(r["split"] == SOURCES[dataset][3] for r in subset if r["role"] in ("smoke", "evaluation"))
    requests = [r for r in all_requests if r["role"] == role]
    counts = Counter(r["dataset"] for r in requests)
    assert set(counts) == set(SOURCES) and min(counts.values()) >= (32 if role == "evaluation" else 8)
    demos = {d: [r for r in all_requests if r["role"] == "demo" and r["dataset"] == d] for d in SOURCES}
    config, summary = read_json(path / "run_config.json"), read_json(path / "summary.json")
    assert config == configuration(role, root / "requests.jsonl"), "changed inputs or measurement source"
    assert summary["status"] == "complete" and summary["configurations"] == 32
    assert summary["samples"] == len(requests) and summary["measured_rows"] == len(requests) * 32
    attempts = list(path.glob("attempt_*.json"))
    assert attempts
    for attempt in attempts:
        meta = read_json(attempt)
        assert meta["model_revision"] == REVISION and meta["model_id"] == MODEL_ID
        assert meta["kv_cache_precision"] == "BF16" and meta["deterministic"]
        assert meta["measurement_sources"] == source_hashes()
        assert meta["run_config_sha256"] == file_hash(path / "run_config.json")
        if role == "evaluation":
            selection = read_json(root / "precision_allocation_selection.json")
            assert meta["allocation_selection_sha256"] == file_hash(root / "precision_allocation_selection.json")
            assert selection["frozen_utc"] < meta["started_utc"], "fixed selection must precede final evaluation"
    frame = pd.read_csv(path / "measurements.csv", keep_default_na=False)
    check_coverage(frame, [r["request_id"] for r in requests])
    assert np.allclose(frame.sensitivity_rank, frame.groupby("request_id").kl_mean.rank(ascending=False))
    parquet = pd.read_parquet(path / "measurements.parquet")
    check_coverage(parquet, [r["request_id"] for r in requests])
    assert np.allclose(parquet.kl_mean, frame.kl_mean, rtol=0, atol=1e-12)
    vectors = pd.read_csv(path / "sensitivity_vectors.csv", index_col="request_id")
    assert list(vectors.columns) == [f"layer_{i}" for i in range(32)] and vectors.index.is_unique
    assert set(vectors.index) == {r["request_id"] for r in requests}
    base_frame = pd.read_csv(path / "baselines.csv", keep_default_na=False).set_index("request_id")
    assert base_frame.index.is_unique and set(base_frame.index) == {r["request_id"] for r in requests}
    index = frame.set_index(["request_id", "layer"])
    sentinels = {next(r["request_id"] for r in requests if r["dataset"] == d) for d in SOURCES}
    for layer in range(32):
        restoration = read_json(path / "checks" / f"layer_{layer}_restoration.json")
        assert restoration["exact"] and set(restoration["sentinels"]) == sentinels
        audit = read_json(path / "quantization" / f"layer_{layer}.json")
        check_audit(audit, layer)
        smoke = root / "runs" / "smoke" / "quantization" / f"layer_{layer}.json"
        if role != "smoke":
            assert audit == read_json(smoke), "different quantizer across roles"
    for request in requests:
        base = read_json(path / "baselines" / f"{key(request)}.json")
        assert base["request_id"] == request["request_id"] and base["dataset"] == request["dataset"]
        assert base["gold"] == request["gold"] and base["role"] == role and base["wording"] == "a"
        prompt = tokenizer.apply_chat_template(messages_for(request, demos[request["dataset"]], "a"),
                                               tokenize=False, add_generation_prompt=True)
        assert base["prompt"] == prompt
        assert tokenizer.encode(prompt, add_special_tokens=False) == base["input_ids"]
        assert len(base["input_ids"]) == base["input_tokens"]
        assert len(base["generated_ids"]) == base["generated_tokens"]
        assert tokenizer.decode(base["generated_ids"], skip_special_tokens=True) == base["answer"]
        assert base_frame.loc[request["request_id"], "answer"] == base["answer"]
        for name, expected in score_answer(base["answer"], request).items():
            assert base[name] == expected
        if request["request_id"] in sentinels:
            check = read_json(path / "checks" / f"{key(request)}_baseline.json")
            assert check["independent_replay_exact"] and check["self_kl_max"] == 0
            check_precision(check["precision"])
        for layer in range(32):
            row = read_json(path / f"layer_{layer}" / f"{key(request)}.json")
            check_measurement(row, base, request)
            assert row["layer"] == layer and row["dataset"] == request["dataset"]
            assert row["role"] == role and row["wording"] == "a"
            assert tokenizer.decode(row["generated_ids"], skip_special_tokens=True) == row["answer"]
            entry = index.loc[(request["request_id"], layer)]
            for field in ("answer", "correct", "bf16_correct", "parsed_answer", "parse_failure", "truncated"):
                assert entry[field] == (row[field] if row[field] is not None else "")
            assert json.loads(entry.token_kl) == row["token_kl"]
            assert np.isclose(entry.kl_mean, row["kl_mean"], rtol=0, atol=1e-12)
            assert np.isclose(vectors.loc[request["request_id"], f"layer_{layer}"], row["kl_mean"], rtol=0, atol=1e-12)
            if request["request_id"] in sentinels:
                check = read_json(path / "checks" / f"layer_{layer}_{key(request)}.json")
                assert check["independent_W4_replay_exact"]
                check_precision(check["precision"])
    return dict(status="verified", role=role, requests=len(requests), counts=dict(counts),
                layers=list(range(32)), measured_rows=len(frame), source_sha256=source_hashes(),
                input_sha256=file_hash(root / "requests.jsonl"),
                measurements_sha256=file_hash(path / "measurements.csv"),
                baselines_sha256=file_hash(path / "baselines.csv"),
                validator_sha256=file_hash(__file__), validated_utc=datetime.now(timezone.utc).isoformat())


def validate_gates(root):
    root = Path(root)
    baseline = read_json(root / "gates/baseline/baseline.json")
    meta = read_json(root / "gates/baseline/metadata.json")
    assert meta["model_revision"] == REVISION and meta["model_id"] == MODEL_ID
    check_precision(meta["precision_check"])
    for name in ["data", "model", "metrics", "quantization"]:
        path = f"quantization_sensitivity/{name}.py"
        assert meta["source_sha256"][path] == file_hash(path)
    for stage, expected_layers in [("single_layer", [0]), ("single_request", list(range(32)))]:
        out = root / "gates" / stage
        result, meta = read_json(out / "result.json"), read_json(out / "metadata.json")
        assert result["status"] == "passed" and result["actual_model"]
        assert result["layers"] == expected_layers and result["n_measurements"] == len(expected_layers)
        assert result["BF16_replay_exact"] and result["BF16_generation_restoration_exact"] and result["self_kl_max"] == 0
        assert meta["measurement_sources"] == source_hashes()
        assert meta["baseline_sha256"] == file_hash(root / "gates/baseline/baseline.json")
        assert meta["gate_source_sha256"] == file_hash("layer_sensitivity/gates.py")
        for layer in expected_layers:
            row = read_json(out / f"layer_{layer}.json")
            check_audit(row["quantization_audit"], layer)
            check_precision(row["precision"])
            assert row["independent_W4_replay_exact"] and row["restored_BF16_logits_exact"]
            assert row["measurement"]["layer"] == layer
            assert row["measurement"]["bf16_generated_ids"] == baseline["result"]["generated_ids"]
            assert np.isclose(np.mean(row["measurement"]["token_kl"]), row["measurement"]["kl_mean"], atol=1e-12)
    fingerprint = read_json(root / "gates/single_layer/parameter_fingerprints.json")
    before, during = fingerprint["before"], fingerprint["during"]
    assert set(before) == set(during)
    changed = {k for k in before if before[k] != during[k]}
    assert changed == {f"model.layers.0.{s}.weight" for s in SUFFIXES}
    assert before == read_json(root / "gates/single_layer/restored_parameter_fingerprints.json")
    return dict(status="verified", single_layer_changed_matrices=7, single_request_layers=32,
                baseline_request_id=baseline["request"]["request_id"], source_sha256=source_hashes())


def validate_allocation(root, tokenizer):
    from .precision_allocation_check import allocation_sources, strategies
    from .selection import BUDGET, random_sets, top_layers
    root = Path(root)
    out, run = root / "precision_allocation_check", root / "runs/evaluation"
    selection = read_json(root / "precision_allocation_selection.json")
    calibration = pd.read_csv(root / "runs/calibration/measurements.csv")
    assert selection["source_role"] == "calibration" and selection["budget"] == BUDGET
    assert selection["calibration_measurements_sha256"] == file_hash(root / "runs/calibration/measurements.csv")
    assert selection["fixed_high_precision_layers"] == top_layers(calibration.groupby("layer").kl_mean.mean().reindex(range(32)))
    assert selection["random_high_precision_layers"] == random_sets()
    assert selection["source_request_ids"] == sorted(calibration.request_id.unique())
    requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    requests = [r for r in requests if r["role"] == "evaluation"]
    assert not set(selection["source_request_ids"]) & {r["request_id"] for r in requests}
    values = pd.read_csv(run / "measurements.csv").pivot(index="request_id", columns="layer", values="kl_mean").reindex(columns=range(32))
    config = read_json(out / "run_config.json")
    assert config["source_sha256"] == allocation_sources()
    assert config["selection_sha256"] == file_hash(root / "precision_allocation_selection.json")
    assert config["evaluation_measurements_sha256"] == file_hash(run / "measurements.csv")
    assert config["requests_sha256"] == file_hash(root / "requests.jsonl") and config["budget"] == BUDGET
    summary = read_json(out / "summary.json")
    assert summary["status"] == "complete" and summary["requests"] == len(requests)
    assert summary["configurations"] == 5 and summary["measurements"] == len(requests) * 5
    assert read_json(out / "checks/restoration.json") == dict(parameters_exact=True, sentinel_logits_exact=True)
    before = read_json(out / "checks/BF16_parameter_fingerprints.json")
    for layer in range(32):
        assert read_json(out / "quantization" / f"layer_{layer}.json") == read_json(run / "quantization" / f"layer_{layer}.json")
    frame = pd.read_csv(out / "measurements.csv", keep_default_na=False)
    names = ["fixed", "random_0", "random_1", "random_2", "request_oracle"]
    expected = {(r["request_id"], s) for r in requests for s in names}
    assert len(frame) == len(expected) and not frame.duplicated(["request_id", "strategy"]).any()
    assert set(zip(frame.request_id, frame.strategy)) == expected
    index = frame.set_index(["request_id", "strategy"])
    for name in names:
        gate = read_json(out / "checks" / f"{name}_isolation.json")
        check_precision(gate["precision"])
        assert set(before) == set(gate["fingerprints"])
        changed = {k for k in before if before[k] != gate["fingerprints"][k]}
        expected_names = {f"model.layers.{i}.{s}.weight" for i in gate["state"]["quantized_layers"] for s in SUFFIXES}
        assert changed == expected_names == set(gate["changed_parameter_names"]) and len(changed) == 24 * 7
    for request in requests:
        base = read_json(run / "baselines" / f"{key(request)}.json")
        choices = strategies(selection, values.loc[request["request_id"]].to_numpy())
        for name in names:
            row = read_json(out / name / f"{key(request)}.json")
            assert row["budget"] == BUDGET and row["strategy"] == name
            state = row["state"]
            assert state["high_precision_layers"] == choices[name]
            assert state["quantized_layers"] == sorted(set(range(32)) - set(choices[name]))
            assert len(state["quantized_layers"]) == 24 and state["copied_weights_exact"]
            check_measurement(row, base, request)
            assert tokenizer.decode(row["generated_ids"], skip_special_tokens=True) == row["answer"]
            assert np.isclose(row["sum_individual_layer_kl"], values.loc[request["request_id"], state["quantized_layers"]].sum(), atol=1e-12)
            entry = index.loc[(request["request_id"], name)]
            assert json.loads(entry.state) == state and json.loads(entry.token_kl) == row["token_kl"]
            assert entry.answer == row["answer"] and entry.correct == row["correct"]
            assert np.isclose(entry.kl_mean, row["kl_mean"], atol=1e-12)
    return dict(status="verified", requests=len(requests), measurements=len(frame), budget=BUDGET,
                strategies=names, measurements_sha256=file_hash(out / "measurements.csv"),
                source_sha256=allocation_sources(), validator_sha256=file_hash(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RESULTS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--role", choices=["smoke", "calibration", "evaluation"])
    group.add_argument("--gates", action="store_true")
    group.add_argument("--allocation", action="store_true")
    args = parser.parse_args()
    if args.gates:
        result, name = validate_gates(args.root), "gates"
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        tokenizer.chat_template = CHAT_TEMPLATE
        result = validate_allocation(args.root, tokenizer) if args.allocation else validate_run(args.root, args.role, tokenizer)
        name = "precision_allocation" if args.allocation else args.role
    save_json(Path(args.root) / f"{name}_validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
