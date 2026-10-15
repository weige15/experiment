"""Offline completion evidence checks, including independent raw-table arithmetic.

This verifies delivered measurements; it is not by itself a scientific-success decision.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.data import save_json
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.runner import MEASUREMENT_SOURCES, file_hash, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_RESULTS)
    p.add_argument("--compare-analysis", help="Optional archived analysis to check exact re-analysis reproducibility")
    args = p.parse_args()
    root = Path(args.root)
    model = read_json(root / "model_provenance.json")
    assert (model["model_id"], model["revision"]) == (MODEL_ID, REVISION)
    assert all(f["verified"] for f in model["files"])
    assert sum(f["file"].endswith(".safetensors") for f in model["files"]) == 4
    manifest = read_json(root / "data_manifest.json")
    assert manifest["requests_sha256"] == file_hash(root / "requests.jsonl")
    assert manifest["counts"] == dict(demo=9, calibration=24, smoke=24, evaluation=96)
    for role, n in [("smoke", 24), ("calibration", 24), ("evaluation", 96)]:
        check = read_json(root / f"{role}_validation.json")
        assert check["status"] == "verified"
        for wording in ["a", "b"]:
            config = read_json(root / "runs" / f"{role}_{wording}" / "run_config.json")
            assert config["requests_sha256"] == manifest["requests_sha256"]
            assert config["code_sha256"] == {f: file_hash(f) for f in MEASUREMENT_SOURCES}, (
                "measurement sources changed: use the original source snapshot to audit archived runs")
            frame = pd.read_csv(root / "runs" / f"{role}_{wording}" / "measurements.csv")
            assert len(frame) == n * 8 and frame.request_id.nunique() == n
    selection = read_json(root / "calibration_selection.json")
    assert not selection["evaluation_data_used"]
    assert all(file_hash(root / f) == h for f,h in selection["input_sha256"].items())
    for wording in ["a", "b"]:
        for path in (root / "runs" / f"evaluation_{wording}").glob("attempt_*.json"):
            assert datetime.fromisoformat(selection["selected_utc"]) < datetime.fromisoformat(read_json(path)["started_utc"])
    one = read_json(root / "steps" / "one_group" / "result.json")
    assert one["exact_restoration"] and one["exact_baseline_reproduction"]
    assert one["before_sha256"] == one["after_sha256"]
    changed = {k for k,v in one["before_sha256"].items() if one["during_sha256"][k] != v}
    assert changed == set(one["changed_parameter_names"]) and len(changed) == 28
    gate = read_json(root / "steps" / "single_sensitivity" / "diagnostics.json")
    assert gate["independent_BF16_replay_exact"] and gate["g0_independent_W4_replay_exact"]
    assert gate["g0_repeat_mean_discrepancy"] == 0 and max(map(abs, gate["reference_self_kl"])) == 0
    eight = read_json(root / "steps" / "eight_groups" / "diagnostics.json")
    assert all(eight[f"g{g}_BF16_restore_logits_exact"] for g in range(8))
    analysis = root / "analysis"
    meta = read_json(analysis / "analysis_metadata.json")
    assert meta["bootstrap_replicates"] >= 2000 and meta["source_requests"] == 96
    assert meta["worded_requests"] == 192 and meta["W4_rows"] == 1536 and meta["actual_answer_rows"] == 1728
    assert all(not x["missing_request_ids"] and x["source_requests"] == 96 for x in meta["coverage"])
    assert all(file_hash(root / f) == h for f,h in meta["input_sha256"].items())
    assert all(file_hash(f) == h for f,h in meta["code_sha256"].items())
    assert meta["calibration_selection_sha256"] == file_hash(root / "calibration_selection.json")
    null = read_json(root / "steps" / "statistical_null_calibration.json")
    assert null["completed"] and null["all_criteria_pass"] and null["trials_per_case"] >= 500
    assert null["code_sha256"].get("quantization_sensitivity/analysis.py") == file_hash(
        "quantization_sensitivity/analysis.py"), "analysis source changed: rerun statistical-method calibration"
    rows = pd.concat([pd.read_csv(root / "runs" / f"evaluation_{w}" / "measurements.csv") for w in ["a","b"]])
    baselines = pd.concat([pd.read_csv(root / "runs" / f"evaluation_{w}" / "baselines.csv") for w in ["a","b"]])
    kl = rows.pivot(index=["request_id","wording"], columns="group", values="kl_mean").to_numpy()
    correct = rows.pivot(index=["request_id","wording"], columns="group", values="correct").to_numpy().astype(float)
    expected = dict(BF16_accuracy=float(baselines.correct.mean()),
        evaluation_KL_best_fixed_ORACLE_KL=float(kl.mean(axis=0).min()),
        per_request_KL_ORACLE_KL=float(kl.min(axis=1).mean()),
        evaluation_accuracy_best_fixed_ORACLE_accuracy=float(correct.mean(axis=0).max()),
        per_request_label_ORACLE_accuracy=float(correct.max(axis=1).mean()),
        per_request_KL_ORACLE_accuracy=float(correct[np.arange(len(kl)),kl.argmin(axis=1)].mean()))
    expected["KL_oracle_gain_vs_KL_best_fixed"] = expected["evaluation_KL_best_fixed_ORACLE_KL"]-expected["per_request_KL_ORACLE_KL"]
    expected["label_oracle_accuracy_gain_vs_accuracy_best_fixed"] = expected["per_request_label_ORACLE_accuracy"]-expected["evaluation_accuracy_best_fixed_ORACLE_accuracy"]
    stats = pd.read_csv(analysis / "statistics.csv")
    overall = stats[(stats.metric=="full")&(stats.scope=="all")&(stats.wording=="paired")].set_index("statistic")
    for name, value in expected.items():
        np.testing.assert_allclose(overall.loc[name,"estimate"], value, atol=1e-12, rtol=1e-10)
    compared = []
    if args.compare_analysis:
        for path in sorted(analysis.glob("*.csv")):
            pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(Path(args.compare_analysis)/path.name), check_exact=True)
            compared.append(path.name)
    for name in ["sensitivity_heatmap", "sensitivity_heatmap_content", "sensitivity_heatmap_normalized", "accuracy_comparison"]:
        assert (root/(name+".png")).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (root/(name+".pdf")).read_bytes().startswith(b"%PDF")
    report = read_json(root / "report_provenance.json")
    assert report["report_sha256"] == file_hash(root / "report.md")
    assert all(file_hash(path)==digest for path,digest in report["inputs"].items())
    assert report["renderer_sha256"] == file_hash("scripts/report_results.py")
    save_json(root / "completion_checks.json", dict(status="verified_artifacts_not_automatic_goal_completion",
        source_requests=96, worded_requests=192, W4_rows=1536, actual_answers=1728,
        independent_raw_table_checks=expected, exact_reanalysis_csvs=compared,
        bounded_evidence_supported=report["bounded_evidence_supported"], scientific_interpretation_requires_review=True))
    print("PASS: complete raw evidence, early fixed choice, immutable source hashes, independent raw arithmetic, figures/report")
    print("Exact re-analysis CSVs:", compared)
    print(expected)


if __name__ == "__main__":
    main()
