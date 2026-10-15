"""CPU-only checks for descriptive entrypoints and immutable measurement provenance."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from quantization_sensitivity import DEFAULT_RESULTS, model, runner
from scripts import analyze_results

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "quantization_sensitivity.runner",
    "scripts.analyze_results",
    "scripts.plot_results",
    "scripts.report_results",
    "scripts.audit_results",
    "scripts.check_content",
    "layer_sensitivity.runner",
    "layer_sensitivity.gates",
    "layer_sensitivity.selection",
    "layer_sensitivity.precision_allocation_check",
    "layer_sensitivity.validate",
    "layer_sensitivity.analysis",
])
def test_entrypoint_help_without_gpu_or_network(module):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPOSITORY,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("change", [None, "requests", "source", "missing_key", "legacy"])
def test_resume_checks_inputs_before_loading_model(tmp_path, monkeypatch, change):
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(dict(dataset=dataset, role="smoke")) + "\n"
                                for dataset in runner.SOURCES for _ in range(8)))
    config = dict(role="smoke", wording="a", requests_sha256=runner.file_hash(requests),
                  code_sha256={name: runner.file_hash(name) for name in runner.MEASUREMENT_SOURCES},
                  quantization=runner.QUANTIZATION)
    if change == "requests":
        # Keep valid JSONL while changing its exact bytes.
        requests.write_text(requests.read_text().rstrip() + " \n")
    elif change == "source":
        config["code_sha256"][runner.MEASUREMENT_SOURCES[0]] = "0" * 64
    elif change == "missing_key":
        config["code_sha256"].pop(runner.MEASUREMENT_SOURCES[0])
    elif change == "legacy":
        config["code_sha256"] = {"m1_run.py": "0" * 64}
    runner.save_json(tmp_path / "run_config.json", config)
    monkeypatch.setattr(sys, "argv", ["runner", "--role", "smoke", "--wording", "a",
                                     "--requests", str(requests), "--out", str(tmp_path)])

    class ModelLoadReached(Exception):
        pass

    def load_model():
        raise ModelLoadReached

    monkeypatch.setattr(runner, "load_model", load_model)
    if change is None:
        with pytest.raises(ModelLoadReached):
            runner.main()
    else:
        with pytest.raises(AssertionError, match="inputs/code changed"):
            runner.main()
    assert runner.read_json(tmp_path / "run_config.json") == config


def test_metadata_hashes_real_package_sources_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(model.torch.cuda, "get_device_name", lambda: "CPU-only test")
    monkeypatch.setattr(model.subprocess, "check_output", lambda *args, **kwargs: "test")
    monkeypatch.chdir(tmp_path)
    hashes = model.metadata()["source_sha256"]
    expected = {f"quantization_sensitivity/{name}.py" for name in
                ["__init__", "data", "model", "quantization", "metrics", "runner", "analysis"]}
    assert hashes.keys() == expected
    assert set(runner.MEASUREMENT_SOURCES) == expected - {"quantization_sensitivity/analysis.py"}
    assert hashes == {name: runner.file_hash(REPOSITORY / name) for name in expected}


def test_analysis_rejects_calibration_from_old_source(tmp_path, monkeypatch):
    runner.save_json(tmp_path / "evaluation_validation.json", dict(status="verified"))
    runner.save_json(tmp_path / "steps" / "statistical_null_calibration.json", dict(
        completed=True, all_criteria_pass=True, trials_per_case=500, wild_replicates=999,
        code_sha256={"m1_analysis.py": "0" * 64}))
    monkeypatch.setattr(sys, "argv", ["analyze_results", "--root", str(tmp_path)])
    with pytest.raises(AssertionError, match="rerun statistical-method calibration"):
        analyze_results.main()


def test_shell_runner_requires_explicit_gpu_and_uses_same_default_results():
    script = REPOSITORY / "scripts" / "run_experiment.sh"
    assert f"OUT=${{EXPERIMENT_OUT:-{DEFAULT_RESULTS}}}" in script.read_text()
    env = {key: value for key, value in os.environ.items() if key != "EXPERIMENT_GPU"}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Set EXPERIMENT_GPU" in result.stderr


def test_layer_shell_requires_explicit_gpu_and_has_separate_modes(tmp_path):
    script = REPOSITORY / "scripts" / "run_layer_sensitivity.sh"
    env = {key: value for key, value in os.environ.items() if key != "LAYER_GPU"}
    env["LAYER_OUT"] = str(tmp_path)
    result = subprocess.run(["bash", str(script), "smoke"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Set LAYER_GPU" in result.stderr
    result = subprocess.run(["bash", str(script), "invalid"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "smoke|full|report" in result.stderr
