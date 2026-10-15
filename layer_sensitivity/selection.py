"""Freeze one train-only 8-layer precision budget and three seeded random controls."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantization_sensitivity.data import save_json
from quantization_sensitivity.runner import file_hash, read_json
from . import DEFAULT_RESULTS
from .runner import source_hashes
from .validate import check_coverage

BUDGET = 8
RANDOM_SEED = 20261016


def top_layers(values, budget=BUDGET):
    values = np.asarray(values, dtype=float)
    if values.shape != (32,) or not np.isfinite(values).all() or not 0 < budget < 32:
        raise ValueError("32 finite sensitivities and a budget in 1..31 required")
    return sorted(np.argsort(-values, kind="stable")[:budget].tolist())


def random_sets():
    return {f"random_{i}": sorted(np.random.default_rng(RANDOM_SEED + i).choice(32, BUDGET, replace=False).tolist())
            for i in range(3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    root = Path(args.root)
    validation = read_json(root / "calibration_validation.json")
    measurements = root / "runs/calibration/measurements.csv"
    assert validation["status"] == "verified" and validation["source_sha256"] == source_hashes()
    assert validation["measurements_sha256"] == file_hash(measurements)
    frame = pd.read_csv(measurements)
    check_coverage(frame, sorted(frame.request_id.unique()))
    assert (frame.role == "calibration").all() and frame.request_id.str.contains(":train:").all()
    means = frame.groupby("layer").kl_mean.mean().reindex(range(32)).to_numpy()
    selection = dict(budget=BUDGET, source_role="calibration", source_request_ids=sorted(frame.request_id.unique()),
        requests_sha256=file_hash(root / "requests.jsonl"), source_sha256=source_hashes(),
        calibration_measurements_sha256=file_hash(measurements), mean_kl_by_layer=means.tolist(),
        fixed_high_precision_layers=top_layers(means), random_high_precision_layers=random_sets(),
        random_seeds=[RANDOM_SEED + i for i in range(3)], tie_break="smaller layer index",
        oracle_label="same-request KL information; not a deployable router or accuracy upper bound",
        selection_source_sha256=file_hash(__file__))
    path = root / "precision_allocation_selection.json"
    if path.exists():
        previous = read_json(path)
        assert {k: v for k, v in previous.items() if k != "frozen_utc"} == selection, "frozen selection changed"
        selection = previous
    else:
        assert not (root / "runs/evaluation/run_config.json").exists(), "freeze before evaluation"
        selection["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(path, selection)
    print(selection, flush=True)


if __name__ == "__main__":
    main()
