"""Freeze metric-matched fixed groups using calibration only, before formal inference."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from m1_data import save_json
from m1_run import file_hash, read_json


def choose_fixed(kl, correct):
    if kl.shape != correct.shape or kl.ndim != 2 or not len(kl) or not np.isfinite(kl).all():
        raise ValueError("finite matching request-by-group arrays required")
    return int(kl.mean(axis=0).argmin()), int(correct.mean(axis=0).argmax())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/m1")
    args = p.parse_args()
    root = Path(args.root)
    assert read_json(root / "calibration_validation.json")["status"] == "verified"
    paths = [root / "runs" / f"calibration_{w}" / "measurements.csv" for w in ["a", "b"]]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    assert set(frame.role) == {"calibration"}
    assert set(frame.wording) == {"a", "b"} and set(frame.group) == set(range(8))
    kl = frame.pivot(index=["request_id", "wording"], columns="group", values="kl_mean")
    correctness = frame.pivot(index=["request_id", "wording"], columns="group", values="correct")
    assert len(kl) == 48 and frame.request_id.nunique() == 24
    kl_group, acc_group = choose_fixed(kl.to_numpy(), correctness.to_numpy())
    result = dict(selected_group_kl=kl_group, selected_group_accuracy=acc_group,
        calibration_source_requests=24, calibration_worded_requests=48,
        group_mean_kl=kl.mean().to_dict(), group_accuracy=correctness.mean().to_dict(),
        tie_break="smallest group index", evaluation_data_used=False,
        input_sha256={str(path.relative_to(root)): file_hash(path) for path in paths},
        selected_utc=datetime.now(timezone.utc).isoformat())
    path = root / "calibration_selection.json"
    if path.exists():
        old = read_json(path)
        for k in ["selected_group_kl", "selected_group_accuracy", "input_sha256"]:
            assert old[k] == result[k], "cannot change a frozen calibration decision"
        print("Already frozen:", old)
    else:
        save_json(path, result)
        print("Frozen calibration-only choices:", result)


if __name__ == "__main__":
    main()
