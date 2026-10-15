"""Post-review influence diagnostic; never modifies/excludes primary observations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from m1_analysis import control_design, controlled_association, cube
from m1_data import save_json
from m1_run import file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/m1")
    p.add_argument("--threshold", type=float, default=.95)
    args = p.parse_args()
    root = Path(args.root)
    path = root / "analysis" / "per_request_measurements.csv"
    frame = pd.read_csv(path)
    ids, _ = cube(frame, "kl_mean")
    ids = np.asarray(ids)
    source = frame.drop_duplicates("request_id").set_index("request_id").loc[ids]
    datasets = source.dataset.to_numpy()
    _, inputs = cube(frame, "input_tokens")
    _, outputs = cube(frame, "kl_tokens")
    inputs, outputs = inputs[:, :, 0], outputs[:, :, 0]
    rows = []
    for metric, column in [("full", "kl_mean"), ("content", "kl_content_mean")]:
        _, values = cube(frame, column)
        for scope in ["all", *np.unique(datasets)]:
            mask = np.ones(len(ids), dtype=bool) if scope == "all" else datasets == scope
            d, inp, out, val, names = datasets[mask], inputs[mask], outputs[mask], values[mask], ids[mask]
            for adjust_output in [False, True]:
                lengths = out if adjust_output else None
                leverages = []
                for w in range(2):
                    x = control_design(d, inp[:, w], out[:, w] if adjust_output else None, nonlinear=True)
                    leverages.append(np.einsum("ij,ji->i", x, np.linalg.pinv(x)))
                high = np.maximum(*leverages) >= args.threshold
                before, _ = controlled_association(val, d, inp, lengths, normalize="shape", nonlinear=True)
                after, _ = controlled_association(val[~high], d[~high], inp[~high],
                    lengths[~high] if lengths is not None else None, normalize="shape", nonlinear=True)
                rows.append(dict(metric=metric, scope=scope, output_adjusted=adjust_output,
                    primary_sources=len(d), diagnostic_sources=int((~high).sum()),
                    high_leverage_source_ids=names[high].tolist(), max_leverage_a=float(max(leverages[0])),
                    max_leverage_b=float(max(leverages[1])), full_sample_correlation=float(before),
                    diagnostic_correlation=float(after), change=float(after-before)))
    save_json(root / "steps" / "leverage_influence.json", dict(
        purpose="POST_REVIEW_EXPLORATORY_INFLUENCE_DIAGNOSTIC_NOT_A_NEW_PRIMARY_ANALYSIS",
        rule="one pass: omit from diagnostic only if either wording's design leverage >= threshold; refit nuisances",
        threshold=args.threshold, primary_observations_removed=0,
        caveat="No new inferential p-values/CIs; high-leverage finite-sample calibration remains uncertain",
        input_sha256=file_hash(path), code_sha256=file_hash(__file__), results=rows))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
