"""SYNTHETIC statistical-method validation; NEVER Llama experimental evidence."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest

from m1_analysis import wild_shape_test
from m1_data import save_json

SIMULATION_SEED = 314159


def synthetic_profiles(rng, n, scenario):
    datasets = np.repeat(["one", "two", "three"], n // 3) if n == 96 else np.repeat("one", n)
    length, output = rng.uniform(-1, 1, (2, n))
    input_lengths = np.column_stack([np.exp(6 + .3 * length), np.exp(6 + .3 * length) + 12])
    output_lengths = np.column_stack([np.exp(3 + .4 * output), np.exp(3 + .4 * output) + 1])
    base = 5 + np.arange(8) * .3
    v1, v2, v3 = np.cos(np.arange(8)), np.sin(np.arange(8)), np.cos(np.arange(8) * 2)
    mean = np.tile(base, (n, 1))
    for i, dataset in enumerate(np.unique(datasets)):
        mean[datasets == dataset] += .1 * i * v2
    sigma = np.full(n, .15)
    if scenario == "heteroskedastic":
        sigma = .08 + .25 * (length + 1) / 2
    elif scenario == "nonlinear_input":
        mean += (.25 * length)[:, None] * v1 + (.15 * length ** 2)[:, None] * v2 + (.1 * length ** 3)[:, None] * v3
    elif scenario == "input_output":
        mean += (.2 * length)[:, None] * v1 + (.2 * output)[:, None] * v2 + (.15 * length ** 2)[:, None] * v3
    noise_a, noise_b = rng.normal(size=(2, n, 8))
    if scenario == "shared_profile_positive_control":
        noise_b = .8 * noise_a + .6 * noise_b
    a, b = mean + sigma[:, None] * noise_a, mean + sigma[:, None] * noise_b
    # Shared arbitrary request-level affine nuisance must not count as group-shape interaction.
    offset = rng.lognormal(1, .7, (n, 1))
    scale = rng.lognormal(0, .7, (n, 1))
    values = np.stack([offset + scale * a, offset * 2 + scale * 1.5 * b], axis=1)
    assert (values > 0).all()
    return values, datasets, input_lengths, output_lengths if scenario == "input_output" else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--wild-replicates", type=int, default=999)
    p.add_argument("--out", default="results/m1/steps/statistical_null_calibration.json")
    args = p.parse_args()
    started = time.monotonic()
    scenarios = ["homoskedastic", "heteroskedastic", "nonlinear_input", "input_output", "shared_profile_positive_control"]
    results = []
    print("SYNTHETIC METHOD VALIDATION ONLY — no model inference or experiment values", flush=True)
    for n in [32, 96]:
        for scenario in scenarios:
            case = len(results)
            rng = np.random.default_rng(SIMULATION_SEED + case)
            rejected, missing, p_values = 0, 0, []
            for trial in range(args.trials):
                values, datasets, lengths, outputs = synthetic_profiles(rng, n, scenario)
                test = wild_shape_test(values, datasets, lengths, outputs, iterations=args.wild_replicates,
                                       bootstrap_iterations=0, seed=SIMULATION_SEED + case * 100000 + trial)
                value = test["wild_p"]
                missing += value is None
                if value is not None:
                    rejected += value <= .05
                    p_values.append(value)
            interval = binomtest(rejected, args.trials).proportion_ci(method="wilson")
            rate = rejected / args.trials
            positive = scenario == "shared_profile_positive_control"
            result = dict(n=n, scenario=scenario, trials=args.trials, rejected=rejected,
                rejection_rate=rate, rate_ci_low=float(interval.low), rate_ci_high=float(interval.high),
                not_estimable=missing, mean_p=float(np.mean(p_values)) if p_values else None,
                criterion="power >= .80" if positive else "false-positive rate <= .08",
                criterion_pass=rate >= .80 if positive else rate <= .08)
            results.append(result)
            print(result, flush=True)
            save_json(args.out, dict(kind="SYNTHETIC_METHOD_VALIDATION_NOT_MODEL_RESULTS", seed=SIMULATION_SEED,
                trials_per_case=args.trials, wild_replicates=args.wild_replicates, results=results,
                completed=len(results) == 10, all_criteria_pass=all(r["criterion_pass"] for r in results),
                code_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                             for f in ["m1_analysis.py", "scripts/calibrate_statistics.py"]},
                elapsed_seconds=time.monotonic() - started))
    assert all(r["criterion_pass"] for r in results), "null calibration/power criterion failed; investigate, do not claim significance"


if __name__ == "__main__":
    main()
