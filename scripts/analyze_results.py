"""Analyze request-dependent quantization sensitivity from actual validated runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.analysis import (add_content_metrics, bootstrap_basic, control_diagnostic, cube, holm_adjust,
                                              pair_signs, percentile_interval, resample_sources, reversal_summary,
                                              row_spearman, strategy_choices, wild_shape_test)
from quantization_sensitivity.data import SEED, SOURCES, save_json
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.runner import file_hash, read_json


def reversal_table(values, datasets, metric):
    rows = []
    pairs = list(zip(*np.triu_indices(8, 1)))
    for scope in ["all", *SOURCES]:
        mask = np.ones(len(values), dtype=bool) if scope == "all" else datasets == scope
        v = values[mask]
        v = v[np.isfinite(v).all(axis=(1, 2))]
        if not len(v):
            continue
        signs = pair_signs(v)
        for p, (i, j) in enumerate(pairs):
            r = dict(metric=metric, scope=scope, group_i=i, group_j=j, source_requests=len(v))
            for w, label in enumerate(["a", "b"]):
                for value, name in [(1, "i_more"), (-1, "j_more"), (0, "near_tie")]:
                    r[f"{label}_{name}"] = int((signs[:, w, p] == value).sum())
            r["replicated_i_more"] = int((signs[:, :, p] == 1).all(axis=1).sum())
            r["replicated_j_more"] = int((signs[:, :, p] == -1).all(axis=1).sum())
            r["replicated_reversal"] = r["replicated_i_more"] > 0 and r["replicated_j_more"] > 0
            rows.append(r)
    return rows


def segment_statistics(frame, datasets, iterations):
    rows = []
    for mode, columns in {"full": ("kl_first_half", "kl_second_half"),
                          "content": ("kl_content_first_half", "kl_content_second_half")}.items():
        _, first = cube(frame, columns[0])
        _, second = cube(frame, columns[1])
        eligible = np.isfinite(first).all(axis=(1, 2)) & np.isfinite(second).all(axis=(1, 2))
        for scope in ["all", *SOURCES]:
            mask = eligible & (np.ones(len(first), dtype=bool) if scope == "all" else datasets == scope)
            a, b, ds = first[mask], second[mask], datasets[mask]
            if not len(a):
                rows.append(dict(mode=mode, scope=scope, source_requests=0, worded_requests=0,
                                 statistic="not_estimable_no_multitoken_content", estimate=None,
                                 ci_low=None, ci_high=None, valid_bootstraps=0))
                continue
            rho = row_spearman(a, b)
            four_way = np.stack([a[:, 0], b[:, 0], a[:, 1], b[:, 1]], axis=1)
            point = dict(mean_spearman=float(np.nanmean(rho)), **reversal_summary(four_way))
            draws = {k: [] for k in point}
            rng = np.random.default_rng(SEED)
            for _ in range(iterations):
                idx = resample_sources(ds, rng)
                stat = dict(mean_spearman=float(np.nanmean(rho[idx])), **reversal_summary(four_way[idx]))
                for k, v in stat.items():
                    draws[k].append(v)
            for k, v in point.items():
                rows.append(dict(mode=mode, scope=scope, source_requests=len(a), worded_requests=2 * len(a),
                                 statistic=k, estimate=v, **percentile_interval(draws[k])))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_RESULTS)
    p.add_argument("--role", choices=["smoke", "evaluation"], default="evaluation")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out")
    args = p.parse_args()
    if args.bootstrap < 1 or (args.role == "evaluation" and args.bootstrap < 2000):
        p.error("formal evaluation requires >=2000 bootstrap/permutation replicates")
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    validation = read_json(root / f"{args.role}_validation.json")
    assert validation["status"] == "verified"
    if args.role == "evaluation":
        null_check = read_json(root / "steps" / "statistical_null_calibration.json")
        assert null_check["completed"] and null_check["all_criteria_pass"]
        assert null_check["trials_per_case"] >= 500 and null_check["wild_replicates"] >= 999
        assert null_check["code_sha256"].get("quantization_sensitivity/analysis.py") == file_hash(
            "quantization_sensitivity/analysis.py"), "analysis source changed: rerun statistical-method calibration"
    selection = read_json(root / "calibration_selection.json")
    fixed_kl, fixed_acc = selection["selected_group_kl"], selection["selected_group_accuracy"]
    paths = [root / "runs" / f"{args.role}_{w}" for w in ["a", "b"]]
    frame = pd.concat([pd.read_csv(path / "measurements.csv") for path in paths], ignore_index=True)
    baseline = pd.concat([pd.read_csv(path / "baselines.csv") for path in paths], ignore_index=True)
    assert set(frame.role) == {args.role} and set(baseline.role) == {args.role}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    frame = add_content_metrics(frame, tokenizer)
    for column in ["kl_content_mean", "kl_first_token"]:
        frame[column + "_rank"] = frame.groupby(["request_id", "wording"])[column].rank(ascending=False)
    frame.to_csv(out / "per_request_measurements.csv", index=False)
    ids, full = cube(frame, "kl_mean")
    _, correct = cube(frame, "correct")
    correct = correct.astype(float)
    source = frame.drop_duplicates("request_id").set_index("request_id").loc[ids]
    datasets, request_lengths = source.dataset.to_numpy(), source.request_tokens.to_numpy()
    _, inputs = cube(frame, "input_tokens")
    _, outputs = cube(frame, "kl_tokens")
    inputs, outputs = inputs[:, :, 0], outputs[:, :, 0]
    bf16 = baseline.pivot(index="request_id", columns="wording", values="correct").loc[ids, ["a", "b"]].to_numpy().astype(float)
    assert np.isfinite(full).all()
    stats_rows, controls, wild_controls, reversals, coverage = [], [], [], [], []
    for metric, column in [("full", "kl_mean"), ("content", "kl_content_mean"), ("first_token", "kl_first_token")]:
        _, kl = cube(frame, column)
        finite = np.isfinite(kl).all(axis=(1, 2))
        coverage.append(dict(metric=metric, source_requests=int(finite.sum()),
                             missing_request_ids=np.asarray(ids)[~finite].tolist()))
        reversals.extend(reversal_table(kl, datasets, metric))
        for scope in ["all", *SOURCES]:
            mask = finite & (np.ones(len(kl), dtype=bool) if scope == "all" else datasets == scope)
            if not mask.any():
                continue
            for wording, axis in [("paired", [0, 1]), ("a", [0]), ("b", [1])]:
                x, c, bf = kl[mask][:, axis], correct[mask][:, axis], bf16[mask][:, axis]
                result = bootstrap_basic(x, c, bf, datasets[mask], fixed_kl, fixed_acc, args.bootstrap)
                for name, value in result.items():
                    stats_rows.append(dict(metric=metric, scope=scope, wording=wording, statistic=name,
                                           source_requests=len(x), worded_requests=x.shape[0] * x.shape[1], **value))
            if metric in ["full", "content"]:
                for include_output in [False, True]:
                    result = control_diagnostic(kl[mask], datasets[mask], inputs[mask], request_lengths[mask],
                                               outputs[mask] if include_output else None, args.bootstrap)
                    controls.append(dict(metric=metric, scope=scope, controls="dataset+log_input" +
                        ("+log_output" if include_output else ""), source_requests=int(mask.sum()),
                        interpretation="exploratory approximate residual permutation; not a confirmatory p-value", **result))
                    if mask.sum() >= 16:
                        wild_result = wild_shape_test(kl[mask], datasets[mask], inputs[mask],
                                                     outputs[mask] if include_output else None,
                                                     iterations=args.bootstrap)
                        wild_controls.append(dict(metric=metric, scope=scope, controls="dataset_specific_cubic_log_input" +
                            ("+dataset_specific_log_output" if include_output else ""), source_requests=int(mask.sum()),
                            profile="row centered and standardized; removes common affine profile", **wild_result))
            print(f"Analyzed {args.role} {metric} {scope}: n={mask.sum()}", flush=True)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(out / "statistics.csv", index=False)
    control_frame = pd.DataFrame(controls)
    control_frame["permutation_p_holm"] = holm_adjust(control_frame.permutation_p.to_numpy())
    control_frame.to_csv(out / "controlled_interaction.csv", index=False)
    wild_frame = pd.DataFrame(wild_controls)
    wild_frame["wild_p_holm"] = holm_adjust(wild_frame.wild_p.to_numpy())
    wild_frame.to_csv(out / "controlled_shape_wild.csv", index=False)
    pd.DataFrame(reversals).to_csv(out / "ranking_reversals.csv", index=False)
    pd.DataFrame(reversal_table(correct, datasets, "correctness_crossover")).to_csv(out / "correctness_crossovers.csv", index=False)
    segments = segment_statistics(frame, datasets, args.bootstrap)
    segments.to_csv(out / "segment_stability.csv", index=False)

    # All actual answers, not a selection of success cases.
    columns = ["request_id", "dataset", "wording", "group", "gold", "answer", "parsed_answer", "correct",
               "parse_failure", "truncated", "generated_tokens", "generated_ids"]
    bf_rows = baseline.copy()
    bf_rows["group"] = -1
    answers = pd.concat([bf_rows[columns], frame[columns]], ignore_index=True)
    answers["configuration"] = answers.group.map(lambda g: "BF16" if g == -1 else f"g{g}_W4")
    answers.sort_values(["dataset", "request_id", "wording", "group"]).to_csv(out / "answers.csv", index=False)
    answers[(~answers.correct) | answers.parse_failure | answers.truncated].to_csv(out / "answer_failures.csv", index=False)
    lookup = frame.set_index(["request_id", "wording", "group"])
    baseline_lookup = baseline.set_index(["request_id", "wording"])
    selected_rows, request_rows = [], []
    choices = strategy_choices(full, correct, fixed_kl, fixed_acc)
    for i, request_id in enumerate(ids):
        rho = row_spearman(full[i, 0:1], full[i, 1:2])[0]
        for w, wording in enumerate(["a", "b"]):
            kls, cs = full[i, w], correct[i, w]
            ordered = np.sort(kls)
            gain = kls[fixed_kl] - kls.min()
            threshold = max(1e-5, .01 * (kls[fixed_kl] + kls.min()) / 2)
            base = baseline_lookup.loc[(request_id, wording)]
            record = dict(request_id=request_id, dataset=datasets[i], wording=wording,
                input_tokens=int(inputs[i, w]), request_tokens=int(request_lengths[i]),
                bf16_tokens=int(outputs[i, w]), bf16_correct=bool(bf16[i, w]), bf16_answer=base.answer,
                highest_group=int(kls.argmax()), lowest_group=int(kls.argmin()),
                KL_by_group=json.dumps(kls.tolist()), correct_by_group=json.dumps(cs.astype(bool).tolist()),
                wording_spearman=float(rho), best_second_gap=float(ordered[1] - ordered[0]),
                KL_oracle_gain_vs_calibration_fixed=float(gain),
                no_material_KL_gain_vs_calibration_fixed=bool(gain <= threshold),
                all_groups_same_correctness=bool(cs.max() == cs.min()),
                KL_oracle_task_loss_vs_calibration_fixed=bool(cs[kls.argmin()] < cs[fixed_kl]))
            request_rows.append(record)
            for name, choice in choices.items():
                group = int(choice[i, w])
                actual = lookup.loc[(request_id, wording, group)]
                selected_rows.append(dict(request_id=request_id, dataset=datasets[i], wording=wording,
                    strategy=name, selected_group=group, oracle="ORACLE" in name,
                    answer=actual.answer, parsed_answer=actual.parsed_answer, gold=actual.gold,
                    correct=bool(actual.correct), kl=float(actual.kl_mean),
                    truncated=bool(actual.truncated), parse_failure=bool(actual.parse_failure)))
    pd.DataFrame(selected_rows).to_csv(out / "oracle_comparisons.csv", index=False)
    requests = pd.DataFrame(request_rows)
    requests.to_csv(out / "request_summary.csv", index=False)
    requests[requests.no_material_KL_gain_vs_calibration_fixed | requests.all_groups_same_correctness |
             requests.KL_oracle_task_loss_vs_calibration_fixed].to_csv(out / "no_benefit_or_loss_cases.csv", index=False)
    save_json(out / "analysis_metadata.json", dict(role=args.role, source_requests=len(ids),
        worded_requests=len(baseline), W4_rows=len(frame), actual_answer_rows=len(answers),
        token_KL_values=sum(len(json.loads(v)) for v in frame.token_kl), coverage=coverage,
        bootstrap_replicates=args.bootstrap, bootstrap_unit="source request, dataset-stratified; wordings/groups paired",
        seed=SEED, calibration_conditioned=True, fixed_groups=dict(KL=fixed_kl, accuracy=fixed_acc),
        permutation_test_family_size=len(control_frame), permutation_interpretation="exploratory approximate only",
        wild_test_family_size=len(wild_frame), multiple_testing="Holm separately across all wild tests and exploratory permutations",
        wild_test_interpretation="approximate HC2 cluster score, nuisance projected; checked in separate synthetic null calibration",
        content_definition="whole non-special non-whitespace tokens preceding next-question delimiter", 
        input_sha256={str(path.relative_to(root)): file_hash(path) for parent in paths
            for path in [parent / "baselines.csv", parent / "measurements.csv"]},
        calibration_selection_sha256=file_hash(root / "calibration_selection.json"),
        code_sha256={f: file_hash(f) for f in ["quantization_sensitivity/analysis.py", "scripts/analyze_results.py"]},
        analyzed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"COMPLETE analysis role={args.role} source_n={len(ids)} W4_rows={len(frame)}", flush=True)


if __name__ == "__main__":
    main()
