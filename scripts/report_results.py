"""Render a bounded quantization-sensitivity report from completed real measurements."""
import argparse
from pathlib import Path
import pandas as pd

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.data import SEED, SOURCES, save_json
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.runner import file_hash, read_json


def table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_RESULTS)
    args = p.parse_args()
    root, analysis = Path(args.root), Path(args.root) / "analysis"
    meta = read_json(analysis / "analysis_metadata.json")
    assert meta["role"] == "evaluation" and meta["source_requests"] >= 96 and meta["bootstrap_replicates"] >= 2000
    assert read_json(root / "evaluation_validation.json")["status"] == "verified"
    stats = pd.read_csv(analysis / "statistics.csv")
    wild = pd.read_csv(analysis / "controlled_shape_wild.csv")
    pairs = pd.read_csv(analysis / "ranking_reversals.csv")
    crossovers = pd.read_csv(analysis / "correctness_crossovers.csv")
    segments = pd.read_csv(analysis / "segment_stability.csv")
    requests = pd.read_csv(analysis / "request_summary.csv")
    answers = pd.read_csv(analysis / "answers.csv", keep_default_na=False)
    oracle = pd.read_csv(analysis / "oracle_comparisons.csv")
    selection = read_json(root / "calibration_selection.json")
    provenance = read_json(root / "model_provenance.json")
    null_check = read_json(root / "steps" / "statistical_null_calibration.json")
    runtime = {w: read_json(sorted((root / "runs" / f"evaluation_{w}").glob("attempt_*.json"))[-1]) for w in ["a", "b"]}
    diagnostic = read_json(root / "steps" / "single_sensitivity" / "diagnostics.json")
    gate_kl = pd.read_csv(root / "steps" / "single_sensitivity" / "measurements.csv").kl_mean.iloc[0]
    null_rates = [r["rejection_rate"] for r in null_check["results"] if r["scenario"] != "shared_profile_positive_control"]
    positive_results = [f"{r['rejected']}/{r['trials']}" for r in null_check["results"] if r["scenario"] == "shared_profile_positive_control"]
    cross_gpu = all((root / "steps" / f"cross_gpu{g}" / "cross_device_comparison.json").exists() for g in [1, 2])
    assert meta["source_requests"] == 96 and meta["worded_requests"] == 192 and len(answers) == meta["actual_answer_rows"] == 1728
    assert requests.groupby("dataset").request_id.nunique().to_dict() == {d: 32 for d in SOURCES}
    assert answers.correct.dtype == bool
    def get(name, metric="full", scope="all", wording="paired"):
        row = stats[(stats.statistic == name) & (stats.metric == metric) & (stats.scope == scope) & (stats.wording == wording)]
        assert len(row) == 1, (name, metric, scope, wording)
        return row.iloc[0]
    def interval(name, metric="full", scope="all", factor=1, digits=6, suffix=""):
        r = get(name, metric, scope)
        return f"{r.estimate*factor:.{digits}f}{suffix} [{r.ci_low*factor:.{digits}f}, {r.ci_high*factor:.{digits}f}]"
    strict = wild[(wild.scope == "all") & wild.controls.str.contains("output")]
    quality = get("label_oracle_accuracy_gain_vs_accuracy_best_fixed")
    transfer = get("cross_wording_KL_relative_gain_vs_KL_best_fixed")
    transfer_note = ("CI 包含零；穩定的整體排名不代表 argmin 可穩定轉移。" if transfer.ci_low <= 0 <= transfer.ci_high
                     else "這仍使用另一 wording 的 BF16/evaluation 訊號，不是部署收益。")
    dataset_gains = {d: get("label_oracle_accuracy_gain_vs_accuracy_best_fixed", scope=d).estimate for d in SOURCES}
    zero_gain = [d for d, value in dataset_gains.items() if abs(value) < 1e-12]
    main_gain_dataset = max(dataset_gains, key=dataset_gains.get)
    correctness_crossovers = int(crossovers[crossovers.scope == "all"].replicated_reversal.sum())
    fixed = oracle[oracle.strategy == "evaluation_accuracy_best_fixed_ORACLE"].set_index(["request_id", "wording"])
    label = oracle[oracle.strategy == "per_request_label_ORACLE"].set_index(["request_id", "wording"])
    gains = label.correct.astype(int) - fixed.correct.astype(int)
    untruncated_gains = int((gains * ~(fixed.truncated | label.truncated)).sum())
    supported = bool(len(strict) == 2 and set(strict.metric) == {"full", "content"}
                     and (strict.ci_low > 0).all() and (strict.wild_p_holm < .05).all()
                     and quality.ci_low > 0 and correctness_crossovers > 0 and untruncated_gains > 0)
    lines = ["# Llama 3.1 8B request-dependent layer quantization sensitivity", "",
        "## 結論與範圍", "",
        ("**證據支持一個有限的量化敏感度結論：在本模型、RTN W4、四層一組與固定 few-shot protocol 下，不同 request 對不同 group 的相對敏感度與實際答案效用存在可重現差異。**"
         if supported else "**目前證據不足以充分支持 request-dependent 量化敏感度的核心命題；不能只憑程式或圖表完成就宣告成功。**"),
        "這**不是**可部署 router、跨 wording 最佳配置可直接轉移、完整模型 W4 品質、或量化 runtime 效能的證據。", "",
        f"- 正式資料：**{meta['source_requests']} 個獨立 source request**（每 dataset 32），兩種 wording 共 {meta['worded_requests']} 個 prompt；**{meta['W4_rows']} 筆 W4 配置量測、{meta['actual_answer_rows']} 筆實際答案、{meta['token_KL_values']:,} 個 token-level KL 值**。未依效果或正確率剔除 request。",
        f"- 全 continuation 與非空白 content-only 分析，各有 {int(pairs[(pairs.metric=='full')&(pairs.scope=='all')].replicated_reversal.sum())}/28、{int(pairs[(pairs.metric=='content')&(pairs.scope=='all')].replicated_reversal.sum())}/28 個 group pair 出現**兩種 wording 都重現**的跨 request 排序反轉；實際 correctness 有 **{correctness_crossovers}/28** 個 pair 的雙向 crossover 也在兩種 wording 重現。",
        f"- 相同 precision budget，label oracle 相對 accuracy-best evaluation-fixed oracle 的改善：**{interval('label_oracle_accuracy_gain_vs_accuracy_best_fixed',factor=100,digits=2,suffix=' pp')}**。這是使用 evaluation labels 的 **oracle upper bound**，不能當作部署收益。",
        f"- 跨 wording 轉移檢查：用另一種 wording 的 KL 最佳組來選配置，full KL 相對 best fixed 的改善為 **{interval('cross_wording_KL_relative_gain_vs_KL_best_fixed',factor=100,digits=2,suffix='%')}**。**{transfer_note}**",  "",
        "## 實驗 protocol、來源與硬體", "",
        f"- 模型：`{MODEL_ID}@{REVISION}`，**base 而非 Instruct**。四個真實 safetensors shards 與 tokenizer/config 的官方 digest 全部匹配：[model_provenance.json](model_provenance.json)。",
        "- Base tokenizer 沒有官方 chat template；使用明確保存的三個 train-only demonstration 的 plain-text Q/A template：[chat_template.jinja](chat_template.jinja)。每個 precision configuration 使用相同 tokenizer、template、prompt IDs。兩種 wording 僅改預定的指令說法，並非題目本體的任意 paraphrase。",
        "- Reference/線性層 input-output/BF16 decode KV cache 均為 BF16；保留標準 Llama 內部 norm/softmax 行為。KL 用相同 full-sequence teacher-forcing、`use_cache=False` 路徑；實際答案用新建 BF16 KV cache 的 greedy generation。",
        "- 32 層分 g0=0–3、g1=4–7、…、g7=28–31。每次只有一組的 attention/MLP 七個 linear weight/層量化；其他組、norm、embedding、LM head 保留 BF16。每組 28 個矩陣、**872,415,232 個量化 weight**，所以所有八種候選配置的 precision budget 相同。BF16 reference 不在 budget-matched oracle 的 action set。",
        "- W4：per-output-row、沿 input dimension 每 128 個 weight 一個 block；FP32 absmax/7 scale，ties-to-even RTN，signed [-7,7]，零 block scale=1；不做 activation-aware calibration、clipping tuning 或 per-request quantizer tuning。實際產生 packed 4-bit codes 並 round-trip decode，將其反量化為 BF16 執行模型（Q/DQ）。**這量測真實量化後權重的誤差，不是 packed-integer kernel 的速度/記憶體測試。**",
        f"- Seed **{SEED}**；GSM8K/ARC-Challenge 使用 test，BoolQ 使用 validation。每種資料 train 中抽 3 demos + 8 獨立 calibration；held-out split 中先抽 8 smoke，再抽 32 disjoint evaluation。問題 hash 去重，完整 IDs/文字/gold： [requests.jsonl](requests.jsonl)、[data_manifest.json](data_manifest.json)。RTN 本身不需 calibration；calibration 僅用來選固定組。",
        f"- 正式 inference 前（{selection['selected_utc']}）封存：KL-fixed **g{selection['selected_group_kl']}**、accuracy-fixed **g{selection['selected_group_accuracy']}**；同分取最小 group index。證據：[calibration_selection.json](calibration_selection.json)。",
        f"- Torch {runtime['a']['versions']['torch']} / Transformers {runtime['a']['versions']['transformers']}；{runtime['a']['gpu']}，CUDA runtime {runtime['a']['cuda_runtime']}。正式 A/B 的最後 attempt 分別用 physical GPU `{runtime['a']['cuda_visible_devices']}` / `{runtime['b']['cuda_visible_devices']}`。逐 attempt 的 driver、硬體、套件、參數與 code hashes 在 `runs/*/attempt_*.json`、`run_config.json`。",
        "- Dataset revisions：" + "; ".join(f"`{repo}@{revision}` ({folder}/{split})" for repo,revision,folder,split in SOURCES.values()),
        "- Greedy、不 sampling；GSM8K 最多 256 new tokens，ARC/BoolQ 16。GSM8K 嚴格解析 `#### <number>`（Decimal 比較）；ARC 解析有效選項標籤；BoolQ 解析 yes/no。解析失敗算錯，truncation 保留，不排除。", "",
        "## 量測正確性與統計方法", "",
        "- KL 方向為 **KL(p_BF16 || p_W4)**，全 vocabulary；FP32 log-softmax/product、FP64 summation。每個 group 在同一 BF16 continuation 的同一 history 上 teacher-force；token t 對應 P+t−1 的 causal logit。W4 的實際答案另外自由 greedy 生成，不能把不同 history 的 logits 當成 KL。",
        "- 原始 primary trace 包含回傳的 EOS/next-question delimiter。次要 content-only 分析一致排除 special、standalone whitespace 與 `Q:` delimiter；不依 label 或效果選 token。Whole token 跨 delimiter 者排除；無 delimiter 的混合文字/尾端空白 token 保留。首個 generated token 可能只是 ARC 數字前的空白，因此 first-token 是格式敏感診斷，不一律等同答案 token。",
        "- 獨立 BF16/W4 same-path replay、逐組 restore logits 及完整選定 weight restore 均 exact；單組完整 parameter digest 證明只改 28 個指定矩陣。這是已檢查路徑的 floor，不是所有硬體/演算法的通用 noise 保證。" + ("原始額外 gate 中 GPU 1/2 也 bitwise 重現 GPU 5；見 `steps/cross_gpu{1,2}/cross_device_comparison.json`。" if cross_gpu else "本次輸出目錄未包含額外跨 GPU gate。"),
        f"- Cached replay 與 full teacher forcing 在 gate request 的 KL 為 **{diagnostic['cached_vs_full_kl']['kl_mean']:.9f} nats/token**；full-route g0 W4 KL 為 {gate_kl:.8f}。所有 primary 比較都使用同一路徑。尤其 oracle gain 可以小於這個跨路徑 discrepancy，**不能據此保證 cached runtime 的最優 KL 配置**；實際答案/correctness crossover 是獨立的 cached-generation 證據。",
        f"- **{meta['bootstrap_replicates']} 次、95% percentile bootstrap**；在 dataset 內重抽 source requests，兩種 wording、全部 groups 共同保留。每次重選 evaluation-best fixed；品質差異使用配對差值，而非看兩個獨立 CI 是否重疊。CIs 條件於固定 demos/calibration，不涵蓋重新抽 calibration 的不確定性。",
        "- 排序反轉忽略差異 <= max(1e−5 nats/token, pair mean 的 1%, 5×已測 replay discrepancy)。報告全部 28 pairs，不對個別 pair 做挑選式 significance test。",
        "- 主 robustness 控制對每個 request 的 group profile 做 row centering/standardization，去除固定 profile 的任意正 affine magnitude；使用 dataset-specific cubic log input length，並另加 dataset-specific log BF16 output length。報告 source-bootstrap residual correlation CI 與 nuisance-projected HC2 source-cluster wild-score 的 **approximate** p-value；16 個 test 做 Holm adjustment。這不是精確 randomization/因果識別；指定的長度模型不能排除任意非線性長度效應或其他未知 confound。",
        "- 原本 coarse-bin residual permutation 沒有嚴格 exchangeability 保證，僅保留為 exploratory diagnostic：[controlled_interaction.csv](analysis/controlled_interaction.csv)，不拿它作確認性證據。修正經獨立 read-only review。",
        f"- **方法驗證的合成資料不是模型實驗結果**：八個預定 null scenario（n=32/96，每個 {null_check['trials_per_case']} trials、{null_check['wild_replicates']} multipliers）拒絕率 {min(null_rates):.3f}–{max(null_rates):.3f}，兩個 strong positive control 為 {'、'.join(positive_results)}；見 [statistical_null_calibration.json](steps/statistical_null_calibration.json)。此結果支持但不證明普遍校準；部分 binomial CI upper bound >0.08，亦未證明較小 Holm 門檻的精確 error rate。",  "",
        "## 完整 evaluation set 的敏感度與穩定性", "",
        "以下列出全體計數；完整逐 request/group 數據及每 token KL：[per_request_measurements.csv](analysis/per_request_measurements.csv)。", ""]
    for name, column in [("最高敏感組（full KL）", "highest_group"), ("最低敏感組（full KL）", "lowest_group")]:
        counts = pd.crosstab([requests.dataset, requests.wording], requests[column]).reindex(columns=range(8), fill_value=0)
        lines += ["### " + name, "", table(["dataset", "wording", *[f"g{i}" for i in range(8)]],
            [[d,w,*counts.loc[(d,w)].tolist()] for d in SOURCES for w in ["a","b"]]), ""]
    lines += [f"全 ranking 的跨 wording Spearman：**{interval('wording_mean_spearman',digits=4)}**；最高組 agreement {interval('wording_top_group_agreement',factor=100,digits=2,suffix='%')}，最低組 agreement {interval('wording_bottom_group_agreement',factor=100,digits=2,suffix='%')}。", "",
        table(["dataset", "full replicated pair reversals /28", "content replicated pair reversals /28"],
              [[d, int(pairs[(pairs.scope==d)&(pairs.metric=='full')].replicated_reversal.sum()),
                int(pairs[(pairs.scope==d)&(pairs.metric=='content')].replicated_reversal.sum())] for d in ["all",*SOURCES]]), "",
        "全部方向計數與 near ties：[ranking_reversals.csv](analysis/ranking_reversals.csv)；actual correctness 的全部 pair 表：[correctness_crossovers.csv](analysis/correctness_crossovers.csv)。這些 pair 數不是獨立樣本數。", "",
        "### 控制 dataset 與長度後", "",
        table(["metric", "scope", "shape residual correlation [95% CI]", "approx. wild p (Holm)"],
          [[r.metric,r.scope,f"{r.shape_residual_correlation:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]",f"{r.wild_p_holm:.6f}"]
           for r in wild[wild.controls.str.contains("output")].itertuples()]), "",
        f"表中是同時控制 input/output length 的版本；input-only 版本及所有 leverage/CI：[controlled_shape_wild.csv](analysis/controlled_shape_wild.csv)。原始 p 的 Monte Carlo 最小可解析值是 1/{meta['bootstrap_replicates']+1}；不得寫成 p=0。",  "",
        "### 評估片段", "",
        table(["片段", "scope", "有效 source / worded requests", "半段 ranking Spearman [95% CI]", "四路 replicated reversals /28"],
          [[mode,scope,f"{int(s.source_requests)} / {int(s.worded_requests)}",
            f"{s.estimate:.4f} [{s.ci_low:.4f}, {s.ci_high:.4f}]",
            int(segments[(segments['mode']==mode)&(segments.scope==scope)&(segments.statistic=='replicated_reversal_pairs')].estimate.iloc[0])]
           for mode in ["full","content"] for scope in ["all",*SOURCES]
           for s in segments[(segments['mode']==mode)&(segments.scope==scope)&(segments.statistic=='mean_spearman')].itertuples()]), "",
        "四路表示兩種 wording × first/second half 的同方向跨 request 反轉。**ARC/BoolQ 只有單一非空白答案 content token，無法估計 content 半段穩定性**；不得把 delimiter 的穩定當成語義答案穩定。Content-only 半段有效 source 僅 GSM8K 32（64 worded）。完整 denominators 包含不可估計項：[segment_stability.csv](analysis/segment_stability.csv)。", "",
        "## 相同 precision budget 與實際答案品質", ""]
    influence_path = root / "steps" / "leverage_influence.json"
    if influence_path.exists():
        influence = read_json(influence_path)
        assert influence["input_sha256"] == file_hash(analysis / "per_request_measurements.csv")
        case = next(r for r in influence["results"] if r["metric"] == "content" and r["scope"] == "all" and r["output_adjusted"])
        lines[-3:-3] = ["", f"**高 leverage 限制與事後影響度診斷：**原始 nuisance fits 的最大 leverage 為 {wild.max_leverage.max():.9f}（約 1；見完整 CSV），因此小樣本 wild inference 特別需要保留限制。獨立審查後，事後影響度檢查用單次規則『任一 wording leverage ≥ {influence['threshold']}』只在診斷副本中排除 {len(case['high_leverage_source_ids'])} 個 sources、重新擬合；content/output-adjusted correlation 從 {case['full_sample_correlation']:.6f} 變為 {case['diagnostic_correlation']:.6f}（診斷 n={case['diagnostic_sources']}）。主分析仍保留全部 {case['primary_sources']} sources，所有原 CI/p 不改。這只檢查點估計影響度，**不是**高-leverage p-value 校準的證明或新的確認性分析。所有模型/排除 ID：[leverage_influence.json](steps/leverage_influence.json)。", ""]
    strategies = [("BF16 reference (非同 budget)", "BF16", "BF16_accuracy"),
        (f"Calibration KL fixed g{selection['selected_group_kl']}","calibration_KL_fixed","calibration_KL_fixed_accuracy"),
        (f"Calibration accuracy fixed g{selection['selected_group_accuracy']}","calibration_accuracy_fixed","calibration_accuracy_fixed_accuracy"),
        ("Evaluation KL-best fixed — ORACLE","evaluation_KL_best_fixed_ORACLE","evaluation_KL_best_fixed_ORACLE_accuracy"),
        ("Evaluation accuracy-best fixed — ORACLE","evaluation_accuracy_best_fixed_ORACLE","evaluation_accuracy_best_fixed_ORACLE_accuracy"),
        ("Per-request KL — BF16-reference ORACLE","per_request_KL_ORACLE","per_request_KL_ORACLE_accuracy"),
        ("Per-request correctness — LABEL ORACLE","per_request_label_ORACLE","per_request_label_ORACLE_accuracy"),
        ("Other-wording KL — ORACLE diagnostic","cross_wording_KL_ORACLE_diagnostic","cross_wording_KL_ORACLE_diagnostic_accuracy")]
    lines += [table(["選擇方式", "實際 correct /192", "accuracy % [95% CI]", "full KL nats/token [95% CI]"],
        [[label_text,int(round(get(stat).estimate*meta['worded_requests'])),interval(stat,factor=100,digits=2),
          "reference" if strategy=="BF16" else interval(strategy+"_KL")]
         for label_text,strategy,stat in strategies]), "",
        "所有 oracle 都要事先看到 evaluation 的 BF16 distributions 或 labels/全部配置結果；**沒有實作或證明可部署 request router**。各 dataset 表內的 evaluation-best fixed 另在該 dataset 範圍取最佳，因此比單一全域固定組更有利。", "",
        table(["dataset", "BF16 % [95% CI]", "accuracy-best fixed oracle %", "label oracle %", "paired oracle gain pp [95% CI]"],
          [[d,interval('BF16_accuracy',scope=d,factor=100,digits=2),
            interval('evaluation_accuracy_best_fixed_ORACLE_accuracy',scope=d,factor=100,digits=2),
            interval('per_request_label_ORACLE_accuracy',scope=d,factor=100,digits=2),
            interval('label_oracle_accuracy_gain_vs_accuracy_best_fixed',scope=d,factor=100,digits=2)] for d in SOURCES]), "",
        table(["配對比較", "full [95% CI]", "content-only [95% CI]"], [
            ["KL oracle vs KL-best fixed: absolute nats/token",interval('KL_oracle_gain_vs_KL_best_fixed'),interval('KL_oracle_gain_vs_KL_best_fixed',metric='content')],
            ["KL oracle vs KL-best fixed: relative reduction %",interval('KL_oracle_relative_gain_vs_KL_best_fixed',factor=100,digits=2),interval('KL_oracle_relative_gain_vs_KL_best_fixed',metric='content',factor=100,digits=2)],
            ["Other-wording KL selection: relative reduction %",interval('cross_wording_KL_relative_gain_vs_KL_best_fixed',factor=100,digits=2),interval('cross_wording_KL_relative_gain_vs_KL_best_fixed',metric='content',factor=100,digits=2)],
            ["KL oracle accuracy vs BF16: pp (非同 budget)",interval('per_request_KL_ORACLE_accuracy_delta_vs_BF16',factor=100,digits=2),interval('per_request_KL_ORACLE_accuracy_delta_vs_BF16',metric='content',factor=100,digits=2)] ]), "",
        f"Label oracle 對最佳固定 accuracy 的 {int(gains.sum())} 個額外 worded correct 中，**{untruncated_gains} 個比較的兩個配置都未 truncation**；{int(gains.sum())-untruncated_gains} 個涉及 truncation。這是全樣本的描述性分解，不是排除截斷案例後重新宣稱效果。KL oracle 的代數最小值改善本身不足以證明命題；此處同時檢查 replicated reversals、controlled shape 和真實答案 crossover。", "",
        "所有 1,728 個實際答案、token IDs、gold、correctness：[answers.csv](analysis/answers.csv)；每個 oracle 的實際選擇/答案：[oracle_comparisons.csv](analysis/oracle_comparisons.csv)。", "",
        "## 失敗、無改善與限制（不移除）", "",
        f"- **{int(requests.all_groups_same_correctness.sum())}/{len(requests)}** worded requests 在八種 W4 下 correctness 全相同；**{int(requests.no_material_KL_gain_vs_calibration_fixed.sum())}/{len(requests)}** 對 calibration KL-fixed 沒有超過 near-tie 門檻的 KL oracle 改善。完整名單：[no_benefit_or_loss_cases.csv](analysis/no_benefit_or_loss_cases.csv)。",
        f"- {int((~answers.correct).sum())}/{len(answers)} 個實際答案錯誤；{int(answers.parse_failure.sum())} parse failures、{int(answers.truncated.sum())} truncated outputs 全保留。完整原文：[answer_failures.csv](analysis/answer_failures.csv)。",
        f"- Label oracle 的 task-quality gain 以 **{main_gain_dataset}** 最大；相對各自最佳固定配置的 point gain 為 0 的 dataset：**{', '.join(zero_gain) or 'none'}**。不能把整體平均改善宣稱為每個 dataset 都改善。",
        f"- KL oracle 相對 BF16 的配對 accuracy 差為 {interval('per_request_KL_ORACLE_accuracy_delta_vs_BF16',factor=100,digits=2,suffix=' pp')}；跨 wording KL choice 相對 accuracy-best fixed 為 {interval('cross_wording_KL_ORACLE_diagnostic_accuracy_delta_vs_accuracy_best_fixed',factor=100,digits=2,suffix=' pp')}。較低 KL 不必然代表更高答案正確率。",
        "- 不做部署、泛化到其他 model/quantizer、全模型 W4、吞吐/壓縮率的宣稱。只有一個固定 seed、每 dataset 32 個獨立 source、固定三個 demos 和小 calibration set，CI 不包含全部設計選擇/模型 pretraining contamination 的不確定性。",
        "- 這個結果足以顯示指定 W4 條件下的 request×group 差異與有限 oracle 潛力，不足以證明可以預測/轉移最佳配置。沒有為取得更好數字升級 W3 或挑樣本；所有原始 W4、pilot、失敗與方法修正均保留。", "",
        "## 圖與完整產物", "", "![全 evaluation set sensitivity heatmap](sensitivity_heatmap.png)", "",
        "![Content-only sensitivity heatmap](sensitivity_heatmap_content.png)", "",
        "![實際答案 accuracy 與 source-cluster CI](accuracy_comparison.png)", "",
        "- 圖也提供 PDF，另有 [row-normalized heatmap](sensitivity_heatmap_normalized.png)。顏色使用 log scale/floor 1e−8 僅為顯示；原始數值與統計未裁切。所有 source 都依 dataset/source index 顯示，不依效果排序。",
        "- 全部 CI：[statistics.csv](analysis/statistics.csv)；逐 request 摘要：[request_summary.csv](analysis/request_summary.csv)；輸入/分析 code hashes：[analysis_metadata.json](analysis/analysis_metadata.json)。",
        "- 原始量測：`runs/evaluation_{a,b}/{baselines.csv,measurements.csv}`、每 group JSON；`reference_logits/*.pt` 本機保留但 Git-ignored。Smoke/calibration raw records 在同目錄對應 role。", "",
        "## 實際執行紀錄與可重現命令", "",
        "安裝：`bash scripts/install.sh`；unit tests：`.venv/bin/python -m pytest -q tests`。一鍵執行：`EXPERIMENT_GPU=1 bash scripts/run_experiment.sh`（選自己的可用 GPU；相同 source hashes 的 raw checkpoints 可驗證/續跑）。完整新推論可指定 `EXPERIMENT_OUT=results/quantization_sensitivity_reproduction EXPERIMENT_GPU=1 bash scripts/run_experiment.sh`。", "",
        "實際執行的命令、exit code 與原始 log 保留在 `steps/`；一鍵命令逐步 `.command/.log/.exit` 的完整 ledger 路徑：[last_one_click_run.txt](last_one_click_run.txt)。以上命令是目前的重現入口，不是對歷史執行命令的重新命名。原分析/報告由一鍵 runner 備份到 ledger，未執行的推論不宣稱為完成。", "",
        "這份報告不以 tests/manifest 的綠燈代替研究結論。逐條 objective→artifact 的完成審核見 repository 的 `doc/completion-audit.md`；執行歷程與原始審查在 `WORK_LOG.md`、`doc/`。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    report_inputs = {str(analysis / name): file_hash(analysis / name) for name in ["statistics.csv", "controlled_shape_wild.csv", "answers.csv"]}
    if influence_path.exists():
        report_inputs[str(influence_path)] = file_hash(influence_path)
    save_json(root / "report_provenance.json", dict(bounded_evidence_supported=supported,
        evidence_is_not_router=True, actual_source_requests=meta["source_requests"],
        correctness_replicated_crossover_pairs=correctness_crossovers, label_oracle_gain_untruncated_worded_cases=untruncated_gains,
        inputs=report_inputs,
        report_sha256=file_hash(root / "report.md"), renderer_sha256=file_hash(__file__)))
    print(f"Rendered {root / 'report.md'}; bounded scientific evidence supported={supported}; completion audit still required")


if __name__ == "__main__":
    main()
