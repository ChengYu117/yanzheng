"""Run the revised P3 contrastive latent interpretation workflow.

The workflow targets stable-core SAE latents and uses Antigravity Gemini 3.5
through file queues rather than direct API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.contrastive_ablation import ProbeSpaceAblationConfig, run_probe_space_ablation
from nlp_re_base.contrastive_evidence_pack import (
    DEFAULT_LABELS,
    ContrastiveEvidenceConfig,
    jsonable,
    run_build_contrastive_evidence_packs,
    write_json,
)
from nlp_re_base.contrastive_explainer import ExplainerTaskConfig, make_explainer_tasks
from nlp_re_base.contrastive_gemini_io import validate_explainer_outputs
from nlp_re_base.contrastive_minimal_pairs import (
    MinimalPairTaskConfig,
    make_minimal_pair_tasks,
    run_minimal_pair_activation_test,
    validate_minimal_pair_outputs,
)
from nlp_re_base.contrastive_scorer import ScorerTaskConfig, make_scorer_tasks, validate_scorer_outputs
from nlp_re_base.contrastive_subconcepts import build_subconcept_table, make_subconcept_tasks


DEFAULT_OUTPUT_DIR = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp"
DEFAULT_LATENTS = "outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv"
DEFAULT_FEATURES = "outputs/misc_full_sae_eval/feature_store/utterance_features.pt"
DEFAULT_LABEL_MATRIX = "outputs/misc_full_sae_eval/label_matrix.csv"
DEFAULT_RECORDS = "outputs/misc_full_sae_eval/records.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return ""
    view = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(_fmt_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_gemini_guide(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    guide_path = output_path / "gemini_antigravity_execution_guide.md"
    lines = [
        "# Antigravity Gemini 3.5 执行指南",
        "",
        "本 P3 流程不假设 Gemini 有可脚本调用的 API。Codex 负责生成任务文件、校验原始 JSON、计算指标；Gemini 3.5 在 Antigravity IDE 中按文件队列执行。",
        "",
        "## 0. 本地先生成证据包",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-packs",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-explainer-tasks",
        "```",
        "",
        "## 1. Explainer 任务",
        "",
        "1. 在 Antigravity 中打开 `outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/ide_tasks/explainer_tasks.jsonl`。",
        "2. 每行是一个任务；复制该行中的 `prompt` 给 Gemini 3.5。",
        "3. 要求 Gemini 只返回一个 JSON object，不要解释性散文。",
        "4. 将原始回答保存到该任务的 `expected_output_path`，路径形如 `explainer_outputs/raw/ctli_0001_explainer_r01.json`。",
        "5. 如果 Gemini 返回 Markdown code fence，原样保存，后续校验脚本会解析。",
        "6. 不要手工改字段名、概率或 sample id；解析失败也保留原始文件。",
        "",
        "校验：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-explainer",
        "```",
        "",
        "失败任务会进入 `explainer_outputs/retry_tasks.jsonl`，把这些任务重新交给 Gemini。",
        "",
        "## 2. Scorer 与 Label Baseline 任务",
        "",
        "生成任务：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-scorer-tasks",
        "```",
        "",
        "Antigravity 中执行：",
        "",
        "1. 打开 `ide_tasks/scorer_tasks.jsonl`，逐条复制 `prompt` 给 Gemini，保存到 `scorer_outputs/raw_explanation_scorer/{task_id}.json`。",
        "2. 打开 `ide_tasks/baseline_tasks.jsonl`，逐条复制 `prompt` 给 Gemini，保存到 `scorer_outputs/raw_label_baseline/{task_id}.json`。",
        "3. Scorer 输出必须是 JSON list，每项包含 `sample_id`, `pred_activate_prob`, `binary_prediction`, `evidence_span`, `reason`。",
        "4. 这里的 ground truth 是 latent 激活检测，不是 MISC 标签分类；不要让 Gemini 看到 answer key。",
        "",
        "校验与指标计算：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-scorer",
        "```",
        "",
        "核心输出是 `scorer_outputs/scorer_metrics.csv`：`accepted` 需要 AUROC >= 0.70 且 `latent_gap > 0`。",
        "",
        "## 3. Minimal Pair 任务",
        "",
        "生成任务：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-minimal-pair-tasks",
        "```",
        "",
        "在 Antigravity 中打开 `ide_tasks/minimal_pair_designer_tasks.jsonl`，让 Gemini 生成 JSON list。保存到 `minimal_pairs/raw_designer_outputs/{task_id}.json`。",
        "",
        "校验：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-minimal-pairs",
        "```",
        "",
        "GPU 前向测试：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step run-minimal-pairs --device cuda --batch-size 4",
        "```",
        "",
        "## 4. 子概念聚合",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-subconcept-tasks",
        "```",
        "",
        "在 Antigravity 中执行 `ide_tasks/subconcept_cluster_tasks.jsonl`，保存到 `subconcepts/raw_cluster_outputs/{task_id}.json`。",
        "",
        "生成表：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-subconcept-table",
        "```",
        "",
        "## 5. 报告与边界",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step ablation",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step report",
        "```",
        "",
        "报告只能写候选解释、held-out predictive sufficiency、minimal-pair 功能敏感性、probe-space dependency；不能写成单个 latent 等同概念或因果机制证明。",
        "",
    ]
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text("\n".join(lines), encoding="utf-8")
    return guide_path


def write_report(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    report_path = output_path / "contrastive_latent_interp_report.md"
    guide_path = write_gemini_guide(output_path)

    evidence_summary = (
        pd.read_csv(output_path / "evidence_packs" / "contrastive_evidence_pack_summary.csv")
        if (output_path / "evidence_packs" / "contrastive_evidence_pack_summary.csv").exists()
        else pd.DataFrame()
    )
    explainer_manifest = _read_json(output_path / "explainer_outputs" / "validation_manifest.json")
    scorer_metrics = (
        pd.read_csv(output_path / "scorer_outputs" / "scorer_metrics.csv")
        if (output_path / "scorer_outputs" / "scorer_metrics.csv").exists()
        else pd.DataFrame()
    )
    baseline_metrics = (
        pd.read_csv(output_path / "scorer_outputs" / "baseline_metrics.csv")
        if (output_path / "scorer_outputs" / "baseline_metrics.csv").exists()
        else pd.DataFrame()
    )
    subconcepts = (
        pd.read_csv(output_path / "subconcepts" / "subconcept_table.csv")
        if (output_path / "subconcepts" / "subconcept_table.csv").exists()
        else pd.DataFrame()
    )
    ablation = (
        pd.read_csv(output_path / "ablation" / "set_ablation.csv")
        if (output_path / "ablation" / "set_ablation.csv").exists()
        else pd.DataFrame()
    )

    lines = [
        "# P3 Contrastive Latent Interpretation Report",
        "",
        "## Scope",
        "",
        "本报告面向 `stable_set_role == stable_core` 的 SAE latent。Gemini 3.5 位于 Antigravity IDE 中，因此本地脚本只生成任务、校验 JSON、计算指标，不直接调用 LLM API。",
        "",
        "结论边界：当前流程提供候选解释、held-out 激活检测、minimal-pair 功能敏感性和 probe-space 依赖证据；不能写成单 latent 等同某 MISC 概念，也不能写成因果机制证明。",
        "",
        "## Step 1 Evidence Packs",
        "",
    ]
    if evidence_summary.empty:
        lines.append("尚未生成 evidence packs。")
    else:
        label_counts = evidence_summary["target_label"].value_counts().sort_index().reset_index()
        label_counts.columns = ["target_label", "n_packs"]
        lines.append(f"- packs: {len(evidence_summary)}")
        lines.append(f"- Gemini guide: `{guide_path}`")
        lines.append("")
        lines.append(_markdown_table(label_counts, ["target_label", "n_packs"]))
        scarce_cols = [col for col in evidence_summary.columns if col.startswith("scarce_")]
        if scarce_cols:
            scarce = pd.DataFrame(
                [{"scarcity_flag": col, "n_packs": int(evidence_summary[col].fillna(False).astype(bool).sum())} for col in scarce_cols]
            )
            scarce = scarce[scarce["n_packs"] > 0]
            if not scarce.empty:
                lines.extend(["", "Scarcity flags:", "", _markdown_table(scarce, ["scarcity_flag", "n_packs"])])

    lines.extend(["", "## Step 2 Explainer", ""])
    if explainer_manifest:
        lines.append(f"- valid explanations: {explainer_manifest.get('n_valid', 0)} / {explainer_manifest.get('n_tasks', 0)}")
        lines.append(f"- retry tasks: {explainer_manifest.get('n_retry', 0)}")
    else:
        lines.append("尚未校验 explainer 输出。")

    lines.extend(["", "## Step 3 Scorer", ""])
    if scorer_metrics.empty:
        lines.append("尚未生成 scorer metrics。")
    else:
        status_counts = scorer_metrics["status"].value_counts().sort_index().reset_index()
        status_counts.columns = ["status", "n_tasks"]
        lines.append(_markdown_table(status_counts, ["status", "n_tasks"]))
        label_summary = (
            scorer_metrics.groupby("target_label", as_index=False)
            .agg(
                n_tasks=("task_id", "count"),
                mean_auroc=("auroc", "mean"),
                mean_baseline_auroc=("baseline_auroc", "mean"),
                mean_latent_gap=("latent_gap", "mean"),
            )
            .sort_values("target_label")
        )
        lines.extend(["", "Label-level scorer summary:", "", _markdown_table(label_summary, ["target_label", "n_tasks", "mean_auroc", "mean_baseline_auroc", "mean_latent_gap"])])
    if not baseline_metrics.empty:
        lines.append("")
        lines.append(f"Baseline tasks scored: {len(baseline_metrics)}")

    lines.extend(["", "## Step 5 Subconcepts", ""])
    if subconcepts.empty:
        lines.append("尚未生成 subconcept table。")
    else:
        lines.append(f"- subconcept rows: {len(subconcepts)}")
        lines.append("")
        lines.append(_markdown_table(subconcepts, ["target_label", "subconcept", "representative_latents", "mean_auroc", "mean_latent_gap", "status"], limit=30))

    lines.extend(["", "## Step 6 Probe-Space Ablation", ""])
    if ablation.empty:
        lines.append("尚未生成 ablation 结果。")
    else:
        lines.append(_markdown_table(ablation, ["ablated_label", "stable_core_count", "target_drop", "random_baseline_drop", "drop_vs_random", "non_target_drop_mean", "status"]))
        lines.append("")
        lines.append("Ablation 只支持 probe-space 选择性依赖表述，不支持因果机制表述。")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable-core contrastive SAE latent interpretation workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "build-packs",
            "make-explainer-tasks",
            "validate-explainer",
            "make-scorer-tasks",
            "validate-scorer",
            "make-minimal-pair-tasks",
            "validate-minimal-pairs",
            "run-minimal-pairs",
            "make-subconcept-tasks",
            "build-subconcept-table",
            "ablation",
            "report",
            "all-local",
        ],
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latents", default=DEFAULT_LATENTS)
    parser.add_argument("--feature-store", default=DEFAULT_FEATURES)
    parser.add_argument("--label-matrix", default=DEFAULT_LABEL_MATRIX)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--expected-stable-core-count", type=int, default=303, help="Use 0 to disable the count guard.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--explainer-repeats", type=int, default=2)
    parser.add_argument("--scorer-best-explanation-only", action="store_true")
    parser.add_argument("--sae-config", default="config/sae_config.json")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--checkpoint-topk-semantics", choices=["disabled", "hard"], default="hard")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    labels = tuple(label.upper() for label in args.labels)
    expected_count = None if int(args.expected_stable_core_count) <= 0 else int(args.expected_stable_core_count)

    packs_path = output_dir / "evidence_packs" / "contrastive_evidence_packs.jsonl"
    explainer_tasks_path = output_dir / "ide_tasks" / "explainer_tasks.jsonl"
    explanations_path = output_dir / "explainer_outputs" / "validated_explanations.jsonl"
    scorer_tasks_path = output_dir / "ide_tasks" / "scorer_tasks.jsonl"
    baseline_tasks_path = output_dir / "ide_tasks" / "baseline_tasks.jsonl"
    answer_key_path = output_dir / "scorer_outputs" / "heldout_answer_key.csv"
    scorer_metrics_path = output_dir / "scorer_outputs" / "scorer_metrics.csv"
    minimal_pair_tasks_path = output_dir / "ide_tasks" / "minimal_pair_designer_tasks.jsonl"
    minimal_pairs_path = output_dir / "minimal_pairs" / "validated_minimal_pairs.jsonl"
    subconcept_tasks_path = output_dir / "ide_tasks" / "subconcept_cluster_tasks.jsonl"
    subconcept_input_path = output_dir / "subconcepts" / "subconcept_cluster_input.csv"

    if args.step in {"build-packs", "all-local"}:
        manifest = run_build_contrastive_evidence_packs(
            latents_path=args.latents,
            feature_store_path=args.feature_store,
            label_matrix_path=args.label_matrix,
            records_path=args.records,
            output_dir=output_dir,
            config=ContrastiveEvidenceConfig(
                labels=labels,
                expected_stable_core_count=expected_count,
                random_state=args.random_state,
            ),
        )
        print(f"Built evidence packs: {manifest['n_packs']} -> {manifest['outputs']['evidence_packs']}")

    if args.step in {"make-explainer-tasks", "all-local"}:
        manifest = make_explainer_tasks(
            packs_path=packs_path,
            output_dir=output_dir,
            config=ExplainerTaskConfig(
                repeats_per_latent=args.explainer_repeats,
                random_state=args.random_state,
            ),
        )
        print(f"Built explainer tasks: {manifest['n_tasks']} -> {manifest['outputs']['explainer_tasks']}")

    if args.step == "validate-explainer":
        manifest = validate_explainer_outputs(tasks_path=explainer_tasks_path, output_dir=output_dir)
        print(f"Validated explainer outputs: {manifest['n_valid']} valid, {manifest['n_retry']} retry")

    if args.step == "make-scorer-tasks":
        manifest = make_scorer_tasks(
            packs_path=packs_path,
            explanations_path=explanations_path,
            output_dir=output_dir,
            config=ScorerTaskConfig(include_all_valid_explanations=not args.scorer_best_explanation_only),
        )
        print(f"Built scorer tasks: {manifest['n_scorer_tasks']} scorer, {manifest['n_baseline_tasks']} baseline")

    if args.step == "validate-scorer":
        manifest = validate_scorer_outputs(
            scorer_tasks_path=scorer_tasks_path,
            baseline_tasks_path=baseline_tasks_path,
            answer_key_path=answer_key_path,
            output_dir=output_dir,
        )
        print(f"Validated scorer outputs: {manifest['n_valid_scorer_predictions']} scorer predictions")

    if args.step == "make-minimal-pair-tasks":
        manifest = make_minimal_pair_tasks(
            packs_path=packs_path,
            explanations_path=explanations_path,
            scorer_metrics_path=scorer_metrics_path,
            output_dir=output_dir,
            config=MinimalPairTaskConfig(),
        )
        print(f"Built minimal-pair tasks: {manifest['n_tasks']}")

    if args.step == "validate-minimal-pairs":
        manifest = validate_minimal_pair_outputs(tasks_path=minimal_pair_tasks_path, output_dir=output_dir)
        print(f"Validated minimal pairs: {manifest['n_pairs']} pairs, {manifest['n_errors']} errors")

    if args.step == "run-minimal-pairs":
        manifest = run_minimal_pair_activation_test(
            pairs_path=minimal_pairs_path,
            output_dir=output_dir,
            sae_config_path=args.sae_config,
            model_config_path=args.model_config,
            model_dir=args.model_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            checkpoint_topk_semantics=args.checkpoint_topk_semantics,
        )
        print(f"Ran minimal-pair activation test: {manifest['n_pairs']} pairs")

    if args.step == "make-subconcept-tasks":
        manifest = make_subconcept_tasks(
            packs_path=packs_path,
            explanations_path=explanations_path,
            scorer_metrics_path=scorer_metrics_path,
            output_dir=output_dir,
        )
        print(f"Built subconcept tasks: {manifest['n_tasks']}")

    if args.step == "build-subconcept-table":
        manifest = build_subconcept_table(
            cluster_tasks_path=subconcept_tasks_path,
            cluster_input_path=subconcept_input_path,
            scorer_metrics_path=scorer_metrics_path,
            output_dir=output_dir,
        )
        print(f"Built subconcept table: {manifest['n_rows']} rows")

    if args.step == "ablation":
        manifest = run_probe_space_ablation(
            latents_path=args.latents,
            feature_store_path=args.feature_store,
            label_matrix_path=args.label_matrix,
            output_dir=output_dir,
            config=ProbeSpaceAblationConfig(
                labels=labels,
                expected_stable_core_count=expected_count,
                random_state=args.random_state,
            ),
        )
        print(f"Ran probe-space ablation: {manifest['n_fold_rows']} fold rows")

    if args.step in {"report", "all-local"}:
        report_path = write_report(output_dir)
        print(f"Wrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
