"""Run the Claude Code LLM-assisted stable-core interpretation workflow."""

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
    read_jsonl,
    run_build_contrastive_evidence_packs,
    write_json,
)
from nlp_re_base.contrastive_explainer import ExplainerTaskConfig, make_explainer_tasks
from nlp_re_base.contrastive_llm_io import validate_explainer_outputs
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


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _stage_status(*, prerequisite_complete: bool, has_artifacts: bool, complete: bool) -> str:
    if complete:
        return "complete"
    if has_artifacts:
        return "partial"
    if not prerequisite_complete:
        return "blocked"
    return "not_started"


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


def write_claude_code_guide(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    guide_path = output_path / "claude_code_llm_execution_guide.md"
    lines = [
        "# Claude Code LLM 执行指南",
        "",
        "Claude Code 同时负责本地实现和 LLM 评估。任务 JSONL 是审计边界：模型必须逐任务阅读 prompt 并直接生成判断，不得用规则、固定词表或硬编码概率代替推理。",
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
        "1. 在 Claude Code 中读取 `outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl`。",
        "2. 每行是一个独立任务；不要让一个任务读取另一个任务的输出。",
        "3. Claude Code 模型只返回一个 JSON object，不输出解释性散文。",
        "4. 将原始回答保存到该任务的 `expected_output_path`，路径形如 `explainer_outputs/raw/ctli_0001_explainer_r01.json`。",
        "5. 如果返回 Markdown code fence，原样保存，后续校验脚本会解析。",
        "6. 不要手工改字段名、概率或 sample id；解析失败也保留原始文件。",
        "",
        "校验：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-explainer",
        "```",
        "",
        "失败任务会进入 `explainer_outputs/retry_tasks.jsonl`，Claude Code 只重做这些任务。",
        "",
        "## 2. Scorer 与 Label Baseline 任务",
        "",
        "生成任务：",
        "",
        "```powershell",
        "conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-scorer-tasks",
        "```",
        "",
        "Claude Code 中执行：",
        "",
        "1. 读取 `llm_tasks/scorer_tasks.jsonl`，逐条完成 prompt，保存到 `scorer_outputs/raw_explanation_scorer/{task_id}.json`。",
        "2. 在隔离上下文中读取 `llm_tasks/baseline_tasks.jsonl`，保存到 `scorer_outputs/raw_label_baseline/{task_id}.json`。",
        "3. Scorer 输出必须是 JSON list，每项包含 `sample_id`, `pred_activate_prob`, `binary_prediction`, `evidence_span`, `reason`。",
        "4. ground truth 是 latent 激活检测，不是 MISC 标签分类；LLM 评估上下文不得读取 answer key。",
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
        "Claude Code 读取 `llm_tasks/minimal_pair_designer_tasks.jsonl` 并生成 JSON list，保存到 `minimal_pairs/raw_designer_outputs/{task_id}.json`。",
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
        "Claude Code 读取 `llm_tasks/subconcept_cluster_tasks.jsonl`，保存到 `subconcepts/raw_cluster_outputs/{task_id}.json`。",
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
    guide_path = write_claude_code_guide(output_path)

    evidence_packs_path = output_path / "evidence_packs" / "contrastive_evidence_packs.jsonl"
    evidence_summary = (
        pd.read_csv(output_path / "evidence_packs" / "contrastive_evidence_pack_summary.csv")
        if (output_path / "evidence_packs" / "contrastive_evidence_pack_summary.csv").exists()
        else pd.DataFrame()
    )
    evidence_manifest = _read_json(output_path / "manifest.json")
    explainer_tasks = _read_jsonl_if_exists(output_path / "llm_tasks" / "explainer_tasks.jsonl")
    explanations = _read_jsonl_if_exists(output_path / "explainer_outputs" / "validated_explanations.jsonl")
    explainer_manifest = _read_json(output_path / "explainer_outputs" / "validation_manifest.json")
    scorer_task_manifest = _read_json(output_path / "scorer_task_manifest.json")
    scorer_validation_manifest = _read_json(output_path / "scorer_outputs" / "validation_manifest.json")
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
    latent_status = (
        pd.read_csv(output_path / "scorer_outputs" / "latent_level_status.csv")
        if (output_path / "scorer_outputs" / "latent_level_status.csv").exists()
        else pd.DataFrame()
    )
    minimal_task_manifest = _read_json(output_path / "minimal_pair_task_manifest.json")
    minimal_validation_manifest = _read_json(
        output_path / "minimal_pairs" / "minimal_pair_validation_manifest.json"
    )
    minimal_activation_manifest = _read_json(
        output_path / "minimal_pairs" / "minimal_pair_activation_manifest.json"
    )
    minimal_results = (
        pd.read_csv(output_path / "minimal_pairs" / "minimal_pair_results.csv")
        if (output_path / "minimal_pairs" / "minimal_pair_results.csv").exists()
        else pd.DataFrame()
    )
    subconcept_task_manifest = _read_json(output_path / "subconcept_task_manifest.json")
    subconcept_manifest = _read_json(output_path / "subconcepts" / "subconcept_manifest.json")
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
    ablation_manifest = _read_json(output_path / "ablation" / "ablation_manifest.json")

    evidence_complete = bool(not evidence_summary.empty and evidence_packs_path.exists())
    evidence_status = _stage_status(
        prerequisite_complete=True,
        has_artifacts=bool(evidence_manifest or not evidence_summary.empty or evidence_packs_path.exists()),
        complete=evidence_complete,
    )
    explainer_tasks_count = int(explainer_manifest.get("n_tasks", len(explainer_tasks)))
    explainer_valid_count = int(explainer_manifest.get("n_valid", len(explanations)))
    explainer_complete = bool(
        explainer_tasks_count > 0
        and explainer_valid_count == explainer_tasks_count
        and int(explainer_manifest.get("n_retry", 0)) == 0
    )
    explainer_status = _stage_status(
        prerequisite_complete=evidence_complete,
        has_artifacts=bool(explainer_tasks or explanations or explainer_manifest),
        complete=explainer_complete,
    )
    scorer_tasks_count = int(scorer_validation_manifest.get("n_scorer_tasks", scorer_task_manifest.get("n_scorer_tasks", 0)))
    baseline_tasks_count = int(scorer_validation_manifest.get("n_baseline_tasks", scorer_task_manifest.get("n_baseline_tasks", 0)))
    scorer_complete = bool(
        scorer_tasks_count > 0
        and baseline_tasks_count > 0
        and int(scorer_validation_manifest.get("n_valid_scorer_tasks", -1)) == scorer_tasks_count
        and int(scorer_validation_manifest.get("n_valid_baseline_tasks", -1)) == baseline_tasks_count
        and int(scorer_validation_manifest.get("n_scorer_errors", 0)) == 0
        and int(scorer_validation_manifest.get("n_baseline_errors", 0)) == 0
        and not latent_status.empty
    )
    scorer_status = _stage_status(
        prerequisite_complete=explainer_complete,
        has_artifacts=bool(scorer_task_manifest or scorer_validation_manifest or not scorer_metrics.empty),
        complete=scorer_complete,
    )
    minimal_expected_tasks = int(minimal_task_manifest.get("expected_tasks", 18))
    minimal_tasks_count = int(minimal_task_manifest.get("n_tasks", 0))
    minimal_complete = bool(
        minimal_tasks_count == minimal_expected_tasks
        and minimal_expected_tasks > 0
        and int(minimal_validation_manifest.get("n_errors", -1)) == 0
        and int(minimal_validation_manifest.get("n_tasks", -1)) == minimal_tasks_count
        and minimal_activation_manifest
        and not minimal_results.empty
    )
    minimal_status = _stage_status(
        prerequisite_complete=scorer_complete,
        has_artifacts=bool(minimal_task_manifest or minimal_validation_manifest or minimal_activation_manifest),
        complete=minimal_complete,
    )
    subconcept_tasks_count = int(subconcept_task_manifest.get("n_tasks", 0))
    subconcept_complete = bool(
        subconcept_tasks_count > 0
        and subconcept_manifest
        and int(subconcept_manifest.get("n_errors", -1)) == 0
        and not subconcepts.empty
    )
    subconcept_status = _stage_status(
        prerequisite_complete=scorer_complete,
        has_artifacts=bool(subconcept_task_manifest or subconcept_manifest or not subconcepts.empty),
        complete=subconcept_complete,
    )
    ablation_complete = bool(ablation_manifest and not ablation.empty)
    ablation_status = _stage_status(
        prerequisite_complete=evidence_complete,
        has_artifacts=bool(ablation_manifest or not ablation.empty),
        complete=ablation_complete,
    )

    stage_rows = [
        {"step": "1", "stage": "Evidence packs", "status": evidence_status, "evidence": f"{len(evidence_summary)} label-latent rows"},
        {"step": "2", "stage": "Explainer", "status": explainer_status, "evidence": f"{explainer_valid_count}/{explainer_tasks_count} valid tasks"},
        {"step": "3", "stage": "Held-out scorer", "status": scorer_status, "evidence": f"{scorer_tasks_count} scorer tasks; {len(latent_status)} distinct label-latents"},
        {"step": "4", "stage": "Minimal pairs", "status": minimal_status, "evidence": f"{minimal_tasks_count}/{minimal_expected_tasks} design tasks; {len(minimal_results)} tested pairs"},
        {"step": "5", "stage": "Subconcepts", "status": subconcept_status, "evidence": f"{subconcept_tasks_count} cluster tasks; {len(subconcepts)} output rows"},
        {"step": "6", "stage": "Probe-space ablation", "status": ablation_status, "evidence": f"{len(ablation)} label rows"},
    ]
    stage_table = pd.DataFrame(stage_rows)
    stage_table.to_csv(output_path / "workflow_stage_status.csv", index=False)

    lines = [
        "# P3 Contrastive Latent Interpretation Report",
        "",
        "## Scope",
        "",
        "本报告面向 `stable_set_role == stable_core` 的 SAE latent。Claude Code 同时承担工程执行和 LLM 评估；任务 JSONL 与 raw JSON 文件保留逐任务审计边界。",
        "",
        "结论边界：当前流程提供候选解释、held-out 激活检测、minimal-pair 功能敏感性和 probe-space 依赖证据；不能写成单 latent 等同某 MISC 概念，也不能写成因果机制证明。",
        "",
        f"- Claude Code LLM execution guide: `{guide_path}`",
        "",
        "## Workflow Status",
        "",
        _markdown_table(stage_table, ["step", "stage", "status", "evidence"]),
        "",
        "状态定义：`not_started` 表示前置条件已满足但尚未开始；`blocked` 表示前置条件未满足；`partial` 表示已有部分产物但未通过完整门禁；`complete` 表示该阶段产物和门禁均满足。",
        "",
        "## Step 1 Evidence Packs",
        "",
    ]
    if evidence_summary.empty:
        lines.append("尚未生成 evidence packs。")
    else:
        label_counts = evidence_summary["target_label"].value_counts().sort_index().reset_index()
        label_counts.columns = ["target_label", "n_packs"]
        distinct_latents = int(evidence_summary["latent_idx"].nunique()) if "latent_idx" in evidence_summary else 0
        lines.append(f"- label-latent rows: {len(evidence_summary)}")
        lines.append(f"- distinct latent ids: {distinct_latents}")
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
        distinct_explained = len({(str(row.get("target_label", "")), int(row.get("latent_idx", -1))) for row in explanations})
        lines.append(f"- valid explanation tasks: {explainer_valid_count} / {explainer_tasks_count}")
        lines.append(f"- distinct explained label-latents: {distinct_explained}")
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
    if not latent_status.empty:
        latent_counts = latent_status["latent_status"].value_counts().sort_index().reset_index()
        latent_counts.columns = ["latent_status", "n_distinct_label_latents"]
        lines.extend(["", "Latent-level gate summary:", "", _markdown_table(latent_counts, ["latent_status", "n_distinct_label_latents"])])
    if not baseline_metrics.empty:
        lines.append("")
        lines.append(f"Baseline tasks scored: {len(baseline_metrics)}")

    lines.extend(["", "## Step 4 Minimal Pairs", ""])
    if minimal_results.empty:
        lines.append("尚无 SAE 前向测试结果，因此本报告不主张 minimal-pair 功能敏感性证据。")
        if minimal_task_manifest:
            lines.append(f"- design tasks planned: {minimal_tasks_count} / {minimal_expected_tasks}")
        if minimal_validation_manifest:
            lines.append(f"- validated pairs: {minimal_validation_manifest.get('n_pairs', 0)}")
            lines.append(f"- validation errors: {minimal_validation_manifest.get('n_errors', 0)}")
    else:
        tested_latents = int(minimal_results[["target_label", "latent_idx"]].drop_duplicates().shape[0])
        lines.append(f"- tested pairs: {len(minimal_results)}")
        lines.append(f"- distinct tested label-latents: {tested_latents}")
        lines.append(f"- pair pass rate: {_fmt_float(minimal_results['pair_passed'].astype(float).mean())}")
        lines.append(f"- mean activation gap: {_fmt_float(minimal_results['activation_gap'].mean())}")
        lines.append("")
        lines.append("该结果仅支持受控文本改动下的 SAE 激活敏感性，不构成因果机制证明。")

    lines.extend(["", "## Step 5 Subconcepts", ""])
    if subconcepts.empty:
        lines.append("尚未生成 subconcept table。")
    else:
        lines.append(f"- cluster tasks: {subconcept_tasks_count}")
        lines.append(f"- distinct validated latent inputs: {subconcept_task_manifest.get('n_distinct_latents', 0)}")
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
    explainer_tasks_path = output_dir / "llm_tasks" / "explainer_tasks.jsonl"
    explanations_path = output_dir / "explainer_outputs" / "validated_explanations.jsonl"
    scorer_tasks_path = output_dir / "llm_tasks" / "scorer_tasks.jsonl"
    baseline_tasks_path = output_dir / "llm_tasks" / "baseline_tasks.jsonl"
    answer_key_path = output_dir / "scorer_outputs" / "heldout_answer_key.csv"
    scorer_metrics_path = output_dir / "scorer_outputs" / "scorer_metrics.csv"
    latent_status_path = output_dir / "scorer_outputs" / "latent_level_status.csv"
    minimal_pair_tasks_path = output_dir / "llm_tasks" / "minimal_pair_designer_tasks.jsonl"
    minimal_pairs_path = output_dir / "minimal_pairs" / "validated_minimal_pairs.jsonl"
    subconcept_tasks_path = output_dir / "llm_tasks" / "subconcept_cluster_tasks.jsonl"
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
            latent_status_path=latent_status_path,
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
