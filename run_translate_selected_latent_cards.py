"""Translate five completed latent cards per stable-core leaf label into Chinese."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.nlp_re_base.codex_latent_cards import (
    CodexLatentCardConfig,
    default_isolation_paths,
    run_codex_latent_card_tasks,
)
from src.nlp_re_base.contrastive_evidence_pack import read_jsonl, write_json, write_jsonl


def _load_selected(stable_path: Path, explanation_path: Path, packet_path: Path) -> list[dict]:
    stable = pd.read_csv(stable_path)
    explanations = {int(row["latent_idx"]): row for row in read_jsonl(explanation_path)}
    packets = {int(row["latent_idx"]): row for row in read_jsonl(packet_path)}
    labels = ["RES", "REC", "QUO", "QUC", "GI", "SU", "AF"]
    selected: list[dict] = []
    for label in labels:
        rows = stable[
            stable["stable_set_role"].astype(str).eq("stable_core")
            & stable["label"].astype(str).eq(label)
        ].copy()
        rows["rank_num"] = pd.to_numeric(rows.get("rank_within_label", 999), errors="coerce").fillna(999)
        rows = rows.sort_values(["rank_num", "latent_idx"])
        seen: set[int] = set()
        for row in rows.itertuples(index=False):
            latent_idx = int(row.latent_idx)
            if latent_idx in seen or latent_idx not in explanations or latent_idx not in packets:
                continue
            seen.add(latent_idx)
            selected.append({
                "label": label,
                "latent_idx": latent_idx,
                "explanation": explanations[latent_idx],
                "packet": packets[latent_idx],
                "rank_within_label": int(getattr(row, "rank_num")),
            })
            if len(seen) == 5:
                break
        if len(seen) != 5:
            raise ValueError(f"Could not select five completed cards for {label}; found {len(seen)}")
    return selected


def _source_text(item: dict) -> str:
    explanation = item["explanation"]
    packet = item["packet"]
    public_packet = {
        "feature_id": explanation["feature_id"],
        "latent_idx": item["latent_idx"],
        "stable_core_label_for_audit": item["label"],
        "rank_within_label": item["rank_within_label"],
        "explanation": explanation,
        "strong_sentences": packet["strong_samples"],
        "weak_sentences": packet["weak_samples"],
        "heldout_sentences": [
            {key: value for key, value in sample.items() if key not in {"row_idx", "true_activation"}}
            for sample in packet["heldout_samples_private"]
        ],
    }
    return json.dumps(public_packet, ensure_ascii=False, indent=2)


def _translation_prompt(task_id: str, item: dict) -> str:
    return f"""Translate the complete scientific card below into Simplified Chinese for human audit.

Task ID: {task_id}

Mandatory output structure inside the markdown string:
1. 候选解释与最窄区分条件
2. 不充分条件
3. 表层/语言学假设
4. 行为/话语功能假设
5. 主要解释、解释类型与置信度
6. 逐条证据（保留每个 evidence item 的 sample_id、group、evidence_level、evidence_role，并翻译 evidence）
7. 备选解释、混淆因素与局限
8. 强组句子（必须逐条保留10条并翻译）
9. 弱组句子（必须逐条保留10条并翻译）
10. Held-out句子（必须逐条保留20条并翻译，保留 high/mid/weak/zero 分层名称）

Do not omit any sentence or evidence item. Keep all IDs and all numeric values unchanged. The markdown content must be Chinese except for preserved IDs, metric names, and code-like field values.

CARD JSON:
{_source_text(item)}"""


def build_tasks(selected: list[dict], output: Path) -> Path:
    task_dir = output / "llm_tasks"; raw_dir = output / "raw_translation"
    task_dir.mkdir(parents=True, exist_ok=True); raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for order, item in enumerate(selected, 1):
        task_id = f"{item['label']}_F{int(item['explanation']['feature_id'][1:]):03d}"
        tasks.append({
            "task_id": task_id,
            "latent_idx": int(item["latent_idx"]),
            "feature_id": item["explanation"]["feature_id"],
            "label": item["label"],
            "prompt": _translation_prompt(task_id, item),
            "expected_output_path": str((raw_dir / f"{task_id}.json").resolve()),
        })
    path = task_dir / "translation_tasks.jsonl"
    write_jsonl(path, tasks)
    write_json(output / "translation_task_manifest.json", {"n_tasks": len(tasks), "labels": {label: 5 for label in sorted({x['label'] for x in selected})}})
    return path


def render_document(selected: list[dict], tasks_path: Path, output: Path) -> Path:
    task_by_id = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    sections = []
    for item in selected:
        task_id = f"{item['label']}_F{int(item['explanation']['feature_id'][1:]):03d}"
        path = Path(task_by_id[task_id]["expected_output_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections.append(
            f"## {item['label']} / Latent {item['latent_idx']} / {item['explanation']['feature_id']}\n\n"
            + str(payload["markdown"]).strip()
        )
    labels = sorted({item["label"] for item in selected})
    lines = [
        "# Stable-core Latent Cards 中文翻译审核文档",
        "",
        "> 来源：已完成的 GPT-5.5-low 对比式解释 card。每个叶标签抽取5个 latent，共35个标签–latent审核条目。解释、证据、强/弱句子及20条 held-out句子均要求逐条翻译；latent可因多标签归属而重复出现。",
        "> 重要边界：这些是候选解释和预测忠实度材料，不等同于MISC标签，也不是因果机制证明。",
        "",
        "## 抽样标签",
        "",
        "、".join(labels),
        "",
        *sections,
        "",
        "## 审核提示",
        "",
        "请重点检查：强组与弱组的最窄区分是否真实存在；译文是否保留原句语义、转录错误和语气；解释是否把表层句式过度升级为行为功能；held-out句子是否支持解释的可预测性。",
        "",
    ]
    path = output / "stable_core_latent_cards_每标签5个_中文翻译审核.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "run", "render", "all"), nargs="?", default="all")
    parser.add_argument("--stable", type=Path, default=Path("outputs/rerun_new_dataset_20260716/min5_words/cross_val/stable_topk_selection_n20_relaxed_leaf7/stable_topk_latent_set.csv"))
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_gpt55_low_full218"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rerun_new_dataset_20260716/min5_words/interpretability/stable_core_latent_cards_translation_review"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(); output = args.output_dir
    selected = _load_selected(args.stable, args.source_dir / "explainer" / "validated_explanations.jsonl", args.source_dir / "private_packets.jsonl")
    tasks_path = build_tasks(selected, output)
    if args.action in {"run", "all"}:
        from src.nlp_re_base.codex_latent_cards import run_codex_latent_card_tasks
        workdir, codex_home = default_isolation_paths(output)
        run_codex_latent_card_tasks(
            tasks_path=tasks_path, output_dir=output / "execution", schema_path="config/latent_card_translation_schema.json",
            instructions_path="config/latent_card_translation_base_instructions.txt", auth_source=Path.home() / ".codex" / "auth.json",
            workdir=workdir, codex_home=codex_home,
            config=CodexLatentCardConfig(model="gpt-5.5", reasoning_effort="low", concurrency=args.concurrency, timeout_seconds=600),
        )
    if args.action in {"render", "all"}:
        print(render_document(selected, tasks_path, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
