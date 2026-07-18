"""CPU-only smoke checks for the isolated Codex latent-card runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.nlp_re_base.codex_latent_cards import (
    audit_event_stream,
    build_codex_latent_card_tasks,
    render_codex_latent_card_catalog,
)


def main() -> None:
    root = Path("outputs/_smoke_codex_latent_cards").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    source = root / "source_tasks.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "card_00001_induction",
                "packet_id": "card_00001",
                "latent_idx": 1,
                "prompt": "test prompt",
                "visible_samples": [{"id": "s001", "text": "text"}],
                "expected_output_path": "old.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    built = build_codex_latent_card_tasks(source_tasks_path=source, output_dir=root / "run")
    assert built["n_tasks"] == 1
    task = json.loads((root / "run/llm_tasks/latent_card_tasks.jsonl").read_text(encoding="utf-8"))
    assert "codex" in task["expected_output_path"]

    clean = audit_event_stream(
        '\n'.join(
            [
                '{"type":"thread.started"}',
                '{"type":"item.completed","item":{"type":"agent_message"}}',
                '{"type":"turn.completed"}',
            ]
        )
    )
    assert clean["tool_event_count"] == 0
    dirty = audit_event_stream(
        '{"type":"item.completed","item":{"type":"command_execution"}}'
    )
    assert dirty["tool_event_count"] == 1

    catalog_root = root / "catalog"
    (catalog_root / "card_outputs").mkdir(parents=True)
    card = {
        "latent_idx": 1,
        "short_name": "test-pattern",
        "explanation_type": "linguistic_structure",
        "confidence": 4,
        "support_count": 45,
        "support_fraction": 0.9,
        "primary_explanation": "A test pattern.",
        "candidate_behavioral_explanation": "A possible test function.",
        "confidence_rationale": "Most samples support it.",
        "possible_confounds": ["test confound"],
        "limitations": ["test limitation"],
        "representative_evidence_ids": ["s001"],
    }
    (catalog_root / "card_outputs" / "validated_cards.jsonl").write_text(
        json.dumps(card) + "\n", encoding="utf-8"
    )
    stable = root / "stable.csv"
    stable.write_text(
        "label,latent_idx,rank_within_label,stable_set_role\nAF,1,1,stable_core\n",
        encoding="utf-8",
    )
    summary = render_codex_latent_card_catalog(
        output_dir=catalog_root, stable_latents_path=stable
    )
    assert summary["n_cards"] == 1
    assert (catalog_root / "analysis" / "all_latent_explanations.md").exists()
    print("test_codex_latent_cards_smoke passed")


if __name__ == "__main__":
    main()
