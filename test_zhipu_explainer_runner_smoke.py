from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import write_jsonl
from src.nlp_re_base.zhipu_explainer_runner import (
    SHORT_SYSTEM_PROMPT,
    ZhipuExplainerConfig,
    run_zhipu_tasks,
    select_pilot_tasks,
)


class _FakeClient:
    def __init__(self, api_key: str) -> None:
        assert api_key == "test-key"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        assert kwargs["model"] == "glm-4.7"
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["content"] == SHORT_SYSTEM_PROMPT
        return SimpleNamespace(
            id="req-test",
            model="glm-4.7",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"latent_idx": 1}'))],
            usage=None,
        )


def test_prepare_and_run_pilot(root: Path) -> None:
    labels = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
    summary_rows = []
    tasks = []
    for label_index, label in enumerate(labels):
        for packet_index in range(1, 3):
            packet_id = f"{label.lower()}_{packet_index}"
            summary_rows.append(
                {
                    "packet_id": packet_id,
                    "target_label": label,
                    "rank_within_label": packet_index,
                    "inclusion_frequency": 1.0 - packet_index / 10,
                    "latent_idx": label_index * 10 + packet_index,
                }
            )
            for repeat in (1, 2):
                tasks.append(
                    {
                        "task_id": f"{packet_id}_explainer_r{repeat:02d}",
                        "packet_id": packet_id,
                        "latent_idx": label_index * 10 + packet_index,
                        "repeat": repeat,
                        "prompt": "Return one JSON object for this anonymous latent.",
                    }
                )
    summary_path = root / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    task_path = root / "source_tasks.jsonl"
    write_jsonl(task_path, tasks)
    selection = select_pilot_tasks(
        tasks_path=task_path,
        summary_path=summary_path,
        output_dir=root / "pilot",
        config=ZhipuExplainerConfig(per_label=2),
    )
    assert selection["n_tasks"] == 36
    assert selection["n_packets"] == 18

    run = run_zhipu_tasks(
        tasks_path=root / "pilot" / "llm_tasks" / "explainer_tasks.jsonl",
        output_dir=root / "pilot",
        api_key="test-key",
        config=ZhipuExplainerConfig(concurrency=2, max_retries=0),
        client_factory=_FakeClient,
        max_tasks=36,
    )
    assert run["counts"]["success"] == 36
    manifest = [
        json.loads(line)
        for line in (root / "pilot" / "llm_execution_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(manifest) == 36
    assert {row["execution_mode"] for row in manifest} == {"zhipu_api"}
    assert len(list((root / "pilot" / "explainer_outputs" / "raw").glob("*.json"))) == 36


def main() -> None:
    root = (Path.cwd() / "outputs" / "_smoke_zhipu_explainer_runner").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        test_prepare_and_run_pilot(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_zhipu_explainer_runner_smoke passed")


if __name__ == "__main__":
    main()
