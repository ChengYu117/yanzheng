from __future__ import annotations

import json
import os
import sys
import tarfile
import traceback
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestDeployConfig(unittest.TestCase):
    def test_cloud_causal_stage_defaults_to_full_misc_dataset(self):
        common = (PROJECT_ROOT / "deploy" / "gce" / "common.sh").read_text(encoding="utf-8")
        run_causal = (PROJECT_ROOT / "deploy" / "gce" / "run_causal.sh").read_text(encoding="utf-8")
        env_example = (PROJECT_ROOT / "deploy" / "gce" / "env.example").read_text(encoding="utf-8")

        self.assertIn(": \"${DATA_DIR:=data/mi_quality_counseling_misc}\"", common)
        self.assertIn(": \"${CAUSAL_DATA_DIR:=${DATA_DIR}}\"", common)
        self.assertIn(": \"${ALLOW_LEGACY_CAUSAL_DATA:=0}\"", common)
        self.assertIn("CAUSAL_DATA_DIR=${DATA_DIR}", env_example)
        self.assertIn("ALLOW_LEGACY_CAUSAL_DATA=0", env_example)
        self.assertIn("--data-dir \"${CAUSAL_DATA_DIR}\"", run_causal)
        self.assertIn("Do not pass --data-dir to deploy/gce/run_causal.sh", run_causal)
        self.assertIn("Refusing to run causal validation on legacy/balanced data", run_causal)
        self.assertIn("label_candidates/${CAUSAL_LABEL}_candidate_latents.csv", run_causal)

    def test_model_dir_precedence(self):
        from nlp_re_base.config import load_model_config

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_name": "test-model",
                        "model_path": "models/from-config",
                        "torch_dtype": "float16",
                        "device_map": "auto",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MODEL_DIR": "/env/model"}, clear=False):
                cfg_env = load_model_config(config_path)
                cfg_cli = load_model_config(config_path, model_dir="/cli/model")

            self.assertEqual(cfg_env["model_path"], "/env/model")
            self.assertEqual(cfg_cli["model_path"], "/cli/model")

    def test_output_root_fallback(self):
        from nlp_re_base.config import resolve_output_dir

        with patch.dict(os.environ, {"OUTPUT_ROOT": "/mnt/disks/data/outputs"}, clear=False):
            output_dir = resolve_output_dir(None, default_subdir="sae_eval")
        self.assertEqual(output_dir, Path("/mnt/disks/data/outputs") / "sae_eval")

    def test_model_loader_falls_back_without_accelerate(self):
        from nlp_re_base.model import load_local_model_and_tokenizer

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            fake_tokenizer = MagicMock()
            fake_tokenizer.pad_token = None
            fake_tokenizer.eos_token = "</s>"

            fake_model = MagicMock()
            fake_model.to.return_value = fake_model

            with patch("nlp_re_base.model.importlib.util.find_spec", return_value=None), \
                 patch("nlp_re_base.model.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), \
                 patch("nlp_re_base.model.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as mocked_load:
                _, _, cfg = load_local_model_and_tokenizer(
                    model_dir=model_dir,
                    device="cuda",
                )

            kwargs = mocked_load.call_args.kwargs
            self.assertNotIn("device_map", kwargs)
            fake_model.to.assert_called_once()
            self.assertEqual(cfg["model_path"], str(model_dir))


class TestPackaging(unittest.TestCase):
    def test_release_archive_includes_deploy_and_excludes_outputs(self):
        from package_project import build_release_archive

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            (root / "src" / "nlp_re_base").mkdir(parents=True)
            (root / "causal").mkdir(parents=True)
            (root / "config").mkdir(parents=True)
            (root / "data" / "mi_re").mkdir(parents=True)
            (root / "data" / "mi_quality_counseling_misc").mkdir(parents=True)
            (root / "deploy" / "gce").mkdir(parents=True)
            (root / "doc").mkdir(parents=True)
            (root / "outputs").mkdir(parents=True)

            (root / "src" / "nlp_re_base" / "__init__.py").write_text("", encoding="utf-8")
            (root / "causal" / "__init__.py").write_text("", encoding="utf-8")
            (root / "config" / "model_config.json").write_text("{}", encoding="utf-8")
            (root / "data" / "mi_re" / "re_dataset.jsonl").write_text("", encoding="utf-8")
            (root / "data" / "mi_quality_counseling_misc" / "README.md").write_text("", encoding="utf-8")
            (root / "deploy" / "gce" / "bootstrap.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (root / "deploy" / "gce" / "run_full_pipeline.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (root / "doc" / "readme.md").write_text("doc", encoding="utf-8")
            (root / "README.md").write_text("root", encoding="utf-8")
            (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (root / "run_sae_evaluation.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_misc_mapping_structure_analysis.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_misc_interpretability_analysis.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_misc_causal_candidate_export.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_misc_label_mapping.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_ai_re_judge.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "run_inference.py").write_text("print('ok')\n", encoding="utf-8")
            (root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
            (root / "outputs" / "should_not_ship.txt").write_text("nope", encoding="utf-8")

            archive_path = build_release_archive(
                project_root=root,
                output_dir=root / "dist",
                archive_name="test.tar.gz",
            )

            with tarfile.open(archive_path, "r:gz") as tar:
                names = set(tar.getnames())

            self.assertIn("deploy/gce/bootstrap.sh", names)
            self.assertIn("deploy/gce/run_full_pipeline.sh", names)
            self.assertIn("run_sae_evaluation.py", names)
            self.assertIn("run_misc_mapping_structure_analysis.py", names)
            self.assertIn("run_misc_interpretability_analysis.py", names)
            self.assertIn("run_misc_causal_candidate_export.py", names)
            self.assertIn("run_misc_label_mapping.py", names)
            self.assertIn("data/mi_quality_counseling_misc/README.md", names)
            self.assertIn("data/mi_re/re_dataset.jsonl", names)
            self.assertNotIn("outputs/should_not_ship.txt", names)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover - manual smoke output
        traceback.print_exc()
        raise
