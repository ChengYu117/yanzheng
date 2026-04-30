from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestPaiDeploySmoke(unittest.TestCase):
    def test_build_job_command_uses_output_root_and_whitelist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["OUTPUT_ROOT"] = tmpdir
            from nlp_re_base.pai import job_runner

            job_runner.OUTPUT_ROOT = Path(tmpdir)
            command, output_dir = job_runner.build_job_command(
                job_type="llamascope_sanity",
                output_subdir="run_20260422",
                args={
                    "max_docs": 8,
                    "batch_size": 4,
                    "max_seq_len": 128,
                    "checkpoint_topk_semantics": "hard",
                },
            )
            self.assertEqual(output_dir, Path(tmpdir) / "run_20260422")
            self.assertIn("run_llamascope_distribution_sanity.py", " ".join(command))
            self.assertIn("--output-dir", command)
            self.assertIn("--max-docs", command)
            self.assertIn("--checkpoint-topk-semantics", command)

    def test_invalid_arg_is_rejected(self):
        from nlp_re_base.pai.job_runner import filter_args

        with self.assertRaises(ValueError):
            filter_args("sae_parity", {"not_allowed": 1})

    def test_service_healthz_if_fastapi_available(self):
        try:
            from fastapi.testclient import TestClient
            from nlp_re_base.pai.service_api import app
        except Exception:
            self.skipTest("FastAPI stack is not installed in the current environment")

        client = TestClient(app)
        response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
