from __future__ import annotations

import argparse
import fnmatch
import tarfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist"
DEFAULT_INCLUDE_DIRS = [
    "src",
    "causal",
    "config",
    "data/cactus",
    "data/mi_quality_counseling_misc",
    "data/mi_re",
    "deploy",
    "doc",
]
DEFAULT_INCLUDE_FILES = [
    ".gitignore",
    "README.md",
    "package_project.py",
    "requirements.txt",
    "pyproject.toml",
    "run_ai_re_judge.py",
    "run_inference.py",
    "run_misc_causal_candidate_export.py",
    "run_misc_interpretability_analysis.py",
    "run_misc_label_mapping.py",
    "run_misc_mapping_structure_analysis.py",
    "run_sae_evaluation.py",
    "run_stage2_activation_extraction.py",
]
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "ENV",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "dist",
    "build",
    "site",
    "models",
    "checkpoints",
    "cache",
    ".cache",
    ".huggingface",
    "hf_cache",
}
EXCLUDED_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.pt.trace.json",
    "*.nvvp",
    "*.nsys-rep",
]


def should_exclude(rel_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
        return True
    return any(fnmatch.fnmatch(rel_path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def iter_release_files(project_root: Path):
    seen: set[Path] = set()

    for dirname in DEFAULT_INCLUDE_DIRS:
        base = project_root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(project_root)
            if should_exclude(rel_path):
                continue
            if rel_path not in seen:
                seen.add(rel_path)
                yield path, rel_path

    for filename in DEFAULT_INCLUDE_FILES:
        path = project_root / filename
        if not path.exists() or not path.is_file():
            continue
        rel_path = path.relative_to(project_root)
        if should_exclude(rel_path):
            continue
        if rel_path not in seen:
            seen.add(rel_path)
            yield path, rel_path


def build_release_archive(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    archive_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive_name is None:
        stamp = datetime.now().strftime("%Y%m%d")
        archive_name = f"nlp-re-gce-src-{stamp}.tar.gz"

    archive_path = output_dir / archive_name
    file_count = 0

    with tarfile.open(archive_path, "w:gz") as tar:
        for path, rel_path in iter_release_files(project_root):
            tar.add(path, arcname=rel_path.as_posix())
            file_count += 1

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Created {archive_path}")
    print(f"Included files: {file_count}")
    print(f"Archive size: {size_mb:.2f} MB")
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Linux-friendly source release archive for GCE deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root to package.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the tar.gz archive will be written.",
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Optional custom tar.gz filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_release_archive(
        project_root=Path(args.project_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        archive_name=args.archive_name,
    )


if __name__ == "__main__":
    main()
