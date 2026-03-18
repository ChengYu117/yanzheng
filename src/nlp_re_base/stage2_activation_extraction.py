"""Stage 2: per-layer activation extraction and linear probe analysis.

This module is the project-compatible version of the user's Stage 2 idea:

1. Run the local Llama base model with `output_hidden_states=True`
2. Extract one utterance-level representation from every hidden layer
3. Train lightweight linear probes per layer to locate the most linearly
   separable RE/NonRE layer

Unlike the original draft, this implementation:
- uses the project's local model loader instead of a hard-coded HF model
- uses `data/mi_re` by default instead of an external directory
- writes outputs under `outputs/` (or `OUTPUT_ROOT`) consistently
- can be imported as a normal module or run via a root CLI wrapper
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import resolve_output_dir
from .data import load_jsonl
from .model import load_local_model_and_tokenizer

DEFAULT_DATA_DIR = Path("data/mi_re")
DEFAULT_OUTPUT_SUBDIR = "stage2_activation_extraction"


def load_stage2_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    """Load RE and NonRE utterances into a binary classification dataset."""
    root = Path(data_dir)
    re_path = root / "re_dataset.jsonl"
    nonre_path = root / "nonre_dataset.jsonl"

    dataset: list[dict[str, Any]] = []

    re_records = load_jsonl(re_path)
    for record in re_records:
        dataset.append(
            {
                "unit_id": f"{record['file_id']}_{record.get('predicted_subcode', 'RE')}",
                "unit_text": record["unit_text"],
                "code": record["predicted_code"],
                "subcode": record.get("predicted_subcode"),
                "confidence": record.get("confidence", 0.0),
                "label_re": 1,
            }
        )

    nonre_records = load_jsonl(nonre_path)
    for record in nonre_records:
        dataset.append(
            {
                "unit_id": (
                    f"{record['file_id']}_"
                    f"{record.get('predicted_subcode', record['predicted_code'])}"
                ),
                "unit_text": record["unit_text"],
                "code": record["predicted_code"],
                "subcode": record.get("predicted_subcode"),
                "confidence": record.get("confidence", 0.0),
                "label_re": 0,
            }
        )

    return dataset


def _get_model_input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def extract_hidden_activations_by_layer(
    model: Any,
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    max_seq_len: int = 128,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extract utterance-level activations from every layer.

    Representation choice:
    - for each utterance, take the last non-padding token's hidden state
    - include embedding layer at index 0, then transformer layers 1..L
    """
    texts = [row["unit_text"] for row in dataset]
    labels = np.asarray([row["label_re"] for row in dataset], dtype=np.int64)

    all_layers: dict[int, list[np.ndarray]] | None = None
    model_device = _get_model_input_device(model)

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Stage2 extract",
        unit="batch",
    ):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=True,
        )
        encoded = {k: v.to(model_device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        last_indices = encoded["attention_mask"].sum(dim=1) - 1

        if all_layers is None:
            all_layers = {layer_idx: [] for layer_idx in range(len(hidden_states))}

        for batch_idx in range(encoded["input_ids"].size(0)):
            pos = int(last_indices[batch_idx].item())
            for layer_idx, hidden in enumerate(hidden_states):
                vector = hidden[batch_idx, pos, :].detach().cpu().float().numpy()
                all_layers[layer_idx].append(vector)

        del outputs, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert all_layers is not None
    stacked = {layer_idx: np.stack(vectors) for layer_idx, vectors in all_layers.items()}
    return stacked, labels


def save_activations_bundle(
    X_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    dataset: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Save extracted per-layer activations and metadata."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    npz_path = out_dir / f"activations_{timestamp}.npz"
    save_dict: dict[str, Any] = {"y": y}
    for layer_idx, matrix in X_by_layer.items():
        save_dict[f"layer_{layer_idx}"] = matrix
    np.savez_compressed(npz_path, **save_dict)

    meta_path = out_dir / f"activations_meta_{timestamp}.json"
    metadata = {
        "num_samples": int(len(y)),
        "num_layers": int(len(X_by_layer)),
        "hidden_dim": int(next(iter(X_by_layer.values())).shape[1]),
        "re_count": int(y.sum()),
        "nonre_count": int(len(y) - y.sum()),
        "model_name": model_name,
        "timestamp": timestamp,
        "unit_ids": [row["unit_id"] for row in dataset],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return npz_path, meta_path


def load_activations_bundle(
    activation_path: str | Path | None = None,
    *,
    output_dir: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, Path]:
    """Load the latest saved activation bundle or a specified file."""
    out_dir = Path(output_dir)
    if activation_path is None:
        files = sorted(out_dir.glob("activations_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"No activation bundle found in {out_dir}. Run Stage 2 with --extract first."
            )
        path = files[-1]
    else:
        path = Path(activation_path)

    data = np.load(path)
    y = data["y"]
    X_by_layer: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_", 1)[1])
            X_by_layer[layer_idx] = data[key]
    return X_by_layer, y, path


def train_linear_probes(
    X_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> list[dict[str, float | int]]:
    """Train one logistic regression probe per layer."""
    results: list[dict[str, float | int]] = []

    for layer_idx in tqdm(sorted(X_by_layer.keys()), desc="Stage2 probes", unit="layer"):
        X = X_by_layer[layer_idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        clf = LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        results.append(
            {
                "layer": int(layer_idx),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred)),
                "auc_roc": float(roc_auc_score(y_test, y_prob)),
            }
        )

    return results


def summarize_probe_results(
    results: list[dict[str, float | int]],
) -> dict[str, dict[str, float | int] | list[dict[str, float | int]]]:
    """Return the best layer summaries under different metrics."""
    best_by_acc = max(results, key=lambda item: item["accuracy"])
    best_by_f1 = max(results, key=lambda item: item["f1_score"])
    best_by_auc = max(results, key=lambda item: item["auc_roc"])
    return {
        "results": results,
        "best_layer_accuracy": best_by_acc,
        "best_layer_f1": best_by_f1,
        "best_layer_auc": best_by_auc,
    }


def save_probe_results(
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist probe summary to JSON and CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"probe_results_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / f"probe_results_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("layer,accuracy,f1_score,auc_roc\n")
        for row in summary["results"]:
            f.write(
                f"{row['layer']},{row['accuracy']:.6f},"
                f"{row['f1_score']:.6f},{row['auc_roc']:.6f}\n"
            )

    return json_path, csv_path


def plot_probe_results(
    results: list[dict[str, float | int]],
    *,
    output_dir: str | Path,
) -> Path | None:
    """Plot per-layer probe scores if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "probe_results_plot.png"

    layers = [int(row["layer"]) for row in results]
    accuracies = [float(row["accuracy"]) for row in results]
    f1_scores = [float(row["f1_score"]) for row in results]
    aucs = [float(row["auc_roc"]) for row in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, accuracies, "o-", label="Accuracy", linewidth=2, markersize=4)
    ax.plot(layers, f1_scores, "s-", label="F1 Score", linewidth=2, markersize=4)
    ax.plot(layers, aucs, "^-", label="AUC-ROC", linewidth=2, markersize=4)

    best_idx = int(np.argmax(accuracies))
    ax.axvline(
        x=layers[best_idx],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Layer = {layers[best_idx]}",
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Performance: RE vs Non-RE by Layer")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def print_probe_summary(summary: dict[str, Any]) -> None:
    """Console summary for interactive use."""
    print("\n" + "=" * 72)
    print("Stage 2 linear probe results (RE vs NonRE)")
    print("=" * 72)
    print(f"{'Layer':<8} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("-" * 72)
    for row in summary["results"]:
        print(
            f"{int(row['layer']):<8} "
            f"{float(row['accuracy']):.4f}       "
            f"{float(row['f1_score']):.4f}       "
            f"{float(row['auc_roc']):.4f}"
        )
    print("-" * 72)
    print(
        "Best layer by accuracy: "
        f"Layer {summary['best_layer_accuracy']['layer']} "
        f"= {summary['best_layer_accuracy']['accuracy']:.4f}"
    )
    print(
        "Best layer by F1:       "
        f"Layer {summary['best_layer_f1']['layer']} "
        f"= {summary['best_layer_f1']['f1_score']:.4f}"
    )
    print(
        "Best layer by AUC:      "
        f"Layer {summary['best_layer_auc']['layer']} "
        f"= {summary['best_layer_auc']['auc_roc']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: activation extraction + per-layer linear probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--extract", action="store_true", help="Extract activations only.")
    parser.add_argument("--probe", action="store_true", help="Train per-layer probes only.")
    parser.add_argument("--all", action="store_true", help="Run extraction and probe training.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max token length.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="RE/NonRE dataset dir.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Falls back to OUTPUT_ROOT/stage2_activation_extraction.",
    )
    parser.add_argument(
        "--activation-path",
        default=None,
        help="Existing activations .npz to reuse for --probe.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional model_config.json path.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override local model directory.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Probe test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and probe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.extract or args.probe or args.all):
        raise SystemExit("Specify one of --extract, --probe, or --all.")

    output_dir = resolve_output_dir(
        args.output_dir,
        default_subdir=DEFAULT_OUTPUT_SUBDIR,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    X_by_layer: dict[int, np.ndarray] | None = None
    y: np.ndarray | None = None

    if args.extract or args.all:
        dataset = load_stage2_dataset(args.data_dir)
        print(
            f"Loaded dataset: total={len(dataset)}, "
            f"RE={sum(row['label_re'] for row in dataset)}, "
            f"NonRE={len(dataset) - sum(row['label_re'] for row in dataset)}"
        )

        model, tokenizer, model_cfg = load_local_model_and_tokenizer(
            args.model_config,
            model_dir=args.model_dir,
        )
        X_by_layer, y = extract_hidden_activations_by_layer(
            model,
            tokenizer,
            dataset,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
        npz_path, meta_path = save_activations_bundle(
            X_by_layer,
            y,
            dataset,
            output_dir=output_dir,
            model_name=model_cfg.get("model_name", str(model_cfg.get("model_path"))),
        )
        print(f"Saved activations: {npz_path}")
        print(f"Saved metadata:    {meta_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.probe or args.all:
        if X_by_layer is None or y is None:
            X_by_layer, y, activation_file = load_activations_bundle(
                args.activation_path,
                output_dir=output_dir,
            )
            print(f"Loaded activations from: {activation_file}")

        results = train_linear_probes(
            X_by_layer,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        summary = summarize_probe_results(results)
        json_path, csv_path = save_probe_results(summary, output_dir=output_dir)
        plot_path = plot_probe_results(results, output_dir=output_dir)
        print_probe_summary(summary)
        print(f"Saved probe summary JSON: {json_path}")
        print(f"Saved probe summary CSV:  {csv_path}")
        if plot_path is not None:
            print(f"Saved probe plot:         {plot_path}")


if __name__ == "__main__":
    main()
