"""Prepare blinded Task 5 materials for matched SAE/PCA human evaluation.

This module stops at the human-review boundary. It selects and matches units
using training-fold data, builds discovery and held-out example packs, and
writes private/blinded artifacts without generating interpretations or scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .cross_val_framework import compute_auc_and_cohens_d, normalize_text


DEFAULT_LABELS = ("REC", "QUO", "GI", "AF")
DEFAULT_PCA_DIMENSIONS = (50, 100)


@dataclass(frozen=True)
class Task5PreparationConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    pca_dimensions: tuple[int, ...] = DEFAULT_PCA_DIMENSIONS
    units_per_label: int = 4
    folds: int = 5
    fold_index: int = 0
    random_state: int = 42
    group_column: str = "file_id"
    discovery_count: int = 10
    heldout_positive_count: int = 10
    heldout_control_count: int = 10
    control_quantile_low: float = 0.35
    control_quantile_high: float = 0.65
    near_duplicate_threshold: float = 0.90
    association_chunk_size: int = 512
    matching_rank_penalty: float = 0.001
    max_auc_difference: float = 0.05

    def validate(self) -> None:
        if not self.labels:
            raise ValueError("At least one label is required")
        if not self.pca_dimensions or any(int(x) < 1 for x in self.pca_dimensions):
            raise ValueError("pca_dimensions must contain positive integers")
        if self.units_per_label < 1:
            raise ValueError("units_per_label must be positive")
        if self.folds < 2 or not 0 <= self.fold_index < self.folds:
            raise ValueError("fold_index must be within the configured folds")
        if self.discovery_count < 1 or self.heldout_positive_count < 1:
            raise ValueError("Example counts must be positive")
        if self.heldout_control_count < 1:
            raise ValueError("heldout_control_count must be positive")
        if not 0.0 <= self.control_quantile_low < self.control_quantile_high <= 1.0:
            raise ValueError("Invalid control quantile interval")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")
        if self.matching_rank_penalty < 0.0:
            raise ValueError("matching_rank_penalty must be non-negative")
        if not 0.0 < self.max_auc_difference <= 0.5:
            raise ValueError("max_auc_difference must be in (0, 0.5]")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(
    sae_features: np.ndarray,
    raw_hidden: np.ndarray,
    label_df: pd.DataFrame,
    stable_core: pd.DataFrame,
    config: Task5PreparationConfig,
) -> None:
    config.validate()
    if sae_features.ndim != 2 or raw_hidden.ndim != 2:
        raise ValueError("SAE and raw hidden inputs must be two-dimensional")
    if not (len(sae_features) == len(raw_hidden) == len(label_df)):
        raise ValueError("SAE, raw hidden, and label rows are not aligned")
    required_labels = set(config.labels).difference(label_df.columns)
    if required_labels:
        raise ValueError(f"Missing label columns: {sorted(required_labels)}")
    required_meta = {config.group_column, "unit_text"}.difference(label_df.columns)
    if required_meta:
        raise ValueError(f"Missing label metadata columns: {sorted(required_meta)}")
    required_stable = {"label", "latent_idx", "stable_set_role"}.difference(stable_core.columns)
    if required_stable:
        raise ValueError(f"Missing stable-core columns: {sorted(required_stable)}")


def _split_for_label(
    label_df: pd.DataFrame, label: str, config: Task5PreparationConfig
) -> tuple[np.ndarray, np.ndarray]:
    y = label_df[label].astype(int).to_numpy()
    groups = label_df[config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_state,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    train_idx, test_idx = splits[config.fold_index]
    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise ValueError(f"Group leakage detected for {label}")
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _stable_candidates_for_label(
    sae_features: np.ndarray,
    stable_core: pd.DataFrame,
    label: str,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    config: Task5PreparationConfig,
) -> pd.DataFrame:
    rows = stable_core[
        stable_core["label"].astype(str).eq(label)
        & stable_core["stable_set_role"].astype(str).eq("stable_core")
    ].copy()
    rows["latent_idx"] = pd.to_numeric(rows["latent_idx"], errors="coerce")
    rows = rows.dropna(subset=["latent_idx"])
    rows["latent_idx"] = rows["latent_idx"].astype(int)
    rows = rows[(rows["latent_idx"] >= 0) & (rows["latent_idx"] < sae_features.shape[1])]
    rows = rows.drop_duplicates("latent_idx").sort_values("latent_idx")
    if len(rows) < config.units_per_label:
        raise ValueError(f"{label} has fewer than {config.units_per_label} stable-core latents")

    latent_ids = rows["latent_idx"].to_numpy(dtype=np.int64)
    x_train = np.asarray(sae_features[np.ix_(train_idx, latent_ids)], dtype=np.float32)
    auc, cohens_d = compute_auc_and_cohens_d(
        x_train,
        y_train.astype(bool),
        chunk_size=config.association_chunk_size,
    )
    scored = pd.DataFrame(
        {
            "label": label,
            "latent_idx": latent_ids,
            "train_auc": auc,
            "train_cohens_d": cohens_d,
            "train_nonzero_count": (x_train != 0).sum(axis=0),
        }
    )
    scored = scored[(scored["train_auc"] > 0.5) & (scored["train_cohens_d"] > 0)].copy()
    scored = scored.sort_values(
        ["train_auc", "train_cohens_d", "latent_idx"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scored["train_rank"] = np.arange(1, len(scored) + 1)
    if len(scored) < config.units_per_label:
        raise ValueError(f"{label} has too few positive training-fold stable-core candidates")
    return scored


def _select_unique_sae_units(
    candidates: pd.DataFrame, config: Task5PreparationConfig
) -> pd.DataFrame:
    labels = list(config.labels)
    latent_ids = sorted(candidates["latent_idx"].astype(int).unique())
    slots = [label for label in labels for _ in range(config.units_per_label)]
    cost = np.full((len(slots), len(latent_ids)), 1e6, dtype=np.float64)
    lookup = {
        (str(row.label), int(row.latent_idx)): row
        for row in candidates.itertuples(index=False)
    }
    for row_idx, label in enumerate(slots):
        for col_idx, latent_idx in enumerate(latent_ids):
            row = lookup.get((label, latent_idx))
            if row is None:
                continue
            cost[row_idx, col_idx] = 1.0 - float(row.train_auc) + 1e-7 * float(row.train_rank)
    selected_rows, selected_cols = linear_sum_assignment(cost)
    if len(selected_rows) != len(slots) or np.any(cost[selected_rows, selected_cols] >= 1e5):
        raise ValueError("Unable to select globally unique SAE units for all label slots")

    records: list[dict[str, Any]] = []
    for slot_idx, col_idx in zip(selected_rows, selected_cols):
        label = slots[int(slot_idx)]
        latent_idx = int(latent_ids[int(col_idx)])
        row = lookup[(label, latent_idx)]
        records.append(
            {
                "label": label,
                "latent_idx": latent_idx,
                "train_auc": float(row.train_auc),
                "train_cohens_d": float(row.train_cohens_d),
                "train_nonzero_count": int(row.train_nonzero_count),
                "train_rank": int(row.train_rank),
            }
        )
    selected = pd.DataFrame(records)
    selected = selected.sort_values(
        ["label", "train_auc", "train_cohens_d", "latent_idx"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    selected["selection_order_within_label"] = selected.groupby("label").cumcount() + 1
    return selected


def _fit_pca_max(
    raw_hidden: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_dimension: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int | str]]:
    x_train = np.asarray(raw_hidden[train_idx], dtype=np.float32)
    x_test = np.asarray(raw_hidden[test_idx], dtype=np.float32)
    input_scaler = StandardScaler()
    x_train = input_scaler.fit_transform(x_train).astype(np.float32)
    x_test = input_scaler.transform(x_test).astype(np.float32)

    n_components = min(int(max_dimension), x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    train_scores = pca.fit_transform(x_train).astype(np.float32)
    test_scores = pca.transform(x_test).astype(np.float32)
    output_scaler = StandardScaler()
    train_scores = output_scaler.fit_transform(train_scores).astype(np.float32)
    test_scores = output_scaler.transform(test_scores).astype(np.float32)
    model = {
        "input_mean": input_scaler.mean_.astype(np.float32),
        "input_scale": input_scaler.scale_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "output_mean": output_scaler.mean_.astype(np.float32),
        "output_scale": output_scaler.scale_.astype(np.float32),
        "n_components": int(n_components),
        "svd_solver": "randomized",
        "random_state": int(random_state),
    }
    return train_scores, test_scores, model


def _pca_candidates(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    label: str,
    dimension: int,
    config: Task5PreparationConfig,
) -> pd.DataFrame:
    scores = np.asarray(train_scores[:, :dimension], dtype=np.float32)
    auc, cohens_d = compute_auc_and_cohens_d(
        scores,
        y_train.astype(bool),
        chunk_size=config.association_chunk_size,
    )
    sign = np.where(auc >= 0.5, 1, -1).astype(np.int8)
    oriented_auc = np.where(sign > 0, auc, 1.0 - auc)
    oriented_d = cohens_d * sign
    frame = pd.DataFrame(
        {
            "label": label,
            "pca_dimension": int(dimension),
            "component_idx": np.arange(dimension, dtype=int),
            "direction_sign": sign,
            "direction": np.where(sign > 0, "+", "-"),
            "raw_train_auc": auc,
            "raw_train_cohens_d": cohens_d,
            "train_auc": oriented_auc,
            "train_cohens_d": oriented_d,
        }
    )
    frame = frame[frame["train_auc"] > 0.5].copy()
    frame = frame.sort_values(
        ["train_auc", "train_cohens_d", "component_idx"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    frame["train_rank"] = np.arange(1, len(frame) + 1)
    return frame


def _match_units(
    sae_candidates: pd.DataFrame,
    pca_candidates: pd.DataFrame,
    label: str,
    dimension: int,
    config: Task5PreparationConfig,
) -> pd.DataFrame:
    sae = sae_candidates[sae_candidates["label"].eq(label)].reset_index(drop=True)
    pca = pca_candidates[
        pca_candidates["label"].eq(label)
        & pca_candidates["pca_dimension"].eq(dimension)
    ].reset_index(drop=True)
    if len(sae) < config.units_per_label or len(pca) < config.units_per_label:
        raise ValueError(f"Insufficient units to match {label}/PCA-{dimension}")

    pair_candidates: list[tuple[float, float, int, int, int, int]] = []
    for sae_idx, sae_row in sae.iterrows():
        for pca_idx, pca_row in pca.iterrows():
            auc_difference = abs(float(sae_row["train_auc"]) - float(pca_row["train_auc"]))
            matching_cost = auc_difference + config.matching_rank_penalty * (
                float(sae_row["train_rank"]) + float(pca_row["train_rank"])
            )
            pair_candidates.append(
                (
                    matching_cost,
                    auc_difference,
                    int(sae_row["train_rank"]),
                    int(pca_row["train_rank"]),
                    int(sae_idx),
                    int(pca_idx),
                )
            )
    pair_candidates.sort()
    chosen: list[tuple[float, float, int, int, int, int]] = []
    used_sae: set[int] = set()
    used_pca: set[int] = set()
    for candidate in pair_candidates:
        sae_idx = candidate[4]
        pca_idx = candidate[5]
        if sae_idx in used_sae or pca_idx in used_pca:
            continue
        chosen.append(candidate)
        used_sae.add(sae_idx)
        used_pca.add(pca_idx)
        if len(chosen) == config.units_per_label:
            break
    if len(chosen) != config.units_per_label:
        raise ValueError(f"Unable to build {config.units_per_label} pairs for {label}/PCA-{dimension}")

    records: list[dict[str, Any]] = []
    for pair_order, candidate in enumerate(chosen, start=1):
        matching_cost, auc_difference, _, _, sae_idx, pca_idx = candidate
        s = sae.iloc[sae_idx]
        p = pca.iloc[pca_idx]
        records.append(
            {
                "pca_dimension": int(dimension),
                "label": label,
                "pair_order": pair_order,
                "sae_latent_idx": int(s["latent_idx"]),
                "sae_train_auc": float(s["train_auc"]),
                "sae_train_cohens_d": float(s["train_cohens_d"]),
                "sae_train_rank": int(s["train_rank"]),
                "pca_component_idx": int(p["component_idx"]),
                "pca_direction": str(p["direction"]),
                "pca_direction_sign": int(p["direction_sign"]),
                "pca_train_auc": float(p["train_auc"]),
                "pca_train_cohens_d": float(p["train_cohens_d"]),
                "pca_train_rank": int(p["train_rank"]),
                "absolute_auc_difference": auc_difference,
                "matching_cost": matching_cost,
                "matching_rank_penalty": config.matching_rank_penalty,
                "within_auc_tolerance": auc_difference <= config.max_auc_difference,
                "selection_metric": "training_fold_raw_auc",
            }
        )
    return pd.DataFrame(records)


def _near_duplicate(text: str, selected: Iterable[str], threshold: float) -> bool:
    for other in selected:
        if text == other:
            return True
        if not text or not other:
            continue
        ratio = min(len(text), len(other)) / max(len(text), len(other))
        if ratio >= threshold and SequenceMatcher(None, text, other).ratio() >= threshold:
            return True
    return False


def _ranked_unique_rows(
    indices: np.ndarray,
    scores: np.ndarray,
    label_df: pd.DataFrame,
    count: int,
    excluded_texts: list[str],
    threshold: float,
) -> list[int]:
    order = np.lexsort((indices, -scores))
    chosen: list[int] = []
    seen = list(excluded_texts)
    for position in order:
        row_idx = int(indices[int(position)])
        text = normalize_text(label_df.iloc[row_idx]["unit_text"])
        if not text or _near_duplicate(text, seen, threshold):
            continue
        chosen.append(row_idx)
        seen.append(text)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Could only select {len(chosen)} of {count} unique examples")
    return chosen


def _control_rows(
    test_idx: np.ndarray,
    test_scores: np.ndarray,
    positives: list[int],
    label_df: pd.DataFrame,
    label: str,
    config: Task5PreparationConfig,
    excluded_texts: list[str],
) -> list[int]:
    low = float(np.quantile(test_scores, config.control_quantile_low))
    high = float(np.quantile(test_scores, config.control_quantile_high))
    candidate_mask = (test_scores >= low) & (test_scores <= high)
    candidate_rows = test_idx[candidate_mask]
    if len(candidate_rows) < config.heldout_control_count:
        candidate_rows = test_idx[test_scores <= np.median(test_scores)]
    score_map = {int(row): float(score) for row, score in zip(test_idx, test_scores)}
    used: set[int] = set(positives)
    seen = list(excluded_texts)
    chosen: list[int] = []

    for positive_idx in positives:
        positive = label_df.iloc[positive_idx]
        positive_words = len(str(positive["unit_text"]).split())
        positive_question = "?" in str(positive["unit_text"])
        ranked: list[tuple[tuple[Any, ...], int, str]] = []
        for row_idx in candidate_rows:
            row_idx = int(row_idx)
            if row_idx in used:
                continue
            row = label_df.iloc[row_idx]
            text = normalize_text(row["unit_text"])
            if not text or _near_duplicate(text, seen, config.near_duplicate_threshold):
                continue
            words = len(str(row["unit_text"]).split())
            question = "?" in str(row["unit_text"])
            key = (
                int(int(row[label]) != int(positive[label])),
                int(str(row.get("source_split", "")) != str(positive.get("source_split", ""))),
                int(question != positive_question),
                abs(words - positive_words),
                abs(score_map[row_idx]),
                row_idx,
            )
            ranked.append((key, row_idx, text))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0])
        _, row_idx, text = ranked[0]
        chosen.append(row_idx)
        used.add(row_idx)
        seen.append(text)
        if len(chosen) == config.heldout_control_count:
            break

    if len(chosen) != config.heldout_control_count:
        raise ValueError(
            f"Could only select {len(chosen)} of {config.heldout_control_count} controls"
        )
    return chosen


def _unit_examples(
    *,
    arm: str,
    blind_unit_id: str,
    representation: str,
    label: str,
    unit_id: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    label_df: pd.DataFrame,
    config: Task5PreparationConfig,
) -> list[dict[str, Any]]:
    discovery = _ranked_unique_rows(
        train_idx,
        train_scores,
        label_df,
        config.discovery_count,
        [],
        config.near_duplicate_threshold,
    )
    discovery_texts = [normalize_text(label_df.iloc[idx]["unit_text"]) for idx in discovery]
    heldout_positive = _ranked_unique_rows(
        test_idx,
        test_scores,
        label_df,
        config.heldout_positive_count,
        discovery_texts,
        config.near_duplicate_threshold,
    )
    positive_texts = [normalize_text(label_df.iloc[idx]["unit_text"]) for idx in heldout_positive]
    controls = _control_rows(
        test_idx,
        test_scores,
        heldout_positive,
        label_df,
        label,
        config,
        discovery_texts + positive_texts,
    )
    score_map_train = {int(row): float(score) for row, score in zip(train_idx, train_scores)}
    score_map_test = {int(row): float(score) for row, score in zip(test_idx, test_scores)}
    records: list[dict[str, Any]] = []
    for phase, truth_role, rows, score_map in (
        ("discovery", "discovery_top", discovery, score_map_train),
        ("heldout", "positive_top", heldout_positive, score_map_test),
        ("heldout", "control", controls, score_map_test),
    ):
        for source_order, row_idx in enumerate(rows, start=1):
            row = label_df.iloc[row_idx]
            records.append(
                {
                    "arm": arm,
                    "blind_unit_id": blind_unit_id,
                    "representation": representation,
                    "target_label": label,
                    "unit_id": unit_id,
                    "phase": phase,
                    "truth_role": truth_role,
                    "source_order": source_order,
                    "row_idx": int(row_idx),
                    "record_id": str(row.get("record_id", "")),
                    "file_id": str(row.get("file_id", "")),
                    "source_split": str(row.get("source_split", "")),
                    "target_label_value": int(row[label]),
                    "unit_score": float(score_map[row_idx]),
                    "unit_text": str(row["unit_text"]),
                    "text_normalized": normalize_text(row["unit_text"]),
                }
            )
    return records


def _blind_materials(
    inventory: pd.DataFrame, arm: str, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arm_rows = inventory[inventory["arm"].eq(arm)].copy()
    rng = np.random.default_rng(random_state)
    unit_ids = arm_rows["blind_unit_id"].drop_duplicates().to_numpy(dtype=object)
    rng.shuffle(unit_ids)
    unit_order = {str(unit_id): i + 1 for i, unit_id in enumerate(unit_ids)}

    discovery_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        unit_rows = arm_rows[arm_rows["blind_unit_id"].eq(unit_id)]
        for phase, destination in (
            ("discovery", discovery_records),
            ("heldout", heldout_records),
        ):
            phase_rows = unit_rows[unit_rows["phase"].eq(phase)].sample(
                frac=1.0,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            for example_order, row in enumerate(phase_rows.itertuples(index=False), start=1):
                destination.append(
                    {
                        "blind_unit_id": str(unit_id),
                        "unit_order": unit_order[str(unit_id)],
                        "example_order": example_order,
                        "sample_id": f"{unit_id}-{'D' if phase == 'discovery' else 'H'}{example_order:02d}",
                        "text": str(row.unit_text),
                    }
                )
    discovery = pd.DataFrame(discovery_records).sort_values(["unit_order", "example_order"])
    heldout = pd.DataFrame(heldout_records).sort_values(["unit_order", "example_order"])
    stage1 = discovery[["blind_unit_id", "unit_order"]].drop_duplicates().copy()
    stage1["semantic_coherence_1_5"] = ""
    stage1["one_sentence_interpretation"] = ""
    stage1["interpretation_type"] = ""
    stage1["behavioral_usefulness_1_5"] = ""
    stage1["surface_artifact_yes_no"] = ""
    stage1["confidence_1_5"] = ""
    stage1["reviewer_notes"] = ""
    stage2 = heldout[["blind_unit_id", "unit_order", "sample_id", "text"]].copy()
    stage2["instantiates_frozen_interpretation_yes_no"] = ""
    stage2["reviewer_notes"] = ""
    return discovery, heldout, stage1, stage2


def _validation_checks(
    split_rows: pd.DataFrame,
    matches: pd.DataFrame,
    inventory: pd.DataFrame,
    blind_files: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: Task5PreparationConfig,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for label in config.labels:
        sub = split_rows[split_rows["label"].eq(label)]
        train_groups = set(sub.loc[sub["split"].eq("train"), "file_id"])
        test_groups = set(sub.loc[sub["split"].eq("test"), "file_id"])
        checks.append(
            {
                "check": f"{label}_file_id_disjoint",
                "status": "PASS" if not train_groups.intersection(test_groups) else "FAIL",
                "value": len(train_groups.intersection(test_groups)),
            }
        )
    expected_matches = len(config.labels) * config.units_per_label * len(config.pca_dimensions)
    checks.append(
        {
            "check": "matched_pair_count",
            "status": "PASS" if len(matches) == expected_matches else "FAIL",
            "value": len(matches),
            "expected": expected_matches,
        }
    )
    excessive = matches["absolute_auc_difference"] > config.max_auc_difference
    checks.append(
        {
            "check": "all_pairs_within_auc_tolerance",
            "status": "PASS" if not excessive.any() else "FAIL",
            "value": int(excessive.sum()),
            "threshold": config.max_auc_difference,
            "maximum_observed": float(matches["absolute_auc_difference"].max()),
        }
    )
    for arm, (discovery, heldout) in blind_files.items():
        forbidden = {
            "representation", "target_label", "unit_id", "unit_score", "row_idx",
            "record_id", "file_id", "truth_role", "pca_dimension", "latent_idx",
        }
        leaked = sorted(forbidden.intersection(discovery.columns).union(forbidden.intersection(heldout.columns)))
        checks.append(
            {
                "check": f"{arm}_blind_column_whitelist",
                "status": "PASS" if not leaked else "FAIL",
                "value": leaked,
            }
        )
        unit_count = inventory[inventory["arm"].eq(arm)]["blind_unit_id"].nunique()
        expected_discovery = unit_count * config.discovery_count
        expected_heldout = unit_count * (
            config.heldout_positive_count + config.heldout_control_count
        )
        checks.extend(
            [
                {
                    "check": f"{arm}_discovery_count",
                    "status": "PASS" if len(discovery) == expected_discovery else "FAIL",
                    "value": len(discovery),
                    "expected": expected_discovery,
                },
                {
                    "check": f"{arm}_heldout_count",
                    "status": "PASS" if len(heldout) == expected_heldout else "FAIL",
                    "value": len(heldout),
                    "expected": expected_heldout,
                },
            ]
        )
    return {
        "overall_status": "PASS" if all(x["status"] == "PASS" for x in checks) else "FAIL",
        "checks": checks,
    }


def prepare_task5_materials(
    *,
    sae_features: np.ndarray,
    raw_hidden: np.ndarray,
    label_df: pd.DataFrame,
    stable_core: pd.DataFrame,
    output_dir: str | Path,
    config: Task5PreparationConfig | None = None,
) -> dict[str, Any]:
    config = config or Task5PreparationConfig()
    _validate_inputs(sae_features, raw_hidden, label_df, stable_core, config)
    output = Path(output_dir)
    private_dir = output / "private"
    blind_dir = output / "blind_materials"
    pending_dir = output / "release_pending"
    model_dir = private_dir / "pca_models"
    for path in (private_dir, blind_dir, pending_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_records: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    split_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in config.labels:
        train_idx, test_idx = _split_for_label(label_df, label, config)
        split_lookup[label] = (train_idx, test_idx)
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for row_idx in indices:
                split_records.append(
                    {
                        "label": label,
                        "row_idx": int(row_idx),
                        "file_id": str(label_df.iloc[int(row_idx)][config.group_column]),
                        "split": split_name,
                        "target_label_value": int(label_df.iloc[int(row_idx)][label]),
                    }
                )
        candidate_frames.append(
            _stable_candidates_for_label(
                sae_features,
                stable_core,
                label,
                train_idx,
                label_df.iloc[train_idx][label].astype(int).to_numpy(),
                config,
            )
        )
    split_rows = pd.DataFrame(split_records)
    sae_candidates = pd.concat(candidate_frames, ignore_index=True)
    pca_train_scores: dict[str, np.ndarray] = {}
    pca_test_scores: dict[str, np.ndarray] = {}
    pca_candidate_frames: list[pd.DataFrame] = []
    max_dimension = max(config.pca_dimensions)
    for label in config.labels:
        train_idx, test_idx = split_lookup[label]
        train_scores, test_scores, model = _fit_pca_max(
            raw_hidden,
            train_idx,
            test_idx,
            max_dimension=max_dimension,
            random_state=config.random_state,
        )
        pca_train_scores[label] = train_scores
        pca_test_scores[label] = test_scores
        np.savez_compressed(model_dir / f"{label}_pca_max{max_dimension}.npz", **model)
        y_train = label_df.iloc[train_idx][label].astype(int).to_numpy()
        for dimension in config.pca_dimensions:
            pca_candidate_frames.append(
                _pca_candidates(train_scores, y_train, label, int(dimension), config)
            )
    pca_candidates = pd.concat(pca_candidate_frames, ignore_index=True)

    match_frames = [
        _match_units(sae_candidates, pca_candidates, label, int(dimension), config)
        for dimension in config.pca_dimensions
        for label in config.labels
    ]
    matches = pd.concat(match_frames, ignore_index=True)
    matches = matches.sort_values(["pca_dimension", "label", "pair_order"]).reset_index(drop=True)
    selected_sae = matches[
        [
            "pca_dimension",
            "label",
            "sae_latent_idx",
            "sae_train_auc",
            "sae_train_cohens_d",
            "sae_train_rank",
        ]
    ].rename(
        columns={
            "sae_latent_idx": "latent_idx",
            "sae_train_auc": "train_auc",
            "sae_train_cohens_d": "train_cohens_d",
            "sae_train_rank": "train_rank",
        }
    )

    rng = np.random.default_rng(config.random_state + 5000)
    inventory_records: list[dict[str, Any]] = []
    mapping_records: list[dict[str, Any]] = []
    for dimension in config.pca_dimensions:
        arm = f"pca_{int(dimension)}"
        arm_matches = matches[matches["pca_dimension"].eq(int(dimension))]
        units: list[dict[str, Any]] = []
        for row in arm_matches.itertuples(index=False):
            units.extend(
                [
                    {
                        "representation": "SAE",
                        "label": str(row.label),
                        "unit_id": f"SAE:{int(row.sae_latent_idx)}",
                        "index": int(row.sae_latent_idx),
                        "direction_sign": 1,
                    },
                    {
                        "representation": f"PCA-{int(dimension)}",
                        "label": str(row.label),
                        "unit_id": f"PCA-{int(dimension)}:{row.label}:PC{int(row.pca_component_idx) + 1}{row.pca_direction}",
                        "index": int(row.pca_component_idx),
                        "direction_sign": int(row.pca_direction_sign),
                    },
                ]
            )
        unique_units = {unit["unit_id"]: unit for unit in units}
        unit_list = list(unique_units.values())
        blind_numbers = np.arange(1, len(unit_list) + 1)
        rng.shuffle(blind_numbers)
        for unit, blind_number in zip(unit_list, blind_numbers):
            blind_id = f"{arm.upper()}-U{int(blind_number):03d}"
            label = str(unit["label"])
            train_idx, test_idx = split_lookup[label]
            if unit["representation"] == "SAE":
                train_scores = np.asarray(sae_features[train_idx, int(unit["index"])], dtype=np.float32)
                test_scores = np.asarray(sae_features[test_idx, int(unit["index"])], dtype=np.float32)
            else:
                sign = int(unit["direction_sign"])
                train_scores = pca_train_scores[label][:, int(unit["index"])] * sign
                test_scores = pca_test_scores[label][:, int(unit["index"])] * sign
            mapping_records.append(
                {
                    "arm": arm,
                    "blind_unit_id": blind_id,
                    **unit,
                }
            )
            inventory_records.extend(
                _unit_examples(
                    arm=arm,
                    blind_unit_id=blind_id,
                    representation=str(unit["representation"]),
                    label=label,
                    unit_id=str(unit["unit_id"]),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_scores=train_scores,
                    test_scores=test_scores,
                    label_df=label_df,
                    config=config,
                )
            )
    inventory = pd.DataFrame(inventory_records)
    blind_mapping = pd.DataFrame(mapping_records).sort_values(["arm", "blind_unit_id"])

    split_path = private_dir / "split_assignments.csv"
    sae_candidates_path = private_dir / "sae_candidates_train.csv"
    selected_sae_path = private_dir / "selected_sae_units.csv"
    pca_candidates_path = private_dir / "pca_candidates_train.csv"
    matches_path = private_dir / "matched_units.csv"
    mapping_path = private_dir / "blind_mapping_key.csv"
    inventory_path = private_dir / "example_inventory.csv"
    split_rows.to_csv(split_path, index=False)
    sae_candidates.to_csv(sae_candidates_path, index=False)
    selected_sae.to_csv(selected_sae_path, index=False)
    pca_candidates.to_csv(pca_candidates_path, index=False)
    matches.to_csv(matches_path, index=False)
    blind_mapping.to_csv(mapping_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    future_review = (
        selected_sae.groupby(["label", "latent_idx"], as_index=False)
        .agg(
            train_auc=("train_auc", "max"),
            train_cohens_d=("train_cohens_d", "max"),
            best_train_rank=("train_rank", "min"),
            pca_dimensions=("pca_dimension", lambda values: ",".join(str(int(x)) for x in sorted(set(values)))),
        )
    )
    future_review["review_status"] = "pending_human_review"
    future_review["reason"] = "selected_for_task5_matched_sae_pca_evaluation"
    future_review_path = output / "selected_sae_for_future_b_review.csv"
    future_review.to_csv(future_review_path, index=False)

    blind_file_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for arm_index, dimension in enumerate(config.pca_dimensions):
        arm = f"pca_{int(dimension)}"
        arm_blind_dir = blind_dir / arm
        arm_pending_dir = pending_dir / arm
        arm_blind_dir.mkdir(parents=True, exist_ok=True)
        arm_pending_dir.mkdir(parents=True, exist_ok=True)
        discovery, heldout, stage1, stage2 = _blind_materials(
            inventory,
            arm,
            config.random_state + 100 * (arm_index + 1),
        )
        discovery.to_csv(arm_blind_dir / "stage1_discovery_examples.csv", index=False)
        stage1.to_csv(arm_blind_dir / "stage1_annotation_template.csv", index=False)
        heldout.to_csv(arm_pending_dir / "stage2_heldout_examples.csv", index=False)
        stage2.to_csv(arm_pending_dir / "stage2_annotation_template.csv", index=False)
        blind_file_frames[arm] = (discovery, heldout)

    validation = _validation_checks(
        split_rows,
        matches,
        inventory,
        blind_file_frames,
        config,
    )
    validation_path = output / "preparation_validation.json"
    _write_json(validation_path, validation)
    if validation["overall_status"] != "PASS":
        raise ValueError("Task 5 preparation validation failed")

    report_path = output / "task5_materials_preparation_report.md"
    report_lines = [
        "# Task 5 SAE/PCA Matched Human-Evaluation Materials",
        "",
        "## Status",
        "",
        "`MATERIALS_PREPARED_PENDING_HUMAN_REVIEW`",
        "",
        "No interpretation, reviewer rating, held-out judgment, adjudication, or inferential result has been generated.",
        "",
        "## Frozen computational setup",
        "",
        f"- Labels: `{', '.join(config.labels)}`",
        f"- PCA variants: `{', '.join(f'PCA-{x}' for x in config.pca_dimensions)}`",
        f"- Split: StratifiedGroupKFold by `{config.group_column}`, folds={config.folds}, fold={config.fold_index}, seed={config.random_state}",
        "- PCA preprocessing: training-fold input scaling, PCA fit, then training-fold post-PCA scaling.",
        "- PCA direction: orient each component using training-fold raw AUC; negative directions are multiplied by -1.",
        "- Matching: minimum absolute training-fold AUC difference, with unique PCA directions within each label/variant.",
        f"- Matching cost: absolute AUC difference + `{config.matching_rank_penalty}` times the two training-fold ranks.",
        f"- Required AUC tolerance: every pair must have absolute difference <= `{config.max_auc_difference}`.",
        "- Unit selection does not use Task 4 card names, explanation types, or apparent interpretability.",
        "",
        "## Counts",
        "",
        f"- Selected unique SAE units across both variants: `{selected_sae['latent_idx'].nunique()}`",
        f"- Matched pairs across both PCA variants: `{len(matches)}`",
    ]
    for dimension in config.pca_dimensions:
        arm = f"pca_{int(dimension)}"
        unit_count = inventory[inventory["arm"].eq(arm)]["blind_unit_id"].nunique()
        report_lines.append(f"- `{arm}` anonymous units: `{unit_count}`")
    report_lines.extend(
        [
            "",
            "## Human-stage boundary",
            "",
            "Only `blind_materials/<arm>/stage1_*` may be released initially. Files under `release_pending/` must remain hidden until each reviewer has frozen the stage-1 interpretation.",
            "The private mapping, association scores, labels, unit IDs, and held-out truth roles must not be accessible to annotators.",
            "",
            "The existing Stable Core registry was frozen before Task 5 but was derived using the full analysis dataset. Therefore the held-out materials test interpretation generalization, not a new confirmatory validation of Stable Core label association.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = {
        "analysis": "task5_matched_sae_pca_human_eval_preparation",
        "status": "materials_prepared_pending_human_review",
        "config": asdict(config),
        "counts": {
            "selected_unique_sae_units": int(selected_sae["latent_idx"].nunique()),
            "selected_sae_label_associations": int(
                selected_sae[["label", "latent_idx"]].drop_duplicates().shape[0]
            ),
            "matched_pairs": int(len(matches)),
            "inventory_rows": int(len(inventory)),
            "arms": {
                arm: {
                    "units": int(inventory[inventory["arm"].eq(arm)]["blind_unit_id"].nunique()),
                    "discovery_rows": int((inventory["arm"].eq(arm) & inventory["phase"].eq("discovery")).sum()),
                    "heldout_rows": int((inventory["arm"].eq(arm) & inventory["phase"].eq("heldout")).sum()),
                }
                for arm in sorted(inventory["arm"].unique())
            },
        },
        "outputs": {
            "matches": str(matches_path),
            "mapping_key": str(mapping_path),
            "example_inventory": str(inventory_path),
            "future_b_review": str(future_review_path),
            "validation": str(validation_path),
            "report": str(report_path),
        },
        "human_stage": {
            "interpretations_generated": False,
            "reviewer_ratings_generated": False,
            "heldout_judgments_generated": False,
            "statistical_comparison_run": False,
        },
    }
    _write_json(manifest_path, manifest)

    checksum_path = output / "SHA256SUMS.txt"
    checksum_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output).as_posix()}" for path in checksum_files)
        + "\n",
        encoding="utf-8",
    )
    return {
        "selected_sae": selected_sae,
        "matches": matches,
        "inventory": inventory,
        "validation": validation,
        "manifest": manifest,
    }
