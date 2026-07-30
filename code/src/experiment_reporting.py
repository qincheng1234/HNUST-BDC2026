"""Read-only reporting utilities for feature-ablation experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASELINE_OOF = {
    "mean_return": 0.011834094740657344,
    "robust_score": 0.002803478328340679,
    "worst_fold_return": -0.037439,
}


def _selected_fold_scores(metadata):
    selected_epoch = int(metadata["selected_epochs"])
    score_index = selected_epoch - 1
    scores = []
    for fold in metadata.get("oof_folds", []):
        epoch_scores = fold.get("epoch_scores", [])
        if score_index >= len(epoch_scores):
            raise ValueError("OOF fold does not contain the selected epoch score")
        scores.append(float(epoch_scores[score_index]))
    if not scores:
        raise ValueError("model metadata contains no OOF fold scores")
    return scores


def build_ablation_row(metadata_path, baseline=BASELINE_OOF):
    """Create one comparable OOF summary row from a model metadata artifact."""
    metadata_path = Path(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    fold_scores = _selected_fold_scores(metadata)
    selection = metadata.get("epoch_selection", {})
    mean_return = float(metadata["oof_top5_mean_return"])
    robust_score = float(selection["robust_score"])
    worst_fold_return = min(fold_scores)
    passes_gate = (
        robust_score > baseline["robust_score"]
        and mean_return >= baseline["mean_return"]
        and worst_fold_return >= baseline["worst_fold_return"]
    )
    return {
        "experiment": metadata.get("feature_experiment", "A0"),
        "model_type": metadata["model_type"],
        "source_feature_set": metadata.get("source_feature_set", "158+39"),
        "feature_count": int(metadata.get("feature_count", len(metadata["features"]))),
        "selected_epochs": int(metadata["selected_epochs"]),
        "oof_top5_mean_return": mean_return,
        "oof_top5_std": float(metadata["oof_top5_std"]),
        "oof_robust_score": robust_score,
        "worst_fold_return": worst_fold_return,
        "positive_fold_rate": sum(score > 0 for score in fold_scores) / len(fold_scores),
        "passes_a_gate": passes_gate,
        "artifact_path": str(metadata_path),
    }


def _latest_metadata_by_experiment(model_root, experiments):
    latest = {}
    for metadata_path in Path(model_root).rglob("model_meta.json"):
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            experiment = metadata.get("feature_experiment", "A0")
        except (OSError, json.JSONDecodeError):
            continue
        if experiment not in experiments:
            continue
        previous = latest.get(experiment)
        if previous is None or metadata_path.stat().st_mtime > previous.stat().st_mtime:
            latest[experiment] = metadata_path
    return latest


def write_ablation_summary(model_root, output_path, experiments=("A1", "A2", "A3")):
    """Write an OOF-only table for the latest completed artifact per experiment."""
    experiments = tuple(experiments)
    metadata_paths = _latest_metadata_by_experiment(model_root, set(experiments))
    rows = []
    for experiment in experiments:
        metadata_path = metadata_paths.get(experiment)
        if metadata_path is None:
            rows.append({"experiment": experiment, "status": "missing_artifact"})
            continue
        try:
            row = build_ablation_row(metadata_path)
            row["status"] = "completed"
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            row = {
                "experiment": experiment,
                "status": "invalid_artifact",
                "error": str(error),
                "artifact_path": str(metadata_path),
            }
        rows.append(row)

    fieldnames = [
        "experiment",
        "status",
        "model_type",
        "source_feature_set",
        "feature_count",
        "selected_epochs",
        "oof_top5_mean_return",
        "oof_top5_std",
        "oof_robust_score",
        "worst_fold_return",
        "positive_fold_rate",
        "passes_a_gate",
        "artifact_path",
        "error",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows
