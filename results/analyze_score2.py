import os
import json
from pathlib import Path
import numpy as np
from analyze_score import compute_ap_for_label, compute_group_map


def load_predictions(json_path):
    """Load predictions_with_scores.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(records):
    """Compute total mAP and grouped mAP using analyze_score logic."""

    # Collect all labels
    all_labels = set()
    for rec in records:
        all_labels.update(rec["pred_scores"].keys())
    all_labels = sorted(all_labels)

    # Compute AP for each label
    aps = {}
    for label in all_labels:
        ap = compute_ap_for_label(records, label)
        if ap is not None:
            aps[label] = ap

    total_map = float(np.mean(list(aps.values()))) if aps else 0.0

    # Groups
    general_class_labels = [lbl for lbl in all_labels if lbl.startswith("general_class_")]
    sub_class_labels     = [lbl for lbl in all_labels if lbl.startswith("sub_class_")]
    color_labels         = [lbl for lbl in all_labels if lbl.startswith("color_")]
    feature_labels       = [
        lbl for lbl in all_labels
        if lbl not in general_class_labels
        and lbl not in sub_class_labels
        and lbl not in color_labels
    ]

    grouped = {
        "general": compute_group_map(general_class_labels, aps),
        "subclass": compute_group_map(sub_class_labels, aps),
        "color": compute_group_map(color_labels, aps),
        "features": compute_group_map(feature_labels, aps),
    }

    return total_map, grouped


def process_prediction_folder(base_dir):
    """Scan blur folders and compute mAP for each."""

    base_dir = Path(base_dir)
    results = {}

    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("blur_"):

            json_path = subdir / "predictions_with_scores.json"
            if not json_path.exists():
                print(f"Skipping {subdir.name}: no predictions_with_scores.json")
                continue

            print(f"Processing: {subdir.name}")

            # Load json and compute metrics
            records = load_predictions(json_path)
            total, grouped = compute_metrics(records)

            # blur_0_run_142623 -> blur0
            blur_level = subdir.name.split("_")[1]
            blur_key = f"blur{blur_level}"

            results[blur_key] = {
                "mAP": {
                    "total": total,
                    "general": grouped["general"],
                    "subclass": grouped["subclass"],
                    "color": grouped["color"],
                    "features": grouped["features"],
                }
            }

    return results


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    prediction_folders = [
        "results/balanced_prediction",
        "results/training_prediction"
    ]

    for folder in prediction_folders:
        print(f"\n=== Processing {folder} ===")
        result = process_prediction_folder(folder)

        out_name = Path(folder).name + "_scores.json"
        save_json(result, out_name)

