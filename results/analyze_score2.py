import json
from pathlib import Path

# ------------------------------
# Load predictions JSON
# ------------------------------
def load_predictions(json_path):
    """Load predictions_with_scores.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# AP for a single label (pure Python)
# ------------------------------
def compute_ap_for_label(records, label):
    """
    Compute AP for a given label using the same formula as analyze_score.py,
    but implemented without NumPy.
    """
    pairs = []   # list of (score, is_positive)
    positives = 0

    for rec in records:
        gt = set(rec["true"])
        is_pos = 1 if label in gt else 0
        score = rec["pred_scores"].get(label, 0.0)

        pairs.append((score, is_pos))
        if is_pos:
            positives += 1

    if positives == 0:
        return None  # label never appears in GT

    # Sort by score descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    tp_cum = 0
    k = 0
    ap_sum = 0.0

    # Precision at each k where the item is positive
    for score, is_pos in pairs:
        k += 1
        if is_pos:
            tp_cum += 1
            precision = tp_cum / k
            ap_sum += precision

    return ap_sum / positives


# ------------------------------
# Grouped mAP helper
# ------------------------------
def compute_group_map(group_labels, aps_dict):
    valid = [aps_dict[lbl] for lbl in group_labels if lbl in aps_dict]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


# ------------------------------
# Compute total and grouped mAP
# ------------------------------
def compute_metrics(records):
    # Collect all labels
    all_labels = set()
    for rec in records:
        all_labels.update(rec["pred_scores"].keys())
    all_labels = sorted(all_labels)

    # AP per label
    aps = {}
    for label in all_labels:
        ap = compute_ap_for_label(records, label)
        if ap is not None:
            aps[label] = ap

    if aps:
        total_map = sum(aps.values()) / len(aps)
    else:
        total_map = 0.0

    # Group definitions (same as analyze_score.py)
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
        "general":  compute_group_map(general_class_labels, aps),
        "subclass": compute_group_map(sub_class_labels, aps),
        "color":    compute_group_map(color_labels, aps),
        "features": compute_group_map(feature_labels, aps),
    }

    return total_map, grouped


# ------------------------------
# Process one prediction root folder
# ------------------------------
def process_prediction_folder(base_dir):
    """
    base_dir: e.g. 'results/balanced_prediction' or 'results/training_prediction'
    Auto-detects blur_* subfolders and computes mAP for each.
    """
    base_dir = Path(base_dir)
    results = {}

    if not base_dir.exists():
        print(f"WARNING: {base_dir} does not exist, skipping.")
        return results

    for subdir in sorted(base_dir.iterdir()):
        if not (subdir.is_dir() and subdir.name.startswith("blur_")):
            continue

        json_path = subdir / "predictions_with_scores.json"
        if not json_path.exists():
            print(f"Skipping {subdir.name}: no predictions_with_scores.json")
            continue

        print(f"Processing: {subdir}")

        records = load_predictions(json_path)
        total, grouped = compute_metrics(records)

        # blur_0_run_142623 -> blur0
        parts = subdir.name.split("_")
        if len(parts) >= 2 and parts[0] == "blur":
            blur_level = parts[1]
        else:
            blur_level = subdir.name  # fallback

        blur_key = f"blur{blur_level}"

        results[blur_key] = {
            "mAP": {
                "total":    total,
                "general":  grouped["general"],
                "subclass": grouped["subclass"],
                "color":    grouped["color"],
                "features": grouped["features"],
            }
        }

    return results


# ------------------------------
# Save JSON
# ------------------------------
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {output_path}")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    prediction_folders = [
        "results/balanced_prediction",
        "results/training_prediction",
    ]

    for folder in prediction_folders:
        print(f"\n=== Processing {folder} ===")
        result = process_prediction_folder(folder)

        out_name = Path(folder).name + "_scores.json"
        save_json(result, out_name)

