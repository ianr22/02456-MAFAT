import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# VARIABLES
run_path = "run_117542"

if len(sys.argv) < 2:
    print("Filepath not entered, defaulting to 117542")
else:
    run_path = sys.argv[1]

# ========= 1. Load JSON file =========
# Change this path to your actual JSON file

json_path = os.path.join(run_path, "predictions.json")
json_path = Path(json_path)

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} image records from run {run_path}")


# ========= 2. Collect all unique labels =========
all_labels = set()

for rec in records:
    all_labels.update(rec["predictions"])
    all_labels.update(rec["ground_truth"])

all_labels = sorted(all_labels)
print(f"Found {len(all_labels)} unique labels")


# ========= 3. Compute per-class TP, FP, FN, TN =========
per_class = {
    label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for label in all_labels
}

for rec in records:
    preds = set(rec["predictions"])
    gts = set(rec["ground_truth"])
    for label in all_labels:
        in_pred = label in preds
        in_gt = label in gts

        if in_pred and in_gt:
            per_class[label]["tp"] += 1
        elif in_pred and not in_gt:
            per_class[label]["fp"] += 1
        elif (not in_pred) and in_gt:
            per_class[label]["fn"] += 1
        else:
            per_class[label]["tn"] += 1


# ========= 4. Helper to compute precision / recall / F1 =========
def prf(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ========= 5. Per-class metrics + "AP" and mAP =========
per_class_metrics = {}
ap_values = []  # we'll define AP(label) = precision(label) at this single threshold

for label in all_labels:
    stats = per_class[label]
    precision, recall, f1 = prf(stats)
    support = stats["tp"] + stats["fn"]  # number of images where label is actually present

    per_class_metrics[label] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "tp": stats["tp"],
        "fp": stats["fp"],
        "fn": stats["fn"],
        "tn": stats["tn"],
    }

    # Only include labels that actually appear in ground truth in the mAP
    if support > 0:
        # NOTE: with only hard predictions (no scores), AP collapses to precision at this threshold
        ap_values.append(precision)

# Mean Average Precision (mAP) in this hard-prediction setting
mAP = sum(ap_values) / len(ap_values) if ap_values else 0.0

print("\nPer-class metrics:")
for label, m in per_class_metrics.items():
    print(
        f"{label:35s} | "
        f"prec={m['precision']:.3f}  "
        f"rec={m['recall']:.3f}  "
        f"f1={m['f1']:.3f}  "
        f"support={m['support']}"
    )

print(f"\nmAP (mean per-class precision over labels with support > 0): {mAP:.4f}")


# ========= 6. (Optional) micro-averaged metrics over all labels =========
total_tp = sum(d["tp"] for d in per_class.values())
total_fp = sum(d["fp"] for d in per_class.values())
total_fn = sum(d["fn"] for d in per_class.values())

micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
micro_f1 = (
    2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    if (micro_precision + micro_recall) > 0
    else 0.0
)

print(
    f"\nMicro-averaged over all labels:"
    f"\n  precision = {micro_precision:.4f}"
    f"\n  recall    = {micro_recall:.4f}"
    f"\n  f1        = {micro_f1:.4f}"
)
