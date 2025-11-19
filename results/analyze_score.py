import json
import os
import sys
from pathlib import Path
import numpy as np

# =============================
# VARIABLES & RUN PATH HANDLING
# =============================

# Default run folder
run_path = "run_124216"

# If user passes a run folder, override default
if len(sys.argv) >= 2:
    run_path = sys.argv[1]
else:
    print("Filepath not entered, defaulting to 124216")

# Path: run_X/predictions_with_scores.json (relative to current dir)
json_path = Path(os.path.join(run_path, "predictions_with_scores.json"))
print(json_path)

print(f"Loading JSON from: {json_path}")

if not json_path.exists():
    print(f"ERROR: File not found at {json_path}")
    sys.exit(1)

# -------------------------------
# Load JSON
# -------------------------------
with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} image records from run {run_path}")

# -------------------------------
# Collect all labels
# -------------------------------
all_labels = set()
for rec in records:
    all_labels.update(rec["pred_scores"].keys())

all_labels = sorted(all_labels)
print(f"Found {len(all_labels)} labels")


# -------------------------------
# Compute AP for a single label
# -------------------------------
def compute_ap_for_label(records, label):
    scores = []
    y_true = []

    for rec in records:
        gt = set(rec["true"])
        score = rec["pred_scores"].get(label, 0.0)
        scores.append(score)
        y_true.append(1 if label in gt else 0)

    scores = np.array(scores)
    y_true = np.array(y_true)

    K = y_true.sum()
    if K == 0:
        return None  # label never appears in GT

    # Sort by score descending
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    # Precision at each k
    tp_cumsum = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    prec_at_k = tp_cumsum / k

    # rel(k) = 1 for positives at each rank
    rel_k = y_sorted

    # AP formula from paper
    AP = (prec_at_k * rel_k).sum() / K
    return float(AP)


# -------------------------------
# Compute mAP across all labels
# -------------------------------
aps = {}
for label in all_labels:
    ap = compute_ap_for_label(records, label)
    if ap is not None:
        aps[label] = ap

mAP = np.mean(list(aps.values()))
print("\n==============================")
print(f"   TRUE mAP = {mAP:.6f}")
print("==============================\n")

# Optional: print per-label AP
for lbl, ap in sorted(aps.items()):
    print(f"{lbl:40s}  AP = {ap:.6f}")


# --------------------------------------------------------
# GROUP DEFINITIONS (COFGA categories)
# --------------------------------------------------------
general_class_labels = [lbl for lbl in all_labels if lbl.startswith("general_class_")]
sub_class_labels     = [lbl for lbl in all_labels if lbl.startswith("sub_class_")]
color_labels         = [lbl for lbl in all_labels if lbl.startswith("color_")]

# Everything NOT in those 3 groups = feature labels
feature_labels = [
    lbl for lbl in all_labels
    if lbl not in general_class_labels
    and lbl not in sub_class_labels
    and lbl not in color_labels
]

groups = {
    "General Class": general_class_labels,
    "Sub-Class": sub_class_labels,
    "Color": color_labels,
    "Features": feature_labels,
}


# --------------------------------------------------------
# Compute mAP per group
# --------------------------------------------------------
def compute_group_map(group_labels, aps_dict):
    valid_aps = [aps_dict[lbl] for lbl in group_labels if lbl in aps_dict]
    if len(valid_aps) == 0:
        return 0.0
    return float(np.mean(valid_aps))


print("\n==============================")
print("       GROUPED mAP")
print("==============================")

for group_name, label_list in groups.items():
    gmap = compute_group_map(label_list, aps)
    print(f"{group_name:15s}  mAP = {gmap:.6f}  (n={len(label_list)})")
