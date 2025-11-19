import json
import os
import sys
from pathlib import Path

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ==============================================================
# 0. Small helpers
# ==============================================================

def short_label(label: str) -> str:
    """
    Make long class names shorter & more readable.
    Examples:
      'sub_class_minivan'              -> 'minivan'
      'general_class_small vehicle'    -> 'small vehicle'
      'color_silver/grey'              -> 'silver/grey'
    """
    prefixes = ["general_class_", "sub_class_", "color_"]
    for p in prefixes:
        if label.startswith(p):
            return label[len(p):]
    return label


def safe_filename(label: str) -> str:
    """Turn a label into something safe to use as a filename."""
    # Replace problematic chars with underscore
    import re
    return re.sub(r"[^0-9a-zA-Z\-]+", "_", label)


# ==============================================================
# 1. Load JSON
# ==============================================================

# usage:
#   python confusion_matric_scatter.py            # defaults to predictions.json in this folder
#   python confusion_matric_scatter.py myfile.json
#   python confusion_matric_scatter.py /abs/path/to/predictions.json

if len(sys.argv) < 2:
    print("Filepath not entered, defaulting to predictions.json in this directory")
    json_path = Path("predictions.json")
else:
    json_path = Path(sys.argv[1])

if not json_path.is_file():
    raise FileNotFoundError(f"Could not find JSON file at: {json_path.resolve()}")

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} image records from {json_path}")


# ==============================================================
# 2. Collect all unique labels
# ==============================================================

all_labels = set()

for rec in records:
    all_labels.update(rec["predictions"])
    all_labels.update(rec["ground_truth"])

all_labels = sorted(all_labels)
print(f"Found {len(all_labels)} unique labels")


# ==============================================================
# 3. Compute TP/FP/FN/TN per class
# ==============================================================

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


# ==============================================================
# 4. Metrics helper
# ==============================================================

def prf(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


# ==============================================================
# 5. Compute metrics for each class
# ==============================================================

per_class_metrics = {}
precisions = []
recalls = []
f1s = []

for label in all_labels:
    stats = per_class[label]
    p, r, f1 = prf(stats)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)
    support = stats["tp"] + stats["fn"]

    per_class_metrics[label] = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
        "tp": stats["tp"],
        "fp": stats["fp"],
        "fn": stats["fn"],
        "tn": stats["tn"],
    }

# Optional: print a small table to stdout
print("\nPer-class metrics:")
for label, m in per_class_metrics.items():
    print(
        f"{short_label(label):25s} | "
        f"prec={m['precision']:.3f}  "
        f"rec={m['recall']:.3f}  "
        f"f1={m['f1']:.3f}  "
        f"support={m['support']}"
    )


# ==============================================================
# 6. Draw per-label confusion matrices
# ==============================================================

def draw_confusion_matrix(label, stats):
    """
    Produces a confusion matrix:
         [[TN, FP],
          [FN, TP]]
    for one label.
    """
    cm = np.array([
        [stats["tn"], stats["fp"]],
        [stats["fn"], stats["tp"]],
    ])

    nice_label = short_label(label)
    fname_label = safe_filename(label)

    plt.figure(figsize=(3.5, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["GT 0", "GT 1"],
    )
    plt.title(f"{nice_label}")
    plt.tight_layout()
    plt.savefig(f"cm_{fname_label}.png", dpi=300)


print("\nSaving per-class confusion matrices...")
for label, stats in per_class.items():
    draw_confusion_matrix(label, stats)
print("  -> cm_<label>.png files created.")


# ==============================================================
# 7. Grouped plots with more readable labels
# ==============================================================

def label_group(lbl: str) -> str:
    if lbl.startswith("general_class_"):
        return "general_class"
    elif lbl.startswith("sub_class_"):
        return "sub_class"
    elif lbl.startswith("color_"):
        return "color"
    else:
        return "other"


from collections import defaultdict

group_to_indices = defaultdict(list)
for i, lbl in enumerate(all_labels):
    g = label_group(lbl)
    group_to_indices[g].append(i)

# ---------- Precision vs Recall (scatter per group) ----------
np.random.seed(0)

for group_name, idxs in group_to_indices.items():
    if not idxs:
        continue

    plt.figure(figsize=(8, 6))
    for j, i in enumerate(idxs):
        x = precisions[i]
        y = recalls[i]
        # tiny horizontal jitter so points with same precision don't fully overlap
        jitter = (j % 5 - 2) * 0.002
        xj = x + jitter
        plt.scatter(xj, y, s=20)

        plt.annotate(
            short_label(all_labels[i]),
            (xj, y),
            textcoords="offset points",
            xytext=(4, (j % 7 - 3) * 2),  # small vertical staggering
            ha="left",
            fontsize=7,
        )

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"Precision vs Recall – {group_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"scatter_precision_recall_{group_name}.png", dpi=150)
    plt.close()

print("Saved grouped P-R plots: scatter_precision_recall_<group>.png")

# ---------- F1 per label (horizontal bar chart per group) ----------

for group_name, idxs in group_to_indices.items():
    if not idxs:
        continue

    labels = [short_label(all_labels[i]) for i in idxs]
    scores = [f1s[i] for i in idxs]

    # sort by F1 so bars are nicely ordered
    order = np.argsort(scores)
    scores = [scores[k] for k in order]
    labels = [labels[k] for k in order]

    plt.figure(figsize=(8, max(4, 0.25 * len(labels))))  # height scales with #labels
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, labels, fontsize=7)
    plt.xlabel("F1 Score")
    plt.title(f"F1 per Label – {group_name}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"bar_f1_{group_name}.png", dpi=150)
    plt.close()

print("Saved grouped F1 bar charts: bar_f1_<group>.png")
print("Done.")