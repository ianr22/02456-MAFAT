import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# save directory
custom_save_dir = "/home/junwoo/cmda4864_capstones/02456-MAFAT/confusion_matric_scatter"
os.makedirs(custom_save_dir, exist_ok=True)

# helper functions

def short_label(label: str) -> str:
    prefixes = ["general_class_", "sub_class_", "color_"]
    for p in prefixes:
        if label.startswith(p):
            return label[len(p):]
    return label

def safe_filename(label: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z\-]+", "_", label)


# loading JSON file

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

# Collect all unique labels

all_labels = set()
for rec in records:
    all_labels.update(rec["predictions"])
    all_labels.update(rec["ground_truth"])

all_labels = sorted(all_labels)
print(f"Found {len(all_labels)} unique labels")


# compute TP/FP/FN/TN per class


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
    
# Metrics helper


def prf(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1

# Compute metrics for each class


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

print("\nPer-class metrics:")
for label, m in per_class_metrics.items():
    print(
        f"{short_label(label):25s} | "
        f"prec={m['precision']:.3f}  "
        f"rec={m['recall']:.3f}  "
        f"f1={m['f1']:.3f}  "
        f"support={m['support']}"
    )

# Draw per-label confusion matrices

def draw_confusion_matrix(label, stats):
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
    save_path = os.path.join(custom_save_dir, f"cm_{fname_label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("\nSaving per-class confusion matrices...")
for label, stats in per_class.items():
    draw_confusion_matrix(label, stats)
print("  -> cm_<label>.png files created.")

# Grouped plots with more readable labels


def label_group(lbl: str) -> str:
    if lbl.startswith("general_class_"):
        return "general_class"
    elif lbl.startswith("sub_class_"):
        return "sub_class"
    elif lbl.startswith("color_"):
        return "color"
    else:
        return "other"

group_to_indices = defaultdict(list)
for i, lbl in enumerate(all_labels):
    g = label_group(lbl)
    group_to_indices[g].append(i)

# combined top 10 confusion matrix

gt_labels = []
pred_labels = []
for rec in records:
    gt_labels.append(rec["ground_truth"][0])
    pred_labels.append(rec["predictions"][0])

label_order = all_labels

cm = confusion_matrix(gt_labels, pred_labels, labels=label_order)
row_sums = cm.sum(axis=1)
top_indices = np.argsort(row_sums)[-10:]  # largest supports (last 10)

cm_top = cm[top_indices][:, top_indices]
row_sums_top = row_sums[top_indices]
cm_normalized_top = np.divide(cm_top, row_sums_top[:, np.newaxis], where=row_sums_top[:, np.newaxis]!=0)
label_order_top = [short_label(label_order[i]) for i in top_indices]

annot_top = np.empty_like(cm_top, dtype='object')
for i in range(cm_top.shape[0]):
    for j in range(cm_top.shape[1]):
        percent = 100.0 * cm_normalized_top[i, j] if row_sums_top[i] else 0
        annot_top[i, j] = f"{cm_top[i, j]}\n({percent:.1f}%)" if cm_top[i, j] > 0 else ""

plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm_normalized_top, annot=annot_top, fmt='', cmap='Blues',
                 xticklabels=label_order_top,
                 yticklabels=label_order_top,
                 cbar_kws={'label': 'Proportion'},
                 linewidths=0.5, linecolor='gray',
                 annot_kws={'size': 13})
ax.set_xlabel("Predicted Label", fontsize=16)
ax.set_ylabel("True Label", fontsize=16)
ax.set_title("Top 10 Classes: Integrated Confusion Matrix", fontsize=18, pad=30)
plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
save_path = os.path.join(custom_save_dir, "confusion_matrix_top10.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved top-10 integrated confusion matrix: {save_path}")

# Precision vs Recall (scatter per group) 

np.random.seed(0)
for group_name, idxs in group_to_indices.items():
    if not idxs:
        continue
    plt.figure(figsize=(8, 6))
    for j, i in enumerate(idxs):
        x = precisions[i]
        y = recalls[i]
        jitter = (j % 5 - 2) * 0.002
        xj = x + jitter
        plt.scatter(xj, y, s=20)
        plt.annotate(
            short_label(all_labels[i]),
            (xj, y),
            textcoords="offset points",
            xytext=(4, (j % 7 - 3) * 2),
            ha="left",
            fontsize=7,
        )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"Precision vs Recall – {group_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(custom_save_dir, f"scatter_precision_recall_{group_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
print("Saved grouped P-R plots: scatter_precision_recall_<group>.png")

# F1 per label (horizontal bar chart per group)

for group_name, idxs in group_to_indices.items():
    if not idxs:
        continue
    labels = [short_label(all_labels[i]) for i in idxs]
    scores = [f1s[i] for i in idxs]
    order = np.argsort(scores)
    scores = [scores[k] for k in order]
    labels = [labels[k] for k in order]
    plt.figure(figsize=(8, max(4, 0.25 * len(labels))))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, labels, fontsize=7)
    plt.xlabel("F1 Score")
    plt.title(f"F1 per Label – {group_name}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(custom_save_dir, f"bar_f1_{group_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
print("Saved grouped F1 bar charts: bar_f1_<group>.png")
print("Done.")
