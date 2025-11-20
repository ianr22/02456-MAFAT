# Use below script to get results for 0, 3, 8, 11 blur levels. 
# python results/analyze_blur_results.py \
#    --results-dir results \
#    --out-dir results/plots \
#    --levels 0,3,8,11

import os
import glob
import json
import argparse
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_fields(entry):
    """Given a dict entry, attempt to extract (y_true, y_pred, score).
    Returns tuple where each element can be None if not present.
    """
    # keep the old extractor for single-label scalar cases but prefer
    # multi-label keys (pred, true, pred_scores) handled in analyze_file.
    # This function is retained for backward compatibility but not used
    # for the standard MAFAT JSON format.
    return None, None, None


def load_json_file(path):
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            # fallback: try reading line by line JSON objects
            data = []
            f.seek(0)
            for line in f:
                line=line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


def analyze_file(path, out_dir, blur_level):
    print(f"Loading {path} for blur {blur_level}")
    data = load_json_file(path)
    # Expecting multi-label JSON with keys like: 'true' (list), 'pred' (list), 'pred_scores' (dict)
    top_preds = []
    top_scores = []
    first_true = []
    if not isinstance(data, list) or len(data) == 0:
        print(f"  No usable entries in {path}; skipping.")
        return

    # collect all labels seen in true sets (for confusion matrix label set)
    label_set = set()
    for rec in data:
        if not isinstance(rec, dict):
            continue
        tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
        pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
        scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')

        # determine top predicted label
        top_label = None
        top_score = np.nan
        if isinstance(pr, list) and len(pr) > 0:
            top_label = pr[0]
            if isinstance(scores, dict) and top_label in scores:
                try:
                    top_score = float(scores[top_label])
                except Exception:
                    top_score = np.nan
        elif isinstance(scores, dict) and len(scores) > 0:
            # choose argmax of pred_scores
            try:
                top_label = max(scores.items(), key=lambda x: float(x[1]))[0]
                top_score = float(scores[top_label])
            except Exception:
                top_label = None
                top_score = np.nan

        # first true label for single-label confusion matrix fallback
        true_first = None
        if isinstance(tr, list) and len(tr) > 0:
            true_first = tr[0]
            label_set.update(tr)
        elif isinstance(tr, str):
            true_first = tr
            label_set.add(tr)

        if top_label is not None:
            label_set.add(top_label)

        top_preds.append(top_label)
        top_scores.append(top_score)
        first_true.append(true_first)

    # convert to arrays for plotting
    # filter only entries where we have a first_true (otherwise cannot compute confusion)
    indices = [i for i, t in enumerate(first_true) if t is not None and top_preds[i] is not None]
    if len(indices) == 0:
        print(f"  No matching top-prediction / true-label pairs in {path}; skipping.")
        return

    idx = np.array(indices)
    scores_plot = np.array([top_scores[i] for i in indices], dtype=float)
    preds_plot = [top_preds[i] for i in indices]
    trues_plot = [first_true[i] for i in indices]

    os.makedirs(out_dir, exist_ok=True)

    # Scatter: top score vs index colored by correct/incorrect
    scatter_path = os.path.join(out_dir, f"blur_{blur_level}_scatter.png")
    correct = np.array([preds_plot[i] == trues_plot[i] for i in range(len(preds_plot))])
    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(len(scores_plot))[~correct], scores_plot[~correct], c='red', s=8, label='incorrect')
    plt.scatter(np.arange(len(scores_plot))[correct], scores_plot[correct], c='tab:green', s=8, label='correct')
    plt.xlabel('example index')
    plt.ylabel('top prediction score')
    plt.title(f'Blur {blur_level} — top-prediction score scatter (n={len(scores_plot)})')
    plt.legend(markerscale=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"  saved scatter -> {scatter_path}")

    # Confusion matrix using first true label vs top predicted label
    cm_path = os.path.join(out_dir, f"blur_{blur_level}_confusion.png")
    labels = sorted(label_set)
    y_true = trues_plot
    y_pred = preds_plot
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    except Exception as e:
        print(f"  could not compute confusion matrix: {e}")
        return

    plt.figure(figsize=(max(4, len(labels)*0.3), max(4, len(labels)*0.3)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title(f'Blur {blur_level} — Confusion matrix (first-true vs top-pred)')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  saved confusion matrix -> {cm_path}")


def find_json_for_blur(results_dir, level):
    pattern = os.path.join(results_dir, f"blur_{level}_run_*", "predictions_with_scores.json")
    matches = glob.glob(pattern)
    if not matches:
        # try without run folder
        alternate = os.path.join(results_dir, f"blur_{level}", "predictions_with_scores.json")
        matches = glob.glob(alternate)
    # return first match
    return matches[0] if matches else None

def plot_confusion_matrices_by_blur(results_dir, out_dir, levels):
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
        axes = axes.flatten()
        for idx, blur_level in enumerate(levels):
            path = find_json_for_blur(results_dir, blur_level)
            if not path:
                axes[idx].axis('off')
                continue
        data = load_json_file(path)
        trues, preds = [], []
        for rec in data:
            tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
            pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
            scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')

            top_label, true_first = None, None
            if isinstance(pr, list) and pr: top_label = pr[0]
            elif isinstance(scores, dict) and scores:
                try:
                    top_label = max(scores, key=lambda k: float(scores[k]))
                except: top_label = None
            if isinstance(tr, list) and tr: true_first = tr[0]
            elif isinstance(tr, str): true_first = tr

            if top_label is not None and true_first is not None:
                preds.append(top_label)
                trues.append(true_first)

            if not trues or not preds:
                axes[idx].axis('off')
                continue

        # Use blur level labels for axes
        blur_labels = [f'Blur {i}' for i in levels]
        cm = confusion_matrix(trues, preds, labels=blur_labels, normalize='true')
        sns.heatmap(cm, ax=axes[idx], cmap='YlOrBr', cbar=False,
                    xticklabels=blur_labels, yticklabels=blur_labels,
                    square=True)
        axes[idx].set_title(f'Blur {blur_level}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].tick_params(left=False, bottom=False)
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, 'all_levels_confusion_grid.png'))
        plt.close()
        print(f"Saved grid of confusion matrices -> {os.path.join(out_dir, 'all_levels_confusion_grid.png')}")
        
    
def plot_score_distributions_by_blur(results_dir, out_dir, levels):
    import pandas as pd
    all_data = []
    for blur_level in levels:
        path = find_json_for_blur(results_dir, blur_level)
        if not path:
            continue
        data = load_json_file(path)
        for rec in data:
            tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
            pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
            scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')
            top_label, top_score, true_first = None, np.nan, None
            if isinstance(pr, list) and pr:
                top_label = pr[0]
                if isinstance(scores, dict) and top_label in scores:
                    try:
                        top_score = float(scores[top_label])
                    except: top_score = np.nan
            elif isinstance(scores, dict) and scores:
                try:
                    top_label = max(scores, key=lambda k: float(scores[k]))
                    top_score = float(scores[top_label])
                except: top_label, top_score = None, np.nan
            if isinstance(tr, list) and tr: true_first = tr[0]
            elif isinstance(tr, str): true_first = tr

            if top_label is not None and true_first is not None:
                all_data.append({'blur': blur_level, 'score': top_score,
                                'correct': top_label == true_first})

    df = pd.DataFrame(all_data)
    plt.figure(figsize=(14,5))
    sns.violinplot(x='blur', y='score', hue='correct', data=df, split=True, palette={True:'green',False:'red'})
    plt.xlabel('Blur Level')
    plt.ylabel('Top Prediction Score')
    plt.title('Prediction Score Distribution by Blur Level')
    plt.legend(title='Correct?')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'score_distributions_by_blur.png'))
    plt.close()
    print(f"Saved blur-level score distributions -> {os.path.join(out_dir, 'score_distributions_by_blur.png')}")
    

def plot_confusion_matrices_by_blur(results_dir, out_dir, levels, TOP_K=8):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) collect global subclass label set ----------
    label_set = set()
    for blur_level in levels:
        path = find_json_for_blur(results_dir, blur_level)
        if not path:
            continue
        data = load_json_file(path)
        for rec in data:
            if not isinstance(rec, dict):
                continue
            tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
            pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
            scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')

            # true labels
            if isinstance(tr, list) and tr:
                label_set.update(tr)
            elif isinstance(tr, str):
                label_set.add(tr)

            # top predicted label
            top_label = None
            if isinstance(pr, list) and pr:
                top_label = pr[0]
            elif isinstance(scores, dict) and scores:
                try:
                    top_label = max(scores, key=lambda k: float(scores[k]))
                except Exception:
                    top_label = None
            if top_label is not None:
                label_set.add(top_label)

    labels = sorted(label_set)

    # ---------- 2) grid of confusion matrices ----------
    n_levels = len(levels)
    ncols = 2
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for idx, blur_level in enumerate(levels):
        ax = axes[idx]
        path = find_json_for_blur(results_dir, blur_level)
        if not path:
            ax.axis('off')
            continue

        data = load_json_file(path)
        trues, preds = [], []

        for rec in data:
            if not isinstance(rec, dict):
                continue

            tr = rec.get('true') or rec.get('ground_truth') or rec.get('gt')
            pr = rec.get('pred') or rec.get('predictions') or rec.get('prediction')
            scores = rec.get('pred_scores') or rec.get('scores') or rec.get('probs')

            # top predicted label
            top_label = None
            if isinstance(pr, list) and pr:
                top_label = pr[0]
            elif isinstance(scores, dict) and scores:
                try:
                    top_label = max(scores, key=lambda k: float(scores[k]))
                except Exception:
                    top_label = None

            # first true label
            true_first = None
            if isinstance(tr, list) and tr:
                true_first = tr[0]
            elif isinstance(tr, str):
                true_first = tr

            if top_label is not None and true_first is not None:
                preds.append(top_label)
                trues.append(true_first)

        if not trues:
            ax.axis('off')
            continue

        cm = confusion_matrix(trues, preds, labels=labels, normalize='true')

        sns.heatmap(
            cm,
            ax=ax,
            cmap='YlOrBr',
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
        )

        ax.set_title(f'Blur {blur_level}', fontsize=12)

    # ---------- 3) show labels only on left column & bottom row ----------
    for i, ax in enumerate(axes[:n_levels]):
        row = i // ncols
        col = i % ncols

        # x-axis labels only on bottom row
        if row == nrows - 1:
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
        else:
            ax.set_xticklabels([])
        ax.set_xlabel('')

        # y-axis labels only on left column
        if col == 0:
            ax.set_yticklabels(labels, rotation=0, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_ylabel('')

    # turn off any unused axes
    for ax in axes[n_levels:]:
        ax.axis('off')

    # global axis labels like in your sketch
    fig.text(0.5, 0.02, 'Predicted', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'True', va='center', rotation='vertical', fontsize=16)

    fig.tight_layout(rect=[0.08, 0.08, 0.98, 0.98])
    out_path = os.path.join(out_dir, 'confusion_blur_0_3_8_11_grid.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved grid of confusion matrices -> {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', help='Base results directory')
    parser.add_argument('--out-dir', default='results/plots', help='Output directory for PNGs')
    parser.add_argument('--levels', default=None, help='Comma-separated blur levels (default 0..11)')
    args = parser.parse_args()

    if args.levels:
        levels = [int(x) for x in args.levels.split(',')]
    else:
        levels = list(range(0, 12))

    for level in levels:
        path = find_json_for_blur(args.results_dir, level)
        if not path:
            print(f"No predictions file found for blur {level} (searched under {args.results_dir})")
            continue
        try:
            analyze_file(path, args.out_dir, level)
        except Exception as e:
            print(f"Error processing blur {level}: {e}")
    
    plot_confusion_matrices_by_blur(args.results_dir, args.out_dir, levels)
    plot_score_distributions_by_blur(args.results_dir, args.out_dir, levels)

if __name__ == '__main__':
    main()
