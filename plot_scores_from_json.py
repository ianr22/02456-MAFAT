import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_scores(json_path):
    """Load a *_prediction_scores.json file and return sorted blur levels + mAP dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract (level, entry) pairs: "blur0" -> 0, etc.
    blur_entries = []
    for key, value in data.items():
        if key.startswith("blur"):
            try:
                level = int(key.replace("blur", ""))
            except ValueError:
                continue
            blur_entries.append((level, value))

    # Sort by numeric blur level: 0,1,2,...,23
    blur_entries.sort(key=lambda x: x[0])

    levels = [lvl for lvl, _ in blur_entries]
    maps   = [entry["mAP"] for _, entry in blur_entries]

    return levels, maps


def compute_sigma(levels, max_sigma=8.0):
    """
    Map blur levels (0..N-1) to sigma values from 0 to max_sigma inclusive,
    using evenly spaced steps (like linspace(0, max_sigma, N)).
    """
    if not levels:
        return []

    n = len(levels)
    if n == 1:
        return [0.0]

    # assume levels are 0..n-1 in order; we only use count, not values
    sigma = [max_sigma * i / (n - 1) for i in range(n)]
    return sigma


def plot_true_map(sigma, mAP_total, title, outfile):
    """Plot TRUE mAP vs sigma."""
    plt.figure(figsize=(8, 4))
    plt.plot(sigma, mAP_total, marker="o")
    plt.xlabel("Sigma (Gaussian Blur)")
    plt.ylabel("TRUE mAP")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def plot_grouped_ap(sigma, general, subclass, color, features, title, outfile):
    """Plot grouped AP vs sigma for the four groups."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(sigma, general,  marker="o", label="General Class")
    plt.plot(sigma, subclass, marker="o", label="Sub-Class")
    plt.plot(sigma, color,    marker="o", label="Color")
    plt.plot(sigma, features, marker="o", label="Features")

    plt.xlabel("Sigma (Gaussian Blur)")
    plt.ylabel("Grouped AP")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def make_plots_for_file(json_path, label_prefix):
    """
    json_path: Path to *_prediction_scores.json
    label_prefix: e.g. "Balanced" or "Training" (used in plot titles / filenames)
    """
    levels, maps = load_scores(json_path)
    sigma = compute_sigma(levels, max_sigma=8.0)

    # Extract series
    total     = [m["total"]    for m in maps]
    general   = [m["general"]  for m in maps]
    subclass  = [m["subclass"] for m in maps]
    color     = [m["color"]    for m in maps]
    features  = [m["features"] for m in maps]

    # Filenames
    base = Path(json_path).stem.replace("_scores", "")
    true_outfile    = f"{base}_true_mAP_vs_sigma.png"
    grouped_outfile = f"{base}_grouped_AP_vs_sigma.png"

    # Titles
    true_title    = f"{label_prefix}: TRUE mAP vs Gaussian Blur Sigma"
    grouped_title = f"{label_prefix}: Grouped AP vs Gaussian Blur Sigma"

    # Make plots
    plot_true_map(sigma, total, true_title, true_outfile)
    plot_grouped_ap(sigma, general, subclass, color, features,
                    grouped_title, grouped_outfile)


if __name__ == "__main__":
    # Adjust paths if your JSON files live somewhere else
    balanced_json = "balanced_prediction_scores.json"
    training_json = "training_prediction_scores.json"

    make_plots_for_file(balanced_json, "Balanced Predictions")
    make_plots_for_file(training_json, "Training Predictions")
