import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def summarize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    experiment = payload.get("experiment", "unknown")
    row = {
        "file": os.path.basename(path),
        "experiment": experiment,
        "encoder_type": payload.get("encoder_type"),
        "pretrained_encoder": payload.get("pretrained_encoder"),
        "roi_width_ratio": payload.get("roi_width_ratio"),
        "label_source": payload.get("label_source"),
        "scrub_prompts": payload.get("scrub_prompts"),
        "mean_default_balanced_accuracy": payload.get("mean_default_balanced_accuracy"),
        "mean_best_threshold_balanced_accuracy": payload.get("mean_best_threshold_balanced_accuracy"),
        "mean_best_balanced_accuracy": payload.get("mean_best_balanced_accuracy"),
    }

    if experiment == "cv5":
        scores = [r["best_score"] for r in payload.get("results", [])]
        thresholds = [r["best_threshold"] for r in payload.get("results", [])]
        row["mean_best_threshold_balanced_accuracy"] = payload.get("mean_best_balanced_accuracy")
        row["fold_scores"] = ", ".join(f"{score:.3f}" for score in scores)
        row["thresholds"] = ", ".join(f"{threshold:.3f}" for threshold in thresholds)
    elif experiment == "feature_baseline":
        scores = [
            r["best_threshold_metrics"]["balanced_accuracy"]
            for r in payload.get("results", [])
        ]
        thresholds = [r["best_threshold"] for r in payload.get("results", [])]
        row["fold_scores"] = ", ".join(f"{score:.3f}" for score in scores)
        row["thresholds"] = ", ".join(f"{threshold:.3f}" for threshold in thresholds)
    elif experiment == "metadata_baseline":
        aggregate = payload.get("aggregate", {})
        for model_name, metrics in aggregate.items():
            row[f"{model_name}_default_bal_acc"] = metrics.get("mean_default_balanced_accuracy")
            row[f"{model_name}_best_bal_acc"] = metrics.get("mean_best_threshold_balanced_accuracy")

    return row


def make_label(row):
    experiment = row["experiment"]
    encoder = row.get("encoder_type") or ""
    pretrain = "pretrained" if row.get("pretrained_encoder") else "scratch"
    roi = row.get("roi_width_ratio")
    if pd.notna(roi):
        return f"{experiment}\n{encoder} {pretrain}\nROI {roi}"
    return f"{experiment}\n{row['file']}"


def save_plot(df, out_dir):
    plot_df = df[df["mean_best_threshold_balanced_accuracy"].notna()].copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values("mean_best_threshold_balanced_accuracy")
    labels = [make_label(row) for _, row in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.25), 5))
    ax.bar(labels, plot_df["mean_best_threshold_balanced_accuracy"], color="#4c78a8")
    if "mean_default_balanced_accuracy" in plot_df:
        default_df = plot_df[plot_df["mean_default_balanced_accuracy"].notna()]
        default_labels = [make_label(row) for _, row in default_df.iterrows()]
        ax.scatter(
            default_labels,
            default_df["mean_default_balanced_accuracy"],
            color="#f58518",
            s=70,
            label="Default threshold",
            zorder=3,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean 5-fold balanced accuracy")
    ax.set_title("X-ray Classifier Experiment Summary")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, "experiment_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    out_dir = "experiment_results"
    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    rows = [summarize_file(path) for path in paths]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    plot_path = save_plot(df, out_dir)

    print("=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"CSV : {csv_path}")
    if plot_path:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
