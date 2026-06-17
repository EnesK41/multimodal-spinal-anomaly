import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from dataset import SpineMultimodalDataset
from train import compute_binary_metrics


HEALTHY_START = 30
HEALTHY_END = 45


def patient_number(base_patient_id):
    match = re.search(r"patient_(\d+)", str(base_patient_id))
    if not match:
        raise ValueError(f"Could not parse patient number from {base_patient_id}")
    return int(match.group(1))


def known_label_from_patient_number(base_patient_id):
    number = patient_number(base_patient_id)
    return 0 if HEALTHY_START <= number <= HEALTHY_END else 1


def prompt_has_diagnostic_leak(prompt):
    prompt = str(prompt).lower()
    diagnostic_terms = [
        "anomaly",
        "block vertebra",
        "scoliosis",
        "hemivertebra",
        "butterfly",
        "fusion",
        "healthy",
        "normal",
        "empty",
    ]
    return int(any(term in prompt for term in diagnostic_terms))


def build_dataset(use_body_roi, roi_width_ratio, scrub_prompts):
    return SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=use_body_roi,
        roi_width_ratio=roi_width_ratio,
        scrub_prompt=scrub_prompts,
    )


def collect_sample_rows(roi_width_ratio, scrub_prompts):
    full_dataset = build_dataset(
        use_body_roi=False,
        roi_width_ratio=roi_width_ratio,
        scrub_prompts=scrub_prompts,
    )
    roi_dataset = build_dataset(
        use_body_roi=True,
        roi_width_ratio=roi_width_ratio,
        scrub_prompts=scrub_prompts,
    )

    full_by_patient = {}
    for idx in range(len(full_dataset)):
        sample = full_dataset[idx]
        full_by_patient[sample["patient_id"]] = sample

    rows = []
    for idx in range(len(roi_dataset)):
        roi_sample = roi_dataset[idx]
        patient_id = roi_sample["patient_id"]
        full_sample = full_by_patient[patient_id]
        base_patient_id = roi_sample["base_patient_id"]
        number = patient_number(base_patient_id)
        known_label = known_label_from_patient_number(base_patient_id)

        full_mask_sum = float(full_sample["mask"].sum().item())
        roi_mask_sum = float(roi_sample["mask"].sum().item())
        full_mask_label = int(full_mask_sum > 0)
        roi_mask_label = int(roi_mask_sum > 0)

        rows.append(
            {
                "patient_id": patient_id,
                "base_patient_id": base_patient_id,
                "patient_number": number,
                "is_augmented": bool(roi_sample["is_augmented"]),
                "known_label_from_range": known_label,
                "known_label_name": "healthy" if known_label == 0 else "anomaly",
                "full_mask_sum": full_mask_sum,
                "roi_mask_sum": roi_mask_sum,
                "full_mask_label": full_mask_label,
                "roi_mask_label": roi_mask_label,
                "roi_lost_positive_mask": int(full_mask_label == 1 and roi_mask_label == 0),
                "prompt": roi_sample["text"],
                "prompt_has_diagnostic_term": prompt_has_diagnostic_leak(roi_sample["text"]),
            }
        )

    return pd.DataFrame(rows)


def aggregate_patient_rows(sample_df):
    agg = (
        sample_df.groupby("base_patient_id")
        .agg(
            patient_number=("patient_number", "first"),
            known_label_from_range=("known_label_from_range", "first"),
            full_mask_label_max=("full_mask_label", "max"),
            roi_mask_label_max=("roi_mask_label", "max"),
            roi_lost_positive_mask_any=("roi_lost_positive_mask", "max"),
            sample_count=("patient_id", "count"),
            augmented_count=("is_augmented", "sum"),
            prompt=("prompt", "first"),
            prompt_has_diagnostic_term=("prompt_has_diagnostic_term", "max"),
        )
        .reset_index()
    )
    agg["known_label_name"] = agg["known_label_from_range"].map({0: "healthy", 1: "anomaly"})
    agg["range_vs_roi_label_match"] = (
        agg["known_label_from_range"] == agg["roi_mask_label_max"]
    ).astype(int)
    return agg


def save_label_range_plot(patient_df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axvspan(HEALTHY_START - 0.5, HEALTHY_END + 0.5, color="#d9ead3", alpha=0.8, label="Known healthy ID range")

    colors = patient_df["known_label_from_range"].map({0: "#2e7d32", 1: "#c62828"})
    ax.scatter(
        patient_df["patient_number"],
        patient_df["known_label_from_range"],
        s=70,
        c=colors,
        edgecolor="black",
        linewidth=0.5,
        label="Patients",
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["healthy", "anomaly"])
    ax.set_xlabel("Patient number")
    ax.set_title("Patient Number Encodes the Label Range")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()

    path = os.path.join(out_dir, "patient_number_label_range.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_prompt_leak_plot(patient_df, out_dir):
    grouped = (
        patient_df.groupby(["known_label_name", "prompt_has_diagnostic_term"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "no diagnostic term", 1: "diagnostic term"})
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax, color=["#90a4ae", "#ef5350"])
    ax.set_title("Prompt Text Contains Diagnostic Label Information")
    ax.set_xlabel("Known patient label")
    ax.set_ylabel("Patient count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    path = os.path.join(out_dir, "prompt_diagnostic_terms.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_roi_label_plot(patient_df, out_dir):
    counts = pd.crosstab(
        patient_df["known_label_name"],
        patient_df["roi_mask_label_max"].map({0: "ROI label healthy", 1: "ROI label anomaly"}),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#66bb6a", "#ef5350"])
    ax.set_title("Known Range Label vs ROI-Mask-Derived Label")
    ax.set_xlabel("Known patient label")
    ax.set_ylabel("Patient count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    path = os.path.join(out_dir, "known_vs_roi_label.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_metadata_summaries():
    paths = sorted(glob.glob(os.path.join("experiment_results", "metadata_baseline*.json")))
    summaries = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = path
        summaries.append(payload)
    return summaries


def save_metadata_baseline_plot(metadata_summaries, out_dir):
    if not metadata_summaries:
        return None

    rows = []
    for summary in metadata_summaries:
        label_source = summary.get("label_source", "mask")
        scrub = "scrubbed" if summary.get("scrub_prompts", False) else "raw"
        for model_name, result in summary["aggregate"].items():
            rows.append(
                {
                    "run": f"{label_source}/{scrub}",
                    "model": model_name,
                    "default_bal_acc": result["mean_default_balanced_accuracy"],
                    "best_threshold_bal_acc": result["mean_best_threshold_balanced_accuracy"],
                }
            )

    df = pd.DataFrame(rows)
    df["label"] = df["run"] + "\n" + df["model"]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.9), 4.5))
    ax.bar(df["label"], df["default_bal_acc"], color="#5c6bc0", label="Default threshold")
    ax.scatter(df["label"], df["best_threshold_bal_acc"], color="#ff7043", s=70, label="Best threshold")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Metadata-Only Baselines Reveal Shortcut Signal")
    ax.set_ylabel("Mean 5-fold balanced accuracy")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, "metadata_baseline_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_notes(sample_df, patient_df, out_dir, plot_paths):
    known_vs_roi = compute_binary_metrics(
        labels=patient_df["known_label_from_range"].tolist(),
        probs=patient_df["roi_mask_label_max"].tolist(),
        threshold=0.5,
    )
    patient_number_rule = compute_binary_metrics(
        labels=patient_df["known_label_from_range"].tolist(),
        probs=[
            known_label_from_patient_number(pid)
            for pid in patient_df["base_patient_id"].tolist()
        ],
        threshold=0.5,
    )

    prompt_leak_rate = patient_df["prompt_has_diagnostic_term"].mean()
    roi_lost = int(sample_df["roi_lost_positive_mask"].sum())

    path = os.path.join(out_dir, "presentation_evidence_notes.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Presentation Evidence Notes\n\n")
        f.write("## Dataset Shortcut Findings\n\n")
        f.write(
            f"- Kept base patients after filtering: {len(patient_df)}.\n"
            f"- Known healthy ID range supplied by dataset owner: patient_{HEALTHY_START:03d} to patient_{HEALTHY_END:03d}.\n"
            f"- Patient-number rule balanced accuracy against known labels: "
            f"{patient_number_rule['balanced_accuracy']:.3f}.\n"
            f"- Prompts with explicit diagnostic terms: {prompt_leak_rate:.1%} of base patients.\n"
        )
        f.write("\n## ROI Label Audit\n\n")
        f.write(
            f"- Samples where full mask is positive but ROI mask becomes empty: {roi_lost}.\n"
            f"- ROI-mask-derived label balanced accuracy against known range label: "
            f"{known_vs_roi['balanced_accuracy']:.3f}.\n"
            f"- ROI label TP/TN/FP/FN vs known range label: "
            f"{known_vs_roi['tp']} / {known_vs_roi['tn']} / {known_vs_roi['fp']} / {known_vs_roi['fn']}.\n"
        )
        f.write("\n## Generated Figures\n\n")
        for name, plot_path in plot_paths.items():
            if plot_path:
                f.write(f"- {name}: `{plot_path}`\n")

    return path


def main():
    parser = argparse.ArgumentParser(description="Generate presentation evidence for shortcut and ROI audits.")
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--scrub_prompts", action="store_true")
    parser.add_argument("--out_dir", default="evidence")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sample_df = collect_sample_rows(args.roi_width_ratio, args.scrub_prompts)
    patient_df = aggregate_patient_rows(sample_df)

    sample_csv = os.path.join(args.out_dir, "sample_label_audit.csv")
    patient_csv = os.path.join(args.out_dir, "patient_label_audit.csv")
    sample_df.to_csv(sample_csv, index=False)
    patient_df.to_csv(patient_csv, index=False)

    metadata_summaries = load_metadata_summaries()
    plot_paths = {
        "patient number label range": save_label_range_plot(patient_df, args.out_dir),
        "prompt diagnostic terms": save_prompt_leak_plot(patient_df, args.out_dir),
        "known vs ROI label": save_roi_label_plot(patient_df, args.out_dir),
        "metadata baseline summary": save_metadata_baseline_plot(metadata_summaries, args.out_dir),
    }
    notes_path = write_notes(sample_df, patient_df, args.out_dir, plot_paths)

    print("=" * 100)
    print("EVIDENCE AUDIT COMPLETE")
    print("=" * 100)
    print(f"Sample audit CSV : {sample_csv}")
    print(f"Patient audit CSV: {patient_csv}")
    print(f"Notes            : {notes_path}")
    for name, path in plot_paths.items():
        if path:
            print(f"{name:28s}: {path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
