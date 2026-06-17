import json
import subprocess
import sys
from pathlib import Path

from train import make_stratified_patient_folds
from train_png_classifier import SpinePngDataset, collect_patient_labels


def main():
    seed = 42
    data_dir = "data/png_patients_roi045"
    tag = "png_roi045_no_extra_roi_seed42_resnet34_pretrained_mask_scrubbed"

    dataset = SpinePngDataset(
        data_dir=data_dir,
        csv_file="data/train_labels_augmented.csv",
        exclude_bad_patients=True,
        use_body_roi=False,
        roi_width_ratio=0.45,
        scrub_prompt=True,
        label_source="mask",
    )
    patient_labels = collect_patient_labels(dataset)
    folds = make_stratified_patient_folds(patient_labels, n_splits=5, seed=seed)

    results = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        run_name = f"{tag}_fold_{fold_idx:02d}"
        cmd = [
            sys.executable,
            "train_png_classifier.py",
            "--data_dir",
            data_dir,
            "--run_name",
            run_name,
            "--epochs",
            "20",
            "--batch_size",
            "4",
            "--seed",
            str(seed),
            "--roi_width_ratio",
            "0.45",
            "--no_body_roi",
            "--encoder_type",
            "resnet34",
            "--pretrained_encoder",
            "--patience",
            "6",
            "--min_save_epoch",
            "3",
            "--demo_holdout",
            *val_patients,
        ]
        print("=" * 100)
        print(f"PNG ROI CV5 FOLD {fold_idx}/5")
        print(f"Validation patients: {', '.join(val_patients)}")
        print("=" * 100)
        subprocess.run(cmd, check=True)

        result_path = Path("experiment_results/png_classifier") / f"{run_name}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["fold"] = fold_idx
        result["val_patients"] = val_patients
        results.append(result)

    scores = [r["best_score_balanced_accuracy"] for r in results]
    default_scores = [r["holdout_metrics_default_threshold"]["balanced_accuracy"] for r in results]
    payload = {
        "experiment": "png_roi045_no_extra_roi_cv5",
        "seed": seed,
        "settings_matched_to": "cv5_resnet34_pretrained_mask_scrubbed_roi045_seed_42",
        "input_format": "png",
        "data_dir": data_dir,
        "use_body_roi": False,
        "mean_best_balanced_accuracy": sum(scores) / len(scores),
        "mean_default_threshold_balanced_accuracy": sum(default_scores) / len(default_scores),
        "results": results,
    }
    out_path = Path("experiment_results/png_classifier/png_roi045_no_extra_roi_cv5_seed42_resnet34_pretrained_mask_scrubbed.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
