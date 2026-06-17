import json
import subprocess
import sys
from pathlib import Path

from train import make_stratified_patient_folds
from train_png_classifier import SpinePngDataset, collect_patient_labels


def main():
    seed = 42
    dataset = SpinePngDataset(
        data_dir="data/png_patients_full",
        csv_file="data/train_labels_augmented.csv",
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=0.45,
        scrub_prompt=True,
        label_source="mask",
    )
    patient_labels = collect_patient_labels(dataset)
    folds = make_stratified_patient_folds(patient_labels, n_splits=5, seed=seed)

    results = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        run_name = f"png_cv5_seed42_resnet34_pretrained_mask_scrubbed_roi045_fold_{fold_idx:02d}"
        cmd = [
            sys.executable,
            "train_png_classifier.py",
            "--data_dir",
            "data/png_patients_full",
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
        print(f"PNG CV5 FOLD {fold_idx}/5")
        print(f"Validation patients: {', '.join(val_patients)}")
        print("=" * 100)
        subprocess.run(cmd, check=True)

        result_path = Path("experiment_results/png_classifier") / f"{run_name}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["fold"] = fold_idx
        result["val_patients"] = val_patients
        results.append(result)

    valid_scores = [r["best_score_balanced_accuracy"] for r in results]
    payload = {
        "experiment": "png_cv5",
        "seed": seed,
        "settings_matched_to": "cv5_resnet34_pretrained_mask_scrubbed_roi045_seed_42",
        "input_format": "png",
        "data_dir": "data/png_patients_full",
        "mean_best_balanced_accuracy": sum(valid_scores) / len(valid_scores),
        "results": results,
    }
    out_path = Path("experiment_results/png_classifier/png_cv5_seed42_resnet34_pretrained_mask_scrubbed_roi045.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
