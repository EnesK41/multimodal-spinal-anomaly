import argparse
import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from augmentation import apply_augmentations


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min > 0:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr


def _load_patient_tensors(patient_id: str, data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    patient_dir = os.path.join(data_dir, patient_id)
    xray_path = os.path.join(patient_dir, f"{patient_id}_xray.nii.gz")
    mask_path = os.path.join(patient_dir, f"{patient_id}_xray_mask.nii.gz")

    if not os.path.exists(xray_path):
        raise FileNotFoundError(f"X-ray not found: {xray_path}")

    xray_data = np.squeeze(nib.load(xray_path).get_fdata())
    if xray_data.ndim != 2:
        raise ValueError(f"Expected 2D xray, got shape={xray_data.shape} for {xray_path}")
    xray_data = _normalize_01(xray_data).astype(np.float32)

    if os.path.exists(mask_path):
        mask_data = np.squeeze(nib.load(mask_path).get_fdata())
        if mask_data.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape={mask_data.shape} for {mask_path}")
        if mask_data.shape != xray_data.shape:
            raise ValueError(
                f"Mask shape mismatch for {patient_id}: xray={xray_data.shape}, mask={mask_data.shape}"
            )
        mask_data = mask_data.astype(np.float32)
    else:
        mask_data = np.zeros_like(xray_data, dtype=np.float32)

    return torch.from_numpy(xray_data).unsqueeze(0), torch.from_numpy(mask_data).unsqueeze(0)


def _resolve_patient_ids(
    patient_id: str | None,
    csv_file: str,
    data_dir: str,
    num_patients: int,
) -> list[str]:
    if patient_id:
        return [patient_id]

    ids: list[str] = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if "patient_id" in df.columns:
            ids = [str(v).strip() for v in df["patient_id"].dropna().tolist()]

    if not ids:
        ids = [d for d in sorted(os.listdir(data_dir)) if d.startswith("patient_")]

    # Keep order but drop duplicates.
    seen = set()
    unique_ids = []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    resolved = []
    for pid in unique_ids:
        xray_path = os.path.join(data_dir, pid, f"{pid}_xray.nii.gz")
        if os.path.exists(xray_path):
            resolved.append(pid)
        if len(resolved) >= num_patients:
            break

    return resolved


def _build_aug_samples(
    xray_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    num_augs: int,
    base_seed: int | None,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    samples = [("original", xray_tensor.clone(), mask_tensor.clone())]
    for i in range(num_augs):
        if base_seed is not None:
            seed = base_seed + i
            random.seed(seed)
            torch.manual_seed(seed)

        aug_xray, aug_mask = apply_augmentations(xray_tensor.clone(), mask_tensor.clone())
        samples.append((f"aug_{i + 1}", aug_xray, aug_mask))

    return samples


def _plot_patient_samples(patient_id: str, samples: list[tuple[str, torch.Tensor, torch.Tensor]], out_path: str) -> None:
    n = len(samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes = np.array(axes).reshape(-1)

    for i, (name, xray_t, mask_t) in enumerate(samples):
        xray = xray_t.squeeze(0).numpy()
        mask = (mask_t.squeeze(0).numpy() > 0.5).astype(np.float32)

        ax = axes[i]
        ax.imshow(xray, cmap="gray", vmin=0.0, vmax=1.0)
        ax.imshow(np.ma.masked_where(mask < 0.5, mask), cmap="Reds", alpha=0.45, vmin=0.0, vmax=1.0)
        ax.set_title(f"{name} | mask%={mask.mean() * 100:.2f}", fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Augmentation preview: {patient_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _print_stats(patient_id: str, samples: list[tuple[str, torch.Tensor, torch.Tensor]]) -> None:
    print(f"\nPatient: {patient_id}")
    for name, xray_t, mask_t in samples:
        xray = xray_t.squeeze(0)
        mask = (mask_t.squeeze(0) > 0.5).float()
        print(
            f"  {name:<10} xray[min={xray.min():.4f}, max={xray.max():.4f}, mean={xray.mean():.4f}] "
            f"mask_coverage={mask.mean() * 100:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview X-ray augmentations before training.")
    parser.add_argument("--data-dir", default="data/original_patients", help="Directory of patient folders")
    parser.add_argument("--csv-file", default="data/train_labels.csv", help="CSV used to pick patients")
    parser.add_argument("--patient-id", default=None, help="Specific patient id (e.g. patient_002)")
    parser.add_argument("--num-patients", type=int, default=3, help="How many patients to preview if --patient-id is not set")
    parser.add_argument("--num-augs", type=int, default=6, help="How many augmented variants per patient")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible previews")
    parser.add_argument("--output-dir", default="augmentation_preview", help="Directory for saved preview images")
    parser.add_argument("--show", action="store_true", help="Open plots after saving")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patient_ids = _resolve_patient_ids(
        patient_id=args.patient_id,
        csv_file=args.csv_file,
        data_dir=args.data_dir,
        num_patients=args.num_patients,
    )

    if not patient_ids:
        raise RuntimeError("No valid patients found for preview.")

    print(f"Selected patients: {patient_ids}")

    for idx, patient_id in enumerate(patient_ids):
        xray_tensor, mask_tensor = _load_patient_tensors(patient_id, args.data_dir)
        samples = _build_aug_samples(
            xray_tensor=xray_tensor,
            mask_tensor=mask_tensor,
            num_augs=args.num_augs,
            base_seed=args.seed + (idx * 1000),
        )

        _print_stats(patient_id, samples)

        out_path = os.path.join(args.output_dir, f"{patient_id}_aug_preview.png")
        _plot_patient_samples(patient_id, samples, out_path)
        print(f"Saved: {out_path}")

        if args.show:
            img = plt.imread(out_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(out_path))
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
