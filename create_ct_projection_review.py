import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import BAD_PATIENTS, SpineMultimodalDataset


def find_file(folder, suffix):
    for path in Path(folder).glob("*"):
        if path.name.lower().endswith(suffix.lower()):
            return str(path)
    return None


def normalize01(array):
    array = np.nan_to_num(array.astype(np.float32))
    low = float(array.min())
    high = float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    return array


def resize_array(array, target_hw, mode="bilinear"):
    array = np.ascontiguousarray(array, dtype=np.float32)
    tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0)
    kwargs = {"mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    tensor = F.interpolate(tensor, size=target_hw, **kwargs)
    return tensor.squeeze().numpy()


def load_volume(path):
    image = nib.as_closest_canonical(nib.load(path))
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    return np.nan_to_num(np.squeeze(volume))


def project_ct(volume, axis, hu_threshold, clip_max, rotate_k, target_hw):
    bone = np.where(
        volume >= float(hu_threshold),
        np.clip(volume, float(hu_threshold), float(clip_max)),
        0.0,
    )
    projection = bone.max(axis=int(axis))
    projection = normalize01(projection)
    if rotate_k:
        projection = np.rot90(projection, k=int(rotate_k)).copy()
    return resize_array(projection, target_hw, mode="bilinear")


def project_mask(mask_volume, axis, rotate_k, target_hw):
    mask = (mask_volume > 0).astype(np.float32)
    projection = mask.max(axis=int(axis))
    if rotate_k:
        projection = np.rot90(projection, k=int(rotate_k)).copy()
    projection = resize_array(projection, target_hw, mode="nearest")
    return (projection > 0.5).astype(np.float32)


def collect_patient_ids(args):
    if args.all_original_ct:
        patients = []
        for folder in sorted(Path(args.original_patients_dir).glob("patient_*")):
            ct_path = find_file(folder, "_ct.nii.gz")
            mask_path = find_file(folder, "_ct_mask.nii.gz")
            if ct_path and mask_path:
                patients.append(folder.name)
        return patients

    labels = pd.read_csv(args.csv_file)
    labels["base_patient_id"] = labels["patient_id"].apply(SpineMultimodalDataset.base_patient_id)
    patients = sorted(labels["base_patient_id"].unique().tolist())
    if args.exclude_bad_patients:
        patients = [pid for pid in patients if pid not in BAD_PATIENTS]
    out = []
    for pid in patients:
        folder = Path(args.original_patients_dir) / pid
        ct_path = find_file(folder, "_ct.nii.gz")
        mask_path = find_file(folder, "_ct_mask.nii.gz")
        if ct_path and mask_path:
            out.append(pid)
    return out


def make_contact_sheet(rows, out_path, title, cols=4):
    n = len(rows)
    fig_rows = math.ceil(n / cols)
    fig, axes = plt.subplots(fig_rows, cols, figsize=(cols * 3.2, fig_rows * 6.0))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, rows):
        ax.imshow(row["drr"], cmap="gray")
        mask = np.ma.masked_where(row["mask"] < 0.5, row["mask"])
        ax.imshow(mask, cmap="Reds", alpha=0.45)
        ax.set_title(
            f"{row['patient_id']} | mask_px={int(row['mask'].sum())}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create HU-thresholded CT projection review sheets")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--original_patients_dir", default="data/original_patients")
    parser.add_argument("--out_dir", default="evidence/ct_hu300_review")
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--rotate_k", type=int, default=1)
    parser.add_argument("--hu_threshold", type=float, default=300.0)
    parser.add_argument("--clip_max", type=float, default=1800.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--exclude_bad_patients", action="store_true", default=True)
    parser.add_argument("--include_bad_patients", dest="exclude_bad_patients", action="store_false")
    parser.add_argument("--all_original_ct", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_hw = (args.height, args.width)
    patients = collect_patient_ids(args)

    rows = []
    for pid in patients:
        folder = Path(args.original_patients_dir) / pid
        ct_path = find_file(folder, "_ct.nii.gz")
        mask_path = find_file(folder, "_ct_mask.nii.gz")
        if not ct_path or not mask_path:
            continue

        ct_volume = load_volume(ct_path)
        mask_volume = load_volume(mask_path)
        drr = project_ct(
            ct_volume,
            axis=args.axis,
            hu_threshold=args.hu_threshold,
            clip_max=args.clip_max,
            rotate_k=args.rotate_k,
            target_hw=target_hw,
        )
        mask = project_mask(
            mask_volume,
            axis=args.axis,
            rotate_k=args.rotate_k,
            target_hw=target_hw,
        )

        rows.append({"patient_id": pid, "drr": drr, "mask": mask})

        fig, ax = plt.subplots(1, 1, figsize=(3.2, 6.0))
        ax.imshow(drr, cmap="gray")
        overlay = np.ma.masked_where(mask < 0.5, mask)
        ax.imshow(overlay, cmap="Reds", alpha=0.45)
        ax.set_title(f"{pid} | HU>={args.hu_threshold:g} | mask_px={int(mask.sum())}", fontsize=9)
        ax.axis("off")
        single_path = Path(args.out_dir) / f"{pid}_hu{args.hu_threshold:g}_axis{args.axis}_rot{args.rotate_k}.png"
        plt.tight_layout()
        plt.savefig(single_path, dpi=180)
        plt.close(fig)

    contact_path = Path(args.out_dir) / (
        f"ct_hu{args.hu_threshold:g}_axis{args.axis}_rot{args.rotate_k}_"
        f"{args.height}x{args.width}_contact_sheet.png"
    )
    make_contact_sheet(
        rows,
        contact_path,
        title=(
            f"CT projections with CT-mask overlay | HU>={args.hu_threshold:g} | "
            f"axis={args.axis} | rot90 k={args.rotate_k} | {args.height}x{args.width}"
        ),
        cols=args.cols,
    )

    summary_path = Path(args.out_dir) / "ct_projection_review_summary.csv"
    pd.DataFrame(
        [
            {
                "patient_id": row["patient_id"],
                "mask_projected_pixels": int(row["mask"].sum()),
                "axis": args.axis,
                "rotate_k": args.rotate_k,
                "hu_threshold": args.hu_threshold,
                "height": args.height,
                "width": args.width,
            }
            for row in rows
        ]
    ).to_csv(summary_path, index=False)

    print(f"Patients projected: {len(rows)}")
    print(f"Contact sheet: {contact_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
