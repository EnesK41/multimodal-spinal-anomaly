import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset import BAD_PATIENTS, SpineMultimodalDataset


def normalize_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    values = arr[finite]
    lo = float(values.min())
    hi = float(values.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr[~finite] = 0
    return np.clip(arr, 0, 1)


def load_nifti_2d(path, rotate_90=False, is_mask=False):
    img_obj = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.squeeze(img_obj.get_fdata())
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got {arr.shape}: {path}")
    if rotate_90:
        arr = np.rot90(arr, k=1)
    if is_mask:
        arr = (arr > 0).astype(np.float32)
    else:
        arr = normalize_array(arr)
    return arr


def save_uint8_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def tensor_to_array(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def make_contact_sheet(rows, out_path, image_key, cols=5, thumb_size=(180, 300), max_items=80):
    subset = [row for row in rows if row.get(image_key)][:max_items]
    if not subset:
        return
    cell_w = thumb_size[0]
    cell_h = thumb_size[1] + 24
    rows_n = (len(subset) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    for idx, row in enumerate(subset):
        img = Image.open(row[image_key]).convert("L")
        img.thumbnail(thumb_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (cell_w, cell_h), "white")
        x = (cell_w - img.width) // 2
        y = (thumb_size[1] - img.height) // 2
        canvas.paste(img.convert("RGB"), (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, thumb_size[1] + 4), row["patient_id"][:28], fill=(0, 0, 0))
        sheet.paste(canvas, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export old training X-ray NIfTI files to clean PNG.")
    parser.add_argument("--data_dir", default="data/augmented_patients")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--out_full", default="data/png_patients_full")
    parser.add_argument("--out_roi", default="data/png_patients_roi045")
    parser.add_argument("--review_dir", default="evidence/png_conversion_review")
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--rotate_90", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_full = Path(args.out_full)
    out_roi = Path(args.out_roi)
    review_dir = Path(args.review_dir)

    if out_full.exists():
        shutil.rmtree(out_full)
    if out_roi.exists():
        shutil.rmtree(out_roi)
    review_dir.mkdir(parents=True, exist_ok=True)

    labels = list(csv.DictReader(open(args.csv_file, encoding="utf-8")))
    rows = []

    roi_dataset = SpineMultimodalDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        img_size=(1024, 512),
        rotate_90=args.rotate_90,
        require_xray=True,
        is_train=False,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=args.roi_width_ratio,
        scrub_prompt=True,
        label_source="mask",
    )
    roi_by_pid = {roi_dataset.patient_ids[i]: i for i in range(len(roi_dataset))}

    for row in labels:
        pid = str(row["patient_id"]).strip()
        base_pid = SpineMultimodalDataset.base_patient_id(pid)
        if base_pid in BAD_PATIENTS:
            continue

        patient_dir = data_dir / pid
        xray_path = patient_dir / f"{pid}_xray.nii.gz"
        mask_path = patient_dir / f"{pid}_xray_mask.nii.gz"
        if not xray_path.exists():
            continue

        full_patient_dir = out_full / pid
        roi_patient_dir = out_roi / pid
        full_xray_png = full_patient_dir / f"{pid}_xray.png"
        roi_xray_png = roi_patient_dir / f"{pid}_xray.png"

        xray_arr = load_nifti_2d(xray_path, rotate_90=args.rotate_90, is_mask=False)
        save_uint8_png(xray_arr, full_xray_png)

        full_mask_png = ""
        if mask_path.exists():
            mask_arr = load_nifti_2d(mask_path, rotate_90=args.rotate_90, is_mask=True)
            full_mask_png = str(full_patient_dir / f"{pid}_xray_mask.png")
            save_uint8_png(mask_arr, full_patient_dir / f"{pid}_xray_mask.png")

        roi_mask_png = ""
        roi_box = ""
        label = ""
        if pid in roi_by_pid:
            sample = roi_dataset[roi_by_pid[pid]]
            save_uint8_png(tensor_to_array(sample["xray"]), roi_xray_png)
            save_uint8_png(tensor_to_array(sample["mask"]), roi_patient_dir / f"{pid}_xray_mask.png")
            roi_mask_png = str(roi_patient_dir / f"{pid}_xray_mask.png")
            roi_box = ",".join(str(int(x)) for x in sample["roi_box"].tolist())
            label = int(float(sample["label"]) > 0.5)

        h, w = xray_arr.shape
        rows.append(
            {
                "patient_id": pid,
                "base_patient_id": base_pid,
                "is_augmented": int(SpineMultimodalDataset.is_augmented(pid)),
                "label": label,
                "source_xray": str(xray_path),
                "source_mask": str(mask_path) if mask_path.exists() else "",
                "full_png": str(full_xray_png),
                "full_mask_png": full_mask_png,
                "roi_png": str(roi_xray_png) if roi_xray_png.exists() else "",
                "roi_mask_png": roi_mask_png,
                "source_height": h,
                "source_width": w,
                "source_orientation_warning": "width_gt_height" if w > h else "",
                "roi_box_y1_y2_x1_x2": roi_box,
                "prompt": row.get("prompt", ""),
            }
        )

    shutil.copyfile(args.csv_file, out_full / "train_labels_augmented.csv")
    shutil.copyfile(args.csv_file, out_roi / "train_labels_augmented.csv")
    write_csv(review_dir / "png_export_manifest.csv", rows)

    originals = [row for row in rows if row["is_augmented"] == 0]
    make_contact_sheet(originals, review_dir / "full_png_original_patients_contact_sheet.png", "full_png")
    make_contact_sheet(originals, review_dir / "roi_png_original_patients_contact_sheet.png", "roi_png")

    summary = {
        "rows_exported": len(rows),
        "original_patients_exported": sum(1 for row in rows if row["is_augmented"] == 0),
        "augmented_samples_exported": sum(1 for row in rows if row["is_augmented"] == 1),
        "width_gt_height_sources": sum(1 for row in rows if row["source_orientation_warning"]),
        "out_full": str(out_full),
        "out_roi": str(out_roi),
        "review_dir": str(review_dir),
        "rotate_90": bool(args.rotate_90),
        "note": "Individual PNGs contain no overlaid text. Contact sheets are for review only.",
    }
    (review_dir / "png_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
