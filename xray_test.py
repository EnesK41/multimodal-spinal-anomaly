import os
import json
import numpy as np
import torch
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F


DATA_DIR = "data/augmented_patients"
CSV_FILE = "data/train_labels_augmented.csv"
OUT_DIR = "debug_roi_coverage"

IMG_SIZE = (1024, 512)

BAD_PATIENTS = [
    "patient_005",
    "patient_019",
    "patient_031",
    "patient_043",
    "patient_044",
    "patient_046",
    "patient_047",
    "patient_048",
    "patient_050",
    "patient_051",
    "patient_054",
]

ROI_WIDTH_RATIOS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

COVERAGE_FAIL_THRESHOLD = 0.90


def base_pid(pid):
    return str(pid).split("_aug")[0]


def normalize_tensor(tensor):
    mn = tensor.min()
    mx = tensor.max()
    if mx > mn:
        return (tensor - mn) / (mx - mn)
    return tensor


def resize_with_padding(tensor, img_size=IMG_SIZE):
    """
    tensor: [1, H, W]
    """
    _, h, w = tensor.shape
    target_h, target_w = img_size

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
    return tensor


def load_2d_nifti(path, is_mask=False):
    img_obj = nib.load(path)
    img_obj = nib.as_closest_canonical(img_obj)
    img = img_obj.get_fdata()
    img = np.squeeze(img)

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    if is_mask:
        tensor = (tensor > 0).float()
        tensor = resize_with_padding(tensor)
        tensor = (tensor > 0.1).float()
    else:
        tensor = normalize_tensor(tensor)
        tensor = resize_with_padding(tensor)

    return tensor


def body_center_roi_box(xray, roi_width_ratio):
    """
    xray: [1, H, W], normalized 0-1
    returns y1, y2, x1, x2
    """
    _, H, W = xray.shape
    img = xray[0]

    threshold = max(0.03, float(img.mean()) * 0.5)
    foreground = img > threshold

    ys, xs = torch.where(foreground)

    if len(ys) == 0:
        return 0, H, 0, W

    y1 = int(ys.min().item())
    y2 = int(ys.max().item())
    x1_body = int(xs.min().item())
    x2_body = int(xs.max().item())

    body_w = x2_body - x1_body + 1
    cx = (x1_body + x2_body) // 2

    roi_w = int(body_w * roi_width_ratio)

    x1 = max(0, cx - roi_w // 2)
    x2 = min(W, cx + roi_w // 2)

    margin_y = int(H * 0.02)
    y1 = max(0, y1 - margin_y)
    y2 = min(H, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, H, 0, W

    return y1, y2, x1, x2


def compute_coverage(mask, roi_box):
    """
    mask: [1, H, W]
    """
    gt_sum = mask.sum().item()

    if gt_sum <= 0:
        return None

    y1, y2, x1, x2 = roi_box
    inside = mask[:, y1:y2, x1:x2].sum().item()

    return inside / gt_sum


def roi_area_ratio(mask, roi_box):
    _, H, W = mask.shape
    y1, y2, x1, x2 = roi_box
    roi_area = max(0, y2 - y1) * max(0, x2 - x1)
    return roi_area / float(H * W)


def save_overlay(xray, mask, roi_box, out_path, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = xray[0].cpu().numpy()
    m = mask[0].cpu().numpy()

    y1, y2, x1, x2 = roi_box

    plt.figure(figsize=(14, 7))

    plt.imshow(x, cmap="gray")
    plt.imshow(np.ma.masked_where(m <= 0, m), cmap="Reds", alpha=0.45)

    xs = [x1, x2, x2, x1, x1]
    ys = [y1, y1, y2, y2, y1]
    plt.plot(xs, ys, linewidth=2)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    records = []

    for _, row in df.iterrows():
        pid = row["patient_id"]
        bpid = base_pid(pid)

        if bpid in BAD_PATIENTS:
            continue

        patient_folder = os.path.join(DATA_DIR, pid)
        xray_path = os.path.join(patient_folder, f"{pid}_xray.nii.gz")
        mask_path = os.path.join(patient_folder, f"{pid}_xray_mask.nii.gz")

        if not os.path.exists(xray_path):
            continue

        xray = load_2d_nifti(xray_path, is_mask=False)

        if os.path.exists(mask_path):
            mask = load_2d_nifti(mask_path, is_mask=True)
        else:
            mask = torch.zeros_like(xray)

        gt_sum = mask.sum().item()

        if gt_sum <= 0:
            continue

        rec = {
            "patient_id": pid,
            "base_id": bpid,
            "gt_pixels": gt_sum,
            "xray": xray,
            "mask": mask,
        }
        records.append(rec)

    print("=" * 90)
    print("ROI COVERAGE SWEEP")
    print("=" * 90)
    print(f"Positive samples after bad-patient filtering: {len(records)}")
    print(f"Bad patients excluded: {BAD_PATIENTS}")
    print("=" * 90)

    summary = []

    for ratio in ROI_WIDTH_RATIOS:
        coverages = []
        area_ratios = []
        failures = []

        ratio_dir = os.path.join(OUT_DIR, f"ratio_{ratio:.2f}")
        os.makedirs(ratio_dir, exist_ok=True)

        for idx, rec in enumerate(records):
            pid = rec["patient_id"]
            xray = rec["xray"]
            mask = rec["mask"]

            roi_box = body_center_roi_box(xray, roi_width_ratio=ratio)

            cov = compute_coverage(mask, roi_box)
            area = roi_area_ratio(mask, roi_box)

            coverages.append(cov)
            area_ratios.append(area)

            if cov < COVERAGE_FAIL_THRESHOLD:
                failures.append((pid, cov, area, roi_box))

            # İlk 30 sample + fail olanları görselleştir
            if idx < 30 or cov < COVERAGE_FAIL_THRESHOLD:
                save_overlay(
                    xray=xray,
                    mask=mask,
                    roi_box=roi_box,
                    out_path=os.path.join(ratio_dir, f"{pid}_cov_{cov:.3f}.png"),
                    title=f"{pid} | ratio={ratio:.2f} | coverage={cov:.3f} | area={area:.3f}"
                )

        coverages = np.array(coverages, dtype=np.float32)
        area_ratios = np.array(area_ratios, dtype=np.float32)

        result = {
            "ratio": ratio,
            "n_positive": len(records),
            "mean_coverage": float(np.mean(coverages)),
            "median_coverage": float(np.median(coverages)),
            "p10_coverage": float(np.percentile(coverages, 10)),
            "min_coverage": float(np.min(coverages)),
            "mean_area_ratio": float(np.mean(area_ratios)),
            "median_area_ratio": float(np.median(area_ratios)),
            "failure_count": len(failures),
            "failure_rate": len(failures) / max(len(records), 1),
        }

        summary.append(result)

        print("\n" + "-" * 90)
        print(f"ROI WIDTH RATIO = {ratio:.2f}")
        print("-" * 90)
        print(f"Mean coverage    : {result['mean_coverage']:.4f}")
        print(f"Median coverage  : {result['median_coverage']:.4f}")
        print(f"P10 coverage     : {result['p10_coverage']:.4f}")
        print(f"Min coverage     : {result['min_coverage']:.4f}")
        print(f"Median area ratio: {result['median_area_ratio']:.4f}")
        print(f"Failure count    : {result['failure_count']} / {len(records)}")
        print(f"Failure rate     : {result['failure_rate']:.4f}")

        if failures:
            print("Worst failures:")
            failures_sorted = sorted(failures, key=lambda x: x[1])[:10]
            for pid, cov, area, box in failures_sorted:
                print(f"  {pid:20s} | coverage={cov:.4f} | area={area:.4f} | box={box}")

    best = sorted(
        summary,
        key=lambda r: (
            -r["p10_coverage"],
            -r["median_coverage"],
            r["median_area_ratio"],
        )
    )[0]

    print("\n" + "=" * 90)
    print("RECOMMENDED RATIO")
    print("=" * 90)
    print(json.dumps(best, indent=2))
    print("=" * 90)

    with open(os.path.join(OUT_DIR, "roi_coverage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved overlays and summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()