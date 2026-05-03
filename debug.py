import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from dataset import SpineMultimodalDataset


DATA_DIR = "data/augmented_patients"
CSV_FILE = "data/train_labels_augmented.csv"
OUT_DIR = "debug_roi_crop"

IMG_SIZE = (1024, 512)

# Şimdilik elle şüpheli gördüklerimizi buraya yazıyoruz.
# İstersen bu listeyi sonra değiştiririz.
BAD_PATIENTS = [
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

VAL_PATIENT_IDS = [
    "patient_004",
    "patient_012",
    "patient_028",
    "patient_045",
    # "patient_051",  # bad listteyse validationdan da çıkar
]


def base_pid(pid):
    return pid.split("_aug")[0]


def save_overlay(xray, mask, roi_box, out_path, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = xray[0].cpu().numpy()
    m = mask[0].cpu().numpy()

    y1, y2, x1, x2 = roi_box

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(x, cmap="gray")
    plt.title("Full X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(x, cmap="gray")
    plt.imshow(m, cmap="Reds", alpha=0.45)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linewidth=2)
    plt.title(f"Full + GT + ROI | mask sum={m.sum():.0f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(x[y1:y2, x1:x2], cmap="gray")
    plt.imshow(m[y1:y2, x1:x2], cmap="Reds", alpha=0.45)
    plt.title("ROI Crop + GT")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def body_center_roi(xray, roi_width_ratio=0.45, margin_y_ratio=0.02):
    """
    Mask kullanmadan X-ray üzerinden kaba gövde/omurga ROI bulur.
    Mantık:
    - Siyah background dışındaki body bbox bulunur.
    - Merkez x alınır.
    - Body width'in belli oranı kadar merkezi dikey band crop edilir.
    """
    x = xray[0]

    # normalize edilmiş input bekliyoruz
    threshold = max(0.03, float(x.mean()) * 0.5)
    fg = x > threshold

    ys, xs = torch.where(fg)

    H, W = x.shape

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

    margin_y = int(H * margin_y_ratio)

    y1 = max(0, y1 - margin_y)
    y2 = min(H, y2 + margin_y)

    return y1, y2, x1, x2


def oracle_mask_roi(mask, margin=128):
    """
    Sadece deney için: GT mask çevresinden ROI çıkarır.
    Gerçek inference için kullanılmaz.
    """
    H, W = mask.shape[1], mask.shape[2]

    ys, xs = torch.where(mask[0] > 0)

    if len(ys) == 0:
        return 0, H, 0, W

    y1 = max(0, int(ys.min().item()) - margin)
    y2 = min(H, int(ys.max().item()) + margin)

    x1 = max(0, int(xs.min().item()) - margin)
    x2 = min(W, int(xs.max().item()) + margin)

    return y1, y2, x1, x2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = SpineMultimodalDataset(
        data_dir=DATA_DIR,
        csv_file=CSV_FILE,
        img_size=IMG_SIZE,
        is_train=False
    )

    kept_indices = []
    removed = []

    for i, pid in enumerate(dataset.patient_ids):
        bpid = base_pid(pid)

        if bpid in BAD_PATIENTS:
            removed.append(pid)
        else:
            kept_indices.append(i)

    kept_positive = 0
    kept_empty = 0

    print("=" * 80)
    print("ROI CROP DEBUG")
    print("=" * 80)
    print(f"Total samples   : {len(dataset)}")
    print(f"Kept samples    : {len(kept_indices)}")
    print(f"Removed samples : {len(removed)}")
    print(f"Bad patients    : {BAD_PATIENTS}")
    print("=" * 80)

    for count, idx in enumerate(kept_indices):
        sample = dataset[idx]

        pid = sample["patient_id"]
        xray = sample["xray"]
        mask = sample["mask"]

        mask_sum = mask.sum().item()

        if mask_sum > 0:
            kept_positive += 1
        else:
            kept_empty += 1

        roi_body = body_center_roi(xray, roi_width_ratio=0.45)
        roi_oracle = oracle_mask_roi(mask, margin=128)

        # Çok fazla dosya basmasın diye:
        # - tüm val hastaları
        # - ilk 40 örnek
        # - pozitif maskeli örneklerin bir kısmı
        should_save = (
            base_pid(pid) in VAL_PATIENT_IDS
            or count < 40
        )

        if should_save:
            save_overlay(
                xray,
                mask,
                roi_body,
                os.path.join(OUT_DIR, "body_center_roi", f"{pid}.png"),
                f"{pid} | BODY CENTER ROI"
            )

            save_overlay(
                xray,
                mask,
                roi_oracle,
                os.path.join(OUT_DIR, "oracle_mask_roi", f"{pid}.png"),
                f"{pid} | ORACLE MASK ROI"
            )

        print(
            f"{pid:20s} | "
            f"mask_sum={mask_sum:8.0f} | "
            f"body_roi={roi_body} | "
            f"oracle_roi={roi_oracle}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Kept positive samples: {kept_positive}")
    print(f"Kept empty samples   : {kept_empty}")
    print(f"Removed samples      : {len(removed)}")
    print(f"Output folder        : {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()