import argparse
import os
import shutil

import nibabel as nib
import numpy as np


def _patient_num(patient_id: str) -> int:
    digits = "".join(ch for ch in patient_id if ch.isdigit())
    return int(digits) if digits else -1


def _rotate_and_save(src_path: str, is_mask: bool) -> None:
    img = nib.as_closest_canonical(nib.load(src_path))
    data = np.squeeze(img.get_fdata())

    if data.ndim != 2:
        raise ValueError(f"Beklenen 2D veri degil: {src_path}, shape={data.shape}")

    data = np.rot90(data, k=1)

    if is_mask:
        data = (data > 0.5).astype(np.uint8)
    else:
        data = data.astype(np.float32)

    out_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(out_img, src_path)


def rotate_dataset_once(data_dir: str, backup_dir: str, max_patient_number: int | None, apply: bool, force: bool) -> None:
    patient_dirs = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("patient_")
    ]

    if max_patient_number is not None:
        patient_dirs = [p for p in patient_dirs if _patient_num(p) <= max_patient_number]

    targets = []
    for patient_id in patient_dirs:
        patient_path = os.path.join(data_dir, patient_id)
        xray = os.path.join(patient_path, f"{patient_id}_xray.nii.gz")
        mask = os.path.join(patient_path, f"{patient_id}_xray_mask.nii.gz")

        if os.path.exists(xray):
            targets.append((xray, False))
        if os.path.exists(mask):
            targets.append((mask, True))

    print(f"Toplam islenecek dosya: {len(targets)}")
    print(f"Mod: {'APPLY' if apply else 'DRY-RUN'}")

    for src_path, is_mask in targets:
        rel = os.path.relpath(src_path, data_dir)
        backup_path = os.path.join(backup_dir, rel)

        if os.path.exists(backup_path) and not force:
            print(f"SKIP (zaten yedek var): {src_path}")
            continue

        print(f"ROTATE: {src_path}")
        if not apply:
            continue

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(src_path, backup_path)
        _rotate_and_save(src_path, is_mask=is_mask)

    if apply:
        print("Bitti. Dosyalar yerinde donduruldu, orijinaller backup klasorunde.")
        print(f"Backup klasoru: {backup_dir}")
    else:
        print("Dry-run tamamlandi. Gercek uygulama icin --apply ekle.")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D X-ray ve maske NIfTI dosyalarini bir kereye mahsus 90 derece saga dondurur.")
    parser.add_argument("--data-dir", default="data", help="Veri klasoru")
    parser.add_argument("--backup-dir", default="data_backup_before_rotate", help="Orijinal dosya yedek klasoru")
    parser.add_argument("--max-patient-number", type=int, default=45, help="Maksimum hasta numarasi (ornek: 45)")
    parser.add_argument("--apply", action="store_true", help="Gercekten uygula")
    parser.add_argument("--force", action="store_true", help="Yedek olsa da tekrar uygula")
    args = parser.parse_args()

    rotate_dataset_once(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        max_patient_number=args.max_patient_number,
        apply=args.apply,
        force=args.force,
    )


if __name__ == "__main__":
    main()
