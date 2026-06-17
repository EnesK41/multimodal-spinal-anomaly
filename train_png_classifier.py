import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from dataset import BAD_PATIENTS, SpineMultimodalDataset
from models.classifier import XrayClassifier
from train import (
    aggregate_patient_level,
    build_patient_split,
    build_xray_encoder,
    compute_binary_metrics,
    count_labels_for_indices,
    find_best_threshold,
    print_metrics,
    save_debug_classification_image,
)


DEFAULT_DEMO_HOLDOUT = [
    "patient_002",
    "patient_004",
    "patient_028",
    "patient_038",
    "patient_040",
    "patient_042",
]


class SpinePngDataset(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file,
        img_size=(1024, 512),
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=0.45,
        scrub_prompt=True,
        label_source="mask",
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.use_body_roi = use_body_roi
        self.roi_width_ratio = roi_width_ratio
        self.scrub_prompt = scrub_prompt
        self.label_source = label_source

        rows = list(csv.DictReader(open(csv_file, encoding="utf-8")))
        filtered = []
        for row in rows:
            pid = str(row["patient_id"]).strip()
            base_pid = SpineMultimodalDataset.base_patient_id(pid)
            if exclude_bad_patients and base_pid in BAD_PATIENTS:
                continue
            if not (self.data_dir / pid / f"{pid}_xray.png").exists():
                continue
            filtered.append(row)

        self.rows = filtered
        self.patient_ids = [str(row["patient_id"]).strip() for row in self.rows]
        print(
            f"[PNG Dataset] samples={len(self.rows)} | data_dir={self.data_dir} | "
            f"use_body_roi={self.use_body_roi} | roi_width_ratio={self.roi_width_ratio}"
        )

    @staticmethod
    def base_patient_id(pid):
        return SpineMultimodalDataset.base_patient_id(pid)

    @staticmethod
    def is_augmented(pid):
        return SpineMultimodalDataset.is_augmented(pid)

    def __len__(self):
        return len(self.rows)

    def normalize_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor

    def resize_with_padding(self, tensor):
        _, h, w = tensor.shape
        target_h, target_w = self.img_size
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
        return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    def body_center_roi_box(self, xray):
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
        roi_w = int(body_w * self.roi_width_ratio)
        x1 = max(0, cx - roi_w // 2)
        x2 = min(W, cx + roi_w // 2)
        margin_y = int(H * 0.02)
        y1 = max(0, y1 - margin_y)
        y2 = min(H, y2 + margin_y)
        if x2 <= x1 or y2 <= y1:
            return 0, H, 0, W
        return y1, y2, x1, x2

    def apply_body_roi(self, xray, mask):
        y1, y2, x1, x2 = self.body_center_roi_box(xray)
        xray_crop = self.resize_with_padding(xray[:, y1:y2, x1:x2])
        mask_crop = self.resize_with_padding(mask[:, y1:y2, x1:x2])
        mask_crop = (mask_crop > 0.1).float()
        return xray_crop, mask_crop, (y1, y2, x1, x2)

    def load_png(self, path, is_mask=False):
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        if is_mask:
            return (tensor > 0.1).float()
        return self.normalize_tensor(tensor)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pid = str(row["patient_id"]).strip()
        base_pid = SpineMultimodalDataset.base_patient_id(pid)
        patient_dir = self.data_dir / pid
        xray = self.load_png(patient_dir / f"{pid}_xray.png", is_mask=False)
        mask_path = patient_dir / f"{pid}_xray_mask.png"
        if mask_path.exists():
            mask = self.load_png(mask_path, is_mask=True)
        else:
            mask = torch.zeros_like(xray)

        xray = self.resize_with_padding(xray)
        mask = self.resize_with_padding(mask)
        mask = (mask > 0.1).float()
        roi_box = torch.tensor([0, xray.shape[1], 0, xray.shape[2]], dtype=torch.long)
        if self.use_body_roi:
            xray, mask, roi_tuple = self.apply_body_roi(xray, mask)
            roi_box = torch.tensor(roi_tuple, dtype=torch.long)

        if self.label_source == "patient_range":
            label = SpineMultimodalDataset.patient_range_label(pid)
        else:
            label = 1.0 if mask.sum().item() > 0 else 0.0

        prompt = row.get("prompt", "") if not self.scrub_prompt else "Spinal x-ray."
        if not prompt:
            prompt = "Spinal x-ray."

        return {
            "patient_id": pid,
            "base_patient_id": base_pid,
            "is_augmented": SpineMultimodalDataset.is_augmented(pid),
            "xray": xray,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.float32),
            "text": prompt,
            "roi_box": roi_box,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collect_patient_labels(dataset):
    labels = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        labels[sample["base_patient_id"]] = max(
            labels.get(sample["base_patient_id"], 0),
            int(sample["label"].item() > 0.5),
        )
    return labels


def write_split_files(out_dir, dataset, train_idx, val_idx, val_patients):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, indices in [("train", train_idx), ("demo_holdout", val_idx)]:
        for idx in indices:
            sample = dataset[idx]
            rows.append(
                {
                    "split": split_name,
                    "patient_id": sample["patient_id"],
                    "base_patient_id": sample["base_patient_id"],
                    "is_augmented": int(sample["is_augmented"]),
                    "label": int(sample["label"].item() > 0.5),
                }
            )
    with open(out_dir / "png_demo_holdout_split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "demo_holdout_patients": val_patients,
        "note": "All augmentations of these base patients are excluded from training. Validation uses original samples only.",
    }
    (out_dir / "png_demo_holdout_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate(xray_enc, classifier, loader, device, threshold):
    xray_enc.eval()
    classifier.eval()
    labels = []
    probs = []
    patient_ids = []
    with torch.no_grad():
        for batch in loader:
            xray = batch["xray"].to(device, non_blocking=True)
            y = batch["label"].float()
            emb, _ = xray_enc(xray)
            logits = classifier(emb).view(-1)
            p = torch.sigmoid(logits).detach().cpu()
            labels.extend(float(v) for v in y)
            probs.extend(float(v) for v in p)
            patient_ids.extend(batch["base_patient_id"])

    sample_metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    pids, patient_labels, patient_probs = aggregate_patient_level(patient_ids, labels, probs, mode="mean")
    default_patient_metrics = compute_binary_metrics(patient_labels, patient_probs, threshold=threshold)
    best_t, best_metrics, _ = find_best_threshold(patient_labels, patient_probs)
    return {
        "sample_metrics": sample_metrics,
        "patient_ids": pids,
        "patient_labels": patient_labels,
        "patient_probs": patient_probs,
        "patient_metrics_default": default_patient_metrics,
        "best_threshold": best_t,
        "patient_metrics_best": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary classifier on exported PNG X-rays.")
    parser.add_argument("--data_dir", default="data/png_patients_full")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--run_name", default="png_full_demo_holdout_seed42")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--no_body_roi", action="store_true")
    parser.add_argument("--encoder_type", choices=["resnet34", "small_cnn"], default="resnet34")
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_save_epoch", type=int, default=3)
    parser.add_argument("--demo_holdout", nargs="*", default=DEFAULT_DEMO_HOLDOUT)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path("experiment_results") / "png_classifier"
    debug_dir = Path("debug_classifier") / args.run_name
    checkpoint_path = Path("checkpoints") / f"{args.run_name}_best.pth"
    result_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = SpinePngDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        img_size=(1024, 512),
        exclude_bad_patients=True,
        use_body_roi=not args.no_body_roi,
        roi_width_ratio=args.roi_width_ratio,
        scrub_prompt=True,
        label_source="mask",
    )

    patient_labels = collect_patient_labels(dataset)
    val_patients = list(args.demo_holdout)
    missing = [pid for pid in val_patients if pid not in patient_labels]
    if missing:
        raise RuntimeError(f"Holdout patients missing from dataset: {missing}")

    train_idx, val_idx = build_patient_split(dataset, val_patients, original_only_val=True)
    write_split_files(result_dir, dataset, train_idx, val_idx, val_patients)
    train_pos, train_neg = count_labels_for_indices(dataset, train_idx, "PNG TRAIN")
    val_pos, val_neg = count_labels_for_indices(dataset, val_idx, "PNG DEMO HOLDOUT")

    print("=" * 100)
    print("PNG CLASSIFIER TRAINING")
    print("=" * 100)
    print(f"Device          : {device}")
    print(f"Run name        : {args.run_name}")
    print(f"Data dir        : {args.data_dir}")
    print(f"Use body ROI    : {not args.no_body_roi}")
    print(f"Holdout patients: {', '.join(val_patients)}")
    print(f"Train samples   : {len(train_idx)} | pos={train_pos} neg={train_neg}")
    print(f"Holdout samples : {len(val_idx)} | pos={val_pos} neg={val_neg}")
    print(f"Checkpoint      : {checkpoint_path}")
    print("=" * 100)

    sample = dataset[train_idx[0]]
    save_debug_classification_image(
        xray=sample["xray"].unsqueeze(0),
        mask=sample["mask"].unsqueeze(0),
        label=sample["label"].item(),
        prob=0.0,
        patient_id=sample["patient_id"],
        epoch=0,
        split="png_sanity",
        out_dir=str(debug_dir),
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    xray_enc = build_xray_encoder(
        encoder_type=args.encoder_type,
        embedding_dim=512,
        pretrained=args.pretrained_encoder,
    ).to(device)
    classifier = XrayClassifier(embedding_dim=512, dropout=0.5).to(device)

    if args.freeze_encoder:
        for param in xray_enc.parameters():
            param.requires_grad = False

    params = [p for p in list(xray_enc.parameters()) + list(classifier.parameters()) if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": xray_enc.parameters(), "lr": args.encoder_lr},
            {"params": classifier.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = -1
    best_payload = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        xray_enc.train(not args.freeze_encoder)
        classifier.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"PNG training {epoch}/{args.epochs}")
        for batch in pbar:
            xray = batch["xray"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            if args.freeze_encoder:
                with torch.no_grad():
                    emb, _ = xray_enc(xray)
            else:
                emb, _ = xray_enc(xray)
            logits = classifier(emb).view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val = evaluate(xray_enc, classifier, val_loader, device, threshold=0.5)
        score = val["patient_metrics_best"]["balanced_accuracy"]
        scheduler.step(score)
        print_metrics(f"PNG HOLDOUT DEFAULT | epoch {epoch}", val["patient_metrics_default"])
        print_metrics(f"PNG HOLDOUT BEST T={val['best_threshold']:.3f} | epoch {epoch}", val["patient_metrics_best"])

        can_save = epoch >= args.min_save_epoch
        if can_save and score > best_score:
            best_score = score
            wait = 0
            best_payload = {
                "epoch": epoch,
                "xray_enc": xray_enc.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "roi_width_ratio": args.roi_width_ratio,
                "use_body_roi": not args.no_body_roi,
                "val_patients": val_patients,
                "original_only_val": True,
                "run_name": args.run_name,
                "encoder_type": args.encoder_type,
                "pretrained_encoder": args.pretrained_encoder,
                "input_format": "png",
                "input_data_dir": args.data_dir,
                "best_threshold": val["best_threshold"],
                "best_score": best_score,
                "holdout_metrics_best_threshold": val["patient_metrics_best"],
                "holdout_metrics_default_threshold": val["patient_metrics_default"],
            }
            torch.save(best_payload, checkpoint_path)
            print(f"SAVED PNG MODEL | epoch={epoch} | balanced_accuracy={best_score:.6f}")
        elif can_save:
            wait += 1
            print(f"Not saved. Patience {wait}/{args.patience}")
            if wait >= args.patience:
                print("Early stopping.")
                break

    if best_payload is None:
        raise RuntimeError("No checkpoint saved. Increase epochs or lower min_save_epoch.")

    summary = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "data_dir": args.data_dir,
        "use_body_roi": not args.no_body_roi,
        "demo_holdout_patients": val_patients,
        "best_epoch": best_payload["epoch"],
        "best_threshold": best_payload["best_threshold"],
        "best_score_balanced_accuracy": best_payload["best_score"],
        "holdout_metrics_best_threshold": best_payload["holdout_metrics_best_threshold"],
        "holdout_metrics_default_threshold": best_payload["holdout_metrics_default_threshold"],
        "split_csv": str(result_dir / "png_demo_holdout_split.csv"),
    }
    out_json = result_dir / f"{args.run_name}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
