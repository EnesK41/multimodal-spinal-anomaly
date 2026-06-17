import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import SpineMultimodalDataset
from models.MaskDecoder import MaskDecoder
from models.XrayEncoder import XrayEncoder


def dice_iou(pred_bin, target_bin):
    pred = pred_bin.float()
    target = target_bin.float()
    intersection = (pred * target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection) / max(pred_sum + target_sum, 1e-8)
    iou = intersection / max(union, 1e-8)
    precision = intersection / max(pred_sum, 1e-8)
    recall = intersection / max(target_sum, 1e-8)
    return dice, iou, precision, recall, pred_sum, target_sum


def save_overlay(xray, target, prob, row, out_path):
    xray_np = xray[0].detach().cpu().numpy()
    target_np = target[0].detach().cpu().numpy()
    prob_np = prob[0].detach().cpu().numpy()
    pred_np = (prob_np >= row["threshold"]).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(xray_np, cmap="gray")
    axes[0].set_title(f"{row['patient_id']} | X-ray")

    axes[1].imshow(xray_np, cmap="gray")
    axes[1].imshow(target_np, cmap="Reds", alpha=0.45)
    axes[1].set_title("GT mask")

    axes[2].imshow(prob_np, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Pred prob")

    axes[3].imshow(xray_np, cmap="gray")
    axes[3].imshow(pred_np, cmap="Blues", alpha=0.45)
    axes[3].set_title(f"Pred mask | Dice={row['dice']:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summarize(rows, threshold):
    fixed = [row for row in rows if abs(row["threshold"] - threshold) < 1e-8]
    positives = [row for row in fixed if row["target_pixels"] > 0]
    healthy = [row for row in fixed if row["target_pixels"] <= 0]

    def mean(values):
        return sum(values) / max(len(values), 1)

    healthy_empty = [
        1.0 if row["pred_pixels"] <= 0 else 0.0
        for row in healthy
    ]
    healthy_empty_specificity = mean(healthy_empty) if healthy else None

    return {
        "threshold": threshold,
        "n": len(fixed),
        "positive_n": len(positives),
        "healthy_n": len(healthy),
        "positive_mean_dice": mean([row["dice"] for row in positives]),
        "positive_mean_iou": mean([row["iou"] for row in positives]),
        "positive_mean_precision": mean([row["precision"] for row in positives]),
        "positive_mean_recall": mean([row["recall"] for row in positives]),
        "healthy_empty_specificity": healthy_empty_specificity,
        "healthy_false_mask_rate": 1.0 - healthy_empty_specificity if healthy else None,
        "mean_pred_pixels_positive": mean([row["pred_pixels"] for row in positives]),
        "mean_target_pixels_positive": mean([row["target_pixels"] for row in positives]),
        "mean_pred_pixels_healthy": mean([row["pred_pixels"] for row in healthy]) if healthy else None,
    }


def best_threshold_summary(rows, thresholds):
    summaries = [summarize(rows, threshold) for threshold in thresholds]
    return max(
        summaries,
        key=lambda row: (
            row["positive_mean_dice"],
            row["healthy_empty_specificity"] if row["healthy_empty_specificity"] is not None else -1.0,
            row["positive_mean_iou"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate X-ray anomaly segmentation checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_roi_045_keep_scoliosis.pth")
    parser.add_argument("--data_dir", default="data/augmented_patients")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--output_dir", default="experiment_results/segmentation_eval")
    parser.add_argument("--img_height", type=int, default=1024)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--roi_width_ratio", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--exclude_bad_patients", action="store_true", default=True)
    parser.add_argument("--include_val_augmentations", action="store_true")
    parser.add_argument("--max_overlays", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    val_patients = ckpt.get("val_patients")
    if not val_patients:
        raise ValueError(
            "Checkpoint does not contain val_patients. Use a checkpoint with patient-level validation metadata."
        )

    roi_width_ratio = args.roi_width_ratio
    if roi_width_ratio is None:
        roi_width_ratio = float(ckpt.get("roi_width_ratio", 0.45))

    dataset = SpineMultimodalDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        img_size=(args.img_height, args.img_width),
        is_train=False,
        exclude_bad_patients=args.exclude_bad_patients,
        use_body_roi=True,
        roi_width_ratio=roi_width_ratio,
        scrub_prompt=True,
        label_source="mask",
    )

    val_set = set(val_patients)
    indices = []
    for idx, pid in enumerate(dataset.patient_ids):
        sample_base = dataset.base_patient_id(pid)
        if sample_base not in val_set:
            continue
        if not args.include_val_augmentations and dataset.is_augmented(pid):
            continue
        indices.append(idx)

    if not indices:
        raise RuntimeError(f"No validation samples found for val_patients={val_patients}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xray_enc = XrayEncoder(embedding_dim=512, pretrained=False).to(device)
    mask_dec = MaskDecoder(embedding_dim=512).to(device)
    xray_enc.load_state_dict(ckpt["xray_enc"])
    mask_dec.load_state_dict(ckpt["mask_dec"])
    xray_enc.eval()
    mask_dec.eval()

    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    thresholds = sorted(set([args.threshold] + [round(x / 20.0, 2) for x in range(1, 20)]))

    rows = []
    overlay_count = 0

    with torch.no_grad():
        for batch in loader:
            xray = batch["xray"].to(device)
            target = batch["mask"].to(device)
            embedding, skip = xray_enc(xray)
            logits = mask_dec(embedding, skip)
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)

            patient_id = batch["patient_id"][0]
            base_patient_id = batch["base_patient_id"][0]
            target_bin = (target > 0.5).float()

            for threshold in thresholds:
                pred_bin = (prob >= threshold).float()
                dice, iou, precision, recall, pred_sum, target_sum = dice_iou(pred_bin, target_bin)
                rows.append(
                    {
                        "patient_id": patient_id,
                        "base_patient_id": base_patient_id,
                        "threshold": threshold,
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                        "pred_pixels": pred_sum,
                        "target_pixels": target_sum,
                    }
                )

            if overlay_count < args.max_overlays:
                fixed_row = next(
                    row for row in rows
                    if row["patient_id"] == patient_id and abs(row["threshold"] - args.threshold) < 1e-8
                )
                out_path = os.path.join(args.output_dir, f"{patient_id}_seg_overlay.png")
                save_overlay(xray[0].cpu(), target[0].cpu(), prob[0].cpu(), fixed_row, out_path)
                overlay_count += 1

    fixed_summary = summarize(rows, args.threshold)
    best_summary = best_threshold_summary(rows, thresholds)

    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val_dice": ckpt.get("best_val_dice"),
        "val_patients": val_patients,
        "indices_evaluated": len(indices),
        "original_only_val": not args.include_val_augmentations,
        "roi_width_ratio": roi_width_ratio,
        "image_size": [args.img_height, args.img_width],
        "fixed_threshold_summary": fixed_summary,
        "best_threshold_summary": best_summary,
        "thresholds": thresholds,
    }

    result_json = os.path.join(args.output_dir, "segmentation_eval_summary.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows_csv = os.path.join(args.output_dir, "segmentation_eval_rows.csv")
    with open(rows_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("SEGMENTATION EVALUATION")
    print("=" * 100)
    print(f"Checkpoint              : {args.checkpoint}")
    print(f"Checkpoint best val Dice: {ckpt.get('best_val_dice')}")
    print(f"Val patients            : {', '.join(val_patients)}")
    print(f"Samples evaluated       : {len(indices)}")
    print(f"Original-only val       : {not args.include_val_augmentations}")
    print(f"ROI width ratio         : {roi_width_ratio}")
    print("-" * 100)
    print(f"Fixed threshold         : {args.threshold}")
    print(f"Positive mean Dice      : {fixed_summary['positive_mean_dice']:.6f}")
    print(f"Positive mean IoU       : {fixed_summary['positive_mean_iou']:.6f}")
    print(f"Positive mean precision : {fixed_summary['positive_mean_precision']:.6f}")
    print(f"Positive mean recall    : {fixed_summary['positive_mean_recall']:.6f}")
    if fixed_summary["healthy_empty_specificity"] is None:
        print("Healthy empty specificity: N/A (no empty-mask validation patients)")
    else:
        print(f"Healthy empty specificity: {fixed_summary['healthy_empty_specificity']:.6f}")
    print("-" * 100)
    print(f"Best diagnostic threshold: {best_summary['threshold']}")
    print(f"Best positive mean Dice  : {best_summary['positive_mean_dice']:.6f}")
    print(f"Best positive mean IoU   : {best_summary['positive_mean_iou']:.6f}")
    if best_summary["healthy_empty_specificity"] is None:
        print("Best healthy specificity : N/A (no empty-mask validation patients)")
    else:
        print(f"Best healthy specificity : {best_summary['healthy_empty_specificity']:.6f}")
    print("-" * 100)
    print(f"Summary JSON            : {result_json}")
    print(f"Rows CSV                : {rows_csv}")
    print(f"Overlays                : {args.output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
