import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import SpineMultimodalDataset, print_dataset_stats
from models.XrayEncoder import XrayEncoder
from train import (
    aggregate_patient_level,
    build_patient_split,
    collect_patient_labels,
    compute_binary_metrics,
    find_best_threshold,
    make_stratified_patient_folds,
    print_metrics,
    print_threshold_sweep,
    save_experiment_results,
    set_global_seed,
    summarize_patient_split,
)


REGION_NAMES = ["cervical", "thoracic", "lumbar"]


class RegionAwareDataset(SpineMultimodalDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        text = sample.get("text", "")
        lowered = text.lower() if isinstance(text, str) else ""

        region = torch.tensor(
            [1.0 if name in lowered else 0.0 for name in REGION_NAMES],
            dtype=torch.float32,
        )
        known = 1.0 if region.sum().item() > 0 and sample["label"].item() > 0.5 else 0.0

        sample["region"] = region
        sample["region_known"] = torch.tensor(known, dtype=torch.float32)
        sample["text"] = "Spinal x-ray."
        return sample


class RegionMultiTaskHead(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.anomaly_head = nn.Linear(256, 1)
        self.region_head = nn.Linear(256, len(REGION_NAMES))

    def forward(self, embedding):
        features = self.shared(embedding)
        anomaly_logits = self.anomaly_head(features).squeeze(1)
        region_logits = self.region_head(features)
        return anomaly_logits, region_logits


def make_dataset(args):
    return RegionAwareDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=args.roi_width_ratio,
        scrub_prompt=False,
        label_source="mask",
    )


def count_region_labels(dataset, indices, title):
    totals = np.zeros(len(REGION_NAMES), dtype=np.int64)
    known = 0
    positive = 0
    for idx in indices:
        sample = dataset[idx]
        if sample["label"].item() > 0.5:
            positive += 1
        if sample["region_known"].item() > 0.5:
            known += 1
            totals += sample["region"].numpy().astype(np.int64)

    print("=" * 100)
    print(f"{title} REGION LABELS")
    print("=" * 100)
    print(f"Positive samples with region labels: {known}/{positive}")
    for name, value in zip(REGION_NAMES, totals):
        print(f"{name:10s}: {int(value)}")
    print("=" * 100)


def compute_region_pos_weight(dataset, indices):
    pos = np.zeros(len(REGION_NAMES), dtype=np.float32)
    known = 0.0
    for idx in indices:
        sample = dataset[idx]
        if sample["region_known"].item() > 0.5:
            known += 1.0
            pos += sample["region"].numpy().astype(np.float32)

    neg = np.maximum(known - pos, 0.0)
    return neg / np.maximum(pos, 1.0)


def compute_region_metrics(region_targets, region_probs, known_mask, threshold=0.5):
    targets = np.asarray(region_targets, dtype=np.float32)
    probs = np.asarray(region_probs, dtype=np.float32)
    known = np.asarray(known_mask, dtype=np.float32) > 0.5

    if targets.size == 0 or known.sum() == 0:
        return {
            "n_known": 0,
            "exact_match": None,
            "macro_f1": None,
            "per_region": {},
        }

    targets = targets[known]
    probs = probs[known]
    preds = (probs >= threshold).astype(np.float32)

    exact = float((preds == targets).all(axis=1).mean())
    per_region = {}
    f1s = []
    for i, name in enumerate(REGION_NAMES):
        y = targets[:, i]
        p = preds[:, i]
        tp = int(((y == 1) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        f1s.append(f1)
        per_region[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    return {
        "n_known": int(known.sum()),
        "exact_match": exact,
        "macro_f1": float(sum(f1s) / len(f1s)),
        "per_region": per_region,
    }


def print_region_metrics(title, metrics):
    print("-" * 100)
    print(title)
    print("-" * 100)
    print(f"Known positives : {metrics['n_known']}")
    if metrics["exact_match"] is None:
        print("Region metrics  : N/A")
    else:
        print(f"Exact match     : {metrics['exact_match']:.6f}")
        print(f"Macro F1        : {metrics['macro_f1']:.6f}")
        for name, row in metrics["per_region"].items():
            print(
                f"{name:10s} | f1={row['f1']:.6f} | "
                f"precision={row['precision']:.6f} | recall={row['recall']:.6f} | "
                f"tp/tn/fp/fn={row['tp']}/{row['tn']}/{row['fp']}/{row['fn']}"
            )
    print("-" * 100)


def run_one_fold(args, dataset, fold_idx, val_patients):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"region_multitask_roi{str(args.roi_width_ratio).replace('.', '')}_fold_{fold_idx:02d}_seed{args.seed}"
    checkpoint_path = os.path.join("checkpoints", f"{run_name}.pth")
    os.makedirs("checkpoints", exist_ok=True)

    train_idx, val_idx = build_patient_split(
        dataset=dataset,
        val_patients=val_patients,
        original_only_val=not args.include_val_augmentations,
    )
    if not val_idx:
        raise RuntimeError("Validation split is empty.")

    train_pos = 0
    train_neg = 0
    for idx in train_idx:
        label = dataset[idx]["label"].item()
        if label > 0.5:
            train_pos += 1
        else:
            train_neg += 1

    count_region_labels(dataset, train_idx, "TRAIN")
    count_region_labels(dataset, val_idx, "VAL")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    xray_enc = XrayEncoder(embedding_dim=512, pretrained=args.pretrained_encoder).to(device)
    head = RegionMultiTaskHead(embedding_dim=512, dropout=args.dropout).to(device)

    optimizer = optim.AdamW(
        [
            {"params": xray_enc.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    pos_weight_val = train_neg / max(train_pos, 1) if train_pos > 0 else 1.0
    anomaly_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )
    region_pos_weight = None
    if args.balance_region_loss:
        region_pos_weight = torch.tensor(
            compute_region_pos_weight(dataset, train_idx),
            dtype=torch.float32,
            device=device,
        )
    region_loss_fn = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=region_pos_weight,
    )

    print("=" * 100)
    print("REGION-AWARE MULTITASK TRAINING")
    print("=" * 100)
    print(f"Run name          : {run_name}")
    print(f"Device            : {device}")
    print(f"Pretrained encoder: {args.pretrained_encoder}")
    print(f"Region loss weight: {args.region_loss_weight}")
    print(f"Balance region BCE: {args.balance_region_loss}")
    if region_pos_weight is not None:
        weights = ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(REGION_NAMES, region_pos_weight.detach().cpu().tolist())
        )
        print(f"Region pos_weight : {weights}")
    print(f"BCE pos_weight    : {pos_weight_val:.6f}")
    print(f"Checkpoint        : {checkpoint_path}")
    print("=" * 100)

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5
    patience = 0
    best_payload = None

    for epoch in range(args.epochs):
        print("\n" + "=" * 100)
        print(f"EPOCH {epoch + 1}/{args.epochs} | {run_name}")
        print("=" * 100)

        xray_enc.train()
        head.train()

        train_labels = []
        train_probs = []
        train_patient_ids = []
        train_region_targets = []
        train_region_probs = []
        train_region_known = []

        total_loss = 0.0
        total_anomaly_loss = 0.0
        total_region_loss = 0.0
        region_batches = 0

        pbar = tqdm(train_loader, desc=f"Training {run_name}")
        for batch in pbar:
            xray = batch["xray"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            regions = batch["region"].to(device, non_blocking=True).float()
            region_known = batch["region_known"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            embedding, _ = xray_enc(xray)
            anomaly_logits, region_logits = head(embedding)

            anomaly_loss = anomaly_loss_fn(anomaly_logits.view(-1), labels)
            region_loss = torch.zeros((), device=device)
            if region_known.sum().item() > 0:
                per_sample_region_loss = region_loss_fn(region_logits, regions).mean(dim=1)
                region_loss = (
                    per_sample_region_loss * region_known
                ).sum() / region_known.sum().clamp_min(1.0)
                region_batches += 1

            loss = anomaly_loss + args.region_loss_weight * region_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(xray_enc.parameters()) + list(head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            probs = torch.sigmoid(anomaly_logits.detach())
            region_probs = torch.sigmoid(region_logits.detach())

            total_loss += loss.item()
            total_anomaly_loss += anomaly_loss.item()
            total_region_loss += region_loss.item()

            train_labels.extend(labels.detach().cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            train_patient_ids.extend(batch["base_patient_id"])
            train_region_targets.extend(regions.detach().cpu().tolist())
            train_region_probs.extend(region_probs.detach().cpu().tolist())
            train_region_known.extend(region_known.detach().cpu().tolist())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{anomaly_loss.item():.4f}",
                    "region": f"{region_loss.item():.4f}",
                    "grad": f"{float(grad_norm):.3f}",
                }
            )

        _, train_patient_labels, train_patient_probs = aggregate_patient_level(
            train_patient_ids,
            train_labels,
            train_probs,
            mode="mean",
        )
        train_metrics = compute_binary_metrics(train_patient_labels, train_patient_probs, threshold=0.5)
        train_region_metrics = compute_region_metrics(
            train_region_targets,
            train_region_probs,
            train_region_known,
        )

        print_metrics("TRAIN PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", train_metrics)
        print_region_metrics("TRAIN REGION METRICS | POSITIVE SAMPLES ONLY", train_region_metrics)
        print(f"Train loss total/class/region: {total_loss:.6f} / {total_anomaly_loss:.6f} / {total_region_loss:.6f}")
        print(f"Train batches with region labels: {region_batches}/{len(train_loader)}")

        xray_enc.eval()
        head.eval()

        val_labels = []
        val_probs = []
        val_patient_ids = []
        val_region_targets = []
        val_region_probs = []
        val_region_known = []
        val_rows = []

        with torch.no_grad():
            for batch in val_loader:
                xray = batch["xray"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True).float()
                regions = batch["region"].to(device, non_blocking=True).float()
                region_known = batch["region_known"].to(device, non_blocking=True).float()

                embedding, _ = xray_enc(xray)
                anomaly_logits, region_logits = head(embedding)
                probs = torch.sigmoid(anomaly_logits)
                region_probs = torch.sigmoid(region_logits)

                label_val = labels.item()
                prob_val = probs.item()
                base_pid = batch["base_patient_id"][0]
                patient_id = batch["patient_id"][0]

                val_labels.append(label_val)
                val_probs.append(prob_val)
                val_patient_ids.append(base_pid)
                val_region_targets.extend(regions.cpu().tolist())
                val_region_probs.extend(region_probs.cpu().tolist())
                val_region_known.extend(region_known.cpu().tolist())

                row = {
                    "patient_id": patient_id,
                    "base_patient_id": base_pid,
                    "label": label_val,
                    "prob": prob_val,
                    "region_known": float(region_known.item()),
                    "region_target": regions.squeeze(0).cpu().tolist(),
                    "region_prob": region_probs.squeeze(0).cpu().tolist(),
                }
                val_rows.append(row)

        val_patient_ids_agg, val_patient_labels, val_patient_probs = aggregate_patient_level(
            val_patient_ids,
            val_labels,
            val_probs,
            mode="mean",
        )
        default_metrics = compute_binary_metrics(val_patient_labels, val_patient_probs, threshold=0.5)
        best_t, best_metrics, threshold_rows = find_best_threshold(val_patient_labels, val_patient_probs)
        region_metrics = compute_region_metrics(
            val_region_targets,
            val_region_probs,
            val_region_known,
        )
        scheduler.step(best_metrics["balanced_accuracy"])

        print_threshold_sweep(threshold_rows)
        print_metrics("VAL PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", default_metrics)
        print_metrics(
            f"VAL PATIENT-LEVEL METRICS | MEAN AGG | BEST THRESHOLD={best_t:.3f}",
            best_metrics,
        )
        print_region_metrics("VAL REGION METRICS | POSITIVE SAMPLES ONLY", region_metrics)

        for row in val_rows:
            target = ",".join(
                name for name, value in zip(REGION_NAMES, row["region_target"]) if value > 0.5
            ) or "none"
            pred = ",".join(
                name for name, value in zip(REGION_NAMES, row["region_prob"]) if value >= 0.5
            ) or "none"
            print(
                f"{row['patient_id']:20s} | base={row['base_patient_id']:12s} | "
                f"label={int(row['label'])} | prob={row['prob']:.6f} | "
                f"region_target={target:24s} | region_pred={pred:24s}"
            )

        score = best_metrics["balanced_accuracy"]
        if epoch + 1 >= args.min_save_epoch and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_threshold = best_t
            patience = 0
            best_payload = {
                "fold": fold_idx,
                "run_name": run_name,
                "val_patients": val_patients,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_threshold": best_threshold,
                "default_threshold_metrics": default_metrics,
                "best_threshold_metrics": best_metrics,
                "region_metrics": region_metrics,
                "val_rows": val_rows,
            }
            torch.save(
                {
                    "epoch": best_epoch,
                    "xray_enc": xray_enc.state_dict(),
                    "head": head.state_dict(),
                    "best_score": best_score,
                    "best_threshold": best_threshold,
                    "region_names": REGION_NAMES,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"MODEL SAVED | best_score={best_score:.6f} | threshold={best_threshold:.3f}")
        elif epoch + 1 >= args.min_save_epoch:
            patience += 1
            print(f"MODEL NOT SAVED | patience={patience}/{args.early_stopping_patience}")

        if epoch + 1 >= args.min_save_epoch and patience >= args.early_stopping_patience:
            print("EARLY STOPPING")
            break

    if best_payload is None:
        best_payload = {
            "fold": fold_idx,
            "run_name": run_name,
            "val_patients": val_patients,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_threshold": best_threshold,
        }

    best_payload["checkpoint"] = checkpoint_path
    return best_payload


def run_cv(args):
    set_global_seed(args.seed)
    dataset = make_dataset(args)
    print_dataset_stats(dataset)
    patient_labels = collect_patient_labels(dataset)
    folds = make_stratified_patient_folds(patient_labels, n_splits=args.folds, seed=args.seed)

    results = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        summarize_patient_split(patient_labels, val_patients, f"REGION MULTITASK CV FOLD {fold_idx}/{args.folds}")
        result = run_one_fold(args, dataset, fold_idx, val_patients)
        results.append(result)

    scores = [row["best_score"] for row in results if row.get("best_score", -1) >= 0]
    region_exact = [
        row.get("region_metrics", {}).get("exact_match")
        for row in results
        if row.get("region_metrics", {}).get("exact_match") is not None
    ]
    region_macro_f1 = [
        row.get("region_metrics", {}).get("macro_f1")
        for row in results
        if row.get("region_metrics", {}).get("macro_f1") is not None
    ]

    payload = {
        "experiment": "region_multitask",
        "seed": args.seed,
        "folds": args.folds,
        "region_names": REGION_NAMES,
        "mean_best_balanced_accuracy": sum(scores) / max(len(scores), 1),
        "mean_region_exact_match": sum(region_exact) / max(len(region_exact), 1),
        "mean_region_macro_f1": sum(region_macro_f1) / max(len(region_macro_f1), 1),
        "args": vars(args),
        "results": results,
    }
    balance_suffix = "_balanced_region" if args.balance_region_loss else ""
    filename = (
        f"region_multitask_roi{str(args.roi_width_ratio).replace('.', '')}_"
        f"w{str(args.region_loss_weight).replace('.', '')}{balance_suffix}_seed{args.seed}.json"
    )
    save_experiment_results(filename, payload)

    print("=" * 100)
    print("REGION MULTITASK CV SUMMARY")
    print("=" * 100)
    print(f"Mean best balanced accuracy: {payload['mean_best_balanced_accuracy']:.6f}")
    print(f"Mean region exact match    : {payload['mean_region_exact_match']:.6f}")
    print(f"Mean region macro F1       : {payload['mean_region_macro_f1']:.6f}")
    for row in results:
        rm = row.get("region_metrics", {})
        print(
            f"Fold {row['fold']}: best_score={row.get('best_score', -1):.6f} | "
            f"epoch={row.get('best_epoch', -1)} | threshold={row.get('best_threshold', 0.5):.3f} | "
            f"region_exact={rm.get('exact_match')}"
        )
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Region-aware X-ray multitask classifier")
    parser.add_argument("--data_dir", default="data/augmented_patients")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--region_loss_weight", type=float, default=0.2)
    parser.add_argument("--balance_region_loss", action="store_true")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min_save_epoch", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--include_val_augmentations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_cv(parse_args())
