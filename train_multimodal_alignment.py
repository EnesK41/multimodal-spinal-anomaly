import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import nibabel as nib

from dataset import SpineMultimodalDataset, print_dataset_stats
from models.XrayEncoder import XrayEncoder
from models.classifier import XrayClassifier
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


class SimpleCTProjectionDataset(SpineMultimodalDataset):
    """
    X-ray dataset with optional CT projection teacher.

    This intentionally does not do registration or manual cropping. For each
    augmented X-ray sample, the base patient id is used to find the original CT.
    If CT is absent, has_ct=0 and the alignment loss is skipped for that sample.
    """

    def __init__(
        self,
        *args,
        original_patients_dir="data/original_patients",
        ct_cache_dir="data/ct_projections_simple",
        ct_projection_axis=0,
        ct_projection_method="max",
        ct_projection_source="ct",
        ct_hu_threshold=None,
        ct_clip_max=1800.0,
        ct_rotate_k=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.original_patients_dir = original_patients_dir
        self.ct_cache_dir = ct_cache_dir
        self.ct_projection_axis = ct_projection_axis
        self.ct_projection_method = ct_projection_method
        self.ct_projection_source = ct_projection_source
        self.ct_hu_threshold = ct_hu_threshold
        self.ct_clip_max = ct_clip_max
        self.ct_rotate_k = ct_rotate_k
        os.makedirs(self.ct_cache_dir, exist_ok=True)

        if self.ct_projection_source not in {"ct", "ct_mask", "masked_ct"}:
            raise ValueError(
                f"Unknown ct_projection_source={self.ct_projection_source}. "
                "Use 'ct', 'ct_mask', or 'masked_ct'."
            )

        self.ct_pairs = {
            pid: self.find_ct_pair(self.base_patient_id(pid))
            for pid in self.patient_ids
        }
        paired = sum(1 for pair in self.ct_pairs.values() if self.has_required_ct_files(pair))
        paired_base = {
            self.base_patient_id(pid)
            for pid, pair in self.ct_pairs.items()
            if self.has_required_ct_files(pair)
        }
        print(
            f"[CT Projection] samples_with_ct={paired}/{len(self.patient_ids)} | "
            f"base_patients_with_ct={len(paired_base)} | "
            f"axis={self.ct_projection_axis} | method={self.ct_projection_method} | "
            f"source={self.ct_projection_source} | "
            f"hu_threshold={self.ct_hu_threshold} | rotate_k={self.ct_rotate_k} | "
            f"cache={self.ct_cache_dir}"
        )

    def find_ct_pair(self, base_patient_id):
        folder = os.path.join(self.original_patients_dir, base_patient_id)
        if not os.path.isdir(folder):
            return {"ct": None, "ct_mask": None}

        ct_path = None
        ct_mask_path = None
        for name in os.listdir(folder):
            lowered = name.lower()
            if lowered.endswith("_ct.nii.gz"):
                ct_path = os.path.join(folder, name)
            elif lowered.endswith("_ct_mask.nii.gz"):
                ct_mask_path = os.path.join(folder, name)
        return {"ct": ct_path, "ct_mask": ct_mask_path}

    def has_required_ct_files(self, pair):
        if pair is None:
            return False
        if self.ct_projection_source == "ct":
            return pair.get("ct") is not None
        if self.ct_projection_source == "ct_mask":
            return pair.get("ct_mask") is not None
        return pair.get("ct") is not None and pair.get("ct_mask") is not None

    def ct_cache_path(self, base_patient_id):
        filename = (
            f"{base_patient_id}_{self.ct_projection_source}_axis{self.ct_projection_axis}_"
            f"{self.ct_projection_method}_"
            f"hu{self.ct_hu_threshold if self.ct_hu_threshold is not None else 'none'}_"
            f"rot{self.ct_rotate_k}_{self.img_size[0]}x{self.img_size[1]}.pt"
        )
        return os.path.join(self.ct_cache_dir, filename)

    def load_volume(self, path):
        img_obj = nib.as_closest_canonical(nib.load(path))
        volume = np.asarray(img_obj.get_fdata(), dtype=np.float32)
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got shape {volume.shape}")
        return np.nan_to_num(volume)

    def load_ct_projection(self, base_patient_id, pair):
        cache_path = self.ct_cache_path(base_patient_id)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        if self.ct_projection_source == "ct":
            volume = self.load_volume(pair["ct"])
        elif self.ct_projection_source == "ct_mask":
            volume = (self.load_volume(pair["ct_mask"]) > 0).astype(np.float32)
        else:
            volume = self.load_volume(pair["ct"])
            mask = (self.load_volume(pair["ct_mask"]) > 0).astype(np.float32)
            if volume.shape != mask.shape:
                raise ValueError(
                    f"CT and CT mask shape mismatch for {base_patient_id}: "
                    f"{volume.shape} vs {mask.shape}"
                )
            volume = volume * mask

        axis = int(self.ct_projection_axis)
        if axis < 0 or axis >= volume.ndim:
            raise ValueError(f"Invalid ct_projection_axis={axis} for shape {volume.shape}")

        if self.ct_projection_source != "ct_mask" and self.ct_hu_threshold is not None:
            volume = np.where(
                volume >= float(self.ct_hu_threshold),
                np.clip(volume, float(self.ct_hu_threshold), float(self.ct_clip_max)),
                0.0,
            )
        elif self.ct_projection_source != "ct_mask":
            low, high = np.percentile(volume, [1, 99])
            if high > low:
                volume = np.clip(volume, low, high)

        if self.ct_projection_method == "max":
            projection = volume.max(axis=axis)
        elif self.ct_projection_method == "mean":
            projection = volume.mean(axis=axis)
        else:
            raise ValueError(f"Unknown ct_projection_method={self.ct_projection_method}")

        tensor = torch.tensor(projection, dtype=torch.float32).unsqueeze(0)
        tensor = self.normalize_tensor(tensor)
        if self.ct_rotate_k != 0:
            tensor = torch.rot90(tensor, k=int(self.ct_rotate_k), dims=(1, 2))
        tensor = self.resize_with_padding(tensor)
        torch.save(tensor, cache_path)
        return tensor

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        base_pid = sample["base_patient_id"]
        pair = self.ct_pairs.get(sample["patient_id"])

        if not self.has_required_ct_files(pair):
            sample["ct_projection"] = torch.zeros_like(sample["xray"])
            sample["has_ct"] = torch.tensor(0.0, dtype=torch.float32)
            sample["ct_path"] = ""
            sample["ct_mask_path"] = ""
            return sample

        sample["ct_projection"] = self.load_ct_projection(base_pid, pair)
        sample["has_ct"] = torch.tensor(1.0, dtype=torch.float32)
        sample["ct_path"] = pair.get("ct") or ""
        sample["ct_mask_path"] = pair.get("ct_mask") or ""
        return sample


def init_identity_embedding_head(encoder):
    if not isinstance(encoder.fc, nn.Linear):
        return
    if encoder.fc.in_features != encoder.fc.out_features:
        return
    with torch.no_grad():
        encoder.fc.weight.zero_()
        encoder.fc.weight.copy_(torch.eye(encoder.fc.in_features))
        encoder.fc.bias.zero_()


def save_multimodal_debug_image(sample, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xray = sample["xray"][0].detach().cpu().numpy()
    ct = sample["ct_projection"][0].detach().cpu().numpy()
    mask = sample["mask"][0].detach().cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(xray, cmap="gray")
    plt.title(f"X-ray | {sample['patient_id']}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ct, cmap="gray")
    plt.title(f"Simple CT projection | has_ct={int(sample['has_ct'].item())}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(xray, cmap="gray")
    if mask.sum() > 0:
        plt.imshow(mask, cmap="Reds", alpha=0.45)
    plt.title(f"Mask overlay | label={int(sample['label'].item())}")
    plt.axis("off")

    plt.tight_layout()
    filename = f"multimodal_sanity_{sample['patient_id']}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


def make_dataset(args):
    return SimpleCTProjectionDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=args.roi_width_ratio,
        scrub_prompt=True,
        label_source="mask",
        original_patients_dir=args.original_patients_dir,
        ct_cache_dir=args.ct_cache_dir,
        ct_projection_axis=args.ct_projection_axis,
        ct_projection_method=args.ct_projection_method,
        ct_projection_source=args.ct_projection_source,
        ct_hu_threshold=args.ct_hu_threshold,
        ct_clip_max=args.ct_clip_max,
        ct_rotate_k=args.ct_rotate_k,
    )


def count_ct_for_indices(dataset, indices, title):
    samples_with_ct = 0
    base_with_ct = set()
    positive_with_ct = 0
    negative_with_ct = 0

    for idx in indices:
        sample = dataset[idx]
        if sample["has_ct"].item() > 0.5:
            samples_with_ct += 1
            base_with_ct.add(sample["base_patient_id"])
            if sample["label"].item() > 0.5:
                positive_with_ct += 1
            else:
                negative_with_ct += 1

    print("=" * 100)
    print(f"{title} CT AVAILABILITY")
    print("=" * 100)
    print(f"Samples with CT          : {samples_with_ct}/{len(indices)}")
    print(f"Base patients with CT    : {len(base_with_ct)}")
    print(f"Positive samples with CT : {positive_with_ct}")
    print(f"Negative samples with CT : {negative_with_ct}")
    print("=" * 100)


def run_one_fold(args, dataset, fold_idx, val_patients):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = (
        f"multimodal_simple_{args.ct_projection_source}_"
        f"{args.ct_projection_method}_axis{args.ct_projection_axis}_"
        f"fold_{fold_idx:02d}_seed{args.seed}"
    )
    checkpoint_path = os.path.join("checkpoints", f"{run_name}.pth")
    debug_dir = os.path.join(args.debug_dir, run_name)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    train_idx, val_idx = build_patient_split(
        dataset=dataset,
        val_patients=val_patients,
        original_only_val=not args.include_val_augmentations,
    )
    if not val_idx:
        raise RuntimeError("Validation split is empty.")

    count_ct_for_indices(dataset, train_idx, "TRAIN")
    count_ct_for_indices(dataset, val_idx, "VAL")

    train_pos = 0
    train_neg = 0
    for idx in train_idx:
        label = dataset[idx]["label"].item()
        if label > 0.5:
            train_pos += 1
        else:
            train_neg += 1

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
    ct_enc = XrayEncoder(embedding_dim=512, pretrained=args.pretrained_encoder).to(device)
    classifier = XrayClassifier(embedding_dim=512, dropout=args.dropout).to(device)

    if args.identity_embedding_head:
        init_identity_embedding_head(xray_enc)
        init_identity_embedding_head(ct_enc)

    for param in ct_enc.parameters():
        param.requires_grad = False
    ct_enc.eval()

    optimizer = optim.AdamW(
        [
            {"params": xray_enc.parameters(), "lr": args.encoder_lr},
            {"params": classifier.parameters(), "lr": args.classifier_lr},
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
    cls_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )

    print("=" * 100)
    print("SIMPLE MULTIMODAL ALIGNMENT TRAINING")
    print("=" * 100)
    print(f"Run name               : {run_name}")
    print(f"Device                 : {device}")
    print(f"Pretrained encoders    : {args.pretrained_encoder}")
    print(f"Identity heads         : {args.identity_embedding_head}")
    print(f"CT encoder frozen      : True")
    print(f"Alignment weight       : {args.alignment_weight}")
    print(
        f"CT projection          : {args.ct_projection_source} "
        f"{args.ct_projection_method} axis={args.ct_projection_axis}"
    )
    print(f"CT HU threshold        : {args.ct_hu_threshold}")
    print(f"CT clip max            : {args.ct_clip_max}")
    print(f"CT rotate k            : {args.ct_rotate_k}")
    print(f"BCE pos_weight         : {pos_weight_val:.6f}")
    print(f"Checkpoint             : {checkpoint_path}")
    print("=" * 100)

    first_ct_sample = next((dataset[idx] for idx in train_idx if dataset[idx]["has_ct"].item() > 0.5), dataset[train_idx[0]])
    save_multimodal_debug_image(first_ct_sample, debug_dir)

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
        classifier.train()
        ct_enc.eval()

        train_labels = []
        train_probs = []
        train_patient_ids = []
        loss_total = 0.0
        cls_total = 0.0
        align_total = 0.0
        align_batches = 0
        align_cos_values = []

        pbar = tqdm(train_loader, desc=f"Training {run_name}")
        for batch in pbar:
            xray = batch["xray"].to(device, non_blocking=True)
            ct_projection = batch["ct_projection"].to(device, non_blocking=True)
            has_ct = batch["has_ct"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            xray_emb, _ = xray_enc(xray)
            logits = classifier(xray_emb).view(-1)
            cls_loss = cls_loss_fn(logits, labels)

            align_loss = torch.zeros((), device=device)
            if has_ct.sum().item() > 0:
                with torch.no_grad():
                    ct_emb, _ = ct_enc(ct_projection)
                mask = has_ct > 0.5
                x_norm = F.normalize(xray_emb[mask], dim=1)
                ct_norm = F.normalize(ct_emb[mask], dim=1)
                cos = (x_norm * ct_norm).sum(dim=1)
                align_loss = (1.0 - cos).mean()
                align_cos_values.extend(cos.detach().cpu().tolist())
                align_batches += 1

            loss = cls_loss + args.alignment_weight * align_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(xray_enc.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            probs = torch.sigmoid(logits.detach())
            loss_total += loss.item()
            cls_total += cls_loss.item()
            align_total += align_loss.item()

            train_labels.extend(labels.detach().cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            train_patient_ids.extend(batch["base_patient_id"])

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "align": f"{align_loss.item():.4f}",
                    "grad": f"{float(grad_norm):.3f}",
                }
            )

        train_patient_ids_agg, train_patient_labels, train_patient_probs = aggregate_patient_level(
            train_patient_ids,
            train_labels,
            train_probs,
            mode="mean",
        )
        train_metrics = compute_binary_metrics(train_patient_labels, train_patient_probs, threshold=0.5)

        print_metrics("TRAIN PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", train_metrics)
        print(f"Train loss total/class/align: {loss_total:.6f} / {cls_total:.6f} / {align_total:.6f}")
        if align_cos_values:
            print(
                f"Train paired CT cosine mean/std: "
                f"{np.mean(align_cos_values):.6f} / {np.std(align_cos_values):.6f}"
            )
        else:
            print("Train paired CT cosine mean/std: N/A")

        xray_enc.eval()
        classifier.eval()
        val_labels = []
        val_probs = []
        val_patient_ids = []
        val_rows = []
        val_align_cos_values = []

        with torch.no_grad():
            for batch in val_loader:
                xray = batch["xray"].to(device, non_blocking=True)
                ct_projection = batch["ct_projection"].to(device, non_blocking=True)
                has_ct = batch["has_ct"].to(device, non_blocking=True).float()
                labels = batch["label"].to(device, non_blocking=True).float()

                xray_emb, _ = xray_enc(xray)
                logits = classifier(xray_emb).view(-1)
                probs = torch.sigmoid(logits)

                align_cos = None
                if has_ct.sum().item() > 0:
                    ct_emb, _ = ct_enc(ct_projection)
                    x_norm = F.normalize(xray_emb, dim=1)
                    ct_norm = F.normalize(ct_emb, dim=1)
                    align_cos = (x_norm * ct_norm).sum(dim=1).item()
                    val_align_cos_values.append(align_cos)

                label_val = labels.item()
                prob_val = probs.item()
                base_pid = batch["base_patient_id"][0]
                patient_id = batch["patient_id"][0]

                val_labels.append(label_val)
                val_probs.append(prob_val)
                val_patient_ids.append(base_pid)
                val_rows.append(
                    {
                        "patient_id": patient_id,
                        "base_patient_id": base_pid,
                        "label": label_val,
                        "prob": prob_val,
                        "has_ct": float(has_ct.item()),
                        "align_cos": align_cos,
                    }
                )

        val_patient_ids_agg, val_patient_labels, val_patient_probs = aggregate_patient_level(
            val_patient_ids,
            val_labels,
            val_probs,
            mode="mean",
        )
        default_metrics = compute_binary_metrics(val_patient_labels, val_patient_probs, threshold=0.5)
        best_t, best_metrics, threshold_rows = find_best_threshold(val_patient_labels, val_patient_probs)
        scheduler.step(best_metrics["balanced_accuracy"])

        print_threshold_sweep(threshold_rows)
        print_metrics("VAL PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", default_metrics)
        print_metrics(
            f"VAL PATIENT-LEVEL METRICS | MEAN AGG | BEST THRESHOLD={best_t:.3f}",
            best_metrics,
        )
        if val_align_cos_values:
            print(
                f"Val paired CT cosine mean/std: "
                f"{np.mean(val_align_cos_values):.6f} / {np.std(val_align_cos_values):.6f}"
            )
        else:
            print("Val paired CT cosine mean/std: N/A")

        for row in val_rows:
            cos_text = "N/A" if row["align_cos"] is None else f"{row['align_cos']:.6f}"
            print(
                f"{row['patient_id']:20s} | base={row['base_patient_id']:12s} | "
                f"label={int(row['label'])} | prob={row['prob']:.6f} | "
                f"has_ct={int(row['has_ct'])} | align_cos={cos_text}"
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
                "val_rows": val_rows,
            }
            torch.save(
                {
                    "epoch": best_epoch,
                    "xray_enc": xray_enc.state_dict(),
                    "ct_enc": ct_enc.state_dict(),
                    "classifier": classifier.state_dict(),
                    "best_score": best_score,
                    "best_threshold": best_threshold,
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
        summarize_patient_split(patient_labels, val_patients, f"MULTIMODAL SIMPLE CV FOLD {fold_idx}/{args.folds}")
        result = run_one_fold(args, dataset, fold_idx, val_patients)
        results.append(result)

    scores = [row["best_score"] for row in results if row.get("best_score", -1) >= 0]
    payload = {
        "experiment": "simple_ct_projection_alignment",
        "seed": args.seed,
        "folds": args.folds,
        "mean_best_balanced_accuracy": sum(scores) / max(len(scores), 1),
        "args": vars(args),
        "results": results,
    }
    filename = (
        f"multimodal_simple_{args.ct_projection_source}_{args.ct_projection_method}_axis{args.ct_projection_axis}_"
        f"hu{args.ct_hu_threshold if args.ct_hu_threshold is not None else 'none'}_"
        f"rot{args.ct_rotate_k}_"
        f"align{str(args.alignment_weight).replace('.', '')}_seed{args.seed}.json"
    )
    save_experiment_results(filename, payload)

    print("=" * 100)
    print("SIMPLE MULTIMODAL CV SUMMARY")
    print("=" * 100)
    print(f"Mean best balanced accuracy: {payload['mean_best_balanced_accuracy']:.6f}")
    for row in results:
        print(
            f"Fold {row['fold']}: best_score={row.get('best_score', -1):.6f} | "
            f"epoch={row.get('best_epoch', -1)} | threshold={row.get('best_threshold', 0.5):.3f}"
        )
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple CT-projection latent alignment experiment")
    parser.add_argument("--data_dir", default="data/augmented_patients")
    parser.add_argument("--csv_file", default="data/train_labels_augmented.csv")
    parser.add_argument("--original_patients_dir", default="data/original_patients")
    parser.add_argument("--ct_cache_dir", default="data/ct_projections_simple")
    parser.add_argument("--debug_dir", default="debug_multimodal_alignment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--ct_projection_axis", type=int, default=0)
    parser.add_argument("--ct_projection_method", choices=["max", "mean"], default="max")
    parser.add_argument("--ct_projection_source", choices=["ct", "ct_mask", "masked_ct"], default="ct")
    parser.add_argument("--ct_hu_threshold", type=float, default=None)
    parser.add_argument("--ct_clip_max", type=float, default=1800.0)
    parser.add_argument("--ct_rotate_k", type=int, default=0)
    parser.add_argument("--alignment_weight", type=float, default=0.1)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min_save_epoch", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--identity_embedding_head", action="store_true")
    parser.add_argument("--include_val_augmentations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_cv(parse_args())
