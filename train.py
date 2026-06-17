import os
import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.XrayEncoder import XrayEncoder
from models.SmallXrayEncoder import SmallXrayEncoder
from models.classifier import XrayClassifier
from dataset import SpineMultimodalDataset, print_dataset_stats


try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# DEBUG IMAGE
# ============================================================

def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_debug_classification_image(
    xray,
    mask,
    label,
    prob,
    patient_id,
    epoch,
    split,
    out_dir="debug_classifier"
):
    os.makedirs(out_dir, exist_ok=True)

    xray_np = xray[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy()

    label_val = float(label)
    prob_val = float(prob)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(xray_np, cmap="gray")
    plt.title(f"{split} X-ray | {patient_id}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(xray_np, cmap="gray")
    if mask_np.sum() > 0:
        plt.imshow(mask_np, cmap="Reds", alpha=0.45)
    plt.title(
        f"GT label={label_val:.0f} | "
        f"pred_prob={prob_val:.4f} | "
        f"mask_sum={mask_np.sum():.0f}"
    )
    plt.axis("off")

    plt.tight_layout()
    filename = (
        f"{split}_epoch_{epoch:03d}_"
        f"{patient_id}_label_{label_val:.0f}_prob_{prob_val:.4f}.png"
    )
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()


# ============================================================
# METRICS
# ============================================================

def compute_binary_metrics(labels, probs, threshold=0.5):
    labels = [float(x) for x in labels]
    probs = [float(x) for x in probs]
    preds = [1.0 if p >= threshold else 0.0 for p in probs]

    total = len(labels)
    correct = sum(1 for y, p in zip(labels, preds) if int(y) == int(p))
    acc = correct / max(total, 1)

    tp = sum(1 for y, p in zip(labels, preds) if y == 1.0 and p == 1.0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0.0 and p == 0.0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0.0 and p == 1.0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1.0 and p == 0.0)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (sensitivity + specificity)

    precision = tp / max(tp + fp, 1)
    recall = sensitivity
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    auc = None
    if SKLEARN_AVAILABLE and len(set(labels)) == 2:
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = None

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": total,
    }


def print_metrics(title, metrics):
    print("-" * 100)
    print(title)
    print("-" * 100)
    print(f"N                 : {metrics['n']}")
    print(f"Accuracy          : {metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.6f}")

    if metrics["auc"] is not None:
        print(f"AUC               : {metrics['auc']:.6f}")
    else:
        print("AUC               : N/A")

    print(f"Sensitivity       : {metrics['sensitivity']:.6f}")
    print(f"Specificity       : {metrics['specificity']:.6f}")
    print(f"Precision         : {metrics['precision']:.6f}")
    print(f"Recall            : {metrics['recall']:.6f}")
    print(f"F1                : {metrics['f1']:.6f}")
    print(f"TP/TN/FP/FN       : {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print("-" * 100)


def aggregate_patient_level(patient_ids, labels, probs, mode="mean"):
    grouped_probs = defaultdict(list)
    grouped_labels = {}

    for pid, y, p in zip(patient_ids, labels, probs):
        grouped_probs[pid].append(float(p))
        grouped_labels[pid] = float(y)

    patient_ids_out = []
    patient_labels = []
    patient_probs = []

    for pid in sorted(grouped_probs.keys()):
        values = grouped_probs[pid]

        if mode == "max":
            agg_prob = max(values)
        elif mode == "mean":
            agg_prob = sum(values) / len(values)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

        patient_ids_out.append(pid)
        patient_labels.append(grouped_labels[pid])
        patient_probs.append(agg_prob)

    return patient_ids_out, patient_labels, patient_probs


def find_best_threshold(labels, probs):
    """
    Validation set küçük olduğu için threshold sweep yapıyoruz.
    Ana kriter: balanced accuracy.
    Tie-breaker: accuracy, sonra specificity.
    """
    thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95,
        0.97, 0.98, 0.99, 0.995, 0.998, 0.999
    ]

    best_t = 0.5
    best_metrics = None
    best_key = None

    rows = []

    for t in thresholds:
        metrics = compute_binary_metrics(labels, probs, threshold=t)

        key = (
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            metrics["specificity"],
            metrics["sensitivity"],
        )

        rows.append((t, metrics))

        if best_key is None or key > best_key:
            best_key = key
            best_t = t
            best_metrics = metrics

    return best_t, best_metrics, rows


def print_threshold_sweep(rows):
    print("-" * 100)
    print("VALIDATION THRESHOLD SWEEP")
    print("-" * 100)
    print(
        f"{'thr':>8s} | {'acc':>8s} | {'bal_acc':>8s} | "
        f"{'sens':>8s} | {'spec':>8s} | {'tp/tn/fp/fn':>15s}"
    )
    print("-" * 100)

    for t, m in rows:
        print(
            f"{t:8.3f} | "
            f"{m['accuracy']:8.4f} | "
            f"{m['balanced_accuracy']:8.4f} | "
            f"{m['sensitivity']:8.4f} | "
            f"{m['specificity']:8.4f} | "
            f"{m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}"
        )

    print("-" * 100)


# ============================================================
# SPLIT HELPERS
# ============================================================

DEFAULT_VAL_PATIENTS = [
    "patient_004",
    "patient_028",
    "patient_032",
    "patient_034",
    "patient_037",
    "patient_045",
]


def build_patient_split(dataset, val_patients, original_only_val=True):
    train_idx = []
    val_idx = []

    for i, pid in enumerate(dataset.patient_ids):
        base_pid = dataset.base_patient_id(pid)
        is_aug = dataset.is_augmented(pid)

        if base_pid in val_patients:
            if original_only_val:
                if not is_aug:
                    val_idx.append(i)
            else:
                val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx


def collect_patient_labels(dataset):
    patient_labels = {}
    conflicts = defaultdict(set)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        base_pid = sample["base_patient_id"]
        label = int(sample["label"].item() > 0.5)
        conflicts[base_pid].add(label)
        patient_labels[base_pid] = max(patient_labels.get(base_pid, 0), label)

    mixed = {pid: labels for pid, labels in conflicts.items() if len(labels) > 1}
    if mixed:
        print("WARNING: Some base patients have mixed labels after ROI cropping:")
        for pid, labels in sorted(mixed.items()):
            print(f"  {pid}: {sorted(labels)}")
        print("Patient-level label uses max(label) across variants for split stratification.")

    return patient_labels


def make_stratified_patient_folds(patient_labels, n_splits=5, seed=42):
    rng = random.Random(seed)
    positives = [pid for pid, label in patient_labels.items() if label == 1]
    negatives = [pid for pid, label in patient_labels.items() if label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    folds = [[] for _ in range(n_splits)]
    for i, pid in enumerate(positives):
        folds[i % n_splits].append(pid)
    for i, pid in enumerate(negatives):
        folds[i % n_splits].append(pid)

    for fold in folds:
        fold.sort()

    return folds


def make_random_patient_splits(
    patient_labels,
    num_splits=5,
    val_pos=3,
    val_neg=3,
    seed=42,
):
    rng = random.Random(seed)
    positives = [pid for pid, label in patient_labels.items() if label == 1]
    negatives = [pid for pid, label in patient_labels.items() if label == 0]

    if val_pos > len(positives) or val_neg > len(negatives):
        raise ValueError(
            f"Requested val_pos={val_pos}, val_neg={val_neg}, but only have "
            f"{len(positives)} positive and {len(negatives)} negative patients."
        )

    splits = []
    seen = set()
    attempts = 0

    while len(splits) < num_splits and attempts < num_splits * 50:
        attempts += 1
        val_patients = sorted(
            rng.sample(positives, val_pos) + rng.sample(negatives, val_neg)
        )
        key = tuple(val_patients)
        if key in seen:
            continue
        seen.add(key)
        splits.append(val_patients)

    if len(splits) < num_splits:
        print(f"WARNING: Only built {len(splits)} unique random splits.")

    return splits


def count_labels_for_indices(dataset, indices, name):
    pos = 0
    neg = 0
    base_patients = set()
    pos_patients = set()
    neg_patients = set()

    for idx in indices:
        sample = dataset[idx]
        label = sample["label"].item()
        bpid = sample["base_patient_id"]

        base_patients.add(bpid)

        if label > 0.5:
            pos += 1
            pos_patients.add(bpid)
        else:
            neg += 1
            neg_patients.add(bpid)

    print("=" * 100)
    print(f"{name} SPLIT STATS")
    print("=" * 100)
    print(f"Samples          : {len(indices)}")
    print(f"Positive samples : {pos}")
    print(f"Negative samples : {neg}")
    print(f"Unique patients  : {len(base_patients)}")
    print(f"Positive patients: {len(pos_patients)}")
    print(f"Negative patients: {len(neg_patients)}")
    print("=" * 100)

    return pos, neg


def build_xray_encoder(encoder_type="resnet34", embedding_dim=512, pretrained=False):
    if encoder_type == "resnet34":
        return XrayEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
    if encoder_type == "small_cnn":
        if pretrained:
            print("WARNING: small_cnn does not support pretrained weights; using random initialization.")
        return SmallXrayEncoder(embedding_dim=embedding_dim)
    raise ValueError(f"Unknown encoder_type={encoder_type}")


# ============================================================
# TRAIN
# ============================================================

def train_model(
    val_patients=None,
    run_name="single",
    checkpoint_path=None,
    epochs=20,
    batch_size=4,
    roi_width_ratio=0.45,
    original_only_val=True,
    min_save_epoch=3,
    early_stopping_patience=6,
    encoder_lr=1e-5,
    classifier_lr=5e-5,
    default_threshold=0.5,
    freeze_encoder=False,
    debug_dir="debug_classifier",
    save_debug_images=True,
    scrub_prompts=False,
    label_source="mask",
    encoder_type="resnet34",
    pretrained_encoder=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VAL_EVERY = 1
    val_patients = list(val_patients or DEFAULT_VAL_PATIENTS)

    if checkpoint_path is None:
        if run_name == "single" and not freeze_encoder:
            checkpoint_path = "checkpoints/best_xray_classifier_roi_045_balacc.pth"
        else:
            checkpoint_path = f"checkpoints/{run_name}_best_xray_classifier_roi_045_balacc.pth"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print("=" * 100)
    print("ROI X-RAY BINARY CLASSIFICATION TRAINING")
    print("=" * 100)
    print(f"Run name               : {run_name}")
    print(f"Device                 : {device}")
    print(f"Batch size              : {batch_size}")
    print(f"Epochs                  : {epochs}")
    print(f"Encoder LR              : {encoder_lr}")
    print(f"Classifier LR           : {classifier_lr}")
    print(f"VAL_EVERY               : {VAL_EVERY}")
    print(f"Default threshold       : {default_threshold}")
    print(f"ROI width ratio         : {roi_width_ratio}")
    print(f"Original-only val       : {original_only_val}")
    print(f"Freeze encoder          : {freeze_encoder}")
    print(f"Min save epoch          : {min_save_epoch}")
    print(f"Early stopping patience : {early_stopping_patience}")
    print(f"Debug dir               : {debug_dir}")
    print(f"Save debug images       : {save_debug_images}")
    print(f"Scrub prompts           : {scrub_prompts}")
    print(f"Label source            : {label_source}")
    print(f"Encoder type            : {encoder_type}")
    print(f"Pretrained encoder      : {pretrained_encoder}")
    print(f"Checkpoint              : {checkpoint_path}")
    print("=" * 100)

    dataset = SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=roi_width_ratio,
        scrub_prompt=scrub_prompts,
        label_source=label_source,
    )

    print_dataset_stats(dataset)

    train_idx, val_idx = build_patient_split(
        dataset=dataset,
        val_patients=val_patients,
        original_only_val=original_only_val,
    )

    train_pos, train_neg = count_labels_for_indices(dataset, train_idx, "TRAIN")
    val_pos, val_neg = count_labels_for_indices(dataset, val_idx, "VAL")

    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty. Check val_patients or bad patient filtering.")

    if val_pos == 0 or val_neg == 0:
        print("WARNING: Validation split has only one class. AUC/specificity/sensitivity will be unreliable.")

    if train_pos == 0 or train_neg == 0:
        print("WARNING: Train split has only one class. BCE classifier cannot learn a meaningful boundary.")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
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
        encoder_type=encoder_type,
        embedding_dim=512,
        pretrained=pretrained_encoder,
    ).to(device)
    classifier = XrayClassifier(embedding_dim=512, dropout=0.5).to(device)

    if freeze_encoder:
        for param in xray_enc.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        [
            {"params": xray_enc.parameters(), "lr": encoder_lr},
            {"params": classifier.parameters(), "lr": classifier_lr},
        ],
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # Pozitif sınıf çoğunlukta olduğu için pos_weight < 1 çıkabilir.
    # Bu normal ama fazla pozitif bias görürsek bunu 1.0 yapmayı da deneyebiliriz.
    if train_pos > 0:
        pos_weight_val = train_neg / max(train_pos, 1)
    else:
        pos_weight_val = 1.0

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("=" * 100)
    print("LOSS SETUP")
    print("=" * 100)
    print(f"Train positive samples : {train_pos}")
    print(f"Train negative samples : {train_neg}")
    print(f"BCE pos_weight         : {pos_weight_val:.6f}")
    print(f"Dropout                : 0.5")
    print(f"Weight decay           : 5e-4")
    print("=" * 100)

    sample = dataset[train_idx[0]]
    print("SANITY CHECK SAMPLE")
    print(f"patient_id : {sample['patient_id']}")
    print(f"xray shape : {tuple(sample['xray'].shape)}")
    print(f"mask sum   : {sample['mask'].sum().item():.0f}")
    print(f"label      : {sample['label'].item():.0f}")
    print(f"text       : {sample['text']}")
    print("=" * 100)

    if save_debug_images:
        save_debug_classification_image(
            xray=sample["xray"].unsqueeze(0),
            mask=sample["mask"].unsqueeze(0),
            label=sample["label"].item(),
            prob=0.0,
            patient_id=sample["patient_id"],
            epoch=0,
            split="sanity",
            out_dir=debug_dir,
        )

    trainable_params = [
        p for p in list(xray_enc.parameters()) + list(classifier.parameters())
        if p.requires_grad
    ]

    best_score = -1.0
    best_epoch = -1
    best_val_accuracy = 0.0
    best_threshold = default_threshold
    patience_counter = 0

    for epoch in range(epochs):
        print("\n" + "=" * 100)
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"Encoder LR   : {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Classifier LR: {optimizer.param_groups[1]['lr']:.8f}")
        print("=" * 100)

        if freeze_encoder:
            xray_enc.eval()
        else:
            xray_enc.train()
        classifier.train()

        running_loss = 0.0
        batch_count = 0

        train_labels = []
        train_probs = []
        train_patient_ids = []

        logit_min_total = 0.0
        logit_mean_total = 0.0
        logit_max_total = 0.0

        prob_min_total = 0.0
        prob_mean_total = 0.0
        prob_max_total = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            xray = batch["xray"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            if freeze_encoder:
                with torch.no_grad():
                    embedding, _ = xray_enc(xray)
            else:
                embedding, _ = xray_enc(xray)
            logits = classifier(embedding)

            if logits.ndim > 1:
                logits = logits.view(-1)

            loss = loss_fn(logits, labels)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            probs = torch.sigmoid(logits.detach())

            running_loss += loss.item()
            batch_count += 1

            train_labels.extend(labels.detach().cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            train_patient_ids.extend(batch["base_patient_id"])

            logit_min_total += logits.detach().min().item()
            logit_mean_total += logits.detach().mean().item()
            logit_max_total += logits.detach().max().item()

            prob_min_total += probs.min().item()
            prob_mean_total += probs.mean().item()
            prob_max_total += probs.max().item()

            preds = (probs >= default_threshold).float()
            batch_acc = (preds == labels.detach()).float().mean().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_acc:.3f}",
                "pmin": f"{probs.min().item():.3f}",
                "pmean": f"{probs.mean().item():.3f}",
                "pmax": f"{probs.max().item():.3f}",
                "grad": f"{float(grad_norm):.3f}",
            })

        avg_train_loss = running_loss / max(batch_count, 1)

        train_sample_metrics = compute_binary_metrics(
            labels=train_labels,
            probs=train_probs,
            threshold=default_threshold,
        )

        _, train_patient_labels, train_patient_probs_mean = aggregate_patient_level(
            patient_ids=train_patient_ids,
            labels=train_labels,
            probs=train_probs,
            mode="mean",
        )

        _, _, train_patient_probs_max = aggregate_patient_level(
            patient_ids=train_patient_ids,
            labels=train_labels,
            probs=train_probs,
            mode="max",
        )

        train_patient_metrics_mean = compute_binary_metrics(
            labels=train_patient_labels,
            probs=train_patient_probs_mean,
            threshold=default_threshold,
        )

        train_patient_metrics_max = compute_binary_metrics(
            labels=train_patient_labels,
            probs=train_patient_probs_max,
            threshold=default_threshold,
        )

        print("\n" + "-" * 100)
        print(f"TRAIN SUMMARY | Epoch {epoch + 1}/{epochs}")
        print("-" * 100)
        print(f"Train Loss                  : {avg_train_loss:.6f}")
        print(
            f"Train Logit min/mean/max    : "
            f"{logit_min_total / max(batch_count, 1):.6f} / "
            f"{logit_mean_total / max(batch_count, 1):.6f} / "
            f"{logit_max_total / max(batch_count, 1):.6f}"
        )
        print(
            f"Train Prob min/mean/max     : "
            f"{prob_min_total / max(batch_count, 1):.6f} / "
            f"{prob_mean_total / max(batch_count, 1):.6f} / "
            f"{prob_max_total / max(batch_count, 1):.6f}"
        )

        print_metrics("TRAIN SAMPLE-LEVEL METRICS | DEFAULT THRESHOLD", train_sample_metrics)
        print_metrics("TRAIN PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", train_patient_metrics_mean)
        print_metrics("TRAIN PATIENT-LEVEL METRICS | MAX AGG | DEFAULT THRESHOLD", train_patient_metrics_max)

        if (epoch + 1) % VAL_EVERY == 0:
            print("\n" + "#" * 100)
            print(f"VALIDATION START | Epoch {epoch + 1}/{epochs}")
            print("#" * 100)

            xray_enc.eval()
            classifier.eval()

            val_loss_total = 0.0
            val_count = 0

            val_labels = []
            val_probs = []
            val_patient_ids = []
            val_rows = []

            with torch.no_grad():
                for val_batch_idx, batch in enumerate(val_loader):
                    patient_id = batch["patient_id"][0]
                    base_patient_id = batch["base_patient_id"][0]

                    xray = batch["xray"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True).float()

                    embedding, _ = xray_enc(xray)
                    logits = classifier(embedding)

                    if logits.ndim > 1:
                        logits = logits.view(-1)

                    loss = loss_fn(logits, labels)
                    probs = torch.sigmoid(logits)

                    val_loss_total += loss.item()
                    val_count += 1

                    label_val = labels.item()
                    prob_val = probs.item()
                    pred_val = 1.0 if prob_val >= default_threshold else 0.0

                    val_labels.append(label_val)
                    val_probs.append(prob_val)
                    val_patient_ids.append(base_patient_id)

                    val_rows.append({
                        "patient_id": patient_id,
                        "base_patient_id": base_patient_id,
                        "label": label_val,
                        "prob": prob_val,
                        "pred": pred_val,
                        "logit": logits.item(),
                        "mask_sum": mask.sum().item(),
                    })

                    if save_debug_images:
                        save_debug_classification_image(
                            xray=xray,
                            mask=mask,
                            label=label_val,
                            prob=prob_val,
                            patient_id=patient_id,
                            epoch=epoch + 1,
                            split="val",
                            out_dir=debug_dir,
                        )

            avg_val_loss = val_loss_total / max(val_count, 1)

            val_sample_metrics_default = compute_binary_metrics(
                labels=val_labels,
                probs=val_probs,
                threshold=default_threshold,
            )

            val_patient_ids_agg, val_patient_labels, val_patient_probs_mean = aggregate_patient_level(
                patient_ids=val_patient_ids,
                labels=val_labels,
                probs=val_probs,
                mode="mean",
            )

            _, _, val_patient_probs_max = aggregate_patient_level(
                patient_ids=val_patient_ids,
                labels=val_labels,
                probs=val_probs,
                mode="max",
            )

            val_patient_metrics_mean_default = compute_binary_metrics(
                labels=val_patient_labels,
                probs=val_patient_probs_mean,
                threshold=default_threshold,
            )

            val_patient_metrics_max_default = compute_binary_metrics(
                labels=val_patient_labels,
                probs=val_patient_probs_max,
                threshold=default_threshold,
            )

            best_t, best_t_metrics, threshold_rows = find_best_threshold(
                labels=val_patient_labels,
                probs=val_patient_probs_mean,
            )

            print("\n" + "-" * 100)
            print(f"VALIDATION ROWS | Epoch {epoch + 1}/{epochs} | DEFAULT THRESHOLD={default_threshold}")
            print("-" * 100)

            for row in val_rows:
                print(
                    f"{row['patient_id']:20s} | "
                    f"base={row['base_patient_id']:12s} | "
                    f"label={row['label']:.0f} | "
                    f"pred={row['pred']:.0f} | "
                    f"prob={row['prob']:.6f} | "
                    f"logit={row['logit']:.6f} | "
                    f"mask_sum={row['mask_sum']:.0f}"
                )

            print("\n" + "-" * 100)
            print(f"VALIDATION SUMMARY | Epoch {epoch + 1}/{epochs}")
            print("-" * 100)
            print(f"Val Loss: {avg_val_loss:.6f}")

            print_metrics("VAL SAMPLE-LEVEL METRICS | DEFAULT THRESHOLD", val_sample_metrics_default)
            print_metrics("VAL PATIENT-LEVEL METRICS | MEAN AGG | DEFAULT THRESHOLD", val_patient_metrics_mean_default)
            print_metrics("VAL PATIENT-LEVEL METRICS | MAX AGG | DEFAULT THRESHOLD", val_patient_metrics_max_default)

            print_threshold_sweep(threshold_rows)

            print_metrics(
                f"VAL PATIENT-LEVEL METRICS | MEAN AGG | BEST THRESHOLD={best_t:.3f}",
                best_t_metrics
            )

            # Save score:
            # Ana metrik balanced accuracy.
            # AUC loglanır ama checkpoint için tek başına kullanılmaz.
            current_auc = val_patient_metrics_mean_default["auc"]
            score_for_scheduler = best_t_metrics["balanced_accuracy"]
            score_for_save = best_t_metrics["balanced_accuracy"]
            score_name = "patient_mean_best_threshold_balanced_accuracy"

            scheduler.step(score_for_scheduler)

            print("-" * 100)
            print(f"Score for scheduler/save : {score_name} = {score_for_save:.6f}")
            if current_auc is not None:
                print(f"Patient mean AUC         : {current_auc:.6f}")
            else:
                print("Patient mean AUC         : N/A")
            print(f"Best threshold this epoch: {best_t:.3f}")
            print(f"Best score so far        : {best_score:.6f}")
            print(f"Patience                 : {patience_counter}/{early_stopping_patience}")
            print(f"Encoder LR after sched   : {optimizer.param_groups[0]['lr']:.8f}")
            print(f"Classifier LR after sched: {optimizer.param_groups[1]['lr']:.8f}")
            print("-" * 100)

            can_save = (epoch + 1) >= min_save_epoch

            if not can_save:
                print(
                    f"MODEL NOT SAVED | Warmup period. "
                    f"Saving starts at epoch {min_save_epoch}."
                )
            elif score_for_save > best_score:
                old_best = best_score
                best_score = score_for_save
                best_val_accuracy = best_t_metrics["accuracy"]
                best_epoch = epoch + 1
                best_threshold = best_t
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "xray_enc": xray_enc.state_dict(),
                        "classifier": classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "roi_width_ratio": roi_width_ratio,
                        "val_patients": val_patients,
                        "original_only_val": original_only_val,
                        "default_threshold": default_threshold,
        "freeze_encoder": freeze_encoder,
        "run_name": run_name,
        "encoder_type": encoder_type,
        "pretrained_encoder": pretrained_encoder,
        "best_threshold": best_threshold,
                        "best_score": best_score,
                        "score_name": score_name,
                        "val_patient_metrics_best_threshold": best_t_metrics,
                        "val_patient_metrics_default_threshold": val_patient_metrics_mean_default,
                    },
                    checkpoint_path,
                )

                print(
                    f"MODEL SAVED | "
                    f"{score_name}: {old_best:.6f} -> {best_score:.6f} | "
                    f"threshold={best_threshold:.3f} | "
                    f"epoch={epoch + 1}"
                )
            else:
                patience_counter += 1
                print(
                    f"MODEL NOT SAVED | Validation balanced accuracy did not improve. "
                    f"Patience: {patience_counter}/{early_stopping_patience}"
                )

            print("#" * 100)

            if can_save and patience_counter >= early_stopping_patience:
                print(
                    f"EARLY STOPPING | No improvement for "
                    f"{early_stopping_patience} validation checks."
                )
                break

        else:
            print(
                f"Validation skipped this epoch. "
                f"Next validation at epoch {((epoch + 1) // VAL_EVERY + 1) * VAL_EVERY}."
            )

    print("\n" + "=" * 100)
    print("CLASSIFICATION TRAINING FINISHED")
    print(f"Best epoch     : {best_epoch}")
    print(f"Best score     : {best_score:.6f}")
    print(f"Best val acc   : {best_val_accuracy:.6f}")
    print(f"Best threshold : {best_threshold:.3f}")
    print(f"Checkpoint     : {checkpoint_path}")
    print("=" * 100)

    return {
        "run_name": run_name,
        "val_patients": val_patients,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_accuracy": best_val_accuracy,
        "best_threshold": best_threshold,
        "checkpoint": checkpoint_path,
        "freeze_encoder": freeze_encoder,
        "original_only_val": original_only_val,
        "encoder_type": encoder_type,
        "pretrained_encoder": pretrained_encoder,
    }


def make_dataset_for_experiments(roi_width_ratio=0.45, scrub_prompts=False, label_source="mask"):
    return SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=roi_width_ratio,
        scrub_prompt=scrub_prompts,
        label_source=label_source,
    )


def summarize_patient_split(patient_labels, val_patients, title):
    val_set = set(val_patients)
    train_patients = [pid for pid in patient_labels if pid not in val_set]

    def counts(patients):
        pos = sum(1 for pid in patients if patient_labels[pid] == 1)
        neg = sum(1 for pid in patients if patient_labels[pid] == 0)
        return pos, neg

    train_pos, train_neg = counts(train_patients)
    val_pos, val_neg = counts(val_patients)

    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"Train patients: {len(train_patients)} | pos={train_pos} | neg={train_neg}")
    print(f"Val patients  : {len(val_patients)} | pos={val_pos} | neg={val_neg}")
    print(f"Val IDs       : {', '.join(val_patients)}")
    print("=" * 100)


def save_experiment_results(filename, payload):
    os.makedirs("experiment_results", exist_ok=True)
    path = os.path.join("experiment_results", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Experiment results saved to: {path}")
    return path


def common_train_kwargs(args, run_name):
    return {
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "roi_width_ratio": args.roi_width_ratio,
        "original_only_val": not args.include_val_augmentations,
        "min_save_epoch": args.min_save_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "freeze_encoder": args.freeze_encoder or args.experiment == "frozen",
        "debug_dir": os.path.join(args.debug_dir, run_name),
        "save_debug_images": not args.no_debug_images,
        "scrub_prompts": args.scrub_prompts,
        "label_source": args.label_source,
        "encoder_type": args.encoder_type,
        "pretrained_encoder": args.pretrained_encoder,
    }


def variant_tag(args):
    pretrain = "pretrained" if args.pretrained_encoder else "scratch"
    scrub = "scrubbed" if args.scrub_prompts else "rawprompt"
    roi = str(args.roi_width_ratio).replace(".", "")
    return f"seed{args.seed}_{args.encoder_type}_{pretrain}_{args.label_source}_{scrub}_roi{roi}"


def run_split_sweep(args):
    dataset = make_dataset_for_experiments(
        args.roi_width_ratio,
        scrub_prompts=args.scrub_prompts,
        label_source=args.label_source,
    )
    patient_labels = collect_patient_labels(dataset)
    splits = make_random_patient_splits(
        patient_labels=patient_labels,
        num_splits=args.num_splits,
        val_pos=args.split_val_pos,
        val_neg=args.split_val_neg,
        seed=args.seed,
    )

    results = []
    for split_idx, val_patients in enumerate(splits, start=1):
        run_name = f"split_sweep_{variant_tag(args)}_{split_idx:02d}"
        summarize_patient_split(patient_labels, val_patients, f"SPLIT SWEEP {split_idx}/{len(splits)}")
        result = train_model(
            val_patients=val_patients,
            checkpoint_path=f"checkpoints/{run_name}_best_xray_classifier_roi_045_balacc.pth",
            **common_train_kwargs(args, run_name),
        )
        results.append(result)

    save_experiment_results(
        f"split_sweep_{variant_tag(args)}_seed_{args.seed}.json",
        {
            "experiment": "split_sweep",
            "seed": args.seed,
            "num_splits": len(splits),
            "results": results,
        },
    )


def run_cv5(args):
    dataset = make_dataset_for_experiments(
        args.roi_width_ratio,
        scrub_prompts=args.scrub_prompts,
        label_source=args.label_source,
    )
    patient_labels = collect_patient_labels(dataset)
    folds = make_stratified_patient_folds(patient_labels, n_splits=5, seed=args.seed)

    results = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        run_name = f"cv5_{variant_tag(args)}_fold_{fold_idx:02d}"
        summarize_patient_split(patient_labels, val_patients, f"CV FOLD {fold_idx}/5")
        result = train_model(
            val_patients=val_patients,
            checkpoint_path=f"checkpoints/{run_name}_best_xray_classifier_roi_045_balacc.pth",
            **common_train_kwargs(args, run_name),
        )
        result["fold"] = fold_idx
        results.append(result)

    valid_scores = [r["best_score"] for r in results if r["best_score"] is not None and r["best_score"] >= 0]
    mean_score = sum(valid_scores) / max(len(valid_scores), 1)

    save_experiment_results(
        f"cv5_{variant_tag(args)}_seed_{args.seed}.json",
        {
            "experiment": "cv5",
            "seed": args.seed,
            "mean_best_balanced_accuracy": mean_score,
            "results": results,
        },
    )

    print("=" * 100)
    print("CV5 SUMMARY")
    print("=" * 100)
    print(f"Mean best balanced accuracy: {mean_score:.6f}")
    for result in results:
        print(
            f"Fold {result['fold']}: best_score={result['best_score']:.6f} | "
            f"best_epoch={result['best_epoch']} | threshold={result['best_threshold']:.3f}"
        )
    print("=" * 100)


def collect_patient_metadata_records(dataset):
    records = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        base_pid = sample["base_patient_id"]
        patient_id = sample["patient_id"]
        label = int(sample["label"].item() > 0.5)

        if base_pid not in records:
            records[base_pid] = {
                "patient_id": base_pid,
                "prompt": sample["text"],
                "label": label,
                "patient_number": int(base_pid.split("_")[-1]),
                "sample_count": 0,
                "augmented_count": 0,
            }

        records[base_pid]["label"] = max(records[base_pid]["label"], label)
        records[base_pid]["sample_count"] += 1
        records[base_pid]["augmented_count"] += int("_aug" in patient_id)

        if "_aug" not in patient_id:
            records[base_pid]["prompt"] = sample["text"]

    return records


def positive_class_probs(model, x_values):
    probs = model.predict_proba(x_values)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0 for _ in range(len(x_values))]
    pos_idx = classes.index(1)
    return [float(row[pos_idx]) for row in probs]


def evaluate_metadata_model(model_name, labels, probs):
    default_metrics = compute_binary_metrics(labels, probs, threshold=0.5)
    best_t, best_metrics, _ = find_best_threshold(labels, probs)

    print_metrics(f"METADATA BASELINE | {model_name} | DEFAULT THRESHOLD", default_metrics)
    print_metrics(f"METADATA BASELINE | {model_name} | BEST THRESHOLD={best_t:.3f}", best_metrics)

    return {
        "model": model_name,
        "default_threshold_metrics": default_metrics,
        "best_threshold": best_t,
        "best_threshold_metrics": best_metrics,
    }


def run_metadata_baseline(args):
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "metadata_baseline requires scikit-learn. Install sklearn or run in the project venv."
        ) from exc

    dataset = make_dataset_for_experiments(
        args.roi_width_ratio,
        scrub_prompts=args.scrub_prompts,
        label_source=args.label_source,
    )
    records = collect_patient_metadata_records(dataset)
    patient_labels = {pid: row["label"] for pid, row in records.items()}
    folds = make_stratified_patient_folds(patient_labels, n_splits=5, seed=args.seed)

    model_summaries = defaultdict(list)

    for fold_idx, val_patients in enumerate(folds, start=1):
        val_set = set(val_patients)
        train_patients = [pid for pid in records if pid not in val_set]

        y_train = [records[pid]["label"] for pid in train_patients]
        y_val = [records[pid]["label"] for pid in val_patients]

        prompt_train = [records[pid]["prompt"] for pid in train_patients]
        prompt_val = [records[pid]["prompt"] for pid in val_patients]

        numeric_train = [
            [
                records[pid]["patient_number"],
                records[pid]["sample_count"],
                records[pid]["augmented_count"],
            ]
            for pid in train_patients
        ]
        numeric_val = [
            [
                records[pid]["patient_number"],
                records[pid]["sample_count"],
                records[pid]["augmented_count"],
            ]
            for pid in val_patients
        ]

        print("=" * 100)
        print(f"METADATA BASELINE FOLD {fold_idx}/5")
        print("=" * 100)
        print(f"Val patients: {', '.join(val_patients)}")

        dummy = DummyClassifier(strategy="prior")
        dummy.fit(prompt_train, y_train)
        dummy_probs = positive_class_probs(dummy, prompt_val)
        model_summaries["dummy_prior"].append(
            evaluate_metadata_model("dummy_prior", y_val, dummy_probs)
        )

        prompt_model = Pipeline(
            [
                ("vectorizer", CountVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        prompt_model.fit(prompt_train, y_train)
        prompt_probs = positive_class_probs(prompt_model, prompt_val)
        model_summaries["prompt_text"].append(
            evaluate_metadata_model("prompt_text", y_val, prompt_probs)
        )

        numeric_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        numeric_model.fit(numeric_train, y_train)
        numeric_probs = positive_class_probs(numeric_model, numeric_val)
        model_summaries["patient_number_aug_counts"].append(
            evaluate_metadata_model("patient_number_aug_counts", y_val, numeric_probs)
        )

    aggregate = {}
    for model_name, rows in model_summaries.items():
        best_bal_acc = [
            row["best_threshold_metrics"]["balanced_accuracy"]
            for row in rows
        ]
        default_bal_acc = [
            row["default_threshold_metrics"]["balanced_accuracy"]
            for row in rows
        ]
        aggregate[model_name] = {
            "mean_default_balanced_accuracy": sum(default_bal_acc) / max(len(default_bal_acc), 1),
            "mean_best_threshold_balanced_accuracy": sum(best_bal_acc) / max(len(best_bal_acc), 1),
            "folds": rows,
        }

    print("=" * 100)
    print("METADATA BASELINE SUMMARY")
    print("=" * 100)
    for model_name, row in aggregate.items():
        print(
            f"{model_name:28s} | "
            f"default_bal_acc={row['mean_default_balanced_accuracy']:.6f} | "
            f"best_thr_bal_acc={row['mean_best_threshold_balanced_accuracy']:.6f}"
        )
    print("=" * 100)

    save_experiment_results(
        (
            f"metadata_baseline_{args.label_source}_"
            f"{'scrubbed' if args.scrub_prompts else 'raw'}_seed_{args.seed}.json"
        ),
        {
            "experiment": "metadata_baseline",
            "seed": args.seed,
            "label_source": args.label_source,
            "scrub_prompts": args.scrub_prompts,
            "aggregate": aggregate,
        },
    )


def extract_embeddings_for_indices(dataset, indices, encoder, device, batch_size=4):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    embeddings = []
    labels = []
    base_patient_ids = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting frozen features"):
            xray = batch["xray"].to(device, non_blocking=True)
            emb, _ = encoder(xray)
            embeddings.extend(emb.detach().cpu().tolist())
            labels.extend([int(x > 0.5) for x in batch["label"].tolist()])
            base_patient_ids.extend(batch["base_patient_id"])

    return embeddings, labels, base_patient_ids


def run_feature_baseline(args):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "feature_baseline requires scikit-learn. Install sklearn or run in the project venv."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset_for_experiments(
        args.roi_width_ratio,
        scrub_prompts=args.scrub_prompts,
        label_source=args.label_source,
    )
    patient_labels = collect_patient_labels(dataset)
    folds = make_stratified_patient_folds(patient_labels, n_splits=5, seed=args.seed)

    encoder = build_xray_encoder(
        encoder_type=args.encoder_type,
        embedding_dim=512,
        pretrained=args.pretrained_encoder,
    ).to(device)

    results = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        print("=" * 100)
        print(f"FROZEN FEATURE BASELINE FOLD {fold_idx}/5")
        print("=" * 100)
        summarize_patient_split(patient_labels, val_patients, f"FEATURE FOLD {fold_idx}/5")

        train_idx, val_idx = build_patient_split(
            dataset=dataset,
            val_patients=val_patients,
            original_only_val=not args.include_val_augmentations,
        )

        x_train, y_train, _ = extract_embeddings_for_indices(
            dataset, train_idx, encoder, device, batch_size=args.batch_size
        )
        x_val, y_val, val_patient_ids = extract_embeddings_for_indices(
            dataset, val_idx, encoder, device, batch_size=args.batch_size
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        probs = positive_class_probs(model, x_val)

        val_patient_ids_agg, val_patient_labels, val_patient_probs_mean = aggregate_patient_level(
            patient_ids=val_patient_ids,
            labels=y_val,
            probs=probs,
            mode="mean",
        )
        default_metrics = compute_binary_metrics(
            val_patient_labels,
            val_patient_probs_mean,
            threshold=0.5,
        )
        best_t, best_metrics, threshold_rows = find_best_threshold(
            val_patient_labels,
            val_patient_probs_mean,
        )

        print_threshold_sweep(threshold_rows)
        print_metrics("FEATURE BASELINE | PATIENT MEAN | DEFAULT THRESHOLD", default_metrics)
        print_metrics(
            f"FEATURE BASELINE | PATIENT MEAN | BEST THRESHOLD={best_t:.3f}",
            best_metrics,
        )

        rows = []
        for pid, y, prob in zip(val_patient_ids_agg, val_patient_labels, val_patient_probs_mean):
            rows.append({"base_patient_id": pid, "label": y, "prob": prob})
            print(f"{pid:12s} | label={int(y)} | prob={prob:.6f}")

        results.append(
            {
                "fold": fold_idx,
                "val_patients": val_patients,
                "default_threshold_metrics": default_metrics,
                "best_threshold": best_t,
                "best_threshold_metrics": best_metrics,
                "rows": rows,
            }
        )

    default_scores = [
        row["default_threshold_metrics"]["balanced_accuracy"]
        for row in results
    ]
    best_scores = [
        row["best_threshold_metrics"]["balanced_accuracy"]
        for row in results
    ]
    payload = {
        "experiment": "feature_baseline",
        "seed": args.seed,
        "label_source": args.label_source,
        "scrub_prompts": args.scrub_prompts,
        "encoder_type": args.encoder_type,
        "pretrained_encoder": args.pretrained_encoder,
        "roi_width_ratio": args.roi_width_ratio,
        "mean_default_balanced_accuracy": sum(default_scores) / max(len(default_scores), 1),
        "mean_best_threshold_balanced_accuracy": sum(best_scores) / max(len(best_scores), 1),
        "results": results,
    }

    filename = (
        f"feature_baseline_{args.encoder_type}_"
        f"{'pretrained' if args.pretrained_encoder else 'scratch'}_"
        f"roi_{str(args.roi_width_ratio).replace('.', '')}_seed_{args.seed}.json"
    )
    save_experiment_results(filename, payload)

    print("=" * 100)
    print("FEATURE BASELINE SUMMARY")
    print("=" * 100)
    print(f"Mean default balanced accuracy   : {payload['mean_default_balanced_accuracy']:.6f}")
    print(f"Mean best-threshold balanced acc : {payload['mean_best_threshold_balanced_accuracy']:.6f}")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="ROI X-ray classifier experiments")
    parser.add_argument(
        "--experiment",
        choices=["single", "frozen", "split_sweep", "cv5", "metadata_baseline", "feature_baseline"],
        default="single",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_val_pos", type=int, default=3)
    parser.add_argument("--split_val_neg", type=int, default=3)
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--label_source", choices=["mask", "patient_range"], default="mask")
    parser.add_argument("--encoder_type", choices=["resnet34", "small_cnn"], default="resnet34")
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--min_save_epoch", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--include_val_augmentations", action="store_true")
    parser.add_argument("--no_debug_images", action="store_true")
    parser.add_argument("--scrub_prompts", action="store_true")
    parser.add_argument("--debug_dir", default="debug_classifier")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.experiment == "metadata_baseline":
        run_metadata_baseline(args)
    elif args.experiment == "feature_baseline":
        run_feature_baseline(args)
    elif args.experiment == "split_sweep":
        run_split_sweep(args)
    elif args.experiment == "cv5":
        run_cv5(args)
    else:
        run_name = (
            f"frozen_{variant_tag(args)}"
            if args.experiment == "frozen"
            else f"single_{variant_tag(args)}"
        )
        train_model(**common_train_kwargs(args, run_name))


if __name__ == "__main__":
    main()
