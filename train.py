import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.XrayEncoder import XrayEncoder
from models.MaskDecoder import MaskDecoder
from dataset import SpineMultimodalDataset
from loss import HybridLoss
from dataset import print_dataset_stats




def initialize_decoder_bias(mask_dec):

    for module in mask_dec.modules():
        if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
            torch.nn.init.constant_(module.bias, -2.0)


def dice_score_from_logits(logits, target, threshold=0.1):

    prob = torch.sigmoid(logits)

    pred = (prob > threshold).float()

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)

    return dice.item(), prob


def save_debug_image(xray, mask, pred=None, name="debug.png"):

    os.makedirs("debug", exist_ok=True)

    xray = xray[0, 0].cpu().numpy()
    mask = mask[0, 0].cpu().numpy()

    cols = 3 if pred is not None else 2

    plt.figure(figsize=(5 * cols, 5))

    plt.subplot(1, cols, 1)
    plt.imshow(xray, cmap="gray")
    plt.title("Xray")
    plt.axis("off")

    plt.subplot(1, cols, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("GT mask")
    plt.axis("off")

    if pred is not None:

        pred = pred[0, 0].cpu().numpy()

        plt.subplot(1, cols, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(f"debug/{name}", dpi=150)

    plt.close()


def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2
    EPOCHS = 50
    LR = 1e-4
    VAL_EVERY = 2
    ROI_WIDTH_RATIO = 0.45
    CHECKPOINT_PATH = "checkpoints/best_model_roi_045_keep_scoliosis.pth"

    train_dataset = SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=ROI_WIDTH_RATIO,
    )

    val_dataset = SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=False,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=ROI_WIDTH_RATIO,
    )

    val_patients = [
        "patient_004",
        "patient_028",
        "patient_045"
    ]

    print_dataset_stats(train_dataset)

    train_idx = [
        i for i, pid in enumerate(train_dataset.patient_ids)
        if pid.split("_aug")[0] not in val_patients
    ]

    val_idx = [
        i for i, pid in enumerate(val_dataset.patient_ids)
        if pid.split("_aug")[0] in val_patients
    ]

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=1
    )

    xray_enc = XrayEncoder(embedding_dim=512).to(device)
    mask_dec = MaskDecoder(embedding_dim=512).to(device)

    initialize_decoder_bias(mask_dec)

    optimizer = optim.AdamW(
        list(xray_enc.parameters()) + list(mask_dec.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    loss_fn = HybridLoss().to(device)

    best_val_dice = 0

    print("TRAIN SIZE:", len(train_idx))
    print("VAL SIZE:", len(val_idx))

    sample = train_dataset[0]

    print("SANITY CHECK")
    print("mask sum:", sample["mask"].sum())

    save_debug_image(
        sample["xray"].unsqueeze(0),
        sample["mask"].unsqueeze(0),
        name="dataset_sample.png"
    )

    for epoch in range(EPOCHS):

        print("\n" + "=" * 100)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        print("=" * 100)

        xray_enc.train()
        mask_dec.train()

        train_loss = 0.0
        train_dice = 0.0
        train_batches = 0

        train_prob_min_total = 0.0
        train_prob_mean_total = 0.0
        train_prob_max_total = 0.0

        train_gt_pixels_total = 0.0
        train_pred_pixels_total = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):

            xray = batch["xray"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            emb, skips = xray_enc(xray)
            pred = mask_dec(emb, skips)

            loss = loss_fn(pred, mask)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(xray_enc.parameters()) + list(mask_dec.parameters()),
                1.0
            )

            optimizer.step()

            dice, prob = dice_score_from_logits(pred.detach(), mask)

            pred_bin = (prob > 0.1).float()

            batch_gt_pixels = mask.sum().item()
            batch_pred_pixels = pred_bin.sum().item()

            train_loss += loss.item()
            train_dice += dice
            train_batches += 1

            train_prob_min_total += prob.min().item()
            train_prob_mean_total += prob.mean().item()
            train_prob_max_total += prob.max().item()

            train_gt_pixels_total += batch_gt_pixels
            train_pred_pixels_total += batch_pred_pixels

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "gt_px": f"{batch_gt_pixels:.0f}",
                "pred_px": f"{batch_pred_pixels:.0f}",
                "pmax": f"{prob.max().item():.3f}",
                "grad": f"{float(grad_norm):.3f}"
            })

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_dice = train_dice / max(train_batches, 1)

        avg_train_prob_min = train_prob_min_total / max(train_batches, 1)
        avg_train_prob_mean = train_prob_mean_total / max(train_batches, 1)
        avg_train_prob_max = train_prob_max_total / max(train_batches, 1)

        avg_train_gt_pixels = train_gt_pixels_total / max(train_batches, 1)
        avg_train_pred_pixels = train_pred_pixels_total / max(train_batches, 1)

        print("\n" + "-" * 100)
        print(f"TRAIN SUMMARY | Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 100)
        print(f"Train Loss              : {avg_train_loss:.6f}")
        print(f"Train Dice              : {avg_train_dice:.6f}")
        print(f"Avg GT pixels / batch   : {avg_train_gt_pixels:.2f}")
        print(f"Avg Pred pixels / batch : {avg_train_pred_pixels:.2f}")
        print(
            f"Train Prob min/mean/max : "
            f"{avg_train_prob_min:.6f} / "
            f"{avg_train_prob_mean:.6f} / "
            f"{avg_train_prob_max:.6f}"
        )
        print("-" * 100)

        if (epoch + 1) % VAL_EVERY == 0:

            print("\n" + "#" * 100)
            print(f"VALIDATION START | Epoch {epoch + 1}/{EPOCHS}")
            print("#" * 100)

            xray_enc.eval()
            mask_dec.eval()

            inter = 0.0
            union = 0.0

            val_patient_results = []

            with torch.no_grad():

                for val_idx, batch in enumerate(val_loader):

                    patient_id = batch["patient_id"][0]

                    xray = batch["xray"].to(device)
                    mask = batch["mask"].to(device)

                    emb, skips = xray_enc(xray)
                    pred = mask_dec(emb, skips)

                    prob = torch.sigmoid(pred)
                    pred_bin = (prob > 0.1).float()

                    patient_inter = (pred_bin * mask).sum().item()
                    patient_union = pred_bin.sum().item() + mask.sum().item()
                    patient_dice = (2.0 * patient_inter + 1e-7) / (patient_union + 1e-7)

                    gt_pixels = mask.sum().item()
                    pred_pixels = pred_bin.sum().item()

                    prob_min = prob.min().item()
                    prob_mean = prob.mean().item()
                    prob_max = prob.max().item()

                    inter += patient_inter
                    union += patient_union

                    val_patient_results.append({
                        "patient_id": patient_id,
                        "dice": patient_dice,
                        "gt_pixels": gt_pixels,
                        "pred_pixels": pred_pixels,
                        "prob_min": prob_min,
                        "prob_mean": prob_mean,
                        "prob_max": prob_max,
                    })

                    save_debug_image(
                        xray,
                        mask,
                        prob,
                        f"epoch_{epoch + 1:03d}_{patient_id}.png"
                    )

            val_dice = (2.0 * inter + 1e-7) / (union + 1e-7)

            print("\n" + "-" * 100)
            print(f"VALIDATION PATIENT RESULTS | Epoch {epoch + 1}/{EPOCHS}")
            print("-" * 100)

            for r in val_patient_results:
                print(
                    f"{r['patient_id']:20s} | "
                    f"Dice={r['dice']:.4f} | "
                    f"GT={r['gt_pixels']:8.0f} | "
                    f"Pred={r['pred_pixels']:8.0f} | "
                    f"Prob min/mean/max="
                    f"{r['prob_min']:.5f}/"
                    f"{r['prob_mean']:.5f}/"
                    f"{r['prob_max']:.5f}"
                )

            print("-" * 100)
            print(f"VAL DICE                 : {val_dice:.6f}")
            print(f"BEST VAL DICE SO FAR    : {best_val_dice:.6f}")
            print("-" * 100)

            scheduler.step(val_dice)

            print(f"LR after scheduler       : {optimizer.param_groups[0]['lr']:.8f}")

            if val_dice > best_val_dice:

                old_best = best_val_dice
                best_val_dice = val_dice

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "xray_enc": xray_enc.state_dict(),
                        "mask_dec": mask_dec.state_dict(),
                        "best_val_dice": best_val_dice,
                        "optimizer": optimizer.state_dict(),
                        "roi_width_ratio": ROI_WIDTH_RATIO,
                        "val_patients": val_patients,
                    },
                    CHECKPOINT_PATH
                )

                print(
                    f"MODEL SAVED | "
                    f"Old Best={old_best:.6f} -> New Best={best_val_dice:.6f}"
                )

            else:
                print("MODEL NOT SAVED | Validation did not improve.")

            print("#" * 100)

        else:
            print(
                f"Validation skipped this epoch. "
                f"Next validation at epoch {((epoch + 1) // VAL_EVERY + 1) * VAL_EVERY}."
            )

    print("\n" + "=" * 100)
    print("TRAINING FINISHED")
    print(f"BEST VAL DICE: {best_val_dice:.6f}")
    print("=" * 100)


if __name__ == "__main__":

    os.makedirs("checkpoints", exist_ok=True)

    train_model()