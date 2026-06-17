import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import SpineMultimodalDataset
from models.classifier import XrayClassifier
from train import build_patient_split, build_xray_encoder


def normalize_image(tensor):
    array = tensor.detach().cpu().numpy()
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val > min_val:
        array = (array - min_val) / (max_val - min_val)
    return array


def compute_gradcam(xray_enc, classifier, xray):
    if not hasattr(xray_enc, "resnet"):
        raise ValueError("Grad-CAM currently supports the ResNet XrayEncoder only.")

    activations = {}
    gradients = {}

    def forward_hook(module, module_input, module_output):
        activations["value"] = module_output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = xray_enc.resnet.layer4.register_forward_hook(forward_hook)
    backward_handle = xray_enc.resnet.layer4.register_full_backward_hook(backward_hook)

    xray_enc.zero_grad(set_to_none=True)
    classifier.zero_grad(set_to_none=True)

    embedding, _ = xray_enc(xray)
    logits = classifier(embedding)
    if logits.ndim > 1:
        logits = logits.view(-1)

    score = logits[0]
    prob = torch.sigmoid(score).item()
    score.backward()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=xray.shape[-2:], mode="bilinear", align_corners=False)
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    forward_handle.remove()
    backward_handle.remove()

    return prob, logits.item(), cam.detach()


def save_gradcam_figure(xray, mask, cam, row, out_path):
    xray_np = normalize_image(xray[0, 0])
    mask_np = mask[0, 0].detach().cpu().numpy()
    cam_np = cam[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    axes[0].imshow(xray_np, cmap="gray")
    axes[0].set_title(f"X-ray | {row['patient_id']}")
    axes[0].axis("off")

    axes[1].imshow(xray_np, cmap="gray")
    if mask_np.sum() > 0:
        axes[1].imshow(mask_np, cmap="Reds", alpha=0.45)
    axes[1].set_title(f"Mask | label={row['label']}")
    axes[1].axis("off")

    axes[2].imshow(xray_np, cmap="gray")
    axes[2].imshow(cam_np, cmap="jet", alpha=0.45)
    axes[2].set_title(
        f"Grad-CAM | prob={row['prob']:.3f} | pred={row['pred']}"
    )
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def load_models(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    encoder_type = ckpt.get("encoder_type", "resnet34")
    xray_enc = build_xray_encoder(
        encoder_type=encoder_type,
        embedding_dim=512,
        pretrained=False,
    ).to(device)
    classifier = XrayClassifier(embedding_dim=512, dropout=0.5).to(device)

    xray_enc.load_state_dict(ckpt["xray_enc"])
    classifier.load_state_dict(ckpt["classifier"])
    xray_enc.eval()
    classifier.eval()

    return ckpt, xray_enc, classifier


def run_checkpoint(checkpoint, args, device):
    ckpt, xray_enc, classifier = load_models(checkpoint, device)
    fold_name = os.path.splitext(os.path.basename(checkpoint))[0]
    out_dir = os.path.join(args.out_dir, fold_name)
    os.makedirs(out_dir, exist_ok=True)

    roi_width_ratio = float(ckpt.get("roi_width_ratio", args.roi_width_ratio))
    val_patients = ckpt["val_patients"]
    threshold = float(ckpt.get("best_threshold", args.threshold))
    original_only_val = bool(ckpt.get("original_only_val", True))

    dataset = SpineMultimodalDataset(
        data_dir="data/augmented_patients",
        csv_file="data/train_labels_augmented.csv",
        img_size=(1024, 512),
        is_train=False,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=roi_width_ratio,
        scrub_prompt=True,
        label_source="mask",
    )
    _, val_idx = build_patient_split(
        dataset=dataset,
        val_patients=val_patients,
        original_only_val=original_only_val,
    )

    rows = []
    for idx in val_idx:
        sample = dataset[idx]
        patient_id = sample["patient_id"]
        xray = sample["xray"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        label = int(sample["label"].item() > 0.5)

        xray_enc.zero_grad(set_to_none=True)
        classifier.zero_grad(set_to_none=True)

        prob, logit, cam = compute_gradcam(xray_enc, classifier, xray)
        pred = int(prob >= threshold)
        correct = int(pred == label)

        row = {
            "checkpoint": checkpoint,
            "fold": fold_name,
            "patient_id": patient_id,
            "base_patient_id": sample["base_patient_id"],
            "label": label,
            "prob": prob,
            "logit": logit,
            "threshold": threshold,
            "pred": pred,
            "correct": correct,
            "mask_sum": float(mask.sum().item()),
        }
        rows.append(row)

        status = "correct" if correct else "error"
        out_path = os.path.join(
            out_dir,
            f"{status}_{patient_id}_label_{label}_pred_{pred}_prob_{prob:.3f}.png",
        )
        save_gradcam_figure(xray, mask, cam, row, out_path)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for classifier checkpoints.")
    parser.add_argument(
        "--checkpoint_glob",
        default="checkpoints/cv5_resnet34_pretrained_mask_scrubbed_roi045_fold_*_best_xray_classifier_roi_045_balacc.pth",
    )
    parser.add_argument("--out_dir", default="evidence/gradcam")
    parser.add_argument("--roi_width_ratio", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {args.checkpoint_glob}")

    all_rows = []
    for checkpoint in checkpoints:
        print("=" * 100)
        print(f"Grad-CAM checkpoint: {checkpoint}")
        print("=" * 100)
        all_rows.extend(run_checkpoint(checkpoint, args, device))

    csv_path = os.path.join(args.out_dir, "gradcam_error_rows.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    correct = sum(row["correct"] for row in all_rows)
    print("=" * 100)
    print("GRAD-CAM COMPLETE")
    print("=" * 100)
    print(f"Rows       : {total}")
    print(f"Correct    : {correct}")
    print(f"Errors     : {total - correct}")
    print(f"CSV        : {csv_path}")
    print(f"Output dir : {args.out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
