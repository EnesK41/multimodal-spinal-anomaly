import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob
from models.XrayEncoder import XrayEncoder
from models.MaskDecoder import MaskDecoder
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

TRAIN_TARGET_SIZE = (512, 256)
RUNTIME_ROTATE_90 = False

def calculate_dice(pred, target, threshold=0.5):
    # Tahmini ve hedefi binary (0-1) formatina getiriyoruz
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    
    intersection = (pred_bin * target_bin).sum()
    dice = (2. * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-7)
    return dice

def _resolve_checkpoint_path(checkpoints_dir="checkpoints"):
    # En son veya en iyi modeli (best_model.pth) oncelikli al
    if os.path.exists(os.path.join(checkpoints_dir, "best_model.pth")):
        return os.path.join(checkpoints_dir, "best_model.pth")
    
    candidates = glob.glob(os.path.join(checkpoints_dir, "spine_model_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint bulunamadi: {checkpoints_dir}")
    
    def epoch_key(path):
        name = os.path.basename(path)
        num = "".join(ch for ch in name if ch.isdigit())
        return int(num) if num else -1
    return sorted(candidates, key=epoch_key)[-1]

def _load_2d_nifti(path, target_size=TRAIN_TARGET_SIZE, rotate_90=True, is_mask=False):
    img_obj = nib.as_closest_canonical(nib.load(path))
    img = np.squeeze(img_obj.get_fdata())
    
    if not is_mask:
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mode = 'nearest' if is_mask else 'bilinear'
    tensor = F.interpolate(tensor, size=target_size, mode=mode, align_corners=False if mode == 'bilinear' else None)
    tensor = tensor.squeeze(0)

    if rotate_90:
        tensor = torch.rot90(tensor, k=1, dims=(1, 2))
    if is_mask:
        tensor = (tensor > 0.5).float()
    return tensor

def run_test_comparison(patient_id, data_dir, text_threshold=0.20, mask_threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint_path()
    print(f"Kullanilan checkpoint: {checkpoint_path}")

    xray_enc = XrayEncoder(embedding_dim=512).to(device)
    mask_dec = MaskDecoder(embedding_dim=512).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    xray_enc.load_state_dict(checkpoint['xray_enc'])
    mask_dec.load_state_dict(checkpoint['mask_dec'])
    xray_enc.eval()
    mask_dec.eval()

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()

    diagnosis_options = [
        "healthy spine with no anomalies",
        "block vertebra anomaly",
        "hemivertebra anomaly",
        "butterfly vertebra anomaly",
        "wedge vertebra anomaly"
    ]

    patient_folder = os.path.join(data_dir, patient_id)
    xray_path = os.path.join(patient_folder, f"{patient_id}_xray.nii.gz")
    mask_path = os.path.join(patient_folder, f"{patient_id}_xray_mask.nii.gz")

    input_tensor = _load_2d_nifti(xray_path, rotate_90=RUNTIME_ROTATE_90, is_mask=False).unsqueeze(0).to(device)
    
    true_mask = None
    if os.path.exists(mask_path):
        true_mask = _load_2d_nifti(mask_path, rotate_90=RUNTIME_ROTATE_90, is_mask=True).squeeze().numpy()

    with torch.no_grad():
        embedding, skip_feats = xray_enc(input_tensor)
        
        # Metin Teshisi
        text_inputs = clip_tokenizer(diagnosis_options, padding=True, return_tensors="pt").to(device)
        text_embeds = clip_model(**text_inputs).text_embeds
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)
        
        # Kosinus benzerligini al ve Softmax ile % yuzdelere cevir (Farklari acmak icin 10 ile carpiyoruz)
        similarities = torch.matmul(embedding_norm, text_embeds_norm.T).squeeze(0)
        probs = torch.nn.functional.softmax(similarities * 10, dim=0) 
        
        detected = []
        for idx, prob in enumerate(probs):
            if prob.item() > 0.15: # Sadece %15'ten yuksek ihtimal verdigi mantikli secenekleri goster
                detected.append(f"- {diagnosis_options[idx]} (%{prob.item()*100:.1f})")
                
        final_diagnosis_text = "AI Teshisi:\n" + "\n".join(detected if detected else ["- Kararsiz (Dusuk Guven)"])

        # Maske Cizimi
        mask_pred = torch.sigmoid(mask_dec(embedding, skip_feats)).squeeze().cpu().numpy()

    # Dice Skoru Hesapla
    dice_score = 0.0
    if true_mask is not None:
        dice_score = calculate_dice(mask_pred, true_mask, threshold=mask_threshold)
        print(f"HASTA {patient_id} ICIN DICE SKORU: {dice_score:.4f}")

    # Gorsellestirme
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 3, 1)
    plt.title(f"{patient_id} - Orijinal")
    plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Gercek Maske (Doktor)")
    plt.imshow(true_mask if true_mask is not None else np.zeros(TRAIN_TARGET_SIZE), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title(f"AI Tahmini (Dice: {dice_score:.4f})")
    plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray')
    
    # Tahmin maskesi overlay
    mask_show = np.ma.masked_where(mask_pred < mask_threshold, mask_pred)
    plt.imshow(mask_show, cmap='Reds', alpha=0.6)
    
    plt.text(5, 40, final_diagnosis_text, color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"final_test_{patient_id}.png")
    plt.show()

if __name__ == "__main__":
    run_test_comparison(patient_id="patient_057", data_dir="data/original_patients")