import os
import random
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF

def apply_augmentations(img_tensor, mask_tensor):
    angle = random.uniform(-10, 10)
    trans_x = int(random.uniform(-0.05, 0.05) * 256) 
    trans_y = int(random.uniform(-0.05, 0.05) * 512)
    scale = random.uniform(0.9, 1.1)

    img_tensor = TF.affine(img_tensor, angle=angle, translate=[trans_x, trans_y], scale=scale, shear=0)
    mask_tensor = TF.affine(mask_tensor, angle=angle, translate=[trans_x, trans_y], scale=scale, shear=0)

    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    img_tensor = TF.adjust_brightness(img_tensor, brightness_factor=brightness)
    img_tensor = TF.adjust_contrast(img_tensor, contrast_factor=contrast)

    if random.random() > 0.5:
        noise = torch.randn_like(img_tensor) * 0.02
        img_tensor = img_tensor + noise
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0) # Artık veri [0,1] olduğu için güvenli!

    mask_tensor = (mask_tensor > 0.5).float()

    return img_tensor, mask_tensor

def generate_offline_dataset(num_aug_per_image=5):
    original_data_dir = "data/original_patients"
    new_data_dir = "data/augmented_patients"
    csv_name = "train_labels.csv"
    data_dir = "data"
    
    os.makedirs(new_data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, csv_name)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"HATA: {csv_path} okunamadi. Hata: {e}")
        return

    new_csv_data = []

    print(f"Veri cogaltma islemi basliyor... Toplam orijinal hasta: {len(df)}")

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        prompt = row['prompt']
        
        orig_patient_dir = os.path.join(original_data_dir, patient_id)
        xray_path = os.path.join(orig_patient_dir, f"{patient_id}_xray.nii.gz")
        mask_path = os.path.join(orig_patient_dir, f"{patient_id}_xray_mask.nii.gz")
        
        if not os.path.exists(xray_path):
            print(f"UYARI: {patient_id} icin X-Ray dosyasi bulunamadi, atlaniyor.")
            continue

        try:
            xray_nii = nib.load(xray_path)
            xray_data = np.squeeze(xray_nii.get_fdata())
            affine = xray_nii.affine
            
            # --- HAYATI KURTARAN YENİ DÜZELTME: NORMALİZASYON ---
            # Görüntüyü augmentasyona sokmadan önce 0 ile 1 arasına sıkıştırıyoruz.
            # Böylece ne parlaklık ayarı ne de clamp işlemi kemik detaylarını bozamaz.
            xray_min = xray_data.min()
            xray_max = xray_data.max()
            if xray_max - xray_min > 0:
                xray_data = (xray_data - xray_min) / (xray_max - xray_min)
            # -----------------------------------------------------

            if os.path.exists(mask_path):
                mask_nii = nib.load(mask_path)
                mask_data = np.squeeze(mask_nii.get_fdata())
            else:
                mask_data = np.zeros_like(xray_data)
                print(f"  -> BILGI: {patient_id} saglikli kabul edildi, bos maske uretildi.")

            xray_tensor = torch.tensor(xray_data, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
            
            new_csv_data.append({"patient_id": patient_id, "prompt": prompt})
            
            new_patient_dir = os.path.join(new_data_dir, patient_id)
            os.makedirs(new_patient_dir, exist_ok=True)
            
            nib.save(nib.Nifti1Image(xray_data, affine), os.path.join(new_patient_dir, f"{patient_id}_xray.nii.gz"))
            nib.save(nib.Nifti1Image(mask_data, affine), os.path.join(new_patient_dir, f"{patient_id}_xray_mask.nii.gz"))
            
            print(f"[{index+1}/{len(df)}] Isleniyor: {patient_id} -> Orijinal kaydedildi.")

            for i in range(num_aug_per_image):
                aug_id = f"{patient_id}_aug{i+1}"
                new_aug_dir = os.path.join(new_data_dir, aug_id)
                os.makedirs(new_aug_dir, exist_ok=True)
                
                aug_xray_tensor, aug_mask_tensor = apply_augmentations(xray_tensor, mask_tensor)
                
                aug_xray_np = aug_xray_tensor.squeeze(0).numpy()
                aug_mask_np = aug_mask_tensor.squeeze(0).numpy()
                
                aug_xray_nii = nib.Nifti1Image(aug_xray_np, affine)
                aug_mask_nii = nib.Nifti1Image(aug_mask_np, affine)
                
                nib.save(aug_xray_nii, os.path.join(new_aug_dir, f"{aug_id}_xray.nii.gz"))
                nib.save(aug_mask_nii, os.path.join(new_aug_dir, f"{aug_id}_xray_mask.nii.gz"))
                
                new_csv_data.append({"patient_id": aug_id, "prompt": prompt})
                
        except Exception as e:
            print(f"HATA: {patient_id} islenirken hata olustu: {e}")

    new_df = pd.DataFrame(new_csv_data)
    new_csv_path = os.path.join(data_dir, "train_labels_augmented.csv")
    new_df.to_csv(new_csv_path, index=False)
    
    print("\n--- ISLEM TAMAM ---")

if __name__ == "__main__":
    generate_offline_dataset()