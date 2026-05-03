import os
import nibabel as nib
import numpy as np

# Senin data klasörünün yolu
data_dir = "data/augmented_patients" 
patient_to_check = "patient_051_aug5" # Uyarı veren herhangi bir hasta

mask_path = os.path.join(data_dir, patient_to_check, f"{patient_to_check}_xray_mask.nii.gz")

if os.path.exists(mask_path):
    img = nib.load(mask_path).get_fdata()
    max_val = np.max(img)
    min_val = np.min(img)
    unique_vals = np.unique(img)
    
    print(f"--- Maske İncelemesi: {patient_to_check} ---")
    print(f"En Yüksek Piksel Değeri: {max_val}")
    print(f"En Düşük Piksel Değeri: {min_val}")
    print(f"Farklı Değerler (İlk 10): {unique_vals[:10]}")
    
    if max_val == 0.0:
        print("🚨 TEŞHİS: Bu dosyanın içi KESİNLİKLE BOŞ (Simsiyah).")
    elif max_val < 0.5:
        print("⚠️ TEŞHİS: Dosya boş değil ama değerler 0.5'in altında. Threshold (Eşik) değerimizi düşürmeliyiz!")
else:
    print("Dosya bulunamadı.")