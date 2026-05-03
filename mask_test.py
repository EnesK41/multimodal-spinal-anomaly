import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def investigate_huge_masks(data_dir="data/augmented_patients", threshold=25000):
    print(f"🔍 {threshold} pikselden büyük maskeler taranıyor...")
    
    suspect_count = 0
    os.makedirs("suspect_masks", exist_ok=True)

    for patient_folder in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        mask_path = os.path.join(patient_path, f"{patient_folder}_xray_mask.nii.gz")
        xray_path = os.path.join(patient_path, f"{patient_folder}_xray.nii.gz")

        if os.path.exists(mask_path) and os.path.exists(xray_path):
            try:
                # Maskeyi oku ve normalize et
                mask_nii = nib.load(mask_path)
                mask_data = np.squeeze(mask_nii.get_fdata())
                
                # 0.5'ten büyük pikselleri say
                pixel_count = np.sum(mask_data > 0.5)

                if pixel_count > threshold:
                    suspect_count += 1
                    print(f"🚨 ŞÜPHELİ: {patient_folder} | Piksel: {pixel_count}")

                    # Röntgeni oku
                    xray_nii = nib.load(xray_path)
                    xray_data = np.squeeze(xray_nii.get_fdata())

                    # Görselleştirme (Matplotlib)
                    plt.figure(figsize=(15, 5))
                    plt.suptitle(f"Hasta: {patient_folder} - Piksel: {pixel_count}", fontsize=16)

                    # 1. Sadece X-Ray
                    plt.subplot(1, 3, 1)
                    plt.title("X-Ray")
                    plt.imshow(xray_data, cmap='gray')
                    plt.axis('off')

                    # 2. Sadece Maske
                    plt.subplot(1, 3, 2)
                    plt.title("Maske")
                    plt.imshow(mask_data, cmap='gray')
                    plt.axis('off')

                    # 3. Overlay (Üst Üste)
                    plt.subplot(1, 3, 3)
                    plt.title("Üst Üste (Overlay)")
                    plt.imshow(xray_data, cmap='gray')
                    # Maskeyi sadece piksellerin olduğu yerde kırmızıya boya
                    masked_overlay = np.ma.masked_where(mask_data < 0.5, mask_data)
                    plt.imshow(masked_overlay, cmap='hsv', alpha=0.5, interpolation='none')
                    plt.axis('off')

                    save_path = os.path.join("suspect_masks", f"{patient_folder}.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    print(f"   📸 Görsel kaydedildi: {save_path}")
                    
            except Exception as e:
                print(f"Hata oluştu ({patient_folder}): {e}")

    if suspect_count == 0:
        print(f"✅ Temiz! {threshold} pikselden büyük maske bulunamadı.")
    else:
        print(f"\n📁 Toplam {suspect_count} adet şüpheli maske 'suspect_masks' klasörüne kaydedildi. Lütfen fotoğrafları incele!")

if __name__ == "__main__":
    investigate_huge_masks()