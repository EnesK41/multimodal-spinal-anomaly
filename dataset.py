import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as F


BAD_PATIENTS = [
    "patient_005",
    "patient_015",
    "patient_017",
    "patient_019",
    "patient_031",
    "patient_043",
    "patient_044",
    "patient_046",
    "patient_047",
    "patient_048",
    "patient_050",
    "patient_051",
    "patient_054",
]


class SpineMultimodalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file,
        img_size=(1024, 512),
        rotate_90=False,
        require_xray=True,
        is_train=True,
        exclude_bad_patients=True,
        use_body_roi=True,
        roi_width_ratio=0.45,
    ):
        self.is_train = is_train
        self.data_dir = data_dir
        self.img_size = img_size
        self.rotate_90 = rotate_90
        self.exclude_bad_patients = exclude_bad_patients
        self.use_body_roi = use_body_roi
        self.roi_width_ratio = roi_width_ratio

        self.labels_df = pd.read_csv(csv_file)
        self.labels_df["patient_id"] = self.labels_df["patient_id"].astype(str).str.strip()

        if require_xray:
            self.labels_df = self.labels_df[
                self.labels_df["patient_id"].apply(
                    lambda pid: os.path.exists(
                        os.path.join(self.data_dir, pid, f"{pid}_xray.nii.gz")
                    )
                )
            ]

        if self.exclude_bad_patients:
            self.labels_df = self.labels_df[
                ~self.labels_df["patient_id"].apply(
                    lambda pid: self.base_patient_id(pid) in BAD_PATIENTS
                )
            ]

        self.labels_df = self.labels_df.reset_index(drop=True)

        self.patient_ids = self.labels_df["patient_id"].tolist()
        self.prompts = self.labels_df["prompt"].tolist()

        print(
            f"[Dataset] samples={len(self.patient_ids)} | "
            f"exclude_bad={self.exclude_bad_patients} | "
            f"use_body_roi={self.use_body_roi} | "
            f"img_size={self.img_size}"
        )

    def __len__(self):
        return len(self.patient_ids)

    @staticmethod
    def base_patient_id(pid):
        return str(pid).split("_aug")[0]

    def normalize_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)

        return tensor

    def resize_with_padding(self, tensor):
        _, h, w = tensor.shape
        target_h, target_w = self.img_size

        scale = min(target_h / h, target_w / w)

        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h = target_h - new_h
        pad_w = target_w - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

        return tensor

    def body_center_roi_box(self, xray):
        """
        Mask kullanmadan kaba body/omurga merkezi ROI bulur.
        xray shape: [1, H, W], normalize edilmiş 0-1 arası.
        """
        _, H, W = xray.shape
        img = xray[0]

        threshold = max(0.03, float(img.mean()) * 0.5)
        foreground = img > threshold

        ys, xs = torch.where(foreground)

        if len(ys) == 0:
            return 0, H, 0, W

        y1 = int(ys.min().item())
        y2 = int(ys.max().item())
        x1_body = int(xs.min().item())
        x2_body = int(xs.max().item())

        body_w = x2_body - x1_body + 1
        cx = (x1_body + x2_body) // 2

        roi_w = int(body_w * self.roi_width_ratio)

        x1 = max(0, cx - roi_w // 2)
        x2 = min(W, cx + roi_w // 2)

        # Dikey eksende full body kalsın. Sadece küçük margin güvenliği.
        margin_y = int(H * 0.02)
        y1 = max(0, y1 - margin_y)
        y2 = min(H, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return 0, H, 0, W

        return y1, y2, x1, x2

    def apply_body_roi(self, xray, mask):
        y1, y2, x1, x2 = self.body_center_roi_box(xray)

        xray_crop = xray[:, y1:y2, x1:x2]
        mask_crop = mask[:, y1:y2, x1:x2]

        xray_crop = self.resize_with_padding(xray_crop)
        mask_crop = self.resize_with_padding(mask_crop)
        mask_crop = (mask_crop > 0.1).float()

        return xray_crop, mask_crop

    def load_2d_nifti_raw(self, path, is_mask=False):
        img_obj = nib.load(path)
        img_obj = nib.as_closest_canonical(img_obj)
        img = img_obj.get_fdata()
        img = np.squeeze(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.rotate_90:
            tensor = torch.rot90(tensor, k=1, dims=(1, 2))

        if is_mask:
            tensor = (tensor > 0).float()
        else:
            tensor = self.normalize_tensor(tensor)

        return tensor

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        prompt = self.prompts[idx]

        if not isinstance(prompt, str) or prompt.strip() == "":
            prompt = "Spinal x-ray showing spinal anomaly."

        patient_folder = os.path.join(self.data_dir, patient_id)

        xray_path = os.path.join(patient_folder, f"{patient_id}_xray.nii.gz")
        mask_path = os.path.join(patient_folder, f"{patient_id}_xray_mask.nii.gz")

        xray_tensor = self.load_2d_nifti_raw(xray_path, is_mask=False)

        if os.path.exists(mask_path):
            mask_tensor = self.load_2d_nifti_raw(mask_path, is_mask=True)
        else:
            mask_tensor = torch.zeros_like(xray_tensor)

        # Önce full image normalize boyuta gelsin
        xray_tensor = self.resize_with_padding(xray_tensor)
        mask_tensor = self.resize_with_padding(mask_tensor)
        mask_tensor = (mask_tensor > 0.1).float()

        # Sonra body-center ROI uygula
        if self.use_body_roi:
            xray_tensor, mask_tensor = self.apply_body_roi(xray_tensor, mask_tensor)

        return {
            "patient_id": patient_id,
            "xray": xray_tensor,
            "mask": mask_tensor,
            "text": prompt,
        }


def print_dataset_stats(dataset):
    positive = 0
    empty = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["mask"].sum().item() > 0:
            positive += 1
        else:
            empty += 1

    print("=" * 60)
    print("DATASET STATS")
    print(f"Total    : {len(dataset)}")
    print(f"Positive : {positive}")
    print(f"Empty    : {empty}")
    print("=" * 60)