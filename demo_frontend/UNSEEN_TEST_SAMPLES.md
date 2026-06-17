# Demo Held-Out Test Samples

Loaded checkpoint:

`checkpoints/png_roi045_no_extra_roi_seed42_resnet34_pretrained_mask_scrubbed_fold_04_best.pth`

This is a fold 04 model. These base patients were held out from training and used as the original-only validation/test fold:

`patient_002`, `patient_007`, `patient_018`, `patient_027`, `patient_032`, `patient_042`, `patient_045`, `patient_052`

Built-in demo buttons use held-out examples:

| Button | Patient | Label | Used in training? |
|---|---|---|---|
| Test Healthy | `patient_032` | Healthy / empty | No |
| Test Anomaly | `patient_002` | Anomaly | No |

All held-out test image files:

| Patient | Label | PNG path |
|---|---|---|
| `patient_002` | Anomaly | `data/png_patients_full/patient_002/patient_002_xray.png` |
| `patient_007` | Anomaly | `data/png_patients_full/patient_007/patient_007_xray.png` |
| `patient_018` | Anomaly | `data/png_patients_full/patient_018/patient_018_xray.png` |
| `patient_027` | Anomaly | `data/png_patients_full/patient_027/patient_027_xray.png` |
| `patient_032` | Healthy / empty | `data/png_patients_full/patient_032/patient_032_xray.png` |
| `patient_042` | Healthy / empty | `data/png_patients_full/patient_042/patient_042_xray.png` |
| `patient_045` | Anomaly | `data/png_patients_full/patient_045/patient_045_xray.png` |
| `patient_052` | Anomaly | `data/png_patients_full/patient_052/patient_052_xray.png` |

Evidence file:

`experiment_results/png_classifier/app_model_fold04_seen_unseen_eval.json`
