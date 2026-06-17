# CT Projection Preprocessing Notes

## Problem Found

The first CT projection alignment experiment used a simple max projection with:

```text
axis=0
no HU threshold
no rotation correction
```

Visual review showed this was not a good X-ray-like frontal projection. The output was effectively lying sideways and did not match the X-ray field of view/orientation.

## Current Correction

The visually correct projection setup is:

```text
projection axis = 1
rotation        = rot90 k=1
HU threshold    = 300
clip max        = 1800
review size     = 512 x 256
training size   = 1024 x 512
```

The training size remains 1024 x 512 because the current X-ray ROI classifier pipeline uses that size. The review size 512 x 256 has the same aspect ratio and is easier to inspect.

## Why Rotation Is Needed

CT is a 3D NIfTI volume, but the array axes and physical display orientation are not the same as an X-ray image. After choosing the correct projection axis, the resulting 2D array can still be rotated relative to the desired radiographic view. This does not mean the CT data itself is wrong; it means the generated 2D projection needs display/preprocessing correction.

We should not modify the original CT files. The safer approach is to keep original NIfTI files unchanged and store preprocessing choices in scripts, cache filenames, result JSONs, and notes.

## HU Threshold Choice

Visual trials showed that HU 150-400 all produce acceptable bone-focused projections. HU 300 was selected as a conservative middle value:

```text
HU >= 300
```

Very high thresholds such as 700-1000 remove too much vertebral detail.

## Evidence Files

HU threshold trials:

```text
evidence/ct_hu_threshold_trials
```

HU300 CT projection review with CT-mask overlay:

```text
evidence/ct_hu300_review/ct_hu300_axis1_rot1_512x256_contact_sheet.png
evidence/ct_hu300_review/ct_projection_review_summary.csv
```

## Training Command

```bash
.venv/Scripts/python.exe train_multimodal_alignment.py --epochs 5 --folds 5 --batch_size 8 --pretrained_encoder --identity_embedding_head --num_workers 0 --ct_projection_source ct --ct_projection_axis 1 --ct_rotate_k 1 --ct_hu_threshold 300
```

## Corrected CT Alignment Result

Result file:

```text
experiment_results/multimodal_simple_ct_max_axis1_hu300.0_rot1_align01_seed42.json
```

Mean best-threshold balanced accuracy:

```text
0.885714
```

Fold balanced accuracies:

```text
0.928571, 1.000000, 0.833333, 0.916667, 0.750000
```

The corrected projection is visually more defensible than the first raw CT projection, but this simple latent-alignment experiment did not improve the classification score. This suggests that orientation and HU correction alone are not enough; the remaining limiting factors are likely CT/X-ray registration, field-of-view mismatch, lack of healthy CT scans, and the weak supervision signal from cosine embedding alignment.

## Corrected CT-Mask Alignment Result

Command:

```bash
.venv/Scripts/python.exe train_multimodal_alignment.py --epochs 5 --folds 5 --batch_size 8 --pretrained_encoder --identity_embedding_head --num_workers 0 --ct_projection_source ct_mask --ct_projection_axis 1 --ct_rotate_k 1
```

Result file:

```text
experiment_results/multimodal_simple_ct_mask_max_axis1_hunone_rot1_align01_seed42.json
```

Mean best-threshold balanced accuracy:

```text
0.919048
```

Fold balanced accuracies:

```text
0.928571, 1.000000, 0.833333, 0.916667, 0.916667
```

This is better than the first sideways CT-mask alignment result, which was 0.902381. Correcting the mask projection orientation therefore helped the CT-mask branch, but it only matched the strongest X-ray-only seed-42 score rather than exceeding it.
