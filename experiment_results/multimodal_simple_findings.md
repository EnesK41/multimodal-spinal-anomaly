# Simple Multimodal CT-Projection Alignment Findings

## Experiment

Implemented a first-pass multimodal experiment without registration or manual cropping.

Pipeline:

```text
X-ray ROI -> trainable ResNet34 encoder -> classifier -> anomaly / healthy
CT volume -> raw 2D max projection -> frozen pretrained ResNet34 encoder -> CT embedding
```

Training loss:

```text
BCE classification loss on all X-ray samples
+ 0.1 * cosine latent-alignment loss only for samples whose base patient has CT
```

Healthy patients do not have CT in this cleaned dataset, so CT is used only as a privileged anomaly-anatomy teacher, not as a balanced healthy/anomaly CT classifier.

## Data Audit

- Total cleaned samples: 252
- Unique base patients: 42
- Positive/anomaly patients: 31
- Healthy/empty patients: 11
- Samples with paired CT: 186 / 252
- Base patients with paired CT: 31 / 42
- Healthy samples with CT: 0

This confirms that CT availability is perfectly tied to anomaly status in this cleaned set.

## Command

```bash
.venv/Scripts/python.exe train_multimodal_alignment.py --epochs 5 --folds 5 --batch_size 8 --pretrained_encoder --identity_embedding_head --num_workers 0
```

## Results

Raw CT projection result file:

```text
experiment_results/multimodal_simple_ct_max_axis0_align01_seed42.json
```

Mean best-threshold balanced accuracy:

```text
0.885714
```

Fold results:

| Fold | Balanced Accuracy | Epoch | Threshold | Sensitivity | Specificity | AUC | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.928571 | 4 | 0.50 | 0.857143 | 1.000000 | 0.904762 | 6/3/0/1 |
| 2 | 1.000000 | 5 | 0.40 | 1.000000 | 1.000000 | 1.000000 | 6/2/0/0 |
| 3 | 0.833333 | 1 | 0.40 | 0.666667 | 1.000000 | 0.666667 | 4/2/0/2 |
| 4 | 0.916667 | 4 | 0.60 | 0.833333 | 1.000000 | 0.916667 | 5/2/0/1 |
| 5 | 0.750000 | 2 | 0.50 | 1.000000 | 0.500000 | 0.583333 | 6/1/1/0 |

CT-mask projection result file:

```text
experiment_results/multimodal_simple_ct_mask_max_axis0_align01_seed42.json
```

Mean best-threshold balanced accuracy:

```text
0.902381
```

Fold results:

| Fold | Balanced Accuracy | Epoch | Threshold | Sensitivity | Specificity | AUC | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.928571 | 4 | 0.50 | 0.857143 | 1.000000 | 0.904762 | 6/3/0/1 |
| 2 | 1.000000 | 5 | 0.40 | 1.000000 | 1.000000 | 1.000000 | 6/2/0/0 |
| 3 | 0.833333 | 1 | 0.40 | 0.666667 | 1.000000 | 0.666667 | 4/2/0/2 |
| 4 | 0.916667 | 4 | 0.60 | 0.833333 | 1.000000 | 0.916667 | 5/2/0/1 |
| 5 | 0.833333 | 4 | 0.60 | 0.666667 | 1.000000 | 0.916667 | 4/2/0/2 |

Corrected HU300 CT projection result file:

```text
experiment_results/multimodal_simple_ct_max_axis1_hu300.0_rot1_align01_seed42.json
```

Projection setup:

```text
axis=1, rot90 k=1, HU threshold=300, clip max=1800
```

Mean best-threshold balanced accuracy:

```text
0.885714
```

Fold results:

| Fold | Balanced Accuracy | Epoch | Threshold | Sensitivity | Specificity | AUC | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.928571 | 4 | 0.55 | 0.857143 | 1.000000 | 0.904762 | 6/3/0/1 |
| 2 | 1.000000 | 5 | 0.40 | 1.000000 | 1.000000 | 1.000000 | 6/2/0/0 |
| 3 | 0.833333 | 1 | 0.40 | 0.666667 | 1.000000 | 0.666667 | 4/2/0/2 |
| 4 | 0.916667 | 4 | 0.70 | 0.833333 | 1.000000 | 0.916667 | 5/2/0/1 |
| 5 | 0.750000 | 2 | 0.55 | 1.000000 | 0.500000 | 0.500000 | 6/1/1/0 |

Corrected CT-mask projection result file:

```text
experiment_results/multimodal_simple_ct_mask_max_axis1_hunone_rot1_align01_seed42.json
```

Projection setup:

```text
axis=1, rot90 k=1, source=ct_mask
```

Mean best-threshold balanced accuracy:

```text
0.919048
```

Fold results:

| Fold | Balanced Accuracy | Epoch | Threshold | Sensitivity | Specificity | AUC | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.928571 | 4 | 0.55 | 0.857143 | 1.000000 | 0.952381 | 6/3/0/1 |
| 2 | 1.000000 | 5 | 0.40 | 1.000000 | 1.000000 | 1.000000 | 6/2/0/0 |
| 3 | 0.833333 | 1 | 0.40 | 0.666667 | 1.000000 | 0.666667 | 4/2/0/2 |
| 4 | 0.916667 | 4 | 0.70 | 0.833333 | 1.000000 | 0.916667 | 5/2/0/1 |
| 5 | 0.916667 | 4 | 0.85 | 0.833333 | 1.000000 | 0.916667 | 5/2/0/1 |

## Method Comparison

| Method | Mean Best Balanced Accuracy | Notes |
|---|---:|---|
| Frozen/scratch feature baseline, ROI 0.45 | 0.726190 | Weakest controlled image baseline |
| Frozen pretrained feature baseline, ROI 0.45 | 0.790476 | ImageNet features help |
| Frozen pretrained feature baseline, ROI 0.35 | 0.857143 | Stronger ROI-only feature baseline |
| Fine-tuned X-ray-only, seed 123 | 0.852381 | Split/seed sensitivity visible |
| Fine-tuned X-ray-only, seed 7 | 0.902381 | Strong but variable |
| Fine-tuned X-ray-only, seed 42 | 0.919048 | Best current single-seed result |
| Simple raw CT projection alignment | 0.885714 | Feasible, not better than X-ray-only |
| Simple CT-mask projection alignment | 0.902381 | Better than raw CT projection, still not above best X-ray-only |
| Corrected HU300 CT projection alignment | 0.885714 | Visually better CT projection, but score did not improve |
| Corrected CT-mask projection alignment | 0.919048 | Best CT branch; matches X-ray-only seed42 but does not exceed it |

## Evidence Artifacts

- CT projection cache: `data/ct_projections_simple`
- Sanity images: `debug_multimodal_alignment`
- Checkpoints: `checkpoints/multimodal_simple_fold_*_seed42.pth`
- Main JSON: `experiment_results/multimodal_simple_ct_max_axis0_align01_seed42.json`
- CT-mask JSON: `experiment_results/multimodal_simple_ct_mask_max_axis0_align01_seed42.json`
- Corrected HU300 CT JSON: `experiment_results/multimodal_simple_ct_max_axis1_hu300.0_rot1_align01_seed42.json`
- Corrected CT-mask JSON: `experiment_results/multimodal_simple_ct_mask_max_axis1_hunone_rot1_align01_seed42.json`
- HU300 CT review sheet: `evidence/ct_hu300_review/ct_hu300_axis1_rot1_512x256_contact_sheet.png`

## Interpretation

The simple multimodal experiment is feasible and runs end-to-end. CT-mask projection improved the multimodal result from 0.886 to 0.902 mean balanced accuracy with the first sideways projection setup, and the corrected mask projection improved it further to 0.919. This suggests that focusing the CT side on spine/bone structure and matching projection orientation both matter. However, it still does not clearly outperform the strongest X-ray-only pipeline, which also reached about 0.919 mean balanced accuracy on seed 42.

That does not mean multimodal failed. It means the simplest possible CT projection is probably too geometrically mismatched to reliably improve the X-ray classifier. The likely issues are:

- CT projections and X-rays are not spatially registered.
- CT field of view often includes anatomy beyond the X-ray field of view.
- Healthy patients have no CT, so CT can teach anomaly anatomy but cannot teach healthy anatomy.
- The CT branch uses raw projection features, not a CT model trained with a balanced CT objective.

## Memorization / Generalization Caution

These scores should be reported as internal patient-level cross-validation results, not clinical validation. The dataset is very small: 42 cleaned base patients, with only 11 healthy/empty patients. Augmented images are kept in the same split as their base patient, which reduces direct augmentation leakage, and validation uses original-only samples by default. Prompts were scrubbed for image experiments, which avoids diagnosis-text leakage.

However, the model can still overfit to dataset-specific visual patterns, acquisition style, preprocessing artifacts, or the limited healthy/anomaly distribution. The high training scores and seed-to-seed variation show that the model is capable of memorizing the training set. The more honest claim is:

```text
The method is promising on internal patient-level cross-validation, but requires more independent patients or an external test set before claiming robust generalization.
```

## Conclusion

This is a successful feasibility implementation, but not yet a decisive performance win over the best X-ray-only pipeline. CT-mask projection is the better multimodal variant so far and is more defensible than raw CT projection because it focuses alignment on segmented anatomy. The remaining likely improvement would be cropping/registration, but that needs more preprocessing effort and possibly manual inspection.
