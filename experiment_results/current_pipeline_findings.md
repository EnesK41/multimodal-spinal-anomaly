# Current X-ray Pipeline Findings

## Best Setup

- Model: ImageNet-pretrained ResNet34 encoder + classifier head.
- Input: body-center ROI X-ray crop.
- ROI width ratio: 0.45.
- Labels: mask-derived binary labels.
- Prompt control: prompts scrubbed to neutral text.
- Split protocol: patient-level 5-fold cross-validation.
- Validation samples: original validation patients only, no validation augmentations.

## Main 5-Fold Result

Seed 42:

- Mean best-threshold balanced accuracy: 0.919
- Fold scores: 0.929, 1.000, 0.833, 0.917, 0.917

## Repeated-Seed Result

Repeated with three CV seeds:

- Seed 42: 0.919
- Seed 7: 0.902
- Seed 123: 0.852

Across-seed estimate:

- Mean: 0.891
- Standard deviation: 0.035
- Range: 0.852 to 0.919

Interpretation: the method is promising, but still sensitive to which patients land in each fold.

## Baseline Comparison

- Scratch frozen ResNet34 features, ROI 0.45: 0.538 default balanced accuracy.
- Pretrained frozen ResNet34 features, ROI 0.45: 0.710 default balanced accuracy.
- Pretrained frozen ResNet34 features, ROI 0.35: 0.774 default balanced accuracy.
- Fine-tuned pretrained ResNet34, ROI 0.45: 0.919 best-threshold balanced accuracy.

Interpretation: ImageNet pretraining helps substantially, and fine-tuning improves over frozen features.

## Grad-CAM/Error Evidence

Grad-CAM generated for the seed-42 ROI 0.45 CV checkpoints:

- Output directory: `evidence/gradcam_roi045_seed42`
- Error CSV: `evidence/gradcam_roi045_seed42/gradcam_error_rows.csv`
- Validation patients analyzed: 42
- Correct: 37
- Errors: 5

Errors:

- patient_026: label 1, prob 0.414, threshold 0.500, predicted 0
- patient_016: label 1, prob 0.655, threshold 0.700, predicted 0
- patient_021: label 1, prob 0.625, threshold 0.700, predicted 0
- patient_052: label 1, prob 0.318, threshold 0.400, predicted 0
- patient_006: label 1, prob 0.340, threshold 0.800, predicted 0

All seed-42 errors are false negatives under the fold-specific best thresholds.

## Presentation Claim

This is a feasible X-ray anomaly classification pipeline under severe data scarcity, not a clinically validated detector.

The defensible claim is:

> A pretrained ROI-based X-ray classifier showed promising patient-level cross-validation performance after shortcut checks and prompt scrubbing, but results remain limited by the very small number of unique patients and threshold instability.

## Recommended Next Direction

For the current X-ray pipeline, stop after Grad-CAM review and repeated-seed reporting unless more patient data becomes available.

The next genuinely different research branch is CT/DRR or vertebra-level geometry, but that is a larger pipeline rather than a small refinement.
