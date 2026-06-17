# Region-Aware Multitask Findings

## Experiment

Implemented a region-aware X-ray classifier without using text at inference.

Input:

```text
X-ray ROI only
```

Training targets:

```text
1. anomaly / healthy
2. auxiliary region labels for positive patients:
   cervical, thoracic, lumbar
```

Loss:

```text
BCE anomaly classification loss on all samples
+ 0.2 * region BCE loss on positive samples with region labels
```

Anomaly type prediction was intentionally not added because the available anomaly-type labels are too fragmented for this dataset size.

## Implementation

Script:

```text
train_region_multitask.py
```

Commands:

```bash
.venv/Scripts/python.exe train_region_multitask.py --epochs 5 --folds 5 --batch_size 8 --pretrained_encoder --num_workers 0 --region_loss_weight 0.2 --seed 42
.venv/Scripts/python.exe train_region_multitask.py --epochs 5 --folds 5 --batch_size 8 --pretrained_encoder --num_workers 0 --region_loss_weight 0.2 --seed 7
```

Result files:

```text
experiment_results/region_multitask_roi045_w02_seed42.json
experiment_results/region_multitask_roi045_w02_seed7.json
```

## Results

| Method | Seeds | Mean Best Balanced Accuracy |
|---|---:|---:|
| X-ray-only fine-tuned ROI classifier | 42, 7, 123 | 0.891270 |
| Region-aware multitask classifier | 42, 7 | 0.911905 |

Seed-level results:

| Method | Seed | Fold Scores | Mean Best Balanced Accuracy |
|---|---:|---|---:|
| X-ray-only | 42 | 0.929, 1.000, 0.833, 0.917, 0.917 | 0.919048 |
| X-ray-only | 7 | 0.929, 0.750, 1.000, 0.833, 1.000 | 0.902381 |
| X-ray-only | 123 | 0.929, 0.833, 1.000, 1.000, 0.500 | 0.852381 |
| Raw CT projection alignment | 42 | 0.929, 1.000, 0.833, 0.917, 0.750 | 0.885714 |
| CT-mask projection alignment | 42 | 0.929, 1.000, 0.833, 0.917, 0.833 | 0.902381 |
| Corrected CT-mask projection alignment | 42 | 0.929, 1.000, 0.833, 0.917, 0.917 | 0.919048 |
| Region-aware multitask | 42 | 1.000, 1.000, 0.833, 1.000, 0.917 | 0.950000 |
| Region-aware multitask | 7 | 0.786, 0.750, 0.917, 0.917, 1.000 | 0.873810 |

Region-head performance:

| Seed | Region Exact Match | Region Macro F1 |
|---:|---:|---:|
| 42 | 0.352381 | 0.214805 |
| 7 | 0.133333 | 0.135873 |

## Interpretation

Region-aware multitask training improved the anomaly classification average across the tested seeds compared with the X-ray-only baseline. However, the region head itself is weak. This means the region labels are probably acting more like an auxiliary regularizer than a reliable localization model.

The correct claim is:

```text
Adding coarse anatomical-region supervision improved internal patient-level CV classification,
but did not produce reliable region localization.
```

This is useful but should not be oversold. The dataset is still small, thresholds are selected on validation folds, and fold scores are sensitive to seed/split. The region-aware result is promising enough to keep, but it still needs external validation or more independent patients before making strong generalization claims.

## Overall Ranking So Far

| Rank | Method | Evidence Strength | Comment |
|---:|---|---|---|
| 1 | Region-aware multitask classifier | Best internal CV so far | Promising, but region predictions weak |
| 2 | X-ray-only pretrained ROI classifier | Strongest simpler baseline | Most stable and clean method |
| 3 | Corrected CT-mask projection alignment | Best CT branch so far | Matches X-ray-only seed42, not above it |
| 4 | Raw CT / old CT-mask alignment | Feasible but limited | Geometry/FOV mismatch likely hurts |
| 5 | Frozen feature baselines | Useful controls | Show ImageNet features and ROI help |

## Conclusion

The next reportable model should be the region-aware multitask classifier, with the X-ray-only model as the main baseline and corrected CT-mask alignment as the multimodal feasibility branch. The honest presentation framing is that region supervision is currently the most useful additional signal, while CT remains scientifically valuable but underused without cropping/registration or balanced healthy CT data.
