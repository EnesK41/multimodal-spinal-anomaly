# PNG vs NIfTI CV5 Comparison

Same settings: patient-level 5-fold CV, original-only validation, ResNet34 pretrained, ROI 0.45, mask labels, scrubbed prompts, 20 epochs, min save epoch 3, patience 6.

| Model input | Mean best balanced accuracy | Fold scores | Best thresholds | Mean balanced accuracy at 0.5 |
|---|---:|---|---|---:|
| NIfTI | 0.919048 | 0.929, 1.000, 0.833, 0.917, 0.917 | 0.5, 0.35, 0.7, 0.4, 0.8 | not recomputed here |
| PNG full export | 0.902381 | 0.929, 1.000, 0.833, 1.000, 0.750 | 0.5, 0.45, 0.9, 0.2, 0.2 | 0.835714 |
| PNG ROI export, no extra ROI | 0.916667 | 1.000, 1.000, 0.833, 1.000, 0.750 | 0.05, 0.4, 0.8, 0.2, 0.2 | 0.773810 |

Interpretation: PNG works and is close to NIfTI. The closest format match is the ROI-PNG export with no extra body ROI at training time, which almost reaches the old NIfTI score. Calibration is still less stable, so threshold choice matters. The previous 6-patient demo holdout was too small and optimistic for reporting as model performance.
