# Multimodal Spinal Anomaly Detection

Research prototype for detecting pediatric spinal anomalies from frontal X-ray images. The final pipeline focuses on binary X-ray classification, while CT-mask projections are used during training as an auxiliary multimodal signal.

## Project Summary

The project started as a segmentation problem, but held-out mask predictions were not reliable enough for a final clinical-style output. The work therefore moved to a controlled anomaly-classification pipeline:

- input: frontal spinal X-ray
- output: anomaly-positive vs healthy/empty probability
- backbone: ResNet34-based X-ray encoder/classifier
- multimodal experiment: CT-mask projection alignment during training
- evaluation: patient-level 5-fold cross-validation to reduce augmentation leakage

The current repository contains code, experiment summaries, report sources, and a local demo frontend. Patient data, model checkpoints, generated medical previews, and private presentation files are intentionally excluded from Git.

## Main Results

Internal validation results should be interpreted as feasibility evidence, not clinical validation.

| Experiment | Mean best balanced accuracy | Interpretation |
| --- | ---: | --- |
| X-ray-only ResNet34, seed 42 | 0.919 | Strong clean baseline under patient-level CV |
| Corrected CT-mask alignment | 0.919 | Selected CT-supported multimodal research result |
| Region-aware auxiliary model | 0.950 | Best single run, but region output was not reliable enough for localization |
| PNG-compatible demo model | 0.917 | Practical model format for live demonstration |

Additional region, anomaly-type, and combined auxiliary experiments were tested, but they are treated as exploratory evidence rather than final deployed outputs.

## Repository Layout

```text
.
├── dataset.py                         # NIfTI dataset loader and label handling
├── train.py                           # Main X-ray classifier training/evaluation script
├── train_multimodal_alignment.py      # CT/CT-mask projection alignment experiments
├── train_region_multitask.py          # Region-aware auxiliary experiments
├── train_auxiliary_multitask.py       # Region/type/CT-mask auxiliary variants
├── train_png_classifier.py            # PNG-compatible classifier training
├── models/                            # X-ray, CT/MR, segmentation, and classifier modules
├── demo_frontend/                     # Local browser demo
├── experiment_results/                # Lightweight experiment summaries and metrics
└── Spinal_Anomaly_Detection_Report/   # LaTeX report source
```

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch entries in `requirements.txt` use CUDA 12.8 wheels. If your machine uses CPU-only PyTorch or another CUDA version, install the matching PyTorch build first from the official PyTorch instructions, then install the remaining packages.

## Typical Commands

Train the main classifier:

```powershell
python train.py --help
```

Run CT-mask alignment experiments:

```powershell
python train_multimodal_alignment.py --help
```

Train/evaluate the PNG-compatible demo model:

```powershell
python train_png_classifier.py --help
python run_png_roi_cv5_experiment.py
```

Start the local demo:

```powershell
cd demo_frontend
.\start_demo.ps1
```

## Data and Privacy

Medical images and patient-derived files are not included in this repository. The expected local data directories are ignored by Git:

- `data/`
- `checkpoints/`
- generated QC previews and medical-image evidence folders

Do not commit raw DICOM, NIfTI, X-ray PNG/JPEG exports, checkpoints, or patient-identifiable files.

## Notes

- Splits must be created by base patient ID, not by augmented sample ID.
- Augmented versions of the same patient must stay in the same fold.
- Reported scores are internal cross-validation results and require external validation before any clinical claim.
