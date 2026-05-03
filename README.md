# multimodal-spinal-anomaly
Multimodal Spinal Anomaly Segmentation
A deep learning architecture that performs anomaly segmentation on 2D spinal X-rays using cross-modal latent space alignment.

Extracts features using 2D and 3D ResNet encoders.

Aligns X-ray, CT/MR volumes, and clinical text concepts (via frozen CLIP) in a shared latent space using GRAM loss.

Generates accurate binary segmentation masks through a UNet-style decoder.