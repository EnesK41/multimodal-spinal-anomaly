"""
DATASET BUSINESS LOGIC & FEATURES TO IMPLEMENT:

1. Expected Directory Structure & Formats:
   - Root: data/ -> patient_001/ -> {xray.png, mask.png, ct.nii.gz, mr.nii.gz, ct_mask.nii.gz, mr_mask.nii.gz}
   - 2D Data: PNG format.
   - 3D Data: NIfTI (.nii.gz) format.

2. Data Consistency & Resolution Enforcement:
   - Enforce strict resolution consistency across all samples.
   - X-ray & 2D Masks: Pad/Crop to fixed [256, 256].
   - CT & MR Volumes: Resample/Pad to fixed depth and spatial resolution (e.g., [128, 256, 256]).
   
3. Quality Filtering (Optional):
   - Implement logic to skip/filter specific patient folders based on a 'quality_flag' in the CSV (e.g., ignoring patients with heavily corrupted scans or previous model failure outliers).

4. Base Modalities (Mandatory):
   - Always load the 2D X-ray image and text concept from the CSV.
   
5. Ground Truth Mask Handling (The "Healthy" Trick):
   - Check if an anomaly mask exists for the specific patient and bone.
   - IF EXISTS: Load and normalize the actual mask.
   - IF NOT EXISTS (Healthy Patient): Dynamically generate a zero-tensor mask to save disk space and teach the network to ignore healthy regions.
   
6. Missing Modalities Handling (CT & MR):
   - Check if the patient folder contains CT scans (set 'has_ct' flag).
     * If True: Load CT volume.
     * If False: Generate a dummy zero-tensor to keep dataloader shape consistent.
   - Repeat exact logic for MR scans (set 'has_mr' flag).
   
7. Output:
   - Return dictionary: {xray, mask, text, ct, mr, has_ct, has_mr}.
"""