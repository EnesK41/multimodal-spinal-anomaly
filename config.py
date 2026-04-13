import os

class Config:
    # ---------------------------------------------------------
    # DIRECTORY PATHS (Absolute paths to prevent working-dir bugs)
    # ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
    
    # Output directories for training artifacts
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    # ---------------------------------------------------------
    # IMAGE & PREPROCESSING PARAMETERS
    # ---------------------------------------------------------
    TARGET_SIZE_2D = 256
    TARGET_SIZE_3D = (128, 256, 256) # Depth, Height, Width for CT/MR
    CROP_MARGIN = 20
    
    # ---------------------------------------------------------
    # ARCHITECTURE & MODEL SETTINGS
    # ---------------------------------------------------------
    EMBEDDING_DIM = 512
    XRAY_BACKBONE = "resnet34"
    CT_MR_BACKBONE = "r3d_18"
    
    # ---------------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # ---------------------------------------------------------
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    EPOCHS = 150 # Sufficient with early stopping
    EARLY_STOPPING_PATIENCE = 15 # Stop if no improvement after 15 epochs
    
    # Loss Weights
    LAMBDA_DICE = 1.0   # Weight for segmentation accuracy
    LAMBDA_GRAM = 1.0   # Weight for latent space alignment
    
    # ---------------------------------------------------------
    # CHECKPOINT & LOGGING SETTINGS
    # ---------------------------------------------------------
    SAVE_EVERY_N_EPOCH = 10
    BEST_MODEL_NAME = "spine_multimodal_best.pth"
    LATEST_MODEL_NAME = "spine_multimodal_latest.pth"
    
    # Validate and generate overlay images every N epochs
    VISUALIZE_EVERY_N_EPOCH = 5