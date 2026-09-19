from pathlib import Path


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# Data Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SPLITS_DIR = DATA_DIR / "splits"


# ============================================================
# AUTSL Dataset
# ============================================================

AUTSL_DIR = RAW_DATA_DIR / "AUTSL"

TRAIN_CSV = RAW_DATA_DIR / "train.csv"

VAL_CSV = RAW_DATA_DIR / "val.csv"

TEST_CSV = RAW_DATA_DIR / "test.csv"


TRAIN_VIDEO_DIR = (
    AUTSL_DIR / "train"
)

VAL_VIDEO_DIR = (
    AUTSL_DIR / "val"
)

TEST_VIDEO_DIR = (
    AUTSL_DIR / "test"
)


# ============================================================
# Artifacts & Logs
# ============================================================

ARTIFACTS_DIR = (
    PROJECT_ROOT / "artifacts"
)

LOGS_DIR = (
    PROJECT_ROOT / "logs"
)


# ============================================================
# Prepared Metadata
# ============================================================

TRAIN_PROCESSED_CSV = (
    ARTIFACTS_DIR
    / "data"
    / "train_processed.csv"
)

VAL_PROCESSED_CSV = (
    ARTIFACTS_DIR
    / "data"
    / "val_processed.csv"
)

TEST_PROCESSED_CSV = (
    ARTIFACTS_DIR
    / "data"
    / "test_processed.csv"
)


# ============================================================
# Dataset Configuration
# ============================================================

NUM_CLASSES = 226

IMAGE_HEIGHT = 160

IMAGE_WIDTH = 160

NUM_FRAMES = 8


# ============================================================
# Development Sampling
# ============================================================

# 0 = use complete official dataset

DEV_SAMPLES_PER_CLASS = 0

DEV_VAL_SAMPLES_PER_CLASS = 0


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 4

EPOCHS = 12

GRAD_ACCUM_STEPS = 2

NUM_WORKERS = 4


# ============================================================
# Loss Configuration
# ============================================================

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.1


# ============================================================
# Gradient Configuration
# ============================================================

GRADIENT_CLIP_VALUE = 1.0


# ============================================================
# Early Stopping
# ============================================================

EARLY_STOPPING_PATIENCE = 8


# ============================================================
# Learning Rates
# ============================================================

LAYER3_LEARNING_RATE = 5e-6

LAYER4_LEARNING_RATE = 1e-5

FC_LEARNING_RATE = 1e-4


# ============================================================
# Scheduler
# ============================================================

SCHEDULER_T0 = 10

SCHEDULER_T_MULT = 2

SCHEDULER_ETA_MIN = 1e-6


# ============================================================
# Model Configuration
# ============================================================

DROPOUT_RATE = 0.5


# ============================================================
# Model Artifacts
# ============================================================

MODEL_DIR = (
    ARTIFACTS_DIR / "model"
)

MODEL_FILE = (
    MODEL_DIR / "sign_language_model.pth"
)


# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = (
    ARTIFACTS_DIR / "checkpoints"
)

CHECKPOINT_FILE = (
    CHECKPOINT_DIR / "best_model.pth"
)


# ============================================================
# Video Configuration
# ============================================================

VIDEO_EXTENSION = ".mp4"