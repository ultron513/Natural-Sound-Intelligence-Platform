from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Important Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Model Files
MODEL_PATH = OUTPUTS_DIR / "rf_advanced_model.pkl"

SCALER_PATH = OUTPUTS_DIR / "advanced_scaler.pkl"

ENCODER_PATH = OUTPUTS_DIR / "label_encoder.pkl"

# Feature Dataset
FEATURES_CSV = OUTPUTS_DIR / "advanced_audio_features.csv"