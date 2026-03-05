from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = ROOT_DIR / "data" / "housing.csv"

# Model storage
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "real_estate_model.joblib"

# Logs
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# ML settings
TEST_SIZE = 0.2
RANDOM_STATE = 42