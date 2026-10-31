import os
from pathlib import Path

# Project root resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"

# File paths
AMIDHA_DATA_PATH = DATA_RAW_DIR / "amidha_database.json"

# Model Configurations
# We will use this clinically tuned model for better symptom matching
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Ensure directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)