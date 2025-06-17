import os
import logging
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data" / "void"

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "out/model.pt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "data/void/vocab.pkl")
META_PATH = os.getenv("META_PATH", "data/void/meta.pkl")

# Model generation settings
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 100))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 10000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "debug.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": LOG_LEVEL,
            "propagate": True
        },
    }
}
