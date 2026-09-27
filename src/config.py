"""
Central configuration for the Plant Disease Classifier.

All constants here were taken directly from the original training notebook
(Untitled114__2_.ipynb) to guarantee inference-time behavior matches
training-time behavior. Do not change IMG_SIZE or the rescale factor
unless the model is retrained — doing so will silently break predictions.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root, so the app works regardless of the
# directory it's launched from, as long as it's launched from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "plant_disease_cnn.keras"
CLASS_MAPPING_PATH = MODEL_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Preprocessing — must exactly match cell 14 / cell 35 of the training
# notebook: ImageDataGenerator(rescale=1./255), target_size=(128, 128).
# ---------------------------------------------------------------------------
IMG_SIZE = (128, 128)          # (width, height) passed to PIL's .resize()
RESCALE = 1.0 / 255.0          # matches ImageDataGenerator(rescale=1./255)
NUM_CLASSES = 38               # Dense(38, activation="softmax") — final layer

# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 15             # soft guard against pathologically large files
MAX_IMAGE_PIXELS = 40_000_000  # guards against decompression-bomb-style images

# ---------------------------------------------------------------------------
# Confidence handling (Phase 6)
#
# This is a heuristic display threshold, NOT a scientifically calibrated
# probability. Softmax outputs are not guaranteed to reflect true likelihood
# of correctness. The model reports ~94% accuracy on its validation split;
# 50% is used only as a rough "the model itself is unsure" signal derived
# from that context, not a statistically derived operating point. If this
# app is used in a higher-stakes setting, that threshold should be set from
# a held-out calibration study instead.
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Health status label used in the training class names (e.g. "...___healthy")
# ---------------------------------------------------------------------------
HEALTHY_TOKEN = "healthy"
