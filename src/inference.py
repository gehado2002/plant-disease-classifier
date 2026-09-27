"""
Core inference pipeline for the Plant Disease Classifier.

This module is intentionally decoupled from Streamlit so it can be:
  - unit tested directly (see tests/test_inference.py)
  - reused from a CLI, notebook, or a different UI framework

Preprocessing here is a direct, faithful port of the training notebook's
final inference cell (cell 35): RGB conversion -> resize to 128x128 ->
rescale to [0, 1] -> batch dimension -> model.predict -> argmax.

No Top-K logic is implemented on purpose — only the single best prediction
is ever returned, per project requirements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src import config


# ---------------------------------------------------------------------------
# Exceptions — distinct types so the UI layer can give useful, specific
# messages instead of a single generic "something went wrong".
# ---------------------------------------------------------------------------
class ArtifactError(RuntimeError):
    """Raised when the model file or class mapping is missing/inconsistent."""


class InvalidImageError(ValueError):
    """Raised when an uploaded file can't be read/decoded as an image."""


class PredictionError(RuntimeError):
    """Raised when the model fails to produce a prediction."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionResult:
    plant: str
    disease: str
    health_status: str
    confidence: float
    raw_class_name: str

    def as_dict(self) -> dict:
        return {
            "plant": self.plant,
            "disease": self.disease,
            "health_status": self.health_status,
            "confidence": self.confidence,
            "raw_class_name": self.raw_class_name,
        }


# ---------------------------------------------------------------------------
# Artifact loading & validation (Phase 8)
# ---------------------------------------------------------------------------
def load_class_mapping(path: Union[str, Path] = config.CLASS_MAPPING_PATH) -> dict[int, str]:
    """Load index -> class_name mapping from class_names.json."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"Class mapping file not found at: {path}")

    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"Class mapping file is invalid: {exc}") from exc


def load_keras_model(path: Union[str, Path] = config.MODEL_PATH):
    """Load the trained Keras model. Import is local so this module can be
    imported (e.g. by tests) without requiring TensorFlow at import time."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"Model file not found at: {path}")

    from tensorflow.keras.models import load_model  # local import

    try:
        return load_model(path)
    except Exception as exc:  # noqa: BLE001 - surface as a clear ArtifactError
        raise ArtifactError(f"Failed to load model from {path}: {exc}") from exc


def validate_artifacts(model, class_mapping: dict[int, str]) -> None:
    """Cross-check the model's output dimension against the class mapping.

    Raising here (rather than silently proceeding) is a deliberate choice:
    a mismatch means predictions would be mapped to the wrong class names
    without any visible error, which is worse than failing loudly.
    """
    output_dim = model.output_shape[-1]
    if output_dim != len(class_mapping):
        raise ArtifactError(
            f"Model/class-mapping mismatch: model outputs {output_dim} classes, "
            f"but class_names.json defines {len(class_mapping)} classes."
        )
    expected_indices = set(range(output_dim))
    actual_indices = set(class_mapping.keys())
    if expected_indices != actual_indices:
        raise ArtifactError(
            "Class mapping indices do not form a contiguous 0..N-1 range "
            f"matching the model's {output_dim} outputs."
        )


# ---------------------------------------------------------------------------
# Preprocessing (must mirror training exactly)
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into the exact array shape/scale the model expects.

    Steps (must match the training notebook):
      1. Convert to RGB (drops alpha / handles grayscale, CMYK, etc.)
      2. Resize to 128x128 (config.IMG_SIZE)
      3. Scale pixel values to [0, 1] (rescale=1./255)
      4. Add a batch dimension
    """
    try:
        rgb_image = image.convert("RGB")
        resized = rgb_image.resize(config.IMG_SIZE)
        array = np.asarray(resized, dtype=np.float32)
        array = array * config.RESCALE
        return np.expand_dims(array, axis=0)
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Could not preprocess image: {exc}") from exc


def parse_class_name(class_name: str) -> tuple[str, str]:
    """Split a raw class name like 'Apple___Apple_scab' into (plant, disease)."""
    if "___" not in class_name:
        # Defensive fallback — should not happen given the known dataset format.
        return class_name, "Unknown"
    plant, disease = class_name.split("___", 1)
    return plant, disease


def _humanize(label: str) -> str:
    """'Apple_scab' / 'Cedar_apple_rust' -> 'Apple Scab' / 'Cedar Apple Rust'."""
    return label.replace("_", " ").replace(",", "").strip().title()


# ---------------------------------------------------------------------------
# Single-image prediction (Phase 2 / Phase 6)
# ---------------------------------------------------------------------------
def predict_single_image(
    image: Image.Image,
    model,
    class_mapping: dict[int, str],
) -> PredictionResult:
    """Run the full pipeline on one PIL image and return the single best
    prediction (no Top-K)."""
    batch = preprocess_image(image)

    try:
        predictions = model.predict(batch, verbose=0)[0]
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Model failed to produce a prediction: {exc}") from exc

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    if predicted_index not in class_mapping:
        raise ArtifactError(
            f"Predicted index {predicted_index} has no entry in the class mapping."
        )

    raw_class_name = class_mapping[predicted_index]
    plant, disease = parse_class_name(raw_class_name)

    is_healthy = disease.strip().lower() == config.HEALTHY_TOKEN
    health_status = "Healthy" if is_healthy else "Diseased"
    disease_display = "None (Healthy)" if is_healthy else _humanize(disease)

    return PredictionResult(
        plant=_humanize(plant),
        disease=disease_display,
        health_status=health_status,
        confidence=confidence,
        raw_class_name=raw_class_name,
    )


def is_low_confidence(confidence: float) -> bool:
    return confidence < config.LOW_CONFIDENCE_THRESHOLD
