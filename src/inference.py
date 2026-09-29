from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src import config


class ArtifactError(RuntimeError):
    """Raised when model or class mapping artifacts are invalid."""


class InvalidImageError(ValueError):
    """Raised when the uploaded image cannot be processed."""


class PredictionError(RuntimeError):
    """Raised when model prediction fails."""


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


def load_class_mapping(
    path: Union[str, Path] = config.CLASS_MAPPING_PATH,
) -> dict[int, str]:

    path = Path(path)

    if not path.exists():
        raise ArtifactError(
            f"Class mapping file not found: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        mapping = {int(k): str(v) for k, v in raw.items()}

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ArtifactError(
            f"Invalid class mapping file: {path}"
        ) from exc

    return mapping


def load_keras_model(
    path: Union[str, Path] = config.MODEL_PATH,
):
    path = Path(path)

    if not path.exists():
        raise ArtifactError(
            f"Model file not found: {path}"
        )

    try:
        from tensorflow.keras.models import load_model

        model = load_model(path)

        return model

    except Exception as exc:
        raise ArtifactError(
            f"Could not load Keras model: {path}"
        ) from exc


def validate_artifacts(
    model,
    class_mapping: dict[int, str],
) -> None:

    try:
        output_dim = int(model.output_shape[-1])
    except Exception as exc:
        raise ArtifactError(
            "Could not determine model output dimension."
        ) from exc

    if output_dim != len(class_mapping):
        raise ArtifactError(
            f"Model outputs {output_dim} classes, "
            f"but class mapping contains {len(class_mapping)} classes."
        )

    expected_indices = set(range(output_dim))
    actual_indices = set(class_mapping.keys())

    if expected_indices != actual_indices:
        raise ArtifactError(
            "Class mapping indices are invalid. "
            f"Expected: {sorted(expected_indices)} "
            f"but found: {sorted(actual_indices)}"
        )


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Match the training preprocessing exactly:

    RGB
    -> resize to 128x128
    -> float32
    -> divide by 255
    -> add batch dimension
    """

    try:
        # Convert to RGB
        rgb_image = image.convert("RGB")

        # Same size used during training
        resized = rgb_image.resize(config.IMG_SIZE)

        # Convert to numpy
        array = np.asarray(
            resized,
            dtype=np.float32,
        )

        # Normalize exactly like ImageDataGenerator(rescale=1./255)
        array = array / 255.0

        # Add batch dimension
        array = np.expand_dims(array, axis=0)

        return array

    except Exception as exc:
        raise InvalidImageError(
            "Could not preprocess the uploaded image."
        ) from exc


def parse_class_name(
    class_name: str,
) -> tuple[str, str]:

    if "___" not in class_name:
        return class_name, "Unknown"

    plant, disease = class_name.split(
        "___",
        1,
    )

    return plant, disease


def _humanize(label: str) -> str:

    return (
        label
        .replace("_", " ")
        .replace(",", "")
        .strip()
        .title()
    )


def predict_single_image(
    image,
    model,
    class_mapping,
) -> PredictionResult:

    batch = preprocess_image(image)

    try:
        predictions = model.predict(
            batch,
            verbose=0,
        )[0]

    except Exception as exc:
        raise PredictionError(
            "Model prediction failed."
        ) from exc

    # Make sure output is valid
    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    if predictions.ndim != 1:
        raise PredictionError(
            f"Unexpected model output shape: {predictions.shape}"
        )

    # Predicted class
    predicted_index = int(
        np.argmax(predictions)
    )

    # Confidence
    confidence = float(
        predictions[predicted_index]
    )

    if predicted_index not in class_mapping:
        raise ArtifactError(
            f"Predicted class index {predicted_index} "
            "does not exist in class mapping."
        )

    raw_class_name = class_mapping[
        predicted_index
    ]

    # Parse plant and disease
    plant, disease = parse_class_name(
        raw_class_name
    )

    # Healthy or diseased
    is_healthy = (
        disease.strip().lower()
        == config.HEALTHY_TOKEN.lower()
    )

    health_status = (
        "Healthy"
        if is_healthy
        else "Diseased"
    )

    disease_display = (
        "None (Healthy)"
        if is_healthy
        else _humanize(disease)
    )

    return PredictionResult(
        plant=_humanize(plant),
        disease=disease_display,
        health_status=health_status,
        confidence=confidence,
        raw_class_name=raw_class_name,
    )


def is_low_confidence(
    confidence: float,
) -> bool:

    return (
        confidence
        < config.LOW_CONFIDENCE_THRESHOLD
    )
