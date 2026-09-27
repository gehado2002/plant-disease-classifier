"""
Lightweight tests for the inference pipeline.

These do NOT require the original dataset. A handful of synthetic images
are generated on the fly to exercise preprocessing, guard logic, and the
end-to-end pipeline against the real model artifacts checked into model/.

Run with:
    python -m pytest tests/test_inference.py -v
or, without pytest:
    python tests/test_inference.py
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.inference import (
    ArtifactError,
    InvalidImageError,
    load_class_mapping,
    load_keras_model,
    parse_class_name,
    predict_single_image,
    preprocess_image,
    validate_artifacts,
)


def _random_image(mode="RGB", size=(300, 200)):
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if mode != "RGB":
        img = img.convert(mode)
    return img


def test_class_mapping_loads():
    mapping = load_class_mapping()
    assert len(mapping) == config.NUM_CLASSES
    assert mapping[0].startswith("Apple")


def test_parse_class_name():
    plant, disease = parse_class_name("Tomato___Late_blight")
    assert plant == "Tomato"
    assert disease == "Late_blight"


def test_parse_class_name_healthy():
    plant, disease = parse_class_name("Blueberry___healthy")
    assert plant == "Blueberry"
    assert disease.lower() == "healthy"


def test_preprocess_shape_and_range():
    img = _random_image()
    batch = preprocess_image(img)
    assert batch.shape == (1, config.IMG_SIZE[1], config.IMG_SIZE[0], 3)
    assert batch.min() >= 0.0 and batch.max() <= 1.0


def test_preprocess_handles_grayscale():
    img = Image.new("L", (100, 100), color=128)
    batch = preprocess_image(img)
    assert batch.shape == (1, config.IMG_SIZE[1], config.IMG_SIZE[0], 3)


def test_preprocess_handles_rgba():
    img = Image.new("RGBA", (100, 100), color=(10, 20, 30, 128))
    batch = preprocess_image(img)
    assert batch.shape == (1, config.IMG_SIZE[1], config.IMG_SIZE[0], 3)


def test_model_and_mapping_are_consistent():
    model = load_keras_model()
    mapping = load_class_mapping()
    validate_artifacts(model, mapping)  # should not raise


def test_mismatched_mapping_raises():
    model = load_keras_model()
    bad_mapping = {i: f"class_{i}" for i in range(5)}  # wrong count
    try:
        validate_artifacts(model, bad_mapping)
        raised = False
    except ArtifactError:
        raised = True
    assert raised


def test_end_to_end_prediction_on_random_image():
    model = load_keras_model()
    mapping = load_class_mapping()
    img = _random_image()
    result = predict_single_image(img, model, mapping)
    assert result.plant
    assert result.disease
    assert result.health_status in ("Healthy", "Diseased")
    assert 0.0 <= result.confidence <= 1.0


def test_end_to_end_prediction_on_unusual_color_mode():
    model = load_keras_model()
    mapping = load_class_mapping()
    img = Image.new("P", (150, 150))  # palette mode
    result = predict_single_image(img, model, mapping)
    assert 0.0 <= result.confidence <= 1.0


def test_missing_model_file_raises_artifact_error():
    try:
        load_keras_model(path="model/does_not_exist.keras")
        raised = False
    except ArtifactError:
        raised = True
    assert raised


def test_missing_class_mapping_raises_artifact_error():
    try:
        load_class_mapping(path="model/does_not_exist.json")
        raised = False
    except ArtifactError:
        raised = True
    assert raised


if __name__ == "__main__":
    # Allow running without pytest for a quick manual smoke test.
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {t.__name__} -> {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
