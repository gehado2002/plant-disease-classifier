"""
Plant Disease Classifier — Streamlit application.

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src import config
from src.inference import (
    ArtifactError,
    InvalidImageError,
    PredictionError,
    is_low_confidence,
    load_class_mapping,
    load_keras_model,
    predict_single_image,
    validate_artifacts,
)

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Cached resource loading — model & class mapping are loaded once per
# session/process, not on every prediction or rerun.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model_and_classes():
    class_mapping = load_class_mapping()
    model = load_keras_model()
    validate_artifacts(model, class_mapping)
    return model, class_mapping


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌿 Plant Disease Classifier")
st.write(
    "Upload a photo of a plant leaf and this app will predict the plant "
    "species and, if present, the disease affecting it, using a "
    "convolutional neural network trained on the New Plant Diseases "
    "Dataset."
)

# Load model/class mapping once, up front, with a clear developer-facing
# error if the artifacts are missing or inconsistent (Phase 8).
try:
    model, class_mapping = get_model_and_classes()
except ArtifactError as exc:
    st.error(
        "⚠️ The application could not start because of a model/artifact "
        f"problem:\n\n**{exc}**\n\nThis is a configuration issue, not "
        "something you can fix by trying a different image."
    )
    st.stop()

st.divider()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=config.ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
    help=f"Supported formats: {', '.join(config.ALLOWED_EXTENSIONS).upper()}",
)

if uploaded_file is None:
    st.info("👆 Upload an image to get a prediction.")
    st.stop()

# --- Guard: file size -------------------------------------------------
size_mb = uploaded_file.size / (1024 * 1024)
if size_mb > config.MAX_UPLOAD_MB:
    st.error(
        f"This file is {size_mb:.1f} MB, which exceeds the "
        f"{config.MAX_UPLOAD_MB} MB limit. Please upload a smaller image."
    )
    st.stop()

# --- Guard: open & decode the image safely -----------------------------
try:
    image = Image.open(uploaded_file)
    image.verify()  # cheap integrity check
    # verify() invalidates the file handle for further reads, so reopen it
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image.load()
except UnidentifiedImageError:
    st.error(
        "This file doesn't look like a valid image. Please upload a "
        "JPG, JPEG, PNG, or WEBP file."
    )
    st.stop()
except Exception:
    st.error(
        "This image appears to be corrupted or unreadable. Please try "
        "a different file."
    )
    st.stop()

# --- Guard: unreasonably large pixel dimensions ------------------------
width, height = image.size
if width * height > config.MAX_IMAGE_PIXELS:
    st.error(
        f"This image is {width}x{height} pixels, which is too large to "
        "process safely. Please upload a smaller image."
    )
    st.stop()

st.image(image, caption="Uploaded image", use_container_width=True)

# --- Prediction ----------------------------------------------------------
with st.spinner("Analyzing image..."):
    try:
        result = predict_single_image(image, model, class_mapping)
    except InvalidImageError:
        st.error(
            "This image's format or color mode couldn't be processed. "
            "Please try a different image (standard JPG or PNG works best)."
        )
        st.stop()
    except PredictionError:
        st.error(
            "The model failed to analyze this image. Please try again, "
            "or try a different image."
        )
        st.stop()
    except ArtifactError as exc:
        st.error(f"⚠️ Internal configuration error: {exc}")
        st.stop()

st.divider()
st.subheader("Prediction")

col1, col2 = st.columns(2)
with col1:
    st.metric("Plant", result.plant)
    st.metric("Health Status", result.health_status)
with col2:
    st.metric("Disease", result.disease)
    st.metric("Confidence", f"{result.confidence:.2%}")

if is_low_confidence(result.confidence):
    st.warning(
        "⚠️ Low-confidence prediction — the model isn't very sure about "
        "this one. Consider uploading a clearer, well-lit, close-up photo "
        "of a single leaf."
    )

with st.expander("About this prediction"):
    st.write(
        "- Confidence is the model's raw softmax output for the predicted "
        "class. It reflects the model's *relative* certainty among the "
        "38 classes it knows, not a calibrated probability of being correct.\n"
        "- This model was trained on a fixed set of plant species and "
        "diseases from the New Plant Diseases Dataset. Plants or "
        "conditions outside that set will still receive a prediction from "
        "the closest matching class, which may be misleading.\n"
        "- This tool is not a substitute for professional agricultural or "
        "plant-pathology advice."
    )

st.divider()
st.caption(
    "Model validation performance (on the project's held-out validation "
    "set): Accuracy 94.44% · Macro F1 94.41% · Weighted F1 94.41%. "
    "Real-world accuracy may differ."
)
