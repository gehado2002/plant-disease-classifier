"""
Plant Disease Classifier — Streamlit application.

Run with:
    streamlit run app.py
"""

import html

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
# HTML helper
# Streamlit markdown treats lines indented by 4+ spaces as code blocks, which
# makes raw HTML show up as text. This strips indentation/blank lines first.
# ---------------------------------------------------------------------------
def render(markup: str) -> None:
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --green-900: #0E2A1B;
    --green-700: #1F5B3A;
    --green-500: #3FA36B;
    --green-300: #8FD3A8;
    --green-100: #E7F5EC;
    --red-700: #9B3B2E;
    --red-100: #FBE9E5;
    --ink: #17231C;
    --muted: #6B7A71;
    --line: #E3EAE5;
}

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #E7F5EC 0%, transparent 40%),
        radial-gradient(circle at 100% 20%, #F1F7E4 0%, transparent 35%),
        #F6F9F7;
}

header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }

.block-container { max-width: 860px; padding-top: 1.5rem; padding-bottom: 3rem; }

/* ---------- HERO ---------- */
.hero {
    position: relative; overflow: hidden; text-align: center;
    padding: 44px 28px 40px; border-radius: 24px;
    background: linear-gradient(135deg, #0E2A1B 0%, #1F5B3A 55%, #3FA36B 100%);
    box-shadow: 0 18px 40px rgba(14, 42, 27, 0.25);
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%;
    background: rgba(255, 255, 255, 0.07);
}
.hero::before { width: 260px; height: 260px; top: -110px; right: -60px; }
.hero::after  { width: 180px; height: 180px; bottom: -90px; left: -40px; }

.hero-badge {
    position: relative; display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #DFF6E7; font-size: 0.78rem; font-weight: 700; letter-spacing: 3px;
}
.hero-title {
    position: relative; margin-top: 18px; color: #FFFFFF;
    font-size: 2.5rem; font-weight: 800; letter-spacing: -1px; line-height: 1.15;
}
.hero-title span { color: var(--green-300); }
.hero-sub { position: relative; margin-top: 10px; color: #C4E6D0; font-size: 0.98rem; }

/* ---------- INTRO ---------- */
.intro {
    margin-top: 22px; padding: 18px 22px; color: #3B4A41;
    font-size: 0.95rem; line-height: 1.6; background: #FFFFFF;
    border: 1px solid var(--line); border-radius: 16px;
}

/* ---------- SECTION TITLE ---------- */
.section-title {
    display: flex; align-items: center; gap: 10px; margin: 34px 0 14px;
    color: var(--ink); font-size: 1.15rem; font-weight: 800;
}
.section-title::before {
    content: ""; width: 5px; height: 22px; border-radius: 4px;
    background: linear-gradient(var(--green-500), var(--green-300));
}

/* ---------- UPLOADER ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important; border: 2px dashed #B9D8C5 !important;
    border-radius: 18px !important; padding: 30px !important;
    transition: all 0.25s ease; box-shadow: 0 6px 20px rgba(31, 91, 58, 0.06);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--green-500) !important; background: #FAFEFB !important;
    transform: translateY(-2px); box-shadow: 0 12px 28px rgba(31, 91, 58, 0.12);
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--ink) !important; font-weight: 600;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #3FA36B, #1F5B3A) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    padding: 8px 18px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { filter: brightness(1.1); }

/* ---------- EMPTY STATE ---------- */
.empty {
    margin-top: 16px; padding: 22px; text-align: center; color: var(--muted);
    font-size: 0.95rem; background: #FFFFFF; border: 1px solid var(--line);
    border-radius: 16px;
}
.empty b { color: var(--green-700); }

/* ---------- IMAGE ---------- */
[data-testid="stImage"] img {
    border-radius: 20px; border: 6px solid #FFFFFF;
    box-shadow: 0 14px 34px rgba(14, 42, 27, 0.16);
}
[data-testid="stImageCaption"] { text-align: center; color: var(--muted); }

/* ---------- ALERTS & EXPANDER ---------- */
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stExpander"] {
    background: #FFFFFF; border: 1px solid var(--line) !important;
    border-radius: 14px; box-shadow: 0 4px 14px rgba(23, 35, 28, 0.04);
}

/* ---------- RESULT CARDS ---------- */
.card {
    background: #FFFFFF; border: 1px solid var(--line); border-radius: 18px;
    padding: 20px 22px; margin-bottom: 16px; min-height: 128px;
    box-shadow: 0 6px 18px rgba(23, 35, 28, 0.05); transition: all 0.25s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 14px 28px rgba(23, 35, 28, 0.10); }
.card-top { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.card-icon {
    width: 36px; height: 36px; display: flex; align-items: center;
    justify-content: center; border-radius: 10px; background: var(--green-100);
    font-size: 1.1rem;
}
.card-label {
    color: var(--muted); font-size: 0.72rem; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
}
.card-value {
    color: var(--ink); font-size: 1.25rem; font-weight: 750;
    line-height: 1.4; overflow-wrap: anywhere;
}

/* ---------- STATUS PILLS ---------- */
.pill {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 14px;
    border-radius: 999px; font-size: 0.9rem; font-weight: 700;
}
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.healthy  { background: var(--green-100); color: var(--green-700); }
.pill.diseased { background: var(--red-100);   color: var(--red-700); }

/* ---------- CONFIDENCE ---------- */
.conf {
    background: #FFFFFF; border: 1px solid var(--line); border-radius: 18px;
    padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(23, 35, 28, 0.05);
}
.conf-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
.conf-label { color: var(--muted); font-size: 0.9rem; font-weight: 600; }
.conf-num   { color: var(--green-700); font-size: 1.6rem; font-weight: 800; }
.track { width: 100%; height: 12px; background: #E8EFEA; border-radius: 999px; overflow: hidden; }
.fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #8FD3A8, #3FA36B, #1F5B3A);
    animation: grow 1s ease-out;
}
.fill.low { background: linear-gradient(90deg, #F3C77B, #E39A3B); }
@keyframes grow { from { width: 0; } }

/* ---------- FOOTER ---------- */
.foot {
    margin-top: 40px; padding-top: 18px; text-align: center; color: #8A9990;
    font-size: 0.78rem; line-height: 1.6; border-top: 1px solid var(--line);
}
.foot b { color: var(--green-700); }

@media (max-width: 640px) {
    .hero { padding: 32px 18px 30px; border-radius: 18px; }
    .hero-title { font-size: 1.8rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


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


def result_card(icon: str, label: str, value_html: str) -> str:
    return f"""
    <div class="card">
        <div class="card-top">
            <div class="card-icon">{icon}</div>
            <div class="card-label">{label}</div>
        </div>
        <div class="card-value">{value_html}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render(
    """
    <div class="hero">
        <div class="hero-badge">🌿 LEAF</div>
        <div class="hero-title">Plant Disease <span>Classifier</span></div>
        <div class="hero-sub">Deep Learning Image Classification</div>
    </div>
    """
)

render(
    """
    <div class="intro">
        Upload a photo of a plant leaf and this app will predict the plant
        species and, if present, the disease affecting it, using a
        convolutional neural network trained on the New Plant Diseases Dataset.
    </div>
    """
)

# Load model/class mapping once, up front, with a clear developer-facing
# error if the artifacts are missing or inconsistent.
try:
    model, class_mapping = get_model_and_classes()
except ArtifactError as exc:
    st.error(
        "⚠️ The application could not start because of a model/artifact "
        f"problem:\n\n**{exc}**\n\nThis is a configuration issue, not "
        "something you can fix by trying a different image."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
render('<div class="section-title">Upload a Leaf Image</div>')

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=config.ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
    label_visibility="collapsed",
    help=f"Supported formats: {', '.join(config.ALLOWED_EXTENSIONS).upper()}",
)

if uploaded_file is None:
    render(
        """
        <div class="empty">
            📷 Upload a <b>leaf image</b> to get a prediction
        </div>
        """
    )
    st.stop()

# --- Guard: file size -------------------------------------------------------
size_mb = uploaded_file.size / (1024 * 1024)
if size_mb > config.MAX_UPLOAD_MB:
    st.error(
        f"This file is {size_mb:.1f} MB, which exceeds the "
        f"{config.MAX_UPLOAD_MB} MB limit. Please upload a smaller image."
    )
    st.stop()

# --- Guard: open & decode the image safely ---------------------------------
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

# --- Guard: unreasonably large pixel dimensions -----------------------------
width, height = image.size
if width * height > config.MAX_IMAGE_PIXELS:
    st.error(
        f"This image is {width}x{height} pixels, which is too large to "
        "process safely. Please upload a smaller image."
    )
    st.stop()

render('<div class="section-title">Uploaded Leaf</div>')
st.image(image, caption="Uploaded image", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
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

# ---- values ----
plant = html.escape(str(result.plant))
disease = html.escape(str(result.disease))
raw_status = str(result.health_status)
health_status = html.escape(raw_status)

confidence = float(result.confidence)
confidence_percent = confidence * 100 if confidence <= 1 else confidence
confidence_percent = max(0.0, min(100.0, confidence_percent))

low_conf = is_low_confidence(result.confidence)

status_class = "healthy" if "healthy" in raw_status.lower() else "diseased"
status_html = f'<span class="pill {status_class}">{health_status}</span>'

# ---- results ----
render('<div class="section-title">Prediction Results</div>')

col1, col2 = st.columns(2, gap="medium")
with col1:
    render(result_card("🌱", "Plant", plant))
with col2:
    render(result_card("🦠", "Disease", disease))

col3, col4 = st.columns(2, gap="medium")
with col3:
    render(result_card("💚", "Health Status", status_html))
with col4:
    render(result_card("🎯", "Confidence", f"{confidence_percent:.2f}%"))

render(
    f"""
    <div class="conf">
        <div class="conf-head">
            <div class="conf-label">Prediction Confidence</div>
            <div class="conf-num">{confidence_percent:.2f}%</div>
        </div>
        <div class="track">
            <div class="fill{' low' if low_conf else ''}"
                 style="width: {confidence_percent:.2f}%;"></div>
        </div>
    </div>
    """
)

if low_conf:
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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render(
    """
    <div class="foot">
        Model validation performance (project's held-out validation set):
        <b>Accuracy 94.44%</b> · <b>Macro F1 94.41%</b> · <b>Weighted F1 94.41%</b><br>
        Real-world accuracy may differ. · Plant Disease Classifier · TensorFlow / Keras
    </div>
    """
)
