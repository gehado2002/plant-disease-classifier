"""
Plant Disease Classifier — Streamlit application.

Run with:
    streamlit run app.py
"""

import base64
import html
from pathlib import Path

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
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
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

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #E7F5EC 0%, transparent 40%),
        radial-gradient(circle at 100% 20%, #F1F7E4 0%, transparent 35%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'%3E%3Cg fill='%233FA36B' fill-opacity='0.08'%3E%3Cpath d='M22 34c11-14 27-11 31 0-5 13-20 16-31 0z'/%3E%3Cpath d='M88 96c11-14 27-11 31 0-5 13-20 16-31 0z'/%3E%3Ccircle cx='104' cy='28' r='3.5'/%3E%3Ccircle cx='38' cy='104' r='3.5'/%3E%3Ccircle cx='70' cy='66' r='2'/%3E%3C/g%3E%3C/svg%3E"),
        #F6F9F7;

    background-attachment: fixed;
}

header[data-testid="stHeader"],
#MainMenu,
footer {
    display: none !important;
}

.block-container {
    max-width: 860px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 44px 28px 40px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #0E2A1B 0%,
        #1F5B3A 55%,
        #3FA36B 100%
    );
    box-shadow: 0 18px 40px rgba(14, 42, 27, 0.25);
}

.hero::before,
.hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.07);
}

.hero::before {
    width: 260px;
    height: 260px;
    top: -110px;
    right: -60px;
}

.hero::after {
    width: 180px;
    height: 180px;
    bottom: -90px;
    left: -40px;
}

.hero-badge {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #DFF6E7;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero-title {
    position: relative;
    margin-top: 18px;
    color: #FFFFFF;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.15;
}

.hero-title span {
    color: var(--green-300);
}

.hero-sub {
    position: relative;
    margin-top: 10px;
    color: #C4E6D0;
    font-size: 0.98rem;
}


/* ---------- HERO ARTWORK ---------- */

.hero-art {
    position: relative;
    max-width: 560px;
    margin: 26px auto 0;
    padding: 14px 18px 8px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(6px);
}

.hero-art svg,
.hero-art img {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 14px;
}

.hero-art img {
    max-height: 240px;
    object-fit: cover;
}


/* ---------- SECTION TITLE ---------- */

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 34px 0 14px;
    color: var(--ink);
    font-size: 1.15rem;
    font-weight: 800;
}

.section-title::before {
    content: "";
    width: 5px;
    height: 22px;
    border-radius: 4px;
    background: linear-gradient(
        var(--green-500),
        var(--green-300)
    );
}


/* ---------- UPLOADER ---------- */

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #B9D8C5 !important;
    border-radius: 18px !important;
    padding: 30px !important;
    transition: all 0.25s ease;
    box-shadow: 0 6px 20px rgba(31, 91, 58, 0.06);
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--green-500) !important;
    background: #FAFEFB !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(31, 91, 58, 0.12);
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--ink) !important;
    font-weight: 600;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(
        135deg,
        #3FA36B,
        #1F5B3A
    ) !important;

    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 8px 18px !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    filter: brightness(1.1);
}


/* ---------- EMPTY STATE ---------- */

.empty {
    margin-top: 16px;
    padding: 22px;
    text-align: center;
    color: var(--muted);
    font-size: 0.95rem;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
}

.empty b {
    color: var(--green-700);
}


/* ---------- IMAGE ---------- */

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 5px solid #FFFFFF;
    box-shadow: 0 10px 26px rgba(14, 42, 27, 0.16);
    max-height: 340px;
    object-fit: cover;
}

[data-testid="stImageCaption"] {
    text-align: center;
    color: var(--muted);
}


/* ---------- ALERTS & EXPANDER ---------- */

[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(23, 35, 28, 0.04);
}

/* Expander title */
[data-testid="stExpander"] summary {
    color: #17231C !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] summary span {
    color: #17231C !important;
}

/* Expander content */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    color: #17231C !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #4F5F56 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    margin-bottom: 14px !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
    color: #1F5B3A !important;
    font-weight: 750 !important;
}


/* ---------- RESULT CARDS ---------- */

.card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 12px;
    min-height: 0;
    box-shadow: 0 6px 18px rgba(23, 35, 28, 0.05);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 28px rgba(23, 35, 28, 0.10);
}

.card-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.card-icon {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: var(--green-100);
    font-size: 1.1rem;
}

.card-label {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.card-value {
    color: var(--ink);
    font-size: 1.25rem;
    font-weight: 750;
    line-height: 1.4;
    overflow-wrap: anywhere;
}


/* ---------- STATUS PILLS ---------- */

.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 700;
}

.pill::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

.pill.healthy {
    background: var(--green-100);
    color: var(--green-700);
}

.pill.diseased {
    background: var(--red-100);
    color: var(--red-700);
}


/* ---------- CONFIDENCE ---------- */

.conf {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(23, 35, 28, 0.05);
}

.conf-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;
}

.conf-label {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 600;
}

.conf-num {
    color: var(--green-700);
    font-size: 1.6rem;
    font-weight: 800;
}

.track {
    width: 100%;
    height: 12px;
    background: #E8EFEA;
    border-radius: 999px;
    overflow: hidden;
}

.fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #8FD3A8,
        #3FA36B,
        #1F5B3A
    );
    animation: grow 1s ease-out;
}

.fill.low {
    background: linear-gradient(
        90deg,
        #F3C77B,
        #E39A3B
    );
}

@keyframes grow {
    from {
        width: 0;
    }
}


/* ---------- FOOTER ---------- */

.foot {
    margin-top: 40px;
    padding-top: 18px;
    text-align: center;
    color: #8A9990;
    font-size: 0.78rem;
    line-height: 1.6;
    border-top: 1px solid var(--line);
}

.foot b {
    color: var(--green-700);
}


@media (max-width: 640px) {

    .hero {
        padding: 32px 18px 30px;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 1.8rem;
    }

    [data-testid="stImage"] img {
        max-width: 260px;
        margin: 0 auto;
        display: block;
    }
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
LEAF = "M0,-62 C42,-46 48,12 0,64 C-48,12 -42,-46 0,-62 Z"

VEINS = """
M0,-22 L22,-36
M0,2 L28,-14
M0,26 L22,12
M0,-22 L-22,-36
M0,2 L-28,-14
M0,26 L-22,12
"""

LEAF_SVG = f"""
<svg viewBox="0 0 600 196" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Healthy leaf, leaf spot disease and blight">

  <defs>

    <linearGradient id="gH" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#8BE0A8"/>
      <stop offset="1" stop-color="#2E8B57"/>
    </linearGradient>

    <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7CCB7A"/>
      <stop offset="1" stop-color="#4E9A45"/>
    </linearGradient>

    <linearGradient id="gB" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#B7C25A"/>
      <stop offset="1" stop-color="#8A6A2F"/>
    </linearGradient>

  </defs>


  <g transform="translate(110,88) rotate(-12) scale(1.15)">

    <path d="{LEAF}" fill="url(#gH)"/>

    <path
        d="M0,-58 L0,62"
        stroke="#1F5B3A"
        stroke-opacity=".55"
        stroke-width="2.2"
        fill="none"
    />

    <path
        d="{VEINS}"
        stroke="#1F5B3A"
        stroke-opacity=".35"
        stroke-width="1.6"
        fill="none"
    />

  </g>


  <text
      x="110"
      y="184"
      text-anchor="middle"
      fill="#DFF6E7"
      font-size="13"
      font-weight="700"
      letter-spacing="2"
      font-family="Inter, sans-serif">
      HEALTHY
  </text>


  <g transform="translate(300,88) rotate(6) scale(1.15)">

    <path d="{LEAF}" fill="url(#gS)"/>

    <path
        d="M0,-58 L0,62"
        stroke="#1F5B3A"
        stroke-opacity=".5"
        stroke-width="2.2"
        fill="none"
    />

    <path
        d="{VEINS}"
        stroke="#1F5B3A"
        stroke-opacity=".3"
        stroke-width="1.6"
        fill="none"
    />

    <g fill="#E6C453" fill-opacity=".55">
      <circle cx="-14" cy="-26" r="11"/>
      <circle cx="16" cy="-6" r="13"/>
      <circle cx="-12" cy="24" r="10"/>
      <circle cx="14" cy="36" r="8"/>
    </g>

    <g fill="#7A4A22">
      <circle cx="-14" cy="-26" r="5.5"/>
      <circle cx="16" cy="-6" r="6.5"/>
      <circle cx="-12" cy="24" r="5"/>
      <circle cx="14" cy="36" r="4"/>
      <circle cx="-2" cy="-44" r="3"/>
      <circle cx="-20" cy="2" r="3.5"/>
    </g>

  </g>


  <text
      x="300"
      y="184"
      text-anchor="middle"
      fill="#DFF6E7"
      font-size="13"
      font-weight="700"
      letter-spacing="2"
      font-family="Inter, sans-serif">
      LEAF SPOT
  </text>


  <g transform="translate(490,88) rotate(-4) scale(1.15)">

    <path d="{LEAF}" fill="url(#gB)"/>

    <path
        d="M0,-58 L0,62"
        stroke="#4A3316"
        stroke-opacity=".5"
        stroke-width="2.2"
        fill="none"
    />

    <path
        d="{VEINS}"
        stroke="#4A3316"
        stroke-opacity=".3"
        stroke-width="1.6"
        fill="none"
    />

    <path
        d="M-26,-4 C-30,-22 -8,-30 -2,-14 C6,-2 -10,12 -22,8 Z"
        fill="#4B2E17"
        fill-opacity=".85"
    />

    <path
        d="M10,18 C6,6 26,2 30,16 C32,30 16,38 8,30 Z"
        fill="#3B2412"
        fill-opacity=".85"
    />

    <path
        d="M-6,-50 C4,-56 18,-46 12,-36 C6,-30 -8,-38 -6,-50 Z"
        fill="#5A3A1E"
        fill-opacity=".8"
    />

  </g>


  <text
      x="490"
      y="184"
      text-anchor="middle"
      fill="#DFF6E7"
      font-size="13"
      font-weight="700"
      letter-spacing="2"
      font-family="Inter, sans-serif">
      BLIGHT
  </text>

</svg>
"""


@st.cache_data
def hero_art_html() -> str:
    """Use assets/hero.(jpg|jpeg|png|webp) if present, else the built-in SVG."""

    assets = Path(__file__).parent / "assets"

    for name in (
        "hero.jpg",
        "hero.jpeg",
        "hero.png",
        "hero.webp",
    ):

        path = assets / name

        if path.exists():

            ext = path.suffix.lstrip(".").lower()

            mime = (
                "image/jpeg"
                if ext in ("jpg", "jpeg")
                else f"image/{ext}"
            )

            b64 = base64.b64encode(
                path.read_bytes()
            ).decode()

            return (
                f'<img src="data:{mime};base64,{b64}" '
                f'alt="Plant disease leaf">'
            )

    return LEAF_SVG


render(
    f"""
    <div class="hero">

        <div class="hero-badge">
            🌿 LEAF
        </div>

        <div class="hero-title">
            Plant Disease <span>Classifier</span>
        </div>

        <div class="hero-sub">
            Deep Learning Image Classification
        </div>

        <div class="hero-art">
            {hero_art_html()}
        </div>

    </div>
    """
)


# ---------------------------------------------------------------------------
# Load model/class mapping once, up front, with a clear developer-facing
# error if the artifacts are missing or inconsistent.
# ---------------------------------------------------------------------------
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
render(
    '<div class="section-title">Upload a Leaf Image</div>'
)

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

    image.verify()

    # verify() invalidates the file handle for further reads,
    # so reopen it.
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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with st.spinner("Analyzing image..."):

    try:

        result = predict_single_image(
            image,
            model,
            class_mapping
        )

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

        st.error(
            f"⚠️ Internal configuration error: {exc}"
        )

        st.stop()


# ---- values ----
plant = html.escape(
    str(result.plant)
)

disease = html.escape(
    str(result.disease)
)

raw_status = str(
    result.health_status
)

health_status = html.escape(
    raw_status
)

confidence = float(
    result.confidence
)

confidence_percent = (
    confidence * 100
    if confidence <= 1
    else confidence
)

confidence_percent = max(
    0.0,
    min(100.0, confidence_percent)
)

low_conf = is_low_confidence(
    result.confidence
)

status_class = (
    "healthy"
    if "healthy" in raw_status.lower()
    else "diseased"
)

status_html = (
    f'<span class="pill {status_class}">'
    f'{health_status}'
    f'</span>'
)


# ---- results ----
render(
    '<div class="section-title">Prediction Results</div>'
)

img_col, info_col = st.columns(
    [1, 1.6],
    gap="large"
)

with img_col:

    st.image(
        image,
        caption="Uploaded leaf",
        use_container_width=True
    )

with info_col:

    render(
        result_card(
            "🌱",
            "Plant",
            plant
        )
    )

    render(
        result_card(
            "🦠",
            "Disease",
            disease
        )
    )

    render(
        result_card(
            "💚",
            "Health Status",
            status_html
        )
    )


render(
    f"""
    <div class="conf">

        <div class="conf-head">

            <div class="conf-label">
                Prediction Confidence
            </div>

            <div class="conf-num">
                {confidence_percent:.2f}%
            </div>

        </div>

        <div class="track">

            <div
                class="fill{' low' if low_conf else ''}"
                style="width: {confidence_percent:.2f}%;">
            </div>

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


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render(
    """
    <div class="foot">

        Model validation performance
        (project's held-out validation set):

        <b>Accuracy 94.44%</b> ·
        <b>Macro F1 94.41%</b> ·
        <b>Weighted F1 94.41%</b>

        <br>

        Real-world accuracy may differ.
        · Plant Disease Classifier
        · TensorFlow / Keras

    </div>
    """
)


# ---------------------------------------------------------------------------
# About (dropdown at the very end)
# ---------------------------------------------------------------------------
with st.expander("About this prediction"):

    st.markdown(
        """
        - **Confidence:** Confidence is the model's raw softmax output for
          the predicted class. It reflects the model's *relative* certainty
          among the 38 classes it knows, not a calibrated probability of
          being correct.

        - **Supported classes:** This model was trained on a fixed set of
          plant species and diseases from the New Plant Diseases Dataset.
          Plants or conditions outside that set will still receive a
          prediction from the closest matching class, which may be
          misleading.

        - **Important:** This tool is not a substitute for professional
          agricultural or plant-pathology advice.
        """
    )
