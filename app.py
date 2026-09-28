import html

import streamlit as st
from PIL import Image

from src.inference import predict_single_image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)


# =========================================================
# HELPER
# ---------------------------------------------------------
# Streamlit's markdown treats any line indented by 4+ spaces
# as a code block, which makes raw HTML show up as text.
# This helper strips indentation and blank lines so the HTML
# is always rendered.
# =========================================================

def render(markup: str) -> None:
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# =========================================================
# CSS
# =========================================================

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
        #F6F9F7;
}

header[data-testid="stHeader"], #MainMenu, footer {
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
    background: linear-gradient(135deg, #0E2A1B 0%, #1F5B3A 55%, #3FA36B 100%);
    box-shadow: 0 18px 40px rgba(14, 42, 27, 0.25);
}
.hero::before, .hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.07);
}
.hero::before { width: 260px; height: 260px; top: -110px; right: -60px; }
.hero::after  { width: 180px; height: 180px; bottom: -90px; left: -40px; }

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
    backdrop-filter: blur(6px);
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
.hero-title span { color: var(--green-300); }
.hero-sub {
    position: relative;
    margin-top: 10px;
    color: #C4E6D0;
    font-size: 0.98rem;
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
    background: linear-gradient(var(--green-500), var(--green-300));
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
    background: linear-gradient(135deg, #3FA36B, #1F5B3A) !important;
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
.empty b { color: var(--green-700); }

/* ---------- IMAGE ---------- */
[data-testid="stImage"] img {
    border-radius: 20px;
    border: 6px solid #FFFFFF;
    box-shadow: 0 14px 34px rgba(14, 42, 27, 0.16);
}

/* ---------- RESULT CARDS ---------- */
.card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 16px;
    min-height: 128px;
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
.pill.healthy  { background: var(--green-100); color: var(--green-700); }
.pill.diseased { background: var(--red-100);   color: var(--red-700); }

/* ---------- CONFIDENCE ---------- */
.conf {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 6px 18px rgba(23, 35, 28, 0.05);
}
.conf-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;
}
.conf-label { color: var(--muted); font-size: 0.9rem; font-weight: 600; }
.conf-num   { color: var(--green-700); font-size: 1.6rem; font-weight: 800; }
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
    background: linear-gradient(90deg, #8FD3A8, #3FA36B, #1F5B3A);
    animation: grow 1s ease-out;
}
@keyframes grow { from { width: 0; } }

/* ---------- FOOTER ---------- */
.foot {
    margin-top: 44px;
    padding-top: 18px;
    text-align: center;
    color: #9AA8A0;
    font-size: 0.78rem;
    border-top: 1px solid var(--line);
}

@media (max-width: 640px) {
    .hero { padding: 32px 18px 30px; border-radius: 18px; }
    .hero-title { font-size: 1.8rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

render(
    """
    <div class="hero">
        <div class="hero-badge">🌿 LEAF</div>
        <div class="hero-title">Plant Disease <span>Classifier</span></div>
        <div class="hero-sub">Deep Learning Image Classification</div>
    </div>
    """
)


# =========================================================
# UPLOAD
# =========================================================

render('<div class="section-title">Upload a Leaf Image</div>')

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:
    render(
        """
        <div class="empty">
            📷 Upload a <b>leaf image</b> to begin the analysis
        </div>
        """
    )


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        render('<div class="section-title">Uploaded Leaf</div>')
        st.image(image, use_container_width=True)

        with st.spinner("Analyzing image..."):
            result = predict_single_image(image)

        # ---- values ----
        plant = html.escape(str(result.plant))
        disease = html.escape(str(result.disease))
        raw_status = str(result.health_status)
        health_status = html.escape(raw_status)

        confidence = float(result.confidence)
        confidence_percent = confidence * 100 if confidence <= 1 else confidence
        confidence_percent = max(0, min(100, confidence_percent))

        status_class = "healthy" if "healthy" in raw_status.lower() else "diseased"
        status_html = f'<span class="pill {status_class}">{health_status}</span>'

        # ---- results ----
        render('<div class="section-title">Prediction Results</div>')

        def card(icon: str, label: str, value_html: str) -> str:
            return f"""
            <div class="card">
                <div class="card-top">
                    <div class="card-icon">{icon}</div>
                    <div class="card-label">{label}</div>
                </div>
                <div class="card-value">{value_html}</div>
            </div>
            """

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            render(card("🌱", "Plant", plant))
        with col2:
            render(card("🦠", "Disease", disease))

        col3, col4 = st.columns(2, gap="medium")
        with col3:
            render(card("💚", "Health Status", status_html))
        with col4:
            render(card("🎯", "Confidence", f"{confidence_percent:.2f}%"))

        # ---- confidence bar ----
        render(
            f"""
            <div class="conf">
                <div class="conf-head">
                    <div class="conf-label">Prediction Confidence</div>
                    <div class="conf-num">{confidence_percent:.2f}%</div>
                </div>
                <div class="track">
                    <div class="fill" style="width: {confidence_percent:.2f}%;"></div>
                </div>
            </div>
            """
        )

    except Exception:
        st.error("Unable to analyze the uploaded image.")


# =========================================================
# FOOTER
# =========================================================

render(
    """
    <div class="foot">
        Plant Disease Classifier · TensorFlow / Keras
    </div>
    """
)
