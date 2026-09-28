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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ==================== APP ==================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(117, 153, 119, 0.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(184, 145, 103, 0.10),
                transparent 30%
            ),
            #F5F7F2;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* ==================== HIDE DEFAULT UI ==================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ==================== HERO ==================== */

    .hero {
        text-align: center;
        padding: 1rem 1rem 2rem;
    }

    .hero-icon {
        width: 74px;
        height: 74px;

        margin: 0 auto 1rem;

        display: flex;
        align-items: center;
        justify-content: center;

        background: linear-gradient(
            145deg,
            #E5F0E0,
            #D1E2CC
        );

        border: 1px solid #C5D8BF;
        border-radius: 22px;

        font-size: 2rem;

        box-shadow:
            0 12px 30px rgba(45, 82, 57, 0.10);
    }

    .hero-title {
        color: #173C2A;

        font-size: 2.7rem;
        font-weight: 800;

        letter-spacing: -1px;
        line-height: 1.2;

        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        max-width: 620px;

        margin: 0 auto;

        color: #6D7C72;

        font-size: 1rem;
        line-height: 1.7;
    }


    /* ==================== UPLOAD ==================== */

    .upload-title {
        color: #244B34;

        font-size: 1.05rem;
        font-weight: 700;

        margin: 0 0 0.7rem 0.2rem;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.2rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #FBFCF9 !important;

        border: 2px dashed #A9C0AA !important;

        border-radius: 22px !important;

        padding: 1.8rem !important;

        transition: all 0.25s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: #F7FAF5 !important;

        border-color: #5F8968 !important;

        box-shadow:
            0 12px 30px rgba(48, 89, 59, 0.08);
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #5E6F63 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #315F43 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #315F43 !important;

        color: white !important;

        border: none !important;
        border-radius: 10px !important;

        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #244B34 !important;
    }


    /* ==================== EMPTY STATE ==================== */

    .empty-state {
        margin-top: 1rem;

        padding: 1rem;

        text-align: center;

        background: rgba(255, 255, 255, 0.55);

        border: 1px solid #DCE5DA;

        border-radius: 16px;

        color: #718078;

        font-size: 0.92rem;
    }


    /* ==================== SECTION TITLE ==================== */

    .section-title {
        color: #244B34;

        font-size: 1.05rem;
        font-weight: 700;

        margin: 1.8rem 0 0.8rem;
    }


    /* ==================== IMAGE ==================== */

    .image-card {
        background: #FBFCF9;

        border: 1px solid #DCE5DA;

        border-radius: 22px;

        padding: 0.6rem;

        box-shadow:
            0 12px 30px rgba(35, 66, 44, 0.07);
    }


    /* ==================== RESULT CARDS ==================== */

    .result-card {
        min-height: 145px;

        background: #FBFCF9;

        border: 1px solid #DCE5DA;

        border-radius: 20px;

        padding: 1.25rem;

        margin-bottom: 1rem;

        box-shadow:
            0 10px 25px rgba(35, 66, 44, 0.055);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-3px);

        box-shadow:
            0 15px 32px rgba(35, 66, 44, 0.09);
    }

    .result-icon {
        font-size: 1.45rem;

        margin-bottom: 0.65rem;
    }

    .result-label {
        color: #78867D;

        font-size: 0.75rem;
        font-weight: 600;

        text-transform: uppercase;

        letter-spacing: 0.8px;

        margin-bottom: 0.35rem;
    }

    .result-value {
        color: #203D2C;

        font-size: 1.12rem;
        font-weight: 750;

        line-height: 1.4;

        overflow-wrap: anywhere;
    }


    /* ==================== STATUS ==================== */

    .status-healthy {
        display: inline-block;

        padding: 0.42rem 0.75rem;

        background: #E4F1E5;

        color: #32633E;

        border: 1px solid #C7DEC9;

        border-radius: 999px;

        font-size: 0.85rem;
        font-weight: 700;
    }

    .status-diseased {
        display: inline-block;

        padding: 0.42rem 0.75rem;

        background: #F6E7E2;

        color: #98594D;

        border: 1px solid #E9CCC4;

        border-radius: 999px;

        font-size: 0.85rem;
        font-weight: 700;
    }


    /* ==================== CONFIDENCE ==================== */

    .confidence-card {
        background: #FBFCF9;

        border: 1px solid #DCE5DA;

        border-radius: 20px;

        padding: 1.25rem;

        margin-top: 0.2rem;

        box-shadow:
            0 10px 25px rgba(35, 66, 44, 0.055);
    }

    .confidence-header {
        display: flex;

        justify-content: space-between;
        align-items: center;

        margin-bottom: 0.7rem;
    }

    .confidence-label {
        color: #66756B;

        font-size: 0.85rem;
        font-weight: 600;
    }

    .confidence-value {
        color: #315F43;

        font-size: 1rem;
        font-weight: 800;
    }

    .confidence-track {
        width: 100%;

        height: 9px;

        background: #E3E9E1;

        border-radius: 999px;

        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;

        background: linear-gradient(
            90deg,
            #7A9E7E,
            #315F43
        );

        border-radius: 999px;
    }


    /* ==================== FOOTER ==================== */

    .footer {
        text-align: center;

        margin-top: 2.8rem;
        padding-top: 1.3rem;

        border-top: 1px solid #DCE5DA;

        color: #819087;

        font-size: 0.78rem;

        line-height: 1.8;
    }

    .footer span {
        color: #A0AAA3;
    }


    /* ==================== MOBILE ==================== */

    @media (max-width: 640px) {

        .block-container {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            font-size: 2.15rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            padding: 1.2rem !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            🌿
        </div>

        <div class="hero-title">
            Plant Disease Classifier
        </div>

        <div class="hero-subtitle">
            Upload a leaf image and discover the predicted
            plant disease with the help of deep learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    """
    <div class="upload-title">
        📤 Upload a Leaf Image
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="empty-state">
            🍃 Upload a leaf image to begin
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# IMAGE + PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # -----------------------------------------------
        # Uploaded image
        # -----------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                🖼️ Uploaded Leaf
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="image-card">',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        # -----------------------------------------------
        # Prediction
        # -----------------------------------------------

        with st.spinner("Analyzing..."):
            result = predict_single_image(image)

        # -----------------------------------------------
        # Safe values
        # -----------------------------------------------

        plant = html.escape(str(result.plant))
        disease = html.escape(str(result.disease))
        health_status = html.escape(str(result.health_status))

        confidence = float(result.confidence)

        if confidence <= 1:
            confidence_percent = confidence * 100
        else:
            confidence_percent = confidence

        confidence_percent = max(
            0,
            min(100, confidence_percent)
        )

        # -----------------------------------------------
        # Health status style
        # -----------------------------------------------

        if "healthy" in health_status.lower():

            status_html = (
                f"""
                <span class="status-healthy">
                    🌱 {health_status}
                </span>
                """
            )

        else:

            status_html = (
                f"""
                <span class="status-diseased">
                    ⚠️ {health_status}
                </span>
                """
            )

        # -----------------------------------------------
        # Results
        # -----------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                🔍 Prediction Results
            </div>
            """,
            unsafe_allow_html=True,
        )

        # First row
        col1, col2 = st.columns(2, gap="medium")

        with col1:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        🌱
                    </div>

                    <div class="result-label">
                        Plant
                    </div>

                    <div class="result-value">
                        {plant}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        🦠
                    </div>

                    <div class="result-label">
                        Disease
                    </div>

                    <div class="result-value">
                        {disease}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # Second row
        col3, col4 = st.columns(2, gap="medium")

        with col3:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        💚
                    </div>

                    <div class="result-label">
                        Health Status
                    </div>

                    <div class="result-value">
                        {status_html}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        ✨
                    </div>

                    <div class="result-label">
                        Confidence
                    </div>

                    <div class="result-value">
                        {confidence_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------
        # Confidence bar
        # -----------------------------------------------

        st.markdown(
            f"""
            <div class="confidence-card">

                <div class="confidence-header">

                    <div class="confidence-label">
                        Prediction Confidence
                    </div>

                    <div class="confidence-value">
                        {confidence_percent:.2f}%
                    </div>

                </div>

                <div class="confidence-track">

                    <div
                        class="confidence-fill"
                        style="width: {confidence_percent:.2f}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception:

        st.error(
            "Unable to analyze this image. "
            "Please upload a valid plant leaf image."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Classifier · TensorFlow / Keras<br>
        <span>Educational & Demonstration Project</span>
    </div>
    """,
    unsafe_allow_html=True,
)
