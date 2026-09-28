import streamlit as st
from PIL import Image

from src.inference import predict_single_image


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide"
)


# =========================================================
# Custom Styling
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       Main Background
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(151, 190, 155, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(205, 180, 120, 0.10),
                transparent 25%
            ),
            #fbfcf9;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       Hide Default Streamlit Decoration
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* =====================================================
       Hero
       ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 3.2rem 2rem;
        margin-bottom: 2rem;

        border-radius: 28px;

        background:
            linear-gradient(
                135deg,
                #e7f2e8 0%,
                #f5f8f1 52%,
                #edf4e9 100%
            );

        border: 1px solid #d7e5d7;

        box-shadow:
            0 12px 35px rgba(45, 83, 53, 0.08);

        text-align: center;
    }

    .hero::before {
        content: "🌿";
        position: absolute;
        left: 28px;
        top: 18px;
        font-size: 3rem;
        opacity: 0.18;
        transform: rotate(-18deg);
    }

    .hero::after {
        content: "🍃";
        position: absolute;
        right: 30px;
        bottom: 15px;
        font-size: 3.5rem;
        opacity: 0.18;
        transform: rotate(15deg);
    }

    .hero-title {
        color: #234f32;
        font-size: 3rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        color: #66756a;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 680px;
        margin: auto;
    }


    /* =====================================================
       Upload Section
       ===================================================== */

    .upload-title {
        color: #315c3b;
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1.5px dashed #a9c5ad;
        border-radius: 22px;
        padding: 0.8rem;
        transition: all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #5d8b67;
        box-shadow: 0 8px 24px rgba(57, 102, 67, 0.08);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #f7faf6;
        border-radius: 16px;
    }


    /* =====================================================
       Image Container
       ===================================================== */

    .image-title {
        color: #315c3b;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }


    /* =====================================================
       Prediction Container
       ===================================================== */

    .prediction-title {
        color: #315c3b;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background: #ffffff;

        border: 1px solid #dce8dd;
        border-radius: 18px;

        padding: 1.15rem 1.35rem;
        margin-bottom: 0.85rem;

        box-shadow:
            0 5px 18px rgba(42, 75, 48, 0.055);

        transition: transform 0.2s ease,
                    box-shadow 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);

        box-shadow:
            0 9px 25px rgba(42, 75, 48, 0.09);
    }

    .result-label {
        color: #829087;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 0.3rem;
    }

    .result-value {
        color: #294f34;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.4;
    }


    /* =====================================================
       Health Status
       ===================================================== */

    .healthy {
        background: #edf7ef;
        border: 1px solid #cce2cf;
        color: #347247;

        border-radius: 16px;

        padding: 0.95rem 1.2rem;
        margin-bottom: 0.85rem;

        text-align: center;

        font-size: 1rem;
        font-weight: 700;
    }

    .diseased {
        background: #faf1e9;
        border: 1px solid #ead7c5;
        color: #96613e;

        border-radius: 16px;

        padding: 0.95rem 1.2rem;
        margin-bottom: 0.85rem;

        text-align: center;

        font-size: 1rem;
        font-weight: 700;
    }


    /* =====================================================
       Progress Bar
       ===================================================== */

    [data-testid="stProgressBar"] {
        background-color: #e7eee8;
        border-radius: 20px;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #6d9b75;
        border-radius: 20px;
    }


    /* =====================================================
       Buttons
       ===================================================== */

    .stButton > button {
        width: 100%;

        height: 2.8rem;

        border-radius: 13px;

        border: 1px solid #b8cfbb;

        background: #eef5ef;

        color: #315c3b;

        font-size: 0.95rem;
        font-weight: 650;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #6f9a77;

        background: #e1eee3;

        color: #234f32;

        transform: translateY(-1px);
    }


    /* =====================================================
       Alerts
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 15px;
    }


    /* =====================================================
       Spinner
       ===================================================== */

    .stSpinner > div {
        border-top-color: #5f8d68 !important;
    }


    /* =====================================================
       Footer
       ===================================================== */

    .footer {
        margin-top: 3rem;
        padding-top: 1.3rem;

        text-align: center;

        color: #8b978e;

        font-size: 0.78rem;

        border-top: 1px solid #e2e9e2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Hero Section
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🌿 Plant Disease Classifier
        </div>

        <div class="hero-subtitle">
            Upload a leaf image and discover the predicted
            plant disease with the help of deep learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Upload
# =========================================================

st.markdown(
    """
    <div class="upload-title">
        📤 Upload a Leaf Image
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        image_col, result_col = st.columns(
            [1.05, 1],
            gap="large"
        )


        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        with image_col:

            st.markdown(
                """
                <div class="image-title">
                    🖼️ Your Leaf
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                image,
                width="stretch"
            )


        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        with result_col:

            st.markdown(
                """
                <div class="prediction-title">
                    🔎 Prediction
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.spinner("Analyzing the leaf..."):

                result = predict_single_image(image)


            # Plant
            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        🌱 Plant
                    </div>

                    <div class="result-value">
                        {result.plant}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # Disease
            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        🦠 Disease
                    </div>

                    <div class="result-value">
                        {result.disease}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # Health Status
            if result.health_status == "Healthy":

                st.markdown(
                    """
                    <div class="healthy">
                        💚 Healthy
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="diseased">
                        ⚠️ Diseased
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Confidence
            confidence = float(result.confidence)

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        📊 Confidence
                    </div>

                    <div class="result-value">
                        {confidence:.2%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)


            # Low confidence warning
            if confidence < 0.70:

                st.warning(
                    "The model has low confidence in this prediction. "
                    "Try uploading a clear image containing a single leaf."
                )


    except Exception:

        st.error(
            "Unable to process the uploaded image. "
            "Please upload a valid JPG, JPEG, PNG, or WEBP image."
        )


else:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:2rem;
            margin-top:1rem;
            color:#87948b;
            background:#f7faf6;
            border-radius:18px;
            border:1px solid #e1e9e2;
        ">
            🍃 Upload a leaf image to begin
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Classifier · TensorFlow / Keras
        <br>
        Educational & Demonstration Project
    </div>
    """,
    unsafe_allow_html=True
)
