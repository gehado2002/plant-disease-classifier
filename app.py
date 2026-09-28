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
# STYLE
# =========================================================

st.markdown(
    """
    <style>

    /* ==============================
       GLOBAL
    ============================== */

    .stApp {
        background: #F4F7F2;
    }

    .block-container {
        max-width: 900px;
        padding: 3rem 2rem 2rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ==============================
       HERO
    ============================== */

    .hero {
        text-align: center;
        padding: 10px 0 35px;
    }

    .hero-icon {
        width: 68px;
        height: 68px;

        margin: 0 auto 18px;

        background: #DCE9D8;

        border: 1px solid #C8DAC4;
        border-radius: 18px;

        display: flex;
        align-items: center;
        justify-content: center;
    }

    .leaf {
        width: 25px;
        height: 38px;

        background: #356B4A;

        border-radius: 100% 0 100% 0;

        transform: rotate(-45deg);

        position: relative;
    }

    .leaf::after {
        content: "";

        position: absolute;

        width: 2px;
        height: 28px;

        background: #B9D0B7;

        left: 12px;
        top: 5px;

        transform: rotate(45deg);

        border-radius: 5px;
    }

    .hero-title {
        color: #183C29;

        font-size: 2.7rem;
        font-weight: 800;

        letter-spacing: -1.2px;

        line-height: 1.15;

        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #718078;

        font-size: 1rem;

        line-height: 1.7;

        max-width: 600px;

        margin: auto;
    }


    /* ==============================
       UPLOAD TITLE
    ============================== */

    .upload-title {
        color: #244B34;

        font-size: 1rem;

        font-weight: 700;

        margin-bottom: 10px;
    }


    /* ==============================
       FILE UPLOADER
    ============================== */

    [data-testid="stFileUploader"] {
        width: 100%;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;

        border: 1.5px dashed #A8BDA9 !important;

        border-radius: 18px !important;

        padding: 28px !important;

        min-height: 150px;

        box-shadow:
            0 8px 25px rgba(36, 75, 52, 0.05);

        transition: 0.25s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4D7858 !important;

        background: #FBFDFB !important;

        box-shadow:
            0 12px 30px rgba(36, 75, 52, 0.09);
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #718078 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #315F43 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #315F43 !important;

        color: #FFFFFF !important;

        border: none !important;

        border-radius: 9px !important;

        padding: 7px 18px !important;

        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #244B34 !important;
    }


    /* ==============================
       EMPTY STATE
    ============================== */

    .empty-state {
        text-align: center;

        margin-top: 12px;

        padding: 13px;

        background: #FFFFFF;

        border: 1px solid #E0E8DF;

        border-radius: 12px;

        color: #89958D;

        font-size: 0.85rem;
    }


    /* ==============================
       SECTION
    ============================== */

    .section-title {
        color: #244B34;

        font-size: 1rem;

        font-weight: 700;

        margin-top: 30px;

        margin-bottom: 12px;
    }


    /* ==============================
       IMAGE CARD
    ============================== */

    .image-wrapper {
        background: #FFFFFF;

        border: 1px solid #DFE7DE;

        border-radius: 18px;

        padding: 8px;

        box-shadow:
            0 8px 25px rgba(36, 75, 52, 0.06);
    }


    /* ==============================
       RESULT CARDS
    ============================== */

    .result-card {
        background: #FFFFFF;

        border: 1px solid #DFE7DE;

        border-radius: 17px;

        padding: 22px;

        min-height: 145px;

        margin-bottom: 15px;

        box-shadow:
            0 7px 22px rgba(36, 75, 52, 0.045);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-3px);

        box-shadow:
            0 12px 28px rgba(36, 75, 52, 0.09);
    }

    .result-label {
        color: #8A958E;

        font-size: 0.72rem;

        font-weight: 700;

        letter-spacing: 0.9px;

        text-transform: uppercase;

        margin-bottom: 9px;
    }

    .result-value {
        color: #203D2C;

        font-size: 1.08rem;

        font-weight: 700;

        line-height: 1.45;

        overflow-wrap: anywhere;
    }


    /* ==============================
       STATUS
    ============================== */

    .healthy {
        display: inline-block;

        background: #E6F1E7;

        color: #356542;

        border: 1px solid #C8DEC9;

        border-radius: 50px;

        padding: 6px 13px;

        font-size: 0.82rem;

        font-weight: 700;
    }

    .diseased {
        display: inline-block;

        background: #F5E8E4;

        color: #985B4F;

        border: 1px solid #E6CEC7;

        border-radius: 50px;

        padding: 6px 13px;

        font-size: 0.82rem;

        font-weight: 700;
    }


    /* ==============================
       CONFIDENCE
    ============================== */

    .confidence-card {
        background: #FFFFFF;

        border: 1px solid #DFE7DE;

        border-radius: 17px;

        padding: 22px;

        margin-top: 2px;

        box-shadow:
            0 7px 22px rgba(36, 75, 52, 0.045);
    }

    .confidence-top {
        display: flex;

        justify-content: space-between;

        align-items: center;

        margin-bottom: 12px;
    }

    .confidence-label {
        color: #718078;

        font-size: 0.82rem;

        font-weight: 600;
    }

    .confidence-number {
        color: #315F43;

        font-size: 1rem;

        font-weight: 800;
    }

    .progress-bg {
        width: 100%;

        height: 8px;

        background: #E5EBE4;

        border-radius: 20px;

        overflow: hidden;
    }

    .progress {
        height: 100%;

        background: #4F805C;

        border-radius: 20px;

        transition: width 0.5s ease;
    }


    /* ==============================
       SPINNER
    ============================== */

    [data-testid="stSpinner"] {
        color: #315F43 !important;
    }


    /* ==============================
       ERROR
    ============================== */

    .stAlert {
        border-radius: 12px !important;
    }


    /* ==============================
       FOOTER
    ============================== */

    .footer {
        text-align: center;

        color: #9AA49E;

        font-size: 0.72rem;

        margin-top: 40px;

        padding-top: 18px;

        border-top: 1px solid #DEE6DD;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 640px) {

        .block-container {
            padding: 2rem 1rem;
        }

        .hero-title {
            font-size: 2.1rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }

        .result-card {
            min-height: 120px;
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
            <div class="leaf"></div>
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
        Upload a Leaf Image
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    label_visibility="collapsed",
)


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="empty-state">
            Upload a leaf image to begin
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        # ---------------------------------------------
        # IMAGE
        # ---------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Uploaded Leaf
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="image-wrapper">',
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


        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        with st.spinner("Analyzing image..."):

            result = predict_single_image(image)


        # ---------------------------------------------
        # VALUES
        # ---------------------------------------------

        plant = html.escape(
            str(result.plant)
        )

        disease = html.escape(
            str(result.disease)
        )

        health_status = html.escape(
            str(result.health_status)
        )

        confidence = float(
            result.confidence
        )


        # Convert confidence to percentage
        if confidence <= 1:
            confidence_percent = confidence * 100
        else:
            confidence_percent = confidence

        confidence_percent = max(
            0,
            min(
                100,
                confidence_percent
            )
        )


        # ---------------------------------------------
        # HEALTH STATUS
        # ---------------------------------------------

        if "healthy" in health_status.lower():

            status = (
                f'<span class="healthy">'
                f'{health_status}'
                f'</span>'
            )

        else:

            status = (
                f'<span class="diseased">'
                f'{health_status}'
                f'</span>'
            )


        # ---------------------------------------------
        # RESULTS TITLE
        # ---------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Prediction Results
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ---------------------------------------------
        # ROW 1
        # ---------------------------------------------

        col1, col2 = st.columns(
            2,
            gap="medium"
        )

        with col1:

            st.markdown(
                f"""
                <div class="result-card">

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


        # ---------------------------------------------
        # ROW 2
        # ---------------------------------------------

        col3, col4 = st.columns(
            2,
            gap="medium"
        )

        with col3:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Health Status
                    </div>

                    <div class="result-value">
                        {status}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        with col4:

            st.markdown(
                f"""
                <div class="result-card">

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


        # ---------------------------------------------
        # CONFIDENCE BAR
        # ---------------------------------------------

        st.markdown(
            f"""
            <div class="confidence-card">

                <div class="confidence-top">

                    <div class="confidence-label">
                        Prediction Confidence
                    </div>

                    <div class="confidence-number">
                        {confidence_percent:.2f}%
                    </div>

                </div>

                <div class="progress-bg">

                    <div
                        class="progress"
                        style="width:{confidence_percent:.2f}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    except Exception as error:

        st.error(
            "Unable to analyze the uploaded image."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Plant Disease Classifier · TensorFlow / Keras
    </div>
    """,
    unsafe_allow_html=True,
)
