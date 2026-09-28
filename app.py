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

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background: #F5F5F5;
    }

    .block-container {
        max-width: 900px;
        padding-top: 0;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =====================================================
       TOP HEADER
    ===================================================== */

    .top-header {
        background: #121212;

        margin-left: -3rem;
        margin-right: -3rem;

        padding: 30px 40px 32px;

        border-bottom: 4px solid #7BAE78;

        box-shadow:
            0 4px 18px rgba(0, 0, 0, 0.15);

        text-align: center;
    }


    /* =====================================================
       TITLE
    ===================================================== */

    .brand {
        display: inline-block;

        background: #7BAE78;

        color: #121212;

        font-size: 2.1rem;

        font-weight: 900;

        letter-spacing: -1px;

        padding: 5px 13px;

        border-radius: 4px;

        line-height: 1;
    }

    .title {
        margin-top: 16px;

        color: #FFFFFF;

        font-size: 2.35rem;

        font-weight: 800;

        letter-spacing: -0.8px;

        line-height: 1.2;
    }

    .subtitle {
        margin-top: 8px;

        color: #BDBDBD;

        font-size: 0.95rem;

        font-weight: 400;
    }


    /* =====================================================
       MAIN CONTENT
    ===================================================== */

    .content {
        padding-top: 32px;
    }


    /* =====================================================
       SECTION TITLE
    ===================================================== */

    .section-title {
        color: #222222;

        font-size: 1.25rem;

        font-weight: 800;

        margin-bottom: 12px;

        padding-left: 10px;

        border-left: 4px solid #7BAE78;
    }


    /* =====================================================
       UPLOAD
    ===================================================== */

    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;

        border: 2px dashed #BDBDBD !important;

        border-radius: 5px !important;

        padding: 25px !important;

        transition: 0.2s ease;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.06);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #7BAE78 !important;

        box-shadow:
            0 4px 14px rgba(0, 0, 0, 0.10);
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #555555 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #222222 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #7BAE78 !important;

        color: #121212 !important;

        border: none !important;

        border-radius: 4px !important;

        font-weight: 700 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #679564 !important;
    }


    /* =====================================================
       IMAGE CARD
    ===================================================== */

    .image-card {
        background: #FFFFFF;

        border-radius: 5px;

        padding: 7px;

        border: 1px solid #DDDDDD;

        box-shadow:
            0 2px 10px rgba(0, 0, 0, 0.08);
    }


    /* =====================================================
       RESULT CARDS
    ===================================================== */

    .result-card {
        background: #FFFFFF;

        border: 1px solid #DDDDDD;

        border-radius: 5px;

        padding: 20px;

        min-height: 125px;

        margin-bottom: 14px;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.06);

        transition: 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);

        box-shadow:
            0 5px 16px rgba(0, 0, 0, 0.10);
    }

    .result-label {
        color: #777777;

        font-size: 0.72rem;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 1px;

        margin-bottom: 9px;
    }

    .result-value {
        color: #222222;

        font-size: 1.12rem;

        font-weight: 750;

        line-height: 1.45;

        overflow-wrap: anywhere;
    }


    /* =====================================================
       HEALTH STATUS
    ===================================================== */

    .healthy {
        display: inline-block;

        background: #E5F1E4;

        color: #35663A;

        border-radius: 4px;

        padding: 5px 10px;

        font-size: 0.82rem;

        font-weight: 700;
    }

    .diseased {
        display: inline-block;

        background: #F4E4E0;

        color: #914E43;

        border-radius: 4px;

        padding: 5px 10px;

        font-size: 0.82rem;

        font-weight: 700;
    }


    /* =====================================================
       CONFIDENCE
    ===================================================== */

    .confidence-card {
        background: #FFFFFF;

        border: 1px solid #DDDDDD;

        border-radius: 5px;

        padding: 20px;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.06);
    }

    .confidence-header {
        display: flex;

        justify-content: space-between;

        align-items: center;

        margin-bottom: 10px;
    }

    .confidence-label {
        color: #555555;

        font-size: 0.85rem;

        font-weight: 600;
    }

    .confidence-number {
        color: #35663A;

        font-size: 1rem;

        font-weight: 800;
    }

    .progress-background {
        width: 100%;

        height: 8px;

        background: #E5E5E5;

        border-radius: 3px;

        overflow: hidden;
    }

    .progress-bar {
        height: 100%;

        background: #7BAE78;

        border-radius: 3px;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        text-align: center;

        margin-top: 35px;

        padding-top: 18px;

        border-top: 1px solid #DDDDDD;

        color: #999999;

        font-size: 0.72rem;
    }


    /* =====================================================
       MOBILE
    ===================================================== */

    @media (max-width: 640px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .top-header {
            margin-left: -1rem;
            margin-right: -1rem;

            padding: 25px 20px 28px;
        }

        .brand {
            font-size: 1.7rem;
        }

        .title {
            font-size: 1.9rem;
        }

        .subtitle {
            font-size: 0.85rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="top-header">

        <div class="brand">
            LEAF
        </div>

        <div class="title">
            Plant Disease Classifier
        </div>

        <div class="subtitle">
            Deep Learning Image Classification
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONTENT
# =========================================================

st.markdown(
    '<div class="content">',
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    """
    <div class="section-title">
        Upload a Leaf Image
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
        <div style="
            text-align:center;
            color:#888888;
            padding:18px;
            background:#FFFFFF;
            border:1px solid #DDDDDD;
            border-radius:5px;
            margin-top:10px;
        ">
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

        # ---------------------------------------------
        # LOAD IMAGE
        # ---------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ---------------------------------------------
        # IMAGE
        # ---------------------------------------------

        st.markdown(
            """
            <div class="section-title"
                 style="margin-top:30px;">
                Uploaded Leaf
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


        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        with st.spinner("Analyzing image..."):

            result = predict_single_image(
                image
            )


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


        # ---------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------

        if confidence <= 1:

            confidence_percent = (
                confidence * 100
            )

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
        # STATUS
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
        # RESULTS
        # ---------------------------------------------

        st.markdown(
            """
            <div class="section-title"
                 style="margin-top:30px;">
                Prediction Results
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ---------------------------------------------
        # FIRST ROW
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
        # SECOND ROW
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

                <div class="confidence-header">

                    <div class="confidence-label">
                        Prediction Confidence
                    </div>

                    <div class="confidence-number">
                        {confidence_percent:.2f}%
                    </div>

                </div>

                <div class="progress-background">

                    <div
                        class="progress-bar"
                        style="
                            width:
                            {confidence_percent:.2f}%;
                        ">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    except Exception:

        st.error(
            "Unable to analyze the uploaded image."
        )


# =========================================================
# CLOSE CONTENT
# =========================================================

st.markdown(
    '</div>',
    unsafe_allow_html=True,
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
