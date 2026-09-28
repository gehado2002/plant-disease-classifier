import streamlit as st
from PIL import Image

from src.inference import predict_single_image


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="🌿 Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
    }

    /* ---------- Main Header ---------- */

    .hero {
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        background: linear-gradient(
            135deg,
            #eaf7ee 0%,
            #f7fbf8 50%,
            #e4f4e9 100%
        );
        border: 1px solid #d5eadb;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #185c37;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #5f6f64;
        max-width: 700px;
        margin: auto;
        line-height: 1.7;
    }

    /* ---------- Section Titles ---------- */

    .section-title {
        font-size: 1.6rem;
        font-weight: 650;
        color: #185c37;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* ---------- Info Cards ---------- */

    .info-card {
        padding: 1.4rem;
        border-radius: 18px;
        background: #f8fbf9;
        border: 1px solid #dcebe0;
        text-align: center;
        min-height: 125px;
    }

    .info-icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }

    .info-title {
        font-size: 0.9rem;
        color: #718078;
    }

    .info-value {
        font-size: 1.25rem;
        font-weight: 650;
        color: #185c37;
        margin-top: 0.25rem;
    }

    /* ---------- Upload Area ---------- */

    [data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    /* ---------- Result Cards ---------- */

    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #dcebe0;
        box-shadow: 0 4px 15px rgba(24, 92, 55, 0.06);
        margin-bottom: 1rem;
    }

    .result-label {
        color: #718078;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .result-value {
        color: #185c37;
        font-size: 1.35rem;
        font-weight: 650;
    }

    /* ---------- Health Status ---------- */

    .healthy-status {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #edf8f0;
        border: 1px solid #cde8d3;
        color: #237342;
        font-weight: 600;
        text-align: center;
    }

    .diseased-status {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fff4ed;
        border: 1px solid #f1d6c2;
        color: #a0522d;
        font-weight: 600;
        text-align: center;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #8a968e;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2ebe5;
    }

    /* ---------- Streamlit Buttons ---------- */

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown("## 🌿 Plant AI")

    st.markdown(
        """
        **Plant Disease Classifier**

        Upload a plant leaf image and let the CNN model analyze it.
        """
    )

    st.markdown("---")

    st.markdown("### 📌 Supported")

    st.markdown(
        """
        - 🌱 14 plant species
        - 🦠 38 disease classes
        - 🖼️ JPG / JPEG / PNG / WEBP
        - 🧠 CNN — TensorFlow / Keras
        """
    )

    st.markdown("---")

    st.caption(
        "For educational and demonstration purposes."
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
            Upload a leaf image to identify the plant species,
            detect possible diseases, and view the model's confidence.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Project Information
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">🌱</div>
            <div class="info-title">Plant Species</div>
            <div class="info-value">14</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">🦠</div>
            <div class="info-title">Disease Classes</div>
            <div class="info-value">38</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">🧠</div>
            <div class="info-title">Model</div>
            <div class="info-value">CNN</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">📊</div>
            <div class="info-title">Validation Accuracy</div>
            <div class="info-value">94.44%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# Navigation
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Prediction"


st.markdown(
    '<div class="section-title">🧭 Navigate</div>',
    unsafe_allow_html=True
)

nav1, nav2, nav3 = st.columns(3)

with nav1:
    if st.button("🔍 Prediction", use_container_width=True):
        st.session_state.page = "Prediction"

with nav2:
    if st.button("📖 About Model", use_container_width=True):
        st.session_state.page = "About"

with nav3:
    if st.button("⚠️ Limitations", use_container_width=True):
        st.session_state.page = "Limitations"


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# Prediction Page
# =========================================================

if st.session_state.page == "Prediction":

    st.markdown(
        '<div class="section-title">📤 Upload Leaf Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image containing a single plant leaf."
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            image_col, result_col = st.columns(
                [1.05, 1],
                gap="large"
            )

            # -------------------------------------------------
            # Image
            # -------------------------------------------------

            with image_col:

                st.markdown(
                    '<div class="section-title">🖼️ Uploaded Image</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width="stretch"
                )


            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            with result_col:

                st.markdown(
                    '<div class="section-title">🔎 Prediction</div>',
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

                # Health
                if result.health_status == "Healthy":

                    st.markdown(
                        """
                        <div class="healthy-status">
                            💚 Healthy Plant
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="diseased-status">
                            ⚠️ Diseased Plant
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence
                confidence = float(result.confidence)

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            📊 Model Confidence
                        </div>

                        <div class="result-value">
                            {confidence:.2%}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(confidence)

                if confidence < 0.70:

                    st.warning(
                        "The model has low confidence in this prediction. "
                        "Try uploading a clearer leaf image."
                    )

        except Exception:

            st.error(
                "Unable to process the uploaded image. "
                "Please upload a valid JPG, JPEG, PNG, or WEBP image."
            )

    else:

        st.info(
            "👆 Upload a leaf image above to start the prediction."
        )


# =========================================================
# About Model Page
# =========================================================

elif st.session_state.page == "About":

    st.markdown(
        '<div class="section-title">🧠 About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### CNN Architecture

        The classifier uses a Convolutional Neural Network built
        with TensorFlow / Keras.

        ```text
        Input Image
             ↓
        Conv2D + ReLU
             ↓
        MaxPooling
             ↓
        Conv2D + ReLU
             ↓
        MaxPooling
             ↓
        Conv2D + ReLU
             ↓
        MaxPooling
             ↓
        Flatten
             ↓
        Dense(128)
             ↓
        Dropout(0.5)
             ↓
        Dense(38, Softmax)
        ```

        ### 📊 Performance

        | Metric | Score |
        |---|---:|
        | Validation Accuracy | **94.44%** |
        | Macro F1 | **94.41%** |
        | Weighted F1 | **94.41%** |

        The model receives **128 × 128 RGB images** and predicts
        one of the 38 plant/disease classes.
        """
    )


# =========================================================
# Limitations Page
# =========================================================

elif st.session_state.page == "Limitations":

    st.markdown(
        '<div class="section-title">⚠️ Limitations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Important Notes

        - 🌱 The model supports **14 plant species** only.
        - 🦠 It predicts **38 known classes** from the training dataset.
        - 📷 Real-world images may differ from the training images.
        - 🔍 Similar diseases may be difficult to distinguish.
        - 📊 Confidence is not a calibrated probability.
        - 🚫 The model does not detect unsupported plants or
          non-plant images.
        - 👨‍🌾 Predictions should not replace professional
          agricultural or plant pathology diagnosis.
        """
    )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">

        🌿 <b>Plant Disease Classifier</b>
        · TensorFlow / Keras

        <br><br>

        Educational & Demonstration Project

    </div>
    """,
    unsafe_allow_html=True
)
