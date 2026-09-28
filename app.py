import streamlit as st
from PIL import Image

from src.inference import predict_single_image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2.5rem;
    }

    /* Upload section */
    .upload-box {
        padding: 1rem;
        border-radius: 16px;
        background: #f8faf8;
        border: 1px solid #dce8df;
        margin-bottom: 1.5rem;
    }

    /* Prediction cards */
    .prediction-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #f8faf8;
        border: 1px solid #dce8df;
        text-align: center;
        height: 100%;
    }

    .card-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }

    .card-value {
        font-size: 1.25rem;
        font-weight: 600;
    }

    /* Confidence */
    .confidence {
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-radius: 16px;
        background: #f8faf8;
        border: 1px solid #dce8df;
        text-align: center;
    }

    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3rem;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌿 Plant Disease Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a leaf image to identify the plant and detect possible diseases.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Upload
# --------------------------------------------------

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        # Display image
        st.image(
            image,
            caption="Uploaded Leaf Image",
            width="stretch"
        )

        # Predict
        with st.spinner("Analyzing the leaf..."):
            result = predict_single_image(image)

        st.markdown("### 🔍 Prediction")

        # Prediction cards
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="card-label">🌱 Plant</div>
                    <div class="card-value">{result.plant}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="card-label">🦠 Disease</div>
                    <div class="card-value">{result.disease}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col3, col4 = st.columns(2)

        with col3:
            status_icon = "💚" if result.health_status == "Healthy" else "⚠️"

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="card-label">Health Status</div>
                    <div class="card-value">
                        {status_icon} {result.health_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="card-label">Confidence</div>
                    <div class="card-value">
                        {result.confidence:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence progress
        st.markdown(
            f"""
            <div class="confidence">
                <div class="card-label">Prediction Confidence</div>
                <div class="confidence-value">
                    {result.confidence:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(result.confidence))

        # Low confidence warning
        if result.confidence < 0.70:
            st.warning(
                "The model is not highly confident in this prediction. "
                "Try uploading a clearer image of a single leaf."
            )

    except Exception:
        st.error(
            "Unable to process this image. "
            "Please upload a valid plant leaf image."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Classifier · CNN · TensorFlow / Keras
        <br>
        For educational and demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True
)
