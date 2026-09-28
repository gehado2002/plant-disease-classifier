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

.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
}

.title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.card {
    padding: 1.2rem;
    border: 1px solid #dfe7e1;
    border-radius: 14px;
    text-align: center;
    background: #f8faf8;
    margin-bottom: 1rem;
}

.label {
    color: #6b7280;
    font-size: 0.9rem;
}

.value {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.8rem;
    margin-top: 2.5rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">🌿 Plant Disease Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a leaf image to identify the plant and detect possible diseases.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf",
            width="stretch"
        )

        with st.spinner("Analyzing the leaf..."):
            result = predict_single_image(image)

        st.markdown("### 🔍 Prediction Results")

        # Plant
        st.markdown(
            f"""
            <div class="card">
                <div class="label">🌱 Plant</div>
                <div class="value">{result.plant}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Disease
        st.markdown(
            f"""
            <div class="card">
                <div class="label">🦠 Disease</div>
                <div class="value">{result.disease}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Health Status
        if result.health_status == "Healthy":
            status = "💚 Healthy"
        else:
            status = "⚠️ Diseased"

        st.markdown(
            f"""
            <div class="card">
                <div class="label">Health Status</div>
                <div class="value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence
        confidence = float(result.confidence)

        st.markdown(
            f"""
            <div class="card">
                <div class="label">📊 Confidence</div>
                <div class="value">{confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(confidence)

        # Low-confidence warning
        if confidence < 0.70:
            st.warning(
                "The model has low confidence in this prediction. "
                "Try uploading a clear image containing a single leaf."
            )

    except Exception as e:
        st.error(
            "Unable to process the uploaded image. "
            "Please upload a valid JPG, JPEG, PNG, or WEBP image."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Classifier · TensorFlow / Keras
        <br>
        For educational and demonstration purposes.
    </div>
    """,
    unsafe_allow_html=True
)
