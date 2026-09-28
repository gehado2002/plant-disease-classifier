# 🌿 Plant Disease Classifier

<p align="center">
  <img src="https://i.ytimg.com/vi/iXpBFxEZktU/maxresdefault.jpg" alt="Plant Disease Classifier" width="100%">
</p>

## 1. Project Overview

A Streamlit application that predicts a plant's **species, health status, and disease** from a single uploaded leaf image.

The app uses a pre-trained CNN model and provides predictions through a simple web interface.

> **Note:** This repository serves the trained model; it does not retrain or modify it.

---

## 2. Model

The model was trained on the **New Plant Diseases Dataset**.

* 🌱 **14 plant species**
* 🦠 **38 plant/disease classes**
* 🧠 **CNN — TensorFlow / Keras**
* 🖼️ Input size: **128 × 128 RGB**
* 📦 Model format: `.keras`

### Architecture

```text
Conv2D → MaxPooling2D
Conv2D → MaxPooling2D
Conv2D → MaxPooling2D
        ↓
      Flatten
        ↓
   Dense(128, ReLU)
        ↓
    Dropout(0.5)
        ↓
   Dense(38, Softmax)
```

---

## 3. Features

* 📤 Upload JPG, JPEG, PNG, or WEBP images
* 🌱 Identify the plant species
* 🦠 Predict the disease
* 💚 Detect Healthy vs. Diseased status
* 📊 Display prediction confidence
* ⚠️ Warn when confidence is low
* 🛡️ Handle invalid or corrupted uploads

---

## 4. Project Structure

```text
plant-disease-classifier/
│
├── app.py                     # Streamlit application
├── src/
│   ├── config.py              # Configuration and paths
│   └── inference.py           # Image preprocessing and prediction
├── model/
│   ├── plant_disease_cnn.keras
│   └── class_names.json
├── tests/
│   └── test_inference.py      # Lightweight tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Installation

From the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Compatibility

The model was saved with **Keras 3.13.2**, so the environment requires **TensorFlow ≥ 2.20**.

---

## 6. Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 7. How to Use

Upload a leaf image through the Streamlit interface. 

The application displays the **Plant**, **Disease**, **Health Status**, and **Confidence** of the prediction.

---

## 8. Model Performance

The model achieved the following results on the project's held-out validation dataset:

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **94.44%** |
| Macro F1    | **94.41%** |
| Weighted F1 | **94.41%** |

These results are based on the validation data used in the original project and **do not guarantee the same performance on real-world images**.

---

## 9. Inference Pipeline

The prediction pipeline in `src/inference.py` follows the same preprocessing used during training:

```text
Uploaded Image
      ↓
Convert to RGB
      ↓
Resize to 128 × 128
      ↓
Normalize pixels to [0, 1]
      ↓
Add batch dimension
      ↓
CNN Prediction
      ↓
Select highest-probability class
      ↓
Map class index → class name
      ↓
Plant + Disease + Health Status
```

The application returns **one best prediction** rather than Top-K predictions.

---

## 10. Limitations

* The model supports only the **14 plant species and 38 classes** included in the training dataset.
* Similar-looking diseases may be confused.
* Real-world photos can be more difficult because of lighting, backgrounds, blur, or multiple leaves.
* The confidence score is **not a calibrated probability** of being correct.
* The model does not currently detect unsupported plants or non-plant images. It will still select one of its 38 known classes.
* This application is **not a substitute for professional agricultural or plant-pathology diagnosis**.

---

## 11. Project Notes

The original notebook contained some duplicated cells and Colab-specific paths.

For deployment, the project:

* Consolidates the inference logic into `src/inference.py`
* Uses project-relative paths instead of `/content/...`
* Uses a single best prediction instead of the notebook's earlier Top-K implementation
* Keeps the trained model unchanged

---

## 12. Future Improvements

* 🌿 Test performance on real-world leaf images
* 📊 Calibrate confidence scores
* 🔍 Add out-of-distribution detection
* 🧠 Improve the model if real-world testing reveals weaknesses
* ☁️ Deploy using Streamlit Community Cloud or another hosting platform

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**. Predictions may be incorrect, especially for images outside the training dataset.
