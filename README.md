# 🌿 Plant Disease Classifier

<p align="center">
  <img src="https://i.ytimg.com/vi/iXpBFxEZktU/maxresdefault.jpg" alt="Plant Disease Classifier" width="100%">
</p>

<details>
<summary>1. 📋 Project Overview</summary>

A Streamlit application that predicts a plant's **species, health status, and disease** from a single uploaded leaf image.

The app uses a trained CNN model to provide predictions through a simple web interface.

> **Note:** This repository serves the trained model; it does not retrain or modify it.

</details>

<br>

<details>
<summary>2. 🧠 Model</summary>

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

</details>

<br>

<details>
<summary>3. ✨ Features</summary>

* 📤 Upload JPG, JPEG, PNG, or WEBP images
* 🌱 Identify the plant species
* 🦠 Predict the disease
* 💚 Detect Healthy vs. Diseased status
* 📊 Display prediction confidence
* ⚠️ Warn when confidence is low
* 🛡️ Handle invalid or corrupted uploads

</details>

<br>

<details>
<summary>4. 📁 Project Structure</summary>

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

</details>

<br>

<details>
<summary>5. ⚙️ Installation</summary>

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

</details>

<br>

<details>
<summary>6. 🚀 Run the Application</summary>

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

</details>

<br>

<details>
<summary>7. 🖼️ Usage</summary>

Upload a leaf image through the Streamlit interface to get the predicted:

* 🌱 **Plant**
* 🦠 **Disease**
* 💚 **Health Status**
* 📊 **Confidence**

</details>

<br>

<details>
<summary>8. 📊 Model Performance</summary>

The model achieved the following results on the project's held-out validation dataset:

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **94.44%** |
| Macro F1    | **94.41%** |
| Weighted F1 | **94.41%** |

These results are based on the validation data used in the original project and **do not guarantee the same performance on real-world images**.

</details>

<br>

<details>
<summary>9. 🔍 Inference Pipeline</summary>

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

### Pipeline Details

1. **Convert to RGB**
   Ensures the image has three color channels.

2. **Resize**
   The image is resized to **128 × 128 pixels**.

3. **Normalize**
   Pixel values are divided by `255.0` to match the preprocessing used during training.

4. **Add Batch Dimension**
   The image changes from `(128, 128, 3)` to `(1, 128, 128, 3)`.

5. **CNN Prediction**
   The processed image is passed to the trained model.

6. **Select Class**
   `argmax` selects the class with the highest predicted probability.

7. **Map Class Name**
   The class index is mapped using `class_names.json`.

8. **Determine Health Status**
   Classes containing `"healthy"` are labeled **Healthy**; otherwise they are labeled **Diseased**.

</details>

<br>

<details>
<summary>10. ⚠️ Limitations</summary>

* The model supports only the **14 plant species and 38 classes** included in the training dataset.
* Similar-looking diseases may be confused.
* Real-world photos can be more difficult because of lighting, backgrounds, blur, or multiple leaves.
* The confidence score is **not a calibrated probability** of being correct.
* The model does not currently detect unsupported plants or non-plant images. It will still select one of its 38 known classes.
* This application is **not a substitute for professional agricultural or plant-pathology diagnosis**.

</details>

<br>

<details>
<summary>11. 📝 Project Notes</summary>

The original notebook contained some duplicated cells and Colab-specific paths.

For deployment, the project:

* Consolidates the inference logic into `src/inference.py`
* Uses project-relative paths instead of `/content/...`
* Uses a single best prediction instead of the notebook's earlier Top-K implementation
* Keeps the trained model unchanged

</details>

<br>

<details>
<summary>12. 🔮 Future Improvements</summary>

* 🌿 Test performance on real-world leaf images
* 📊 Calibrate confidence scores
* 🔍 Add out-of-distribution detection
* 🧠 Improve the model if real-world testing reveals weaknesses
* ☁️ Deploy using Streamlit Community Cloud or another hosting platform

</details>

<br>

<details>
<summary>⚠️ Disclaimer</summary>

This project is intended for **educational and demonstration purposes**. Predictions may be incorrect, especially for images outside the training dataset.

</details>
