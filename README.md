# 🌿 Plant Disease Classifier

## 1. Project Overview

This application predicts a plant's species and health status — and, if
diseased, the specific disease — from a single uploaded leaf photo. It wraps
an already-trained convolutional neural network in a clean Streamlit
interface, so a user can upload an image and immediately see a prediction.

## 2. Model

The application uses a CNN trained on the **New Plant Diseases Dataset**
(an augmented version of the PlantVillage dataset), covering 14 plant
species and 38 plant/disease classes (including healthy classes). The model
was trained and evaluated in the original project notebook; this repository
does not retrain or modify it — it only serves it.

- Architecture: 3 × (Conv2D + MaxPooling2D) → Flatten → Dense(128, ReLU) →
  Dropout(0.5) → Dense(38, Softmax)
- Input size: 128 × 128 RGB
- Framework: TensorFlow / Keras 3 (`.keras` format)

## 3. Features

- Drag-and-drop image upload (JPG, JPEG, PNG, WEBP)
- Plant species identification (14 supported species)
- Disease prediction across 38 known plant/disease classes
- Healthy vs. Diseased classification
- Confidence score, with a low-confidence warning
- Graceful handling of invalid, corrupted, or oversized uploads

## 4. Project Structure

```
plant-disease-classifier/
│
├── app.py                     # Streamlit UI
├── src/
│   ├── config.py              # Paths, image size, thresholds
│   └── inference.py           # Preprocessing, prediction, validation
├── model/
│   ├── plant_disease_cnn.keras
│   └── class_names.json
├── tests/
│   └── test_inference.py      # Lightweight tests (no dataset required)
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

```bash
# From the plant-disease-classifier/ directory
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Compatibility note:** `model/plant_disease_cnn.keras` was saved with
Keras 3.13.2, which requires TensorFlow ≥ 2.20. `requirements.txt` pins a
floor version accordingly. If model loading fails in your environment,
install a TensorFlow version whose bundled Keras is ≥ 3.13.

## 6. Running Locally

From the `plant-disease-classifier/` directory:

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) —
open it in your browser.

## 7. Usage

1. Open the app in your browser.
2. Click the upload widget and select a leaf photo (JPG/JPEG/PNG/WEBP).
3. The uploaded image is displayed for confirmation.
4. The app shows **Plant**, **Disease**, **Health Status**, and
   **Confidence**.
5. If confidence is low, a warning suggests trying a clearer photo.

## 8. Model Performance

Measured on the project's held-out validation dataset:

| Metric | Value |
|---|---|
| Accuracy | 94.44% |
| Macro F1 | 94.41% |
| Weighted F1 | 94.41% |

These numbers describe performance on the validation split used during the
original project, **not** a guarantee of real-world accuracy. Images from
outside the training distribution (different lighting, backgrounds, camera
quality, disease stages, or plants not in the training set) may be
classified less reliably.

## 9. Limitations

- Performance is dataset-dependent; the model only knows the 14 plant
  species and 38 classes it was trained on.
- Visually similar diseases (e.g., across different blight types) can be
  confused with each other.
- Real-world phone photos often differ from the dataset's images
  (background clutter, lighting, blur, multiple leaves in frame), which can
  reduce accuracy.
- The softmax confidence score is **not** a calibrated probability of
  correctness — it reflects relative certainty among the known classes, not
  an objective likelihood.
- The model will always return one of its 38 known classes, even for
  images of unsupported plants or non-plant images. There is no built-in
  "unknown / not a supported plant" detection in this version.
- This tool is not a substitute for professional agricultural or
  plant-pathology diagnosis.

---

## Final Inference Pipeline — How It Works

`src/inference.py` implements a single, reusable pipeline (`predict_single_image`)
that mirrors the training notebook's final inference cell exactly:

1. **Convert to RGB** — normalizes grayscale, RGBA, palette, etc. to 3
   channels, matching what `ImageDataGenerator` produced during training.
2. **Resize to 128×128** — the exact size used by `flow_from_directory`.
3. **Rescale to [0, 1]** — `pixel / 255.0`, matching
   `ImageDataGenerator(rescale=1./255)`.
4. **Add batch dimension** — shape becomes `(1, 128, 128, 3)`.
5. **Run `model.predict`** and take `argmax` — only the single best class is
   used; no Top-K logic exists anywhere in the pipeline.
6. **Map index → class name** via `class_names.json`, then split on `___`
   into `plant` and `disease`.
7. **Determine health status** — `"healthy"` (case-insensitive) → Healthy,
   otherwise Diseased.
8. **Return a structured result** (`PredictionResult`) with plant, disease,
   health status, and confidence.

`app.py` wraps this pipeline with Streamlit-specific concerns only: cached
model loading (`st.cache_resource`), file upload handling, input
validation, and display.

## Issues Discovered in the Original Notebook

- The notebook contains several **duplicate cells** (class-mapping saving
  and inference-pipeline construction each appear more than once as the
  project evolved). This repository consolidates that logic into one
  canonical version in `src/inference.py`; no behavior was changed by this
  consolidation.
- An earlier inference function in the notebook (cell 35) returns **Top-K**
  predictions by default (`top_k=3`). Per the project requirements, this
  repository's pipeline always returns only the single best prediction —
  this is an intentional simplification for deployment, not a bug fix.
- Colab-specific absolute paths (`/content/...`) were used throughout the
  notebook. This repository uses paths resolved relative to the project
  root (`src/config.py`) so the app runs identically on any machine.

## Remaining Limitations

See "Limitations" above. In particular: no out-of-distribution / "not a
supported plant" detection, and confidence is uncalibrated.

## Next Steps

- Test the model against real-world (non-dataset) photos to see how
  accuracy holds up outside the validation set.
- Explore confidence calibration (e.g., temperature scaling) if a
  calibration/held-out set becomes available.
- Add out-of-distribution detection so clearly unrelated images (e.g., a
  photo of a car) are flagged as "not a supported plant" rather than forced
  into one of the 38 known classes.
- If real-world testing reveals systematic weaknesses, revisit the model
  (architecture, data augmentation, or additional training data) — out of
  scope for this deployment pass.
- Prepare for cloud deployment (e.g., Streamlit Community Cloud): this
  project already uses relative paths, a `requirements.txt`, and requires
  no secrets/API keys, so it should deploy with minimal changes. Watch the
  model file size (~38 MB) against the hosting platform's repo/storage
  limits.
