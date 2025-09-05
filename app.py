from __future__ import annotations
import os
from io import BytesIO
from pathlib import Path
from typing import List

from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

from predict import load_labels as predict_load_labels, load_model as predict_load_model, predict_from_pil

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.h5"
LABELS_PATH = APP_DIR / "labels.txt"

app = Flask(__name__)


def load_labels() -> List[str]:
    return predict_load_labels(LABELS_PATH)


def safe_load_model():
    """Try to import tensorflow and load the Keras model. If TensorFlow is not installed,
    return None and an error string instead of raising.
    """
    try:
        from tensorflow.keras.models import load_model
    except Exception as e:  # pragma: no cover - environment dependent
        return None, f"TensorFlow import failed: {e}"

    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"

    try:
        model = load_model(str(MODEL_PATH))
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def prepare_image(img: Image.Image, target_size):
    # Convert to RGB and resize
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((target_size[1], target_size[0]))
    import numpy as np
    arr = np.asarray(img).astype('float32') / 255.0
    # add batch dim
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr


@app.route('/', methods=['GET'])
def index():
    labels = load_labels()
    # expose last uploaded image if present so index can show a persistent preview
    last_img_path = APP_DIR / 'static' / 'uploads' / 'last_upload.jpg'
    has_last = last_img_path.exists()
    img_url = url_for('static', filename='uploads/last_upload.jpg') if has_last else None
    return render_template('index.html', labels_available=bool(labels), last_image=has_last, last_image_url=img_url)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return redirect(url_for('index'))

    img = Image.open(BytesIO(file.read()))

    # load model (may raise ImportError/FileNotFoundError which we catch)
    try:
        model = predict_load_model(MODEL_PATH)
    except Exception as e:
        return render_template('result.html', error=str(e))

    labels = load_labels()
    try:
        preds = predict_from_pil(model, img, topk=5, labels=labels)
    except Exception as e:
        return render_template('result.html', error=f'Erreur durant la prédiction: {e}')

    # Ensure scores are native Python floats (not numpy types) for Jinja filters
    results = [{'label': name, 'score': float(score)} for name, score in preds]

    # Save uploaded image to static for display (overwrite safe)
    out_path = APP_DIR / 'static' / 'uploads'
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / 'last_upload.jpg'
    img.convert('RGB').save(save_path)

    # expose the saved image url for the template
    img_url = url_for('static', filename='uploads/last_upload.jpg')
    return render_template('result.html', results=results, img_url=img_url)


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
