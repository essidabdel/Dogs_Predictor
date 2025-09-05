from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Callable

MODEL_FN = Path(__file__).parent / "model.h5"


def load_labels(path: Path = Path(__file__).parent / "labels.txt") -> List[str]:
    if path.exists():
        return [l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]
    return []


def build_custom_objects():
    # From notebook: small Cast layer used when saving under mixed precision
    try:
        import tensorflow as tf

        class Cast(tf.keras.layers.Layer):
            def __init__(self, dtype="float32", **kwargs):
                super().__init__(**kwargs)
                self.target_dtype = tf.as_dtype(dtype)

            def call(self, inputs):
                return tf.cast(inputs, self.target_dtype)

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"dtype": self.target_dtype.name})
                return cfg

        return {"Cast": Cast}
    except Exception:
        return {}


def _get_preprocess_fn_from_model(model) -> Optional[Callable]:
    # Heuristic: inspect model name for backbone used in notebook
    try:
        name = getattr(model, 'name', '') or ''
        name = name.lower()
        if 'effv2' in name or 'efficientnetv2' in name:
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as fn
                return fn
            except Exception:
                return None
        if 'efficientnet' in name or 'effb' in name:
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input as fn
                return fn
            except Exception:
                return None
    except Exception:
        return None
    return None


def load_model(model_path: Path = MODEL_FN):
    try:
        from tensorflow.keras.models import load_model as _load
    except Exception as e:
        raise ImportError(f"TensorFlow not available: {e}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    custom = build_custom_objects()
    model = _load(str(model_path), custom_objects=custom, compile=False)
    return model


def prepare_image_pil(img, target_size=(320, 320), preprocess_fn: Optional[Callable] = None):
    # Resize and normalize. If a preprocess_fn is provided, apply it.
    from PIL import Image
    import numpy as np

    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((target_size[1], target_size[0]))
    arr = np.asarray(img).astype('float32')
    if preprocess_fn is not None:
        try:
            arr = preprocess_fn(arr)
        except Exception:
            # fallback to simple scaling
            arr = arr / 255.0
    else:
        arr = arr / 255.0

    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr


def predict_from_pil(model, img, topk=5, labels: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    import numpy as np

    # Try to infer input size
    try:
        shape = model.input_shape
        if len(shape) == 4:
            _, h, w, _ = shape
            if h is None or w is None:
                h, w = 320, 320
        else:
            h, w = 320, 320
    except Exception:
        h, w = 320, 320

    preprocess = _get_preprocess_fn_from_model(model)
    x = prepare_image_pil(img, (h, w), preprocess_fn=preprocess)
    preds = model.predict(x)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    else:
        probs = np.asarray(preds).ravel()

    idx = list(np.argsort(probs)[::-1][:topk])
    results = []
    for i in idx:
        name = labels[i] if labels and i < len(labels) else str(i)
        results.append((name, float(probs[i])))
    return results


if __name__ == '__main__':
    # quick CLI smoke test
    import sys
    if len(sys.argv) < 2:
        print("Usage: predict.py <image_path>")
        raise SystemExit(1)
    imgp = sys.argv[1]
    model = load_model()
    labels = load_labels()
    res = predict_from_pil(model, imgp, labels=labels)
    for name, score in res:
        print(name, score)
