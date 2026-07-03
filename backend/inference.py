import io
import json
import os

import joblib
import numpy as np
import requests
import torch
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from catboost import CatBoostRegressor
from torchvision import models, transforms

from preprocessing import parse_catalog_content, clean_text

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load text vectorizers / encoders ----------
name_vectorizer = joblib.load(os.path.join(MODELS_DIR, "name_vectorizer.pkl"))
bullet_vectorizer = joblib.load(os.path.join(MODELS_DIR, "bullet_vectorizer.pkl"))
desc_vectorizer = joblib.load(os.path.join(MODELS_DIR, "desc_vectorizer.pkl"))
brand_encoder = joblib.load(os.path.join(MODELS_DIR, "brand_encoder.pkl"))
unit_encoder = joblib.load(os.path.join(MODELS_DIR, "unit_encoder.pkl"))

with open(os.path.join(MODELS_DIR, "value_median.json")) as f:
    VALUE_MEDIAN = json.load(f)["value_median"]

# ---------- Load the two regressors ----------
lgb_model = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))

cat_model = CatBoostRegressor()
cat_model.load_model(os.path.join(MODELS_DIR, "catboost.cbm"))

# ---------- Load EfficientNet-B0 image feature extractor ----------
_eff = models.efficientnet_b0(weights="DEFAULT")
feature_extractor = torch.nn.Sequential(*list(_eff.children())[:-1]).to(DEVICE)
feature_extractor.eval()

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_image_embedding(image: Image.Image) -> np.ndarray:
    """Runs EfficientNet-B0 and returns a 1280-d feature vector."""
    img = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(tensor)
    return feat.squeeze().cpu().numpy().reshape(1, -1)


def _load_image(image_file=None, image_url=None) -> Image.Image:
    if image_file is not None:
        return Image.open(image_file)
    if image_url:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    raise ValueError("Either image_file or image_url must be provided")


def _predict_core(item_name, brand_name, bullet_points, product_description,
                   value, unit, image_file=None, image_url=None) -> dict:
    """Shared inference logic. All text fields must already be CLEANED
    (lowercased, punctuation/stopwords stripped) except brand_name, which
    is used raw — same as during training."""
    item_name = item_name or ""
    brand_name = brand_name or "Unknown"
    bullet_points = bullet_points or ""
    product_description = product_description or ""
    value = value if value is not None else VALUE_MEDIAN

    X_name = name_vectorizer.transform([item_name])
    X_bullet = bullet_vectorizer.transform([bullet_points])
    X_desc = desc_vectorizer.transform([product_description])

    brand_encoded = brand_encoder.transform([[brand_name]])
    unit_encoded = unit_encoder.transform([[unit]])

    image = _load_image(image_file=image_file, image_url=image_url)
    image_features = _get_image_embedding(image)

    numeric = np.array([[value, unit_encoded[0][0], brand_encoded[0][0]]], dtype=np.float32)

    X = hstack([X_name, X_bullet, X_desc, csr_matrix(image_features), csr_matrix(numeric)]).tocsr()

    lgb_pred_log = lgb_model.predict(X)[0]
    cat_pred_log = cat_model.predict(X)[0]
    ensemble_log = 0.5 * lgb_pred_log + 0.5 * cat_pred_log
    price = float(np.expm1(ensemble_log))

    return {
        "predicted_price": round(price, 2),
        "lgb_price": round(float(np.expm1(lgb_pred_log)), 2),
        "catboost_price": round(float(np.expm1(cat_pred_log)), 2),
        "parsed": {
            "item_name": item_name,
            "brand_name": brand_name,
            "bullet_points": bullet_points[:200],
            "product_description": product_description[:200],
            "value": value,
            "unit": unit,
        },
    }


def predict(catalog_content: str, image_file=None, image_url=None) -> dict:
    """
    catalog_content: raw text blob (Item Name / Bullet Point / Product
        Description / Value / Unit — same format as training data)
    image_file: file-like object (e.g. from Flask request.files)
    image_url: URL string, used if image_file is not provided
    Returns dict with predicted_price and the parsed fields (for debugging/UI).
    """
    parsed = parse_catalog_content(catalog_content)
    return _predict_core(
        item_name=parsed["item_name"],
        brand_name=parsed["brand_name"],
        bullet_points=parsed["bullet_points"],
        product_description=parsed["product_description"],
        value=parsed["value"],
        unit=parsed["unit"],
        image_file=image_file,
        image_url=image_url,
    )


def predict_from_fields(item_name, brand_name, bullet_points, product_description,
                         value, unit, image_file=None, image_url=None) -> dict:
    """
    Same pipeline as `predict`, but takes each feature as a separate
    already-typed-by-the-user field instead of one raw catalog_content
    blob. Text fields are cleaned here (lowercase, strip punctuation/
    stopwords) exactly like the training-time `clean_text` step.

    item_name / bullet_points / product_description: plain strings
        (bullet_points may contain multiple points separated by newlines)
    brand_name: plain string, used as-is (not cleaned) — matches training
    value: float or None
    unit: string or None
    """
    bullets_joined = " ".join(
        line.strip() for line in (bullet_points or "").splitlines() if line.strip()
    )

    return _predict_core(
        item_name=clean_text(item_name),
        brand_name=(brand_name or "Unknown").strip(),
        bullet_points=clean_text(bullets_joined),
        product_description=clean_text(product_description),
        value=value,
        unit=unit,
        image_file=image_file,
        image_url=image_url,
    )