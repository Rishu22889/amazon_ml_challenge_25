"""
Flask backend for the Amazon ML Challenge price predictor.
"""
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import predict, predict_from_fields

app = Flask(__name__)
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=[o.strip() for o in _origins.split(",")] if _origins != "*" else "*")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accepts multipart/form-data:
        catalog_content : str (required)
        image           : file (optional if image_url given)
        image_url       : str  (optional if image file given)
    """
    catalog_content = request.form.get("catalog_content", "")
    image_url = request.form.get("image_url", "").strip() or None
    image_file = request.files.get("image")

    if not catalog_content:
        return jsonify({"error": "catalog_content is required"}), 400
    if image_file is None and not image_url:
        return jsonify({"error": "Provide either an image file or image_url"}), 400

    try:
        result = predict(
            catalog_content=catalog_content,
            image_file=image_file if image_file else None,
            image_url=image_url,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_fields", methods=["POST"])
def predict_fields_route():
    """
    Accepts multipart/form-data with each feature as its own field:
        item_name            : str (required)
        brand_name           : str (optional, defaults to "Unknown")
        bullet_points         : str (optional, one bullet per line)
        product_description  : str (optional)
        value                : float (optional, falls back to training median)
        unit                 : str (optional)
        image                : file (optional if image_url given)
        image_url            : str  (optional if image file given)
    """
    item_name = request.form.get("item_name", "")
    brand_name = request.form.get("brand_name", "")
    bullet_points = request.form.get("bullet_points", "")
    product_description = request.form.get("product_description", "")
    unit = request.form.get("unit", "").strip() or None

    value_raw = request.form.get("value", "").strip()
    value = float(value_raw) if value_raw else None

    image_url = request.form.get("image_url", "").strip() or None
    image_file = request.files.get("image")

    if not item_name.strip():
        return jsonify({"error": "item_name is required"}), 400
    if image_file is None and not image_url:
        return jsonify({"error": "Provide either an image file or image_url"}), 400

    try:
        result = predict_from_fields(
            item_name=item_name,
            brand_name=brand_name,
            bullet_points=bullet_points,
            product_description=product_description,
            value=value,
            unit=unit,
            image_file=image_file if image_file else None,
            image_url=image_url,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
