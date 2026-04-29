import json
import os
import pickle
from pathlib import Path

import numpy as np
import pymysql
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "model" / "artifacts"
CHART_DIR = BASE_DIR / "report" / "charts"

app = Flask(__name__)
CORS(app)


FRONTEND_DIR = BASE_DIR / "frontend"


@app.get("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/index.html")
def serve_index_page():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/analytics.html")
def serve_analytics_page():
    return send_from_directory(FRONTEND_DIR, "analytics.html")


@app.get("/predictor.html")
def serve_predictor_page():
    return send_from_directory(FRONTEND_DIR, "predictor.html")


@app.get("/script.js")
def serve_script():
    return send_from_directory(FRONTEND_DIR, "script.js")


@app.get("/styles.css")
def serve_styles():
    return send_from_directory(FRONTEND_DIR, "styles.css")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def risk_name(class_id: int) -> str:
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    return mapping.get(int(class_id), "Unknown")


linear = load_pickle(ARTIFACT_DIR / "linear.pkl") if (ARTIFACT_DIR / "linear.pkl").exists() else None
poly = load_pickle(ARTIFACT_DIR / "poly.pkl") if (ARTIFACT_DIR / "poly.pkl").exists() else None
poly_transformer = (
    load_pickle(ARTIFACT_DIR / "poly_transformer.pkl")
    if (ARTIFACT_DIR / "poly_transformer.pkl").exists()
    else None
)
log_model = load_pickle(ARTIFACT_DIR / "log.pkl") if (ARTIFACT_DIR / "log.pkl").exists() else None
power_model = load_pickle(ARTIFACT_DIR / "power.pkl") if (ARTIFACT_DIR / "power.pkl").exists() else None
risk_model = load_pickle(ARTIFACT_DIR / "risk.pkl") if (ARTIFACT_DIR / "risk.pkl").exists() else None


def predict_magnitude_by_model(model_name: str, features: np.ndarray) -> float:
    model_name = model_name.lower()
    if model_name == "linear":
        return float(linear.predict(features)[0])
    if model_name == "polynomial":
        if poly is None or poly_transformer is None:
            raise ValueError("Polynomial model not available.")
        return float(poly.predict(poly_transformer.transform(features))[0])
    if model_name == "logarithmic":
        if log_model is None:
            raise ValueError("Log model not available.")
        return float(np.expm1(log_model.predict(features))[0])
    if model_name == "power":
        if power_model is None:
            raise ValueError("Power model not available.")
        transformed = np.log1p(np.clip(features, 0, None))
        return float(np.expm1(power_model.predict(transformed))[0])
    raise ValueError("Unknown model. Use linear/polynomial/logarithmic/power.")


def get_mysql_connection():
    host = os.getenv("MYSQL_HOST", "localhost")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DB", "earthquake_db")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    return pymysql.connect(host=host, user=user, password=password, database=db, port=port)


def insert_prediction(lat: float, lon: float, depth: float, magnitude: float, risk: str) -> None:
    try:
        conn = get_mysql_connection()
    except Exception:
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (latitude, longitude, depth, magnitude, risk)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (lat, lon, depth, magnitude, risk),
            )
        conn.commit()
    finally:
        conn.close()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/metrics")
def metrics():
    metrics_file = ARTIFACT_DIR / "metrics.json"
    if not metrics_file.exists():
        return jsonify({"error": "metrics.json not found. Run training first."}), 404
    with open(metrics_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/feature-importance")
def feature_importance():
    fi_file = ARTIFACT_DIR / "feature_importance.json"
    if not fi_file.exists():
        return jsonify({"error": "feature_importance.json not found. Run training first."}), 404
    with open(fi_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return jsonify(payload)


@app.get("/generated-charts/<path:filename>")
def generated_chart(filename: str):
    chart_file = CHART_DIR / filename
    if not chart_file.exists():
        return jsonify({"error": f"Chart '{filename}' not found. Run model/train.py first."}), 404
    return send_from_directory(CHART_DIR, filename)


@app.get("/forecast-comparison")
def forecast_comparison():
    if linear is None:
        return jsonify({"error": "Model artifacts not found. Run model/train.py first."}), 400

    lat = float(request.args.get("lat", "0"))
    lon = float(request.args.get("lon", "0"))
    depth_max = float(request.args.get("depth_max", "700"))
    depth_points = int(request.args.get("points", "30"))
    depth_points = max(10, min(depth_points, 100))

    depths = np.linspace(0, max(depth_max, 1), depth_points)
    series = {"Linear": [], "Polynomial": [], "Logarithmic": [], "Power": []}

    for d in depths:
        row = np.array([[lat, lon, float(d)]])
        series["Linear"].append(predict_magnitude_by_model("linear", row))
        series["Polynomial"].append(predict_magnitude_by_model("polynomial", row))
        series["Logarithmic"].append(predict_magnitude_by_model("logarithmic", row))
        series["Power"].append(predict_magnitude_by_model("power", row))

    return jsonify(
        {
            "reference": "Predicted magnitude vs depth for fixed latitude and longitude.",
            "lat": lat,
            "lon": lon,
            "depths": [float(x) for x in depths],
            "series": series,
        }
    )


@app.get("/predict-visuals")
def predict_visuals():
    if linear is None or risk_model is None:
        return jsonify({"error": "Model artifacts not found. Run model/train.py first."}), 400

    lat = float(request.args.get("lat", "0"))
    lon = float(request.args.get("lon", "0"))
    depth = float(request.args.get("depth", "10"))
    features = np.array([[lat, lon, depth]])

    model_predictions = {
        "Linear": predict_magnitude_by_model("linear", features),
        "Polynomial": predict_magnitude_by_model("polynomial", features),
        "Logarithmic": predict_magnitude_by_model("logarithmic", features),
        "Power": predict_magnitude_by_model("power", features),
    }
    risk_class = int(risk_model.predict(features)[0])

    risk_probabilities = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    if hasattr(risk_model, "predict_proba"):
        probs = risk_model.predict_proba(features)[0]
        classes = risk_model.classes_
        for idx, cls in enumerate(classes):
            risk_probabilities[risk_name(int(cls))] = float(probs[idx])

    return jsonify(
        {
            "reference": "Visual analytics for a single predicted event input.",
            "input": {"lat": lat, "lon": lon, "depth": depth},
            "magnitude_by_model": model_predictions,
            "risk_level": risk_name(risk_class),
            "risk_class": risk_class,
            "risk_probabilities": risk_probabilities,
        }
    )


@app.post("/predict")
def predict():
    if linear is None or risk_model is None:
        return jsonify({"error": "Model artifacts not found. Run model/train.py first."}), 400

    data = request.get_json(force=True)
    lat = float(data["lat"])
    lon = float(data["lon"])
    depth = float(data["depth"])
    model_name = data.get("model", "linear").lower()

    features = np.array([[lat, lon, depth]])

    try:
        magnitude = predict_magnitude_by_model(model_name, features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    risk_class = int(risk_model.predict(features)[0])
    risk = risk_name(risk_class)
    insert_prediction(lat, lon, depth, magnitude, risk)

    return jsonify(
        {
            "selected_model": model_name,
            "predicted_magnitude": magnitude,
            "risk_level": risk,
            "risk_class": risk_class,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
