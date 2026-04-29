import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ARTIFACT_DIR = ROOT / "model" / "artifacts"
CHART_DIR = ROOT / "report" / "charts"


def get_risk_label(magnitude: float) -> str:
    if magnitude < 4:
        return "Low"
    if magnitude < 6:
        return "Medium"
    return "High"


def get_risk_class(magnitude: float) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}[get_risk_label(magnitude)]


def evaluate_regression(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def resolve_dataset_path() -> Path:
    # Prefer the user-provided dataset name first.
    preferred_names = [
        "USG_earthquake_data.csv",
        "earthquake.csv",
    ]
    for name in preferred_names:
        candidate = DATASET_DIR / name
        if candidate.exists():
            return candidate

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"No CSV dataset found in {DATASET_DIR}. Add USG_earthquake_data.csv."
    )


def main() -> None:
    data_path = resolve_dataset_path()
    df = pd.read_csv(data_path)
    if "magnitude" not in df.columns and "mag" in df.columns:
        df = df.rename(columns={"mag": "magnitude"})

    required_cols = {"latitude", "longitude", "depth", "magnitude"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.dropna(subset=["latitude", "longitude", "depth", "magnitude"]).copy()
    df["risk_level"] = df["magnitude"].apply(get_risk_label)
    df["risk_class"] = df["magnitude"].apply(get_risk_class)

    X = df[["latitude", "longitude", "depth"]]
    y = df["magnitude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear regression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred_linear = linear.predict(X_test)

    # Polynomial regression
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_transformer.fit_transform(X_train)
    X_test_poly = poly_transformer.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    # Logarithmic model (log target)
    y_train_log = np.log1p(y_train)
    log_model = LinearRegression()
    log_model.fit(X_train, y_train_log)
    y_pred_log = np.expm1(log_model.predict(X_test))

    # Power model (log features + log target)
    X_train_log = np.log1p(X_train.clip(lower=0))
    X_test_log = np.log1p(X_test.clip(lower=0))
    power_model = LinearRegression()
    power_model.fit(X_train_log, y_train_log)
    y_pred_power = np.expm1(power_model.predict(X_test_log))

    regression_results = {
        "Linear": evaluate_regression(y_test, y_pred_linear),
        "Polynomial": evaluate_regression(y_test, y_pred_poly),
        "Logarithmic": evaluate_regression(y_test, y_pred_log),
        "Power": evaluate_regression(y_test, y_pred_power),
    }

    # Classification model for risk level
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, df.loc[X_train.index, "risk_class"])
    y_risk_test = df.loc[X_test.index, "risk_class"]
    y_risk_pred = clf.predict(X_test)
    classification_accuracy = float(accuracy_score(y_risk_test, y_risk_pred))

    # Dataset-driven feature importances for risk classification
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_cols = {"magnitude", "risk_class"}
    numeric_feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
    X_importance = numeric_df[numeric_feature_cols].fillna(
        numeric_df[numeric_feature_cols].median(numeric_only=True)
    )
    y_importance = df["risk_class"]

    class_counts = y_importance.value_counts()
    stratify_target = y_importance if class_counts.min() >= 2 else None
    X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(
        X_importance, y_importance, test_size=0.2, random_state=42, stratify=stratify_target
    )
    importance_model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    importance_model.fit(X_imp_train, y_imp_train)
    importance_accuracy = float(accuracy_score(y_imp_test, importance_model.predict(X_imp_test)))
    feature_importances = (
        pd.DataFrame(
            {
                "feature": X_importance.columns,
                "importance": importance_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACT_DIR / "linear.pkl", "wb") as f:
        pickle.dump(linear, f)
    with open(ARTIFACT_DIR / "poly_transformer.pkl", "wb") as f:
        pickle.dump(poly_transformer, f)
    with open(ARTIFACT_DIR / "poly.pkl", "wb") as f:
        pickle.dump(poly_model, f)
    with open(ARTIFACT_DIR / "log.pkl", "wb") as f:
        pickle.dump(log_model, f)
    with open(ARTIFACT_DIR / "power.pkl", "wb") as f:
        pickle.dump(power_model, f)
    with open(ARTIFACT_DIR / "risk.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(ARTIFACT_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "RandomForestClassifier",
                "reference": "Top 10 numeric feature importances for predicting risk_class",
                "accuracy": importance_accuracy,
                "features": feature_importances.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    metrics_payload = {
        "regression": regression_results,
        "classification": {"accuracy": classification_accuracy},
    }
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="depth", y="magnitude", hue="risk_level", alpha=0.7)
    plt.title("Magnitude vs Depth")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "magnitude_vs_depth.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="risk_level", order=["Low", "Medium", "High"])
    plt.title("Risk Level Distribution")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "risk_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    model_names = list(regression_results.keys())
    rmses = [regression_results[m]["rmse"] for m in model_names]
    sns.barplot(x=model_names, y=rmses, hue=model_names, palette="viridis", legend=False)
    plt.ylabel("RMSE")
    plt.title("Regression Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "rmse_comparison.png", dpi=140)
    plt.close()

    print("Training completed successfully.")
    print(f"Dataset used: {data_path}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
