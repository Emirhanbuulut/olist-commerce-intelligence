"""Train leakage-aware late-delivery classifiers with chronological validation."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR


NUMERIC_FEATURES = [
    "item_count", "unique_product_count", "seller_count", "total_price", "total_freight",
    "avg_product_weight_g", "avg_product_volume_cm3", "avg_distance_km",
    "payment_installments", "estimated_days", "purchase_month", "purchase_weekday", "purchase_hour",
]
CATEGORICAL_FEATURES = [
    "customer_state", "primary_category", "primary_seller_state", "primary_payment_type"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def metrics(y_true, probabilities, threshold=0.5) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def best_f1_threshold(y_true, probabilities) -> float:
    candidates = np.linspace(0.05, 0.80, 76)
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def main() -> None:
    orders = pd.read_parquet(PROCESSED_DIR / "orders_enriched.parquet")
    model_data = (
        orders.query("order_status == 'delivered'")
        .dropna(subset=["is_late", "order_purchase_timestamp"])
        .sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )
    model_data["is_late"] = model_data["is_late"].astype(int)
    split_at = int(len(model_data) * 0.80)
    train = model_data.iloc[:split_at]
    test = model_data.iloc[split_at:]
    validation_at = int(len(train) * 0.80)
    fit, validation = train.iloc[:validation_at], train.iloc[validation_at:]

    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation_results = {}
    fitted = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipeline.fit(fit[FEATURES], fit["is_late"])
        probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
        threshold = best_f1_threshold(validation["is_late"], probabilities) if name != "dummy" else 0.5
        validation_results[name] = {**metrics(validation["is_late"], probabilities, threshold), "threshold": threshold}
        fitted[name] = pipeline

    best_name = max(
        (name for name in estimators if name != "dummy"),
        key=lambda name: validation_results[name]["pr_auc"],
    )
    best_threshold = validation_results[best_name]["threshold"]
    best_pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimators[best_name])])
    best_pipeline.fit(train[FEATURES], train["is_late"])
    test_probabilities = best_pipeline.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metrics(test["is_late"], test_probabilities, best_threshold)

    baseline_pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", DummyClassifier(strategy="prior"))])
    baseline_pipeline.fit(train[FEATURES], train["is_late"])
    baseline_probabilities = baseline_pipeline.predict_proba(test[FEATURES])[:, 1]
    baseline_metrics = metrics(test["is_late"], baseline_probabilities, 0.5)

    predictions = test[["order_id", "order_purchase_timestamp", "is_late"]].copy()
    predictions["late_probability"] = test_probabilities
    predictions["predicted_late"] = (test_probabilities >= best_threshold).astype(int)
    predictions.to_csv(PROCESSED_DIR / "model_test_predictions.csv", index=False)

    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
    model = best_pipeline.named_steps["model"]
    importance = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
    pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).head(30).to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    results = {
        "target": "Delivery after the promised date",
        "positive_rate_train": float(train["is_late"].mean()),
        "positive_rate_test": float(test["is_late"].mean()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(train["order_purchase_timestamp"].max()),
        "test_start": str(test["order_purchase_timestamp"].min()),
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "validation": validation_results,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "features": FEATURES,
        "leakage_policy": "Only fields known at purchase or checkout are used; delivery and review outcomes are excluded.",
    }
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "late_delivery_model.joblib")
    (ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected {best_name}: test PR-AUC={test_metrics['pr_auc']:.3f}, recall={test_metrics['recall']:.3f}")


if __name__ == "__main__":
    main()

