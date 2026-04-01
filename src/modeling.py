from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.data_factory import build_dataset

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - environment dependent
    XGBClassifier = None


def _metric_block(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_project(base_dir: str | Path) -> dict:
    df = build_dataset()
    X = df.drop(columns=["target_default"])
    y = df["target_default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logistic = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2500, class_weight="balanced"))]
    )
    logistic.fit(X_train, y_train)
    logistic_scores = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = _metric_block(y_test.to_numpy(), logistic_scores, threshold=0.45)

    if XGBClassifier is not None:
        selected_model_name = "xgboost"
        booster = XGBClassifier(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2,
            eval_metric="logloss",
            random_state=42,
        )
        booster.fit(X_train, y_train)
        booster_scores = booster.predict_proba(X_test)[:, 1]
        selected_metrics = _metric_block(y_test.to_numpy(), booster_scores, threshold=0.45)
        importances = booster.feature_importances_
        runtime_mode = "xgboost"
    else:
        selected_model_name = "hist_gradient_boosting_fallback"
        booster = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=5,
            max_iter=260,
            random_state=42,
        )
        booster.fit(X_train, y_train)
        booster_scores = booster.predict_proba(X_test)[:, 1]
        selected_metrics = _metric_block(y_test.to_numpy(), booster_scores, threshold=0.45)
        permutation = permutation_importance(
            booster,
            X_test,
            y_test,
            n_repeats=8,
            random_state=42,
            n_jobs=1,
        )
        importances = permutation.importances_mean
        runtime_mode = "fallback_without_xgboost"

    feature_importance = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(8)
    )

    report = {
        "project_name": "loan_default_xgboost",
        "dataset_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
        "runtime_mode": runtime_mode,
        "baseline_model": "logistic_regression",
        "baseline_metrics": logistic_metrics,
        "selected_model": selected_model_name,
        "selected_model_metrics": selected_metrics,
        "top_feature_importance": feature_importance.to_dict(orient="records"),
    }

    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "loan_default_xgboost_report.json").write_text(
        pd.Series(report).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
