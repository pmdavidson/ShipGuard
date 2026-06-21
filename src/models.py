"""
Three complementary ML models targeting different analytics goals:

  1. AnomalyDetector: unsupervised Isolation Forest that flags unusual sensor readings without labels
  2. FaultClassifier: XGBoost multi-class classifier that identifies fault type (0 = normal, 1 - 3 = fault)
  3. RULPredictor: XGBoost regressor that estimates remaining-useful-life in timesteps

Each class exposes: fit(X), predict(X), evaluate(X, y), feature_importance()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# sklearn / xgboost — imported here so failure is loud and early
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
import xgboost as xgb

"""
Isolation Forest trained only on normal-operation data
Predicts -1 (anomaly) or +1 (normal)
"""
class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns array of 1 (anomaly) or 0 (normal)."""
        raw = self.model.predict(X)          # sklearn: -1 or +1
        return (raw == -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw anomaly score (lower = more anomalous)."""
        return self.model.score_samples(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        y_true: 0 = normal, 1 = anomaly (any fault).
        Returns precision, recall, f1 for the anomaly class.
        """
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred,
                                       target_names=["normal", "anomaly"],
                                       output_dict=True)
        print("\n── Anomaly Detector ──")
        print(classification_report(y_true, y_pred,
                                    target_names=["normal", "anomaly"]))
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "report"          : report,
            "predictions"     : y_pred,
            "scores"          : self.score(X),
        }


"""
XGBoost multi-class classifier

Labels: 0 = normal, 1 - 3 = fault type
"""
FAULT_NAMES = {0: "Normal", 1: "Cooling Failure",
               2: "Fuel Degradation", 3: "Bearing Wear"}

class FaultClassifier:
    def __init__(self, **kwargs):
        params = dict(
            objective       = "multi:softprob",
            num_class       = 4,
            n_estimators    = 400,
            max_depth       = 6,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            use_label_encoder=False,
            eval_metric     = "mlogloss",
            random_state    = 42,
            n_jobs          = -1,
        )
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray,
            y: pd.Series | np.ndarray,
            X_val: pd.DataFrame | np.ndarray | None = None,
            y_val: pd.Series | np.ndarray | None = None) -> "FaultClassifier":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X, y_true: np.ndarray) -> dict:
        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)
        classes = list(range(4))
        y_bin   = label_binarize(y_true, classes=classes)
        try:
            auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        report = classification_report(
            y_true, y_pred,
            target_names=[FAULT_NAMES[i] for i in classes],
            output_dict=True,
        )
        print("\n── Fault Classifier ──")
        print(classification_report(y_true, y_pred,
                                    target_names=[FAULT_NAMES[i] for i in classes]))
        print(f"   Macro ROC-AUC: {auc:.4f}")
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "report"          : report,
            "roc_auc"         : auc,
            "predictions"     : y_pred,
            "probabilities"   : y_proba,
        }

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        names = (self.feature_names_
                 if self.feature_names_ else [f"f{i}" for i in range(len(imp))])
        return pd.Series(imp, index=names).sort_values(ascending=False)


"""
XGBoost regressor that estimates remaining-useful-life in timesteps.
Trained only on pre-fault and normal rows (excludes active-fault rows
where RUL = 0) to focus on early detection.
"""
class RULPredictor:
    def __init__(self, **kwargs):
        params = dict(
            objective    = "reg:squarederror",
            n_estimators = 500,
            max_depth    = 5,
            learning_rate= 0.04,
            subsample    = 0.8,
            random_state = 42,
            n_jobs       = -1,
        )
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)
        self.feature_names_: list[str] | None = None

    def fit(self, X, y, X_val=None, y_val=None) -> "RULPredictor":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X).clip(0, 144)

    def evaluate(self, X, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        mae    = mean_absolute_error(y_true, y_pred)
        r2     = r2_score(y_true, y_pred)
        print(f"\n── RUL Predictor ──  MAE={mae:.1f} steps  R²={r2:.4f}")
        return {
            "mae"        : mae,
            "r2"         : r2,
            "predictions": y_pred,
            "actuals"    : y_true,
        }

    def feature_importance(self) -> pd.Series:
        imp   = self.model.feature_importances_
        names = (self.feature_names_
                 if self.feature_names_ else [f"f{i}" for i in range(len(imp))])
        return pd.Series(imp, index=names).sort_values(ascending=False)
