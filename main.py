"""
main.py
-------
ShipGuard – Marine Vessel Predictive Maintenance & Anomaly Detection
End-to-end pipeline runner.

Usage:
    python main.py               # full run
    python main.py --no-plots    # skip visualisations (faster)
    python main.py --vessel VS_Haida_Gwaii   # restrict sensor plots to one vessel
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import generate
from src.pipeline       import run as pipeline_run
from src.features       import build as build_features, get_feature_cols
from src.models         import AnomalyDetector, FaultClassifier, RULPredictor
from src import visualizations as viz

OUTPUT_DIR   = Path("reports/figures")
DATA_DIR     = Path("data")
RAW_CSV      = DATA_DIR / "sensor_data_raw.csv"
VESSELS      = ["VS_Haida_Gwaii", "VS_Spirit_Sound", "VS_Pacific_Wren"]


def _arrays(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].values
    y_class = df["fault_label"].values.astype(int)
    y_rul   = df["rul"].values
    return X, y_class, y_rul


def main(plots: bool = True, vessel_filter: str | None = None):
    print("=" * 60)
    print("  ShipGuard – Marine Predictive Maintenance")
    print("=" * 60)

    # Generate data
    if not RAW_CSV.exists():
        generate(DATA_DIR)
    else:
        print(f"[main] Using cached data → {RAW_CSV}")

    # Pipeline
    train_raw, test_raw, stats = pipeline_run(RAW_CSV)

    # Feature engineering
    train_fe = build_features(train_raw)
    test_fe  = build_features(test_raw)

    feature_cols = get_feature_cols(train_fe)
    X_train, y_class_train, y_rul_train = _arrays(train_fe, feature_cols)
    X_test,  y_class_test,  y_rul_test  = _arrays(test_fe,  feature_cols)

    # binary anomaly labels (any fault type → 1)
    y_anom_train = (y_class_train > 0).astype(int)
    y_anom_test  = (y_class_test  > 0).astype(int)

    # EDA visualisations (pre-model)
    if plots:
        print("\n[main] Generating EDA plots …")
        viz.eda_distributions(train_raw, output_dir=OUTPUT_DIR)

        for v in (VESSELS if vessel_filter is None else [vessel_filter]):
            viz.sensor_overview(train_raw, vessel=v, output_dir=OUTPUT_DIR)

    # Anomaly Detector (Isolation Forest)
    print("\n[main] Training Anomaly Detector …")
    # train only on normal-operation rows (no labels required in practice)
    normal_mask = y_class_train == 0
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X_train[normal_mask])
    anom_results = detector.evaluate(X_test, y_anom_test)

    if plots:
        for v in (VESSELS if vessel_filter is None else [vessel_filter]):
            v_mask  = test_fe["vessel_id"] == v
            if v_mask.sum() == 0:
                continue
            v_df    = test_fe[v_mask]
            scores  = detector.score(v_df[feature_cols].values)
            viz.anomaly_timeline(
                timestamps    = v_df["timestamp"],
                anomaly_scores= scores,
                fault_labels  = v_df["fault_label"].values,
                vessel        = v,
                output_dir    = OUTPUT_DIR,
            )

    # Fault Classifier (XGBoost)
    print("\n[main] Training Fault Classifier …")
    classifier = FaultClassifier()
    classifier.fit(
        pd.DataFrame(X_train, columns=feature_cols), y_class_train,
        X_val=pd.DataFrame(X_test,  columns=feature_cols), y_val=y_class_test,
    )
    clf_results = classifier.evaluate(
        pd.DataFrame(X_test, columns=feature_cols), y_class_test
    )

    if plots:
        viz.confusion_matrix_plot(clf_results["confusion_matrix"],
                                  output_dir=OUTPUT_DIR)
        viz.feature_importance_plot(
            classifier.feature_importance(), top_n=25,
            title="Fault Classifier – Top Feature Importances",
            filename="feature_importance_clf.png",
            output_dir=OUTPUT_DIR,
        )

    # RUL Predictor (XGBoost Regressor)
    print("\n[main] Training RUL Predictor …")
    # exclude active-fault rows (RUL=0) so model learns pre-fault patterns
    pre_fault_train = y_rul_train > 0
    pre_fault_test  = y_rul_test  > 0

    rul_predictor = RULPredictor()
    rul_predictor.fit(
        pd.DataFrame(X_train[pre_fault_train], columns=feature_cols),
        y_rul_train[pre_fault_train],
        X_val=pd.DataFrame(X_test[pre_fault_test],  columns=feature_cols),
        y_val=y_rul_test[pre_fault_test],
    )
    rul_results = rul_predictor.evaluate(
        pd.DataFrame(X_test[pre_fault_test], columns=feature_cols),
        y_rul_test[pre_fault_test],
    )

    if plots:
        viz.rul_scatter(rul_results["actuals"], rul_results["predictions"],
                        output_dir=OUTPUT_DIR)
        viz.feature_importance_plot(
            rul_predictor.feature_importance(), top_n=25,
            title="RUL Predictor – Top Feature Importances",
            filename="feature_importance_rul.png",
            output_dir=OUTPUT_DIR,
        )

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    anom_report = anom_results["report"]["anomaly"]
    print(f"  Anomaly Detection - Precision: {anom_report['precision']:.3f} "
          f"| Recall: {anom_report['recall']:.3f} "
          f"| F1: {anom_report['f1-score']:.3f}")
    print(f"  Fault Classifier - Macro F1: "
          f"{clf_results['report']['macro avg']['f1-score']:.3f} "
          f"| ROC-AUC: {clf_results['roc_auc']:.3f}")
    print(f"  RUL Predictor - MAE: {rul_results['mae']:.1f} steps "
          f"({rul_results['mae'] * 10 / 60:.1f} h) "
          f"| R²: {rul_results['r2']:.3f}")
    if plots:
        print(f"\n  Figures saved to → {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShipGuard pipeline runner")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all visualisation output")
    parser.add_argument("--vessel", type=str, default=None,
                        help="Restrict sensor / anomaly plots to one vessel ID")
    args = parser.parse_args()
    main(plots=not args.no_plots, vessel_filter=args.vessel)
