"""Smoke tests for frozen-model batch inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from missed_delivery.inference import predict_batch


def test_frozen_artifacts_load_and_predict() -> None:
    artifact_dir = REPOSITORY_ROOT / "models"
    for artifact_name in [
        "md_xgboost_pipeline.joblib",
        "calibrator.joblib",
        "model_metadata.json",
        "feature_schema.json",
    ]:
        assert (artifact_dir / artifact_name).is_file()

    data_path = REPOSITORY_ROOT / "data" / "processed" / "dataset_features.csv"
    sample = pd.read_csv(data_path, parse_dates=["Fecha"]).tail(12)
    predictions = predict_batch(sample)

    assert len(predictions) == len(sample)
    assert {
        "raw_score",
        "calibrated_probability",
        "threshold",
        "alert_flag",
    }.issubset(predictions.columns)
    assert predictions["raw_score"].between(0, 1).all()
    assert predictions["calibrated_probability"].between(0, 1).all()
    assert np.allclose(predictions["threshold"], 0.08)
    assert set(predictions["alert_flag"].unique()).issubset({0, 1})


def test_missing_feature_is_rejected() -> None:
    data_path = REPOSITORY_ROOT / "data" / "processed" / "dataset_features.csv"
    sample = pd.read_csv(data_path, nrows=2).drop(columns=["CumpStock_D7"])

    with pytest.raises(ValueError, match="missing required model features"):
        predict_batch(sample)
