"""Inference must tolerate categories not observed during model fitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from missed_delivery.inference import predict_batch


def test_unseen_context_categories_do_not_break_inference() -> None:
    data_path = REPOSITORY_ROOT / "data" / "processed" / "dataset_features.csv"
    sample = pd.read_csv(data_path, parse_dates=["Fecha"], nrows=3)
    sample.loc[:, "Grupo_raiz"] = "NEW_GROUP"
    sample.loc[:, "customer"] = "NEW_CUSTOMER"
    sample.loc[:, "uat"] = "NEW_UAT"
    sample.loc[:, "destination"] = "NEW_DESTINATION"

    predictions = predict_batch(sample)

    assert len(predictions) == len(sample)
    assert predictions["raw_score"].between(0, 1).all()
    assert predictions["calibrated_probability"].between(0, 1).all()
    assert set(predictions["alert_flag"].unique()).issubset({0, 1})
