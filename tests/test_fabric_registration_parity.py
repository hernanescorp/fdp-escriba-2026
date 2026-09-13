"""Parity checks for the corrected Fabric registration wrapper."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from missed_delivery.inference import predict_batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "notebooks"
    / "fabric"
    / "07_fabric_model_registration.ipynb"
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cell_source(cell_index: int) -> str:
    return "".join(_notebook()["cells"][cell_index]["source"])


def test_output_example_uses_logit_before_sigmoid_calibration() -> None:
    """The signature example must follow the same chain as real inference."""
    source = _cell_source(15)

    assert "score_to_logit(raw_score)" in source
    assert "np.clip" in source
    assert "raw_score.reshape(-1, 1)" not in source


def test_fabric_wrapper_matches_canonical_predict_batch(monkeypatch) -> None:
    """Execute Notebook 07's wrapper and compare it row-for-row locally."""

    class PythonModel:
        """Minimal stand-in for the MLflow base class used in the notebook."""

    mlflow_stub = types.ModuleType("mlflow")
    mlflow_stub.pyfunc = SimpleNamespace(PythonModel=PythonModel)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    namespace: dict[str, object] = {}
    exec(_cell_source(19), namespace)
    wrapper_class = namespace["MissedDeliveryModel"]

    artifact_dir = REPOSITORY_ROOT / "models"
    context = SimpleNamespace(
        artifacts={
            "pipeline": str(artifact_dir / "md_xgboost_pipeline.joblib"),
            "calibrator": str(artifact_dir / "calibrator.joblib"),
        }
    )
    wrapper = wrapper_class()
    wrapper.load_context(context)

    sample = pd.read_csv(
        REPOSITORY_ROOT / "data" / "processed" / "dataset_features.csv",
        parse_dates=["Fecha"],
    ).tail(64)
    schema = json.loads(
        (artifact_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    model_input = sample.loc[:, schema["expected_features"]]

    expected = predict_batch(sample).loc[
        :, ["raw_score", "calibrated_probability", "threshold", "alert_flag"]
    ]
    actual = wrapper.predict(None, model_input)

    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(actual["raw_score"], expected["raw_score"])
    np.testing.assert_allclose(
        actual["calibrated_probability"],
        expected["calibrated_probability"],
    )
    np.testing.assert_allclose(actual["threshold"], expected["threshold"])
    np.testing.assert_array_equal(actual["alert_flag"], expected["alert_flag"])

