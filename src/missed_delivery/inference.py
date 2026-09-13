"""Batch inference for the frozen Missed Delivery model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


_DEFAULT_ARTIFACT_RELATIVE_PATH = Path("models")
_DEFAULT_THRESHOLD_RELATIVE_PATH = Path("configs") / "thresholds.yaml"
_IDENTIFIER_COLUMNS = [
    "Fecha",
    "target_date",
    "Grupo_raiz",
    "customer",
    "uat",
    "destination",
]


def _find_local_project_path(relative_path: Path) -> Path:
    """Resolve a project-local default when the package is source or wheel installed."""
    current_path = Path.cwd().resolve()
    for project_candidate in (current_path, *current_path.parents):
        candidate_path = project_candidate / relative_path
        if candidate_path.exists():
            return candidate_path

    source_repository_candidate = Path(__file__).resolve().parents[2] / relative_path
    return source_repository_candidate


def _score_to_logit(score: np.ndarray) -> np.ndarray:
    """Convert XGBoost probabilities to the input used by locked Sigmoid."""
    clipped_score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped_score / (1 - clipped_score)).reshape(-1, 1)


@lru_cache(maxsize=4)
def _load_artifacts(artifact_dir: str) -> tuple[object, object, dict, dict]:
    """Load and cache the frozen pipeline, calibrator, metadata and schema."""
    artifact_path = Path(artifact_dir)
    required_paths = {
        "pipeline": artifact_path / "md_xgboost_pipeline.joblib",
        "calibrator": artifact_path / "calibrator.joblib",
        "metadata": artifact_path / "model_metadata.json",
        "schema": artifact_path / "feature_schema.json",
    }
    missing_artifacts = [
        str(path) for path in required_paths.values() if not path.is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "Missing frozen model artifacts: " + ", ".join(missing_artifacts)
        )

    pipeline = joblib.load(required_paths["pipeline"])
    calibrator = joblib.load(required_paths["calibrator"])
    metadata = json.loads(required_paths["metadata"].read_text(encoding="utf-8"))
    feature_schema = json.loads(
        required_paths["schema"].read_text(encoding="utf-8")
    )
    return pipeline, calibrator, metadata, feature_schema


def _load_threshold(config_path: Path) -> tuple[float, dict]:
    """Read the operational threshold from the external YAML configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Threshold configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        threshold_config = yaml.safe_load(config_file)
    threshold = float(threshold_config["selected_candidate"]["threshold"])
    return threshold, threshold_config


def predict_batch(
    df: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    threshold_config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score a feature-engineered batch with the frozen model configuration.

    Extra input columns are ignored. Required model features are selected in
    their training order. Identifier columns are retained when present.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    resolved_artifact_dir = Path(
        artifact_dir
        if artifact_dir is not None
        else _find_local_project_path(_DEFAULT_ARTIFACT_RELATIVE_PATH)
    ).resolve()
    resolved_threshold_config = Path(
        threshold_config_path
        if threshold_config_path is not None
        else _find_local_project_path(_DEFAULT_THRESHOLD_RELATIVE_PATH)
    ).resolve()

    pipeline, calibrator, metadata, feature_schema = _load_artifacts(
        str(resolved_artifact_dir)
    )
    expected_features = feature_schema["expected_features"]
    missing_features = [
        feature for feature in expected_features if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "Input is missing required model features: "
            + ", ".join(missing_features)
        )

    threshold, threshold_config = _load_threshold(resolved_threshold_config)
    if not np.isclose(threshold, float(metadata["operational_threshold"])):
        raise ValueError("Threshold YAML and frozen model metadata do not match")
    if threshold_config["calibration_method"].lower() != "sigmoid":
        raise ValueError("Threshold configuration does not reference Sigmoid")

    model_input = df.loc[:, expected_features]
    raw_score = pipeline.predict_proba(model_input)[:, 1]
    calibrated_probability = calibrator.predict_proba(
        _score_to_logit(raw_score)
    )[:, 1]
    alert_flag = (calibrated_probability >= threshold).astype(np.int8)

    identifiers = [column for column in _IDENTIFIER_COLUMNS if column in df.columns]
    predictions = df.loc[:, identifiers].copy()
    predictions["raw_score"] = raw_score
    predictions["calibrated_probability"] = calibrated_probability
    predictions["threshold"] = threshold
    predictions["alert_flag"] = alert_flag
    return predictions


__all__ = ["predict_batch"]
