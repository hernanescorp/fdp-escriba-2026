"""Time-aware backtesting utilities for binary classification models.

The module intentionally keeps model construction outside the backtest. This
allows callers to supply a complete scikit-learn-compatible pipeline containing
both preprocessing and the estimator, preventing preprocessing leakage between
the training and validation periods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class TemporalFold:
    """Definition of one expanding-window temporal validation fold.

    Boundaries use half-open intervals. Training includes observations from
    ``train_start`` (when supplied) up to, but excluding, ``train_end``.
    Validation includes observations from ``validation_start`` up to, but
    excluding, ``validation_end``.
    """

    name: str
    train_end: pd.Timestamp | str
    validation_start: pd.Timestamp | str
    validation_end: pd.Timestamp | str
    train_start: pd.Timestamp | str | None = None

    def __post_init__(self) -> None:
        """Normalize timestamps and reject overlapping or empty periods."""
        train_start = (
            pd.Timestamp(self.train_start) if self.train_start is not None else None
        )
        train_end = pd.Timestamp(self.train_end)
        validation_start = pd.Timestamp(self.validation_start)
        validation_end = pd.Timestamp(self.validation_end)

        if train_start is not None and train_start >= train_end:
            raise ValueError("train_start must be earlier than train_end")
        if train_end > validation_start:
            raise ValueError("training and validation periods must not overlap")
        if validation_start >= validation_end:
            raise ValueError("validation_start must be earlier than validation_end")

        object.__setattr__(self, "train_start", train_start)
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "validation_start", validation_start)
        object.__setattr__(self, "validation_end", validation_end)


def split_temporal_fold(
    data: pd.DataFrame,
    fold: TemporalFold,
    *,
    date_col: str = "Fecha",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent training and validation frames for ``fold``."""
    if date_col not in data.columns:
        raise KeyError(f"Missing date column: {date_col}")

    dates = pd.to_datetime(data[date_col], errors="coerce")
    if dates.isna().any():
        invalid_rows = int(dates.isna().sum())
        raise ValueError(f"{date_col} contains {invalid_rows} invalid dates")

    train_mask = dates < fold.train_end
    if fold.train_start is not None:
        train_mask &= dates >= fold.train_start

    validation_mask = (
        (dates >= fold.validation_start)
        & (dates < fold.validation_end)
    )

    train = data.loc[train_mask].copy()
    validation = data.loc[validation_mask].copy()

    if train.empty:
        raise ValueError(f"Fold {fold.name!r} has no training observations")
    if validation.empty:
        raise ValueError(f"Fold {fold.name!r} has no validation observations")
    if train[date_col].max() >= validation[date_col].min():
        raise ValueError(f"Fold {fold.name!r} leaks validation time into training")

    return train, validation


def _validate_binary_target(target: pd.Series, fold_name: str, period: str) -> None:
    """Ensure both binary classes are represented in a fold period."""
    values = set(target.dropna().unique())
    if values != {0, 1}:
        raise ValueError(
            f"Fold {fold_name!r} {period} target must contain both 0 and 1; "
            f"found {sorted(values)}"
        )


def run_temporal_backtest(
    data: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    folds: Iterable[TemporalFold],
    estimator_factory: Callable[[], object],
    threshold: float = 0.5,
    date_col: str = "Fecha",
) -> pd.DataFrame:
    """Fit and evaluate a fresh probabilistic classifier in every fold.

    Parameters
    ----------
    data:
        Dataset containing dates, features and the binary target.
    feature_cols:
        Predictor columns. Post-event variables must be excluded by the caller.
    target_col:
        Binary target column containing zero and one.
    folds:
        Temporal fold definitions. Each fold receives a newly constructed model.
    estimator_factory:
        Zero-argument callable returning an unfitted estimator or complete
        preprocessing/model pipeline with ``fit`` and ``predict_proba`` methods.
    threshold:
        Fixed operational threshold evaluated across all folds. Threshold
        selection must be performed separately to avoid optimistic estimates.
    date_col:
        Event-date column used for chronological splitting.

    Returns
    -------
    pandas.DataFrame
        One record per fold with imbalance-aware and operational metrics.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    required_cols = {date_col, target_col, *feature_cols}
    missing_cols = sorted(required_cols.difference(data.columns))
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one predictor")

    fold_list = list(folds)
    if not fold_list:
        raise ValueError("At least one temporal fold is required")

    records: list[dict[str, object]] = []

    for fold in fold_list:
        train, validation = split_temporal_fold(data, fold, date_col=date_col)

        y_train = train[target_col]
        y_validation = validation[target_col]
        _validate_binary_target(y_train, fold.name, "training")
        _validate_binary_target(y_validation, fold.name, "validation")

        estimator = estimator_factory()
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict_proba"):
            raise TypeError(
                "estimator_factory must return an estimator with fit and "
                "predict_proba methods"
            )

        estimator.fit(train[list(feature_cols)], y_train)
        probabilities = np.asarray(
            estimator.predict_proba(validation[list(feature_cols)])
        )[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        records.append(
            {
                "fold": fold.name,
                "train_start": train[date_col].min(),
                "train_end": train[date_col].max(),
                "validation_start": validation[date_col].min(),
                "validation_end": validation[date_col].max(),
                "n_train": len(train),
                "n_validation": len(validation),
                "train_md": int(y_train.sum()),
                "validation_md": int(y_validation.sum()),
                "validation_md_rate": float(y_validation.mean()),
                "threshold": threshold,
                "precision": precision_score(
                    y_validation, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_validation, predictions, zero_division=0
                ),
                "f1": f1_score(y_validation, predictions, zero_division=0),
                "pr_auc": average_precision_score(y_validation, probabilities),
                "roc_auc": roc_auc_score(y_validation, probabilities),
                "alert_rate": float(predictions.mean()),
                "n_alerts": int(predictions.sum()),
                "md_detected": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )

    return pd.DataFrame.from_records(records)
