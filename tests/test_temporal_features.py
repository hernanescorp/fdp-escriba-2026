"""Controlled tests for exact temporal feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missed_delivery.features import (
    OPERATIONAL_COLUMNS,
    build_temporal_features,
)
from missed_delivery.validation import validate_temporal_features


def _history_row(date: str, value: float) -> dict[str, object]:
    row: dict[str, object] = {"Fecha": date, "Grupo_raiz": "G1"}
    row.update({column: value for column in OPERATIONAL_COLUMNS})
    return row


def test_exact_lags_are_used_and_nearby_dates_are_not_substituted() -> None:
    target = pd.DataFrame(
        {
            "Fecha": ["2026-02-01"],
            "Grupo_raiz": ["G1"],
            "customer": ["C1"],
            "uat": ["U1"],
            "destination": ["D1"],
        }
    )
    history = pd.DataFrame(
        [
            _history_row("2026-01-25", 7.0),
            _history_row("2026-01-18", 14.0),
            _history_row("2026-01-11", 21.0),
            _history_row("2026-01-04", 28.0),
            _history_row("2026-01-26", 999.0),
        ]
    )

    features = build_temporal_features(target, history)

    assert features.loc[0, "FabricacionSemana_D7"] == 7.0
    assert features.loc[0, "FabricacionSemana_D14"] == 14.0
    assert features.loc[0, "FabricacionSemana_D21"] == 21.0
    assert features.loc[0, "FabricacionSemana_D28"] == 28.0
    assert features.loc[0, "CumpStock_delta_D7_D14"] == -7.0
    assert features.loc[0, "available_lag_count"] == 4


def test_missing_exact_lag_remains_missing() -> None:
    target = pd.DataFrame(
        {
            "Fecha": ["2026-02-01"],
            "Grupo_raiz": ["G1"],
            "customer": ["C1"],
            "uat": ["U1"],
            "destination": ["D1"],
        }
    )
    history = pd.DataFrame(
        [
            _history_row("2026-01-25", 7.0),
            _history_row("2026-01-19", 999.0),
            _history_row("2026-01-11", 21.0),
            _history_row("2026-01-04", 28.0),
        ]
    )

    features = build_temporal_features(target, history)

    assert np.isnan(features.loc[0, "FabricacionSemana_D14"])
    assert features.loc[0, "history_available_D14"] == 0
    assert features.loc[0, "available_lag_count"] == 3


def test_future_or_non_exact_history_timestamp_is_rejected() -> None:
    audit = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-02-01"]),
            "_history_date_D7": pd.to_datetime(["2026-02-02"]),
            "_history_date_D14": pd.to_datetime(["2026-01-18"]),
            "_history_date_D21": pd.to_datetime(["2026-01-11"]),
            "_history_date_D28": pd.to_datetime(["2026-01-04"]),
        }
    )
    with pytest.raises(ValueError, match="future or same-day"):
        validate_temporal_features(audit)
