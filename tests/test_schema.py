"""Tests for the minimum scoring schema."""

from __future__ import annotations

import pandas as pd
import pytest

from missed_delivery.validation import validate_schema


def _valid_scoring_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fecha": pd.to_datetime(["2026-09-01", "2026-09-02"]),
            "Grupo_raiz": ["G1", "G1"],
            "customer": ["C1", "C1"],
            "uat": ["U1", "U1"],
            "destination": ["D1", "D1"],
        }
    )


def test_valid_schema_passes() -> None:
    validate_schema(_valid_scoring_rows())


def test_missing_required_column_is_rejected() -> None:
    invalid = _valid_scoring_rows().drop(columns=["destination"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(invalid)


def test_duplicate_business_key_is_rejected() -> None:
    invalid = pd.concat(
        [_valid_scoring_rows().iloc[[0]], _valid_scoring_rows().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match=r"Duplicate Fecha \+ Grupo_raiz"):
        validate_schema(invalid)
