"""Tests for leakage-safe predictor selection."""

from __future__ import annotations

import pytest

from missed_delivery.validation import validate_no_leakage


def test_historical_predictors_pass() -> None:
    validate_no_leakage(
        ["Grupo_raiz", "CumpStock_D7", "stock_gap_D14", "week_of_year"]
    )


@pytest.mark.parametrize(
    "forbidden_column",
    [
        "lost",
        "delivered",
        "requested",
        "loss_rate",
        "complete_failure",
        "high_volume_md",
        "missed_delivery",
        "CumpStock",
        "stock_gap",
    ],
)
def test_forbidden_predictor_is_rejected(forbidden_column: str) -> None:
    with pytest.raises(ValueError, match="Forbidden leakage"):
        validate_no_leakage(["Grupo_raiz", forbidden_column])
