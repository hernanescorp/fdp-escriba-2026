"""Minimal data and temporal validation for Missed Delivery scoring."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


MINIMUM_SCORING_COLUMNS = (
    "Fecha",
    "Grupo_raiz",
    "customer",
    "uat",
    "destination",
)

FORBIDDEN_PREDICTOR_COLUMNS = frozenset(
    {
        # Target and post-event outcomes.
        "missed_delivery",
        "lost",
        "delivered",
        "requested",
        "loss_rate",
        "complete_failure",
        "high_volume_md",
        # Same-day operational values excluded in Notebook 04.
        "FabricacionSemana",
        "RitmoSemana",
        "CumpFabRit",
        "ExpedidasSemana",
        "Demanda",
        "CumpExpDem",
        "StockFin",
        "StockReal",
        "CumpStock",
        "AcumFab",
        "AcumRitmo",
        "PorcenCump",
        "AcumEnvio",
        "AcumDemanda",
        "PorcenCumpto",
        "EntregasPrev",
        "DemandActual",
        "CumpTotal",
        # Same-day derived values.
        "production_gap",
        "shipment_demand_gap",
        "stock_gap",
        "accumulated_production_gap",
        "accumulated_delivery_gap",
        "planned_delivery_gap",
    }
)


def validate_schema(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] = MINIMUM_SCORING_COLUMNS,
    check_unique_key: bool = True,
) -> None:
    """Validate minimum columns and the Fecha + Grupo_raiz business key."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if check_unique_key and {"Fecha", "Grupo_raiz"}.issubset(df.columns):
        duplicate_mask = df.duplicated(["Fecha", "Grupo_raiz"], keep=False)
        if duplicate_mask.any():
            raise ValueError(
                "Duplicate Fecha + Grupo_raiz keys found: "
                f"{int(duplicate_mask.sum())} rows"
            )


def validate_no_leakage(columns: Iterable[str]) -> None:
    """Reject target, post-event, same-day, and same-day-derived predictors."""
    column_list = list(columns)
    forbidden_by_lower = {
        column.lower(): column for column in FORBIDDEN_PREDICTOR_COLUMNS
    }
    leakage_columns = sorted(
        {
            forbidden_by_lower[column.lower()]
            for column in column_list
            if column.lower() in forbidden_by_lower
        }
    )
    if leakage_columns:
        raise ValueError(
            "Forbidden leakage or same-day predictor columns: "
            + ", ".join(leakage_columns)
        )


def validate_temporal_features(
    df: pd.DataFrame,
    *,
    target_date_col: str = "Fecha",
    lag_days: Sequence[int] = (7, 14, 21, 28),
    history_date_template: str = "_history_date_D{lag}",
) -> None:
    """Ensure every non-missing lag timestamp is exact and strictly historical."""
    if target_date_col not in df.columns:
        raise ValueError(f"Missing target date column: {target_date_col}")

    target_dates = pd.to_datetime(df[target_date_col], errors="coerce")
    if target_dates.isna().any():
        raise ValueError(f"Invalid dates found in {target_date_col}")

    for lag in lag_days:
        history_date_col = history_date_template.format(lag=lag)
        if history_date_col not in df.columns:
            raise ValueError(f"Missing history audit column: {history_date_col}")

        history_dates = pd.to_datetime(df[history_date_col], errors="coerce")
        available = history_dates.notna()
        if (history_dates.loc[available] >= target_dates.loc[available]).any():
            raise ValueError(f"{history_date_col} contains future or same-day data")

        expected_dates = target_dates - pd.to_timedelta(lag, unit="D")
        if not history_dates.loc[available].equals(expected_dates.loc[available]):
            raise ValueError(
                f"{history_date_col} must equal target date minus exactly {lag} days"
            )


__all__ = [
    "FORBIDDEN_PREDICTOR_COLUMNS",
    "MINIMUM_SCORING_COLUMNS",
    "validate_no_leakage",
    "validate_schema",
    "validate_temporal_features",
]
