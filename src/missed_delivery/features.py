"""Temporal feature construction reproduced from Notebook 04."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .validation import (
    validate_no_leakage,
    validate_schema,
    validate_temporal_features,
)


LAG_DAYS = (7, 14, 21, 28)

CONTEXT_COLUMNS = ("Grupo_raiz", "customer", "uat", "destination")

OPERATIONAL_COLUMNS = (
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
)

GAP_DEFINITIONS = {
    "production_gap": ("FabricacionSemana", "RitmoSemana"),
    "shipment_demand_gap": ("ExpedidasSemana", "Demanda"),
    "stock_gap": ("StockReal", "StockFin"),
    "accumulated_production_gap": ("AcumFab", "AcumRitmo"),
    "accumulated_delivery_gap": ("AcumEnvio", "AcumDemanda"),
    "planned_delivery_gap": ("EntregasPrev", "DemandActual"),
}

TREND_BASE_COLUMNS = (
    "CumpStock",
    "stock_gap",
    "CumpExpDem",
    "shipment_demand_gap",
    "CumpFabRit",
    "production_gap",
    "PorcenCumpto",
    "accumulated_delivery_gap",
    "PorcenCump",
    "accumulated_production_gap",
    "CumpTotal",
    "planned_delivery_gap",
    "Demanda",
    "DemandActual",
    "EntregasPrev",
)

AVAILABILITY_FEATURE_COLUMNS = tuple(
    [f"history_available_D{lag}" for lag in LAG_DAYS]
    + ["available_lag_count", "history_records_before_D"]
)

LAG_FEATURE_COLUMNS = tuple(
    f"{column}_D{lag}" for lag in LAG_DAYS for column in OPERATIONAL_COLUMNS
)

HISTORICAL_GAP_COLUMNS = tuple(
    f"{gap}_D{lag}" for lag in LAG_DAYS for gap in GAP_DEFINITIONS
)

TREND_COLUMNS = tuple(
    f"{column}_{delta}"
    for column in TREND_BASE_COLUMNS
    for delta in ("delta_D7_D14", "delta_D14_D28", "delta_D7_D28")
)

CALENDAR_COLUMNS = ("month", "day_of_week", "week_of_year")

FINAL_FEATURE_COLUMNS = tuple(
    CONTEXT_COLUMNS
    + AVAILABILITY_FEATURE_COLUMNS
    + LAG_FEATURE_COLUMNS
    + HISTORICAL_GAP_COLUMNS
    + TREND_COLUMNS
    + CALENDAR_COLUMNS
)

assert len(FINAL_FEATURE_COLUMNS) == 154


def _validate_history_schema(history: pd.DataFrame) -> None:
    required_history_columns = ["Fecha", "Grupo_raiz", *OPERATIONAL_COLUMNS]
    missing_columns = [
        column for column in required_history_columns if column not in history
    ]
    if missing_columns:
        raise ValueError(
            "Operational history is missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_mask = history.duplicated(["Fecha", "Grupo_raiz"], keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "Operational history contains duplicate Fecha + Grupo_raiz keys: "
            f"{int(duplicate_mask.sum())} rows"
        )


def _add_history_record_count(
    features: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    target_temp = features[["_target_row_id", "Fecha", "Grupo_raiz"]].copy()
    target_temp["_lookup_date"] = (
        target_temp["Fecha"].astype("datetime64[ns]")
        - pd.Timedelta(nanoseconds=1)
    )

    history_temp = (
        history[["Fecha", "Grupo_raiz"]]
        .drop_duplicates()
        .sort_values(["Grupo_raiz", "Fecha"])
        .copy()
    )
    history_temp["history_records_before_D"] = (
        history_temp.groupby("Grupo_raiz").cumcount() + 1
    )
    history_temp = history_temp.rename(columns={"Fecha": "_history_date"})

    target_temp = target_temp.sort_values(["_lookup_date", "Grupo_raiz"])
    history_temp = history_temp.sort_values(["_history_date", "Grupo_raiz"])
    support_match = pd.merge_asof(
        target_temp,
        history_temp,
        left_on="_lookup_date",
        right_on="_history_date",
        by="Grupo_raiz",
        direction="backward",
        allow_exact_matches=True,
    )
    support_count = (
        support_match.set_index("_target_row_id")["history_records_before_D"]
        .fillna(0)
        .astype(int)
    )

    result = features.copy()
    result["history_records_before_D"] = result["_target_row_id"].map(
        support_count
    ).astype(int)
    return result


def build_temporal_features(
    target: pd.DataFrame,
    history: pd.DataFrame,
    *,
    lag_days: Sequence[int] = LAG_DAYS,
) -> pd.DataFrame:
    """Build the exact leakage-safe 154-feature contract used by the model.

    A lag is joined only when history contains exactly ``Fecha - lag`` for the
    same ``Grupo_raiz``. Missing exact dates remain missing; no previous-value
    or nearest-date substitution is performed.
    """
    if tuple(lag_days) != LAG_DAYS:
        raise ValueError("The frozen model requires exact lags D-7/D-14/D-21/D-28")

    validate_schema(target)
    _validate_history_schema(history)

    target_work = target.copy()
    history_work = history.copy()
    target_work["Fecha"] = pd.to_datetime(target_work["Fecha"], errors="coerce")
    history_work["Fecha"] = pd.to_datetime(history_work["Fecha"], errors="coerce")
    if target_work["Fecha"].isna().any() or history_work["Fecha"].isna().any():
        raise ValueError("Fecha contains invalid dates")

    target_work["Fecha"] = target_work["Fecha"].astype("datetime64[ns]")
    history_work["Fecha"] = history_work["Fecha"].astype("datetime64[ns]")
    target_work["_target_row_id"] = np.arange(len(target_work))

    history_lag_base = history_work[
        ["Fecha", "Grupo_raiz", *OPERATIONAL_COLUMNS]
    ].copy()
    features = target_work.copy()

    for lag in LAG_DAYS:
        history_date_column = f"_history_date_D{lag}"
        features[history_date_column] = features["Fecha"] - pd.Timedelta(days=lag)
        history_lag = history_lag_base.rename(
            columns={
                "Fecha": history_date_column,
                **{
                    column: f"{column}_D{lag}"
                    for column in OPERATIONAL_COLUMNS
                },
            }
        )
        features = features.merge(
            history_lag,
            on=[history_date_column, "Grupo_raiz"],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    validate_temporal_features(features)

    for lag in LAG_DAYS:
        lag_columns = [f"{column}_D{lag}" for column in OPERATIONAL_COLUMNS]
        features[f"history_available_D{lag}"] = (
            features[lag_columns].notna().any(axis=1).astype(int)
        )
    features["available_lag_count"] = features[
        [f"history_available_D{lag}" for lag in LAG_DAYS]
    ].sum(axis=1)

    features = _add_history_record_count(features, history_work)

    for lag in LAG_DAYS:
        for gap_name, (minuend, subtrahend) in GAP_DEFINITIONS.items():
            features[f"{gap_name}_D{lag}"] = (
                features[f"{minuend}_D{lag}"]
                - features[f"{subtrahend}_D{lag}"]
            )

    for column in TREND_BASE_COLUMNS:
        features[f"{column}_delta_D7_D14"] = (
            features[f"{column}_D7"] - features[f"{column}_D14"]
        )
        features[f"{column}_delta_D14_D28"] = (
            features[f"{column}_D14"] - features[f"{column}_D28"]
        )
        features[f"{column}_delta_D7_D28"] = (
            features[f"{column}_D7"] - features[f"{column}_D28"]
        )

    features["month"] = features["Fecha"].dt.month.astype(int)
    features["day_of_week"] = features["Fecha"].dt.dayofweek.astype(int)
    features["week_of_year"] = (
        features["Fecha"].dt.isocalendar().week.astype(int)
    )

    validate_no_leakage(FINAL_FEATURE_COLUMNS)
    output_columns = ["Fecha", *FINAL_FEATURE_COLUMNS]
    if "missed_delivery" in target_work.columns:
        # Notebook 04 appends calendar columns after creating model_df, so its
        # exported target appears immediately before the three calendars.
        output_columns = [
            "Fecha",
            *FINAL_FEATURE_COLUMNS[: -len(CALENDAR_COLUMNS)],
            "missed_delivery",
            *CALENDAR_COLUMNS,
        ]

    # Notebook 04 exports the analytical dataset ordered by group and date.
    # Keep that deterministic ordering so this production function reproduces
    # the notebook output exactly.
    result = (
        features.sort_values(["Grupo_raiz", "Fecha"])
        .loc[:, output_columns]
        .reset_index(drop=True)
    )
    return result


__all__ = [
    "FINAL_FEATURE_COLUMNS",
    "LAG_DAYS",
    "OPERATIONAL_COLUMNS",
    "build_temporal_features",
]
