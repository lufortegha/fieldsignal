"""Transparent seasonal anomaly features built from daily NASA POWER data.

Historical baselines are period totals/means for the same month/day window in
each requested year. A baseline sample is one complete year-window; incomplete
windows are omitted for that variable. Z-scores use sample standard deviation
(ddof=1), and are undefined with fewer than two baseline years or zero spread.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Sequence

import pandas as pd

RAIN_EPSILON_MM = 1e-6
STD_EPSILON = 1e-9


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def _period_dates(start: date, end: date, year: int) -> list[pd.Timestamp]:
    """Make the same month/day span in a given year; skip impossible leap days."""
    result = []
    cursor = pd.Timestamp(year, start.month, start.day)
    stop = pd.Timestamp(year, end.month, end.day)
    while cursor <= stop:
        result.append(cursor)
        cursor += pd.Timedelta(1, unit="D")
    return result


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _window_values(frame: pd.DataFrame, dates: list[pd.Timestamp], column: str) -> list[float | None]:
    values = frame.set_index("date")[column].to_dict()
    return [_number(values.get(day)) for day in dates]


def _complete(values: list[float | None]) -> bool:
    return bool(values) and all(value is not None for value in values)


def _feature(value: float | None, valid: bool, reason: str | None, unit: str,
             baseline: list[float], baseline_mean: float | None = None,
             baseline_std: float | None = None) -> dict[str, Any]:
    return {
        "value": round(value, 6) if value is not None and math.isfinite(value) else None,
        "unit": unit,
        "baseline_mean": round(baseline_mean, 6) if baseline_mean is not None else None,
        "baseline_std_sample": round(baseline_std, 6) if baseline_std is not None else None,
        "baseline_sample_size": len(baseline),
        "statistically_valid": bool(valid),
        "reason": reason,
    }


def build_environmental_features(
    daily: pd.DataFrame,
    target_start: str | date,
    target_end: str | date,
    historical_years: Sequence[int],
    *,
    recent_days: int = 7,
    wet_day_threshold_mm: float = 1.0,
) -> dict[str, Any]:
    """Return seasonal environmental features and per-feature quality metadata.

    ``daily`` is the clean DataFrame from ``fieldsignal_power.fetch_daily``. It
    must contain dates for the target and requested historical years. For each
    historical year, the target's exact month/day interval is used. A year with
    an impossible date (such as Feb 29 in a non-leap year) has no baseline
    window. Rain persistence is the fraction of the most recent ``recent_days``
    in the target window with rainfall >= ``wet_day_threshold_mm``.
    """
    start, end = _parse_date(target_start), _parse_date(target_end)
    if end < start:
        raise ValueError("target_end must be on or after target_start")
    if start.year != end.year:
        raise ValueError("target window must be within one calendar year")
    if recent_days < 1:
        raise ValueError("recent_days must be at least 1")
    if wet_day_threshold_mm < 0 or not math.isfinite(wet_day_threshold_mm):
        raise ValueError("wet_day_threshold_mm must be finite and non-negative")

    columns = {"rainfall": "PRECTOTCORR", "mean_temperature": "T2M", "maximum_temperature": "T2M_MAX"}
    required = {"date", *columns.values()}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily DataFrame is missing columns: {', '.join(sorted(missing))}")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("daily DataFrame must have unique, valid dates")
    frame = frame.sort_values("date").reset_index(drop=True)

    requested_years = list(dict.fromkeys(int(year) for year in historical_years if int(year) != start.year))
    target_dates = list(pd.date_range(start, end, freq="D"))
    baseline_windows: dict[int, list[pd.Timestamp]] = {}
    for year in requested_years:
        try:
            baseline_windows[year] = _period_dates(start, end, year)
        except ValueError:  # e.g. Feb 29 in a non-leap year
            continue

    target_complete: dict[str, bool] = {}
    target_values: dict[str, list[float | None]] = {}
    baseline_values: dict[str, list[float]] = {}
    baseline_rows: dict[str, list[dict[str, float]]] = {}
    missing_fraction: dict[str, dict[str, float]] = {}
    for label, column in columns.items():
        values = _window_values(frame, target_dates, column)
        target_values[label] = values
        target_complete[label] = _complete(values)
        samples: list[float] = []
        per_year: list[dict[str, float]] = []
        all_baseline = [value for window in baseline_windows.values()
                        for value in _window_values(frame, window, column)]
        for year, dates in baseline_windows.items():
            year_values = _window_values(frame, dates, column)
            if _complete(year_values):
                numeric = [float(value) for value in year_values]
                period_value = sum(numeric) if label == "rainfall" else sum(numeric) / len(numeric)
                samples.append(period_value)
                per_year.append({"year": float(year), "value": period_value})
        baseline_values[label] = samples
        baseline_rows[label] = per_year
        target_missing = sum(value is None for value in values) / len(target_dates)
        baseline_missing = sum(value is None for value in all_baseline) / len(all_baseline) if all_baseline else 0.0
        missing_fraction[label] = {"target": round(target_missing, 6), "historical": round(baseline_missing, 6)}

    features: dict[str, dict[str, Any]] = {}
    for label, suffix, unit in (("rainfall", "rainfall", "mm"),):
        target = target_values[label]
        total = sum(float(value) for value in target) if target_complete[label] else None
        features["accumulated_rainfall"] = _feature(
            total, target_complete[label], None if target_complete[label] else "target window contains missing rainfall",
            unit, baseline_values[label])

    def add_baseline_features(label: str, anomaly_name: str, z_name: str, unit: str,
                              target_aggregate: float | None) -> None:
        samples = baseline_values[label]
        baseline_mean = sum(samples) / len(samples) if samples else None
        baseline_std = float(pd.Series(samples).std(ddof=1)) if len(samples) >= 2 else None
        target_ok = target_aggregate is not None
        anomaly = target_aggregate - baseline_mean if target_ok and baseline_mean is not None else None
        features[anomaly_name] = _feature(
            anomaly, target_ok and baseline_mean is not None,
            None if target_ok and baseline_mean is not None else (
                "target window contains missing data" if not target_ok else "no complete historical baseline windows"),
            unit, samples, baseline_mean, baseline_std)

        if label == "rainfall":
            if baseline_mean is None:
                pct, pct_valid, pct_reason = None, False, "no complete historical baseline windows"
            elif abs(baseline_mean) <= RAIN_EPSILON_MM:
                pct, pct_valid, pct_reason = None, False, "historical rainfall baseline is zero or near zero"
            elif not target_ok:
                pct, pct_valid, pct_reason = None, False, "target window contains missing rainfall"
            else:
                pct = 100.0 * (target_aggregate - baseline_mean) / baseline_mean
                pct_valid, pct_reason = True, None
            features["rainfall_anomaly_percent"] = _feature(pct, pct_valid, pct_reason, "%", samples, baseline_mean, baseline_std)

        if baseline_std is None:
            z, z_valid, z_reason = None, False, "fewer than two complete historical baseline windows"
        elif baseline_std <= STD_EPSILON:
            z, z_valid, z_reason = None, False, "historical baseline standard deviation is zero or near zero"
        elif not target_ok:
            z, z_valid, z_reason = None, False, "target window contains missing data"
        else:
            z, z_valid, z_reason = (target_aggregate - baseline_mean) / baseline_std, True, None
        features[z_name] = _feature(z, z_valid, z_reason, "z-score", samples, baseline_mean, baseline_std)

    rainfall_total = sum(float(value) for value in target_values["rainfall"]) if target_complete["rainfall"] else None
    add_baseline_features("rainfall", "rainfall_anomaly_mm", "rainfall_z_score", "mm", rainfall_total)
    for label, anomaly_name, z_name in (
        ("mean_temperature", "mean_temperature_anomaly_c", "mean_temperature_z_score"),
        ("maximum_temperature", "maximum_temperature_anomaly_c", "maximum_temperature_z_score"),
    ):
        values = target_values[label]
        mean = sum(float(value) for value in values) / len(values) if target_complete[label] else None
        add_baseline_features(label, anomaly_name, z_name, "°C", mean)

    recent_dates = target_dates[-min(recent_days, len(target_dates)):]
    recent_rain = _window_values(frame, recent_dates, columns["rainfall"])
    persistence = (sum(value >= wet_day_threshold_mm for value in recent_rain if value is not None) / len(recent_dates)
                   if _complete(recent_rain) else None)
    features["recent_rainfall_persistence"] = _feature(
        persistence, persistence is not None,
        None if persistence is not None else "recent target days contain missing rainfall",
        "fraction of days at/above threshold", [],
    )

    # Reuse the same seasonal aggregates and sample-standard-deviation
    # convention for the unsupervised model's training vectors.
    model_dimensions = ["rainfall_z_score", "mean_temperature_z_score", "maximum_temperature_z_score"]
    model_labels = ["rainfall", "mean_temperature", "maximum_temperature"]
    target_vector = [features[name]["value"] for name in model_dimensions]
    historical_vectors = []
    if all(value is not None and features[name]["statistically_valid"]
           for name, value in zip(model_dimensions, target_vector)):
        means = {label: sum(baseline_values[label]) / len(baseline_values[label])
                 for label in model_labels if baseline_values[label]}
        deviations = {label: float(pd.Series(baseline_values[label]).std(ddof=1))
                      for label in model_labels if len(baseline_values[label]) >= 2}
        years = sorted(set.intersection(*(set(row["year"] for row in baseline_rows[label])
                                          for label in model_labels))) if all(
            baseline_rows[label] for label in model_labels) else []
        for year in years:
            values = []
            for label in model_labels:
                std = deviations.get(label)
                record = next(row for row in baseline_rows[label] if row["year"] == year)
                if std is None or std <= STD_EPSILON:
                    values = []
                    break
                values.append((record["value"] - means[label]) / std)
            if len(values) == len(model_dimensions):
                historical_vectors.append({"year": int(year), "values": [round(value, 6) for value in values]})

    return {
        "source": daily.attrs.get("source", "unknown"),
        "coordinates": daily.attrs.get("coordinates"),
        "target_window": {"start": start.isoformat(), "end": end.isoformat()},
        "historical_years": requested_years,
        "comparison": "same month/day window per historical year; incomplete year-windows omitted per variable",
        "features": features,
        "model_inputs": {
            "feature_names": model_dimensions,
            "target_vector": [round(float(value), 6) for value in target_vector] if all(value is not None for value in target_vector) else None,
            "historical_vectors": historical_vectors,
        },
        "data_quality": {
            "missing_value_fraction": missing_fraction,
            "complete_baseline_windows_by_feature": {
                key: [int(row["year"]) for row in rows] for key, rows in baseline_rows.items()
            },
            "target_window_complete_by_feature": target_complete,
            "recent_days": len(recent_dates),
            "wet_day_threshold_mm": wet_day_threshold_mm,
        },
    }
