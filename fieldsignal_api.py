"""Minimal FastAPI orchestration for the FieldSignal environmental pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
import pandas as pd

from fieldsignal_attention import infer_environmental_attention
from fieldsignal_features import build_environmental_features
from fieldsignal_power import PARAMETERS, SENTINELS, fetch_daily

logger = logging.getLogger(__name__)
VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "examples" / "demo_fixtures"
DEFAULT_MAX_DATA_AGE_DAYS = 7
FRESHNESS_STATUSES = ("CURRENT_DATA_AVAILABLE", "LATEST_AVAILABLE_DATA_DELAYED", "UPSTREAM_UNAVAILABLE")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    crop: str = Field(min_length=1, max_length=80)
    analysis_date: date
    window_days: int = Field(default=7, ge=1, le=60)
    historical_years: list[int] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_years_and_window(self):
        target_start = self.analysis_date - timedelta(days=self.window_days - 1)
        if target_start.year != self.analysis_date.year:
            raise ValueError("analysis window must stay within one calendar year")
        if self.historical_years is not None:
            if len(self.historical_years) != len(set(self.historical_years)):
                raise ValueError("historical_years cannot contain duplicates")
            if any(year >= self.analysis_date.year or year < 1981 for year in self.historical_years):
                raise ValueError("historical_years must be between 1981 and the year before analysis_date")
        return self


class Provenance(BaseModel):
    source: str
    location: dict[str, float | None]
    crop: str
    analysis_period: dict[str, str]
    historical_period: dict[str, Any]
    mode: Literal["live", "cached", "fixture"]
    cache_status: str | None = None
    retrieved_at: str | None = None
    requested_analysis_date: date | None = None
    actual_analysis_end_date: date | None = None
    latest_data_date: date | None = None
    data_age_days: int | None = None
    freshness_status: Literal[
        "CURRENT_DATA_AVAILABLE", "LATEST_AVAILABLE_DATA_DELAYED", "UPSTREAM_UNAVAILABLE"
    ] | None = None
    max_data_age_days: int | None = None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["NORMAL", "ATTENTION", "INSUFFICIENT_EVIDENCE"]
    environmental_attention_score: int | None
    confidence: Literal["HIGH", "MODERATE", "LOW"]
    model: dict[str, Any]
    statistical_baseline_score: float | None
    primary_signal: str
    signals: dict[str, Any]
    evidence: list[str]
    data_quality: dict[str, Any]
    limitations: list[str]
    provenance: Provenance
    methodology: dict[str, Any]
    environmental_features: dict[str, Any]


app = FastAPI(title="FieldSignal Environmental API", version=VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


def _analysis_wrapper(
    inferred: dict[str, Any], environmental_features: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        **inferred,
        "provenance": provenance,
        "methodology": {
            "seasonal_comparison": environmental_features.get("comparison"),
            "feature_names": environmental_features.get("model_inputs", {}).get("feature_names", []),
            "model": "IsolationForest trained on complete historical equivalent-season standardized vectors; random_state=42",
            "statistical_baseline": "root mean square of Task 2 rainfall, mean-temperature, and maximum-temperature z-scores",
            "attention_score_mapping": "70% capped statistical magnitude and 30% empirical IsolationForest anomaly rank; five equal-width bands map to scores 1–5",
            "state_policy": "INSUFFICIENT_EVIDENCE overrides; otherwise score >= 3 is ATTENTION and scores 1–2 are NORMAL",
        },
        "environmental_features": environmental_features,
    }


def _max_data_age_days() -> int:
    """Maximum allowed date lag for an explicitly disclosed delayed window."""
    configured = os.environ.get("FIELDSIGNAL_MAX_DATA_AGE_DAYS", str(DEFAULT_MAX_DATA_AGE_DAYS))
    try:
        value = int(configured)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="FIELDSIGNAL_MAX_DATA_AGE_DAYS must be a non-negative integer") from exc
    if value < 0:
        raise HTTPException(status_code=500, detail="FIELDSIGNAL_MAX_DATA_AGE_DAYS must be a non-negative integer")
    return value


def _unavailable(message: str, requested_date: date, max_age: int, *, status_code: int = 502,
                  latest_date: date | None = None, age: int | None = None,
                  freshness_status: str = "UPSTREAM_UNAVAILABLE") -> HTTPException:
    return HTTPException(status_code=status_code, detail={
        "code": "UPSTREAM_DATA_UNAVAILABLE" if freshness_status == "UPSTREAM_UNAVAILABLE" else "FRESHNESS_LIMIT_EXCEEDED",
        "message": message,
        "freshness_status": freshness_status,
        "requested_analysis_date": requested_date.isoformat(),
        "latest_data_date": latest_date.isoformat() if latest_date else None,
        "data_age_days": age,
        "max_data_age_days": max_age,
    })


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    analysis_end = request.analysis_date
    analysis_start = analysis_end - timedelta(days=request.window_days - 1)
    years = request.historical_years if request.historical_years is not None else list(
        range(max(1981, analysis_end.year - 10), analysis_end.year)
    )
    years = sorted(set(years))
    if not years:
        raise HTTPException(status_code=422, detail="at least one historical year is required")
    if any(year >= analysis_end.year or year < 1981 for year in years):
        raise HTTPException(status_code=422, detail="historical_years must be between 1981 and the year before analysis_date")

    # POWER accepts a contiguous request. The feature engine later selects only
    # exact equivalent calendar windows from those years.
    earliest_year = min(years)
    max_age = _max_data_age_days()
    retrieval_start = date(earliest_year, analysis_start.month, analysis_start.day) - timedelta(days=max_age)
    try:
        daily = fetch_daily(
            request.latitude, request.longitude, retrieval_start, analysis_end,
            cache_dir=os.environ.get("FIELDSIGNAL_CACHE_DIR", "data/cache"),
            allow_trailing_missing=True,
        )
    except Exception as exc:
        logger.warning("Environmental retrieval failed (%s)", type(exc).__name__)
        raise _unavailable(
            "NASA POWER data could not be loaded from the requested source or a valid matching cache.",
            analysis_end, max_age,
        ) from exc

    latest_value = daily.attrs.get("latest_data_date", daily["date"].max())
    try:
        latest_date = (datetime.strptime(str(latest_value), "%Y%m%d").date()
                       if len(str(latest_value)) == 8 and str(latest_value).isdigit()
                       else pd.Timestamp(latest_value).date())
    except (TypeError, ValueError):
        raise _unavailable("NASA POWER response did not identify a valid latest data date.", analysis_end, max_age)
    age_days = (analysis_end - latest_date).days
    if age_days < 0:
        raise _unavailable("NASA POWER returned data dated after the requested analysis date.", analysis_end, max_age)
    freshness_status = "CURRENT_DATA_AVAILABLE" if age_days == 0 else "LATEST_AVAILABLE_DATA_DELAYED"
    if age_days > max_age:
        raise _unavailable(
            "The latest valid data exceed the configured maximum data age; no stale window was analyzed.",
            analysis_end, max_age, status_code=503, latest_date=latest_date, age=age_days,
            freshness_status="LATEST_AVAILABLE_DATA_DELAYED",
        )

    actual_end = latest_date
    actual_start = actual_end - timedelta(days=request.window_days - 1)
    if actual_start.year != actual_end.year:
        raise _unavailable("No complete within-year analysis window can be recovered.", analysis_end, max_age,
                           latest_date=latest_date, age=age_days)
    target_dates = list(pd.date_range(actual_start, actual_end, freq="D"))
    normalized_dates = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    mask = normalized_dates.isin(target_dates)
    target_frame = daily.loc[mask]
    observed_dates = sorted(normalized_dates.loc[mask].tolist())
    def valid_value(value: Any) -> bool:
        if value is None or pd.isna(value):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and number not in SENTINELS

    if observed_dates != target_dates or any(
        not valid_value(value) for column in PARAMETERS for value in target_frame[column]
    ):
        raise _unavailable("No complete valid analysis window exists at the latest available date.", analysis_end,
                           max_age, latest_date=latest_date, age=age_days)

    environmental_features = build_environmental_features(daily, actual_start, actual_end, years)
    result = infer_environmental_attention(environmental_features)
    cache_state = daily.attrs.get("cache", "unknown")
    mode = "cached" if cache_state == "hit" else "live"
    provenance = {
        "source": daily.attrs.get("source", "unknown"),
        "location": {"latitude": request.latitude, "longitude": request.longitude},
        "crop": request.crop,
        "analysis_period": {"start": actual_start.isoformat(), "end": actual_end.isoformat()},
        "historical_period": {
            "years_requested": years,
            "calendar_window": {
                "start_month_day": actual_start.strftime("%m-%d"),
                "end_month_day": actual_end.strftime("%m-%d"),
            },
            "comparison": "same calendar window in each requested year",
        },
        "mode": mode,
        "cache_status": cache_state,
        "retrieved_at": daily.attrs.get("retrieved_at"),
        "requested_analysis_date": analysis_end.isoformat(),
        "actual_analysis_end_date": actual_end.isoformat(),
        "latest_data_date": latest_date.isoformat(),
        "data_age_days": age_days,
        "freshness_status": freshness_status,
        "max_data_age_days": max_age,
    }
    return _analysis_wrapper(result, environmental_features, provenance)


@app.get("/demo/{scenario}", response_model=AnalyzeResponse)
def demo(scenario: str) -> dict[str, Any]:
    if scenario not in {"normal", "attention", "insufficient"}:
        raise HTTPException(status_code=404, detail="scenario must be normal, attention, or insufficient")
    path = FIXTURE_DIR / f"{scenario}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Demo fixture unavailable: %s", scenario)
        raise HTTPException(status_code=500, detail="local demo fixture is unavailable or invalid") from exc
    return payload
