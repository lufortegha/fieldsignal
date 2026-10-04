"""Minimal FastAPI orchestration for the FieldSignal environmental pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fieldsignal_attention import infer_environmental_attention
from fieldsignal_features import build_environmental_features
from fieldsignal_power import fetch_daily

logger = logging.getLogger(__name__)
VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "examples" / "demo_fixtures"


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
    start_day = min(analysis_start.day, 28) if analysis_start.month == 2 else analysis_start.day
    retrieval_start = date(earliest_year, analysis_start.month, start_day)
    try:
        daily = fetch_daily(
            request.latitude, request.longitude, retrieval_start, analysis_end,
            cache_dir=os.environ.get("FIELDSIGNAL_CACHE_DIR", "data/cache"),
        )
    except Exception as exc:
        logger.warning("Environmental retrieval failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail={"code": "UPSTREAM_DATA_UNAVAILABLE", "message": "NASA POWER data could not be loaded from the requested source or a valid matching cache."},
        ) from exc

    environmental_features = build_environmental_features(daily, analysis_start, analysis_end, years)
    result = infer_environmental_attention(environmental_features)
    cache_state = daily.attrs.get("cache", "unknown")
    mode = "cached" if cache_state == "hit" else "live"
    provenance = {
        "source": daily.attrs.get("source", "unknown"),
        "location": {"latitude": request.latitude, "longitude": request.longitude},
        "crop": request.crop,
        "analysis_period": {"start": analysis_start.isoformat(), "end": analysis_end.isoformat()},
        "historical_period": {
            "years_requested": years,
            "calendar_window": {
                "start_month_day": analysis_start.strftime("%m-%d"),
                "end_month_day": analysis_end.strftime("%m-%d"),
            },
            "comparison": "same calendar window in each requested year",
        },
        "mode": mode,
        "cache_status": cache_state,
        "retrieved_at": daily.attrs.get("retrieved_at"),
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
