"""Small, reproducible environmental anomaly and attention engine.

IsolationForest is unsupervised: historical matched-season windows are its
only training rows. Scores describe environmental unusualness, not crop impact.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except ImportError as exc:  # keep module import errors readable for new users
    raise ImportError("Install project dependencies with `python -m pip install -r requirements.txt`.") from exc


MODEL_FEATURES = ["rainfall_z_score", "mean_temperature_z_score", "maximum_temperature_z_score"]
MIN_BASELINE_YEARS = 5
RANDOM_STATE = 42
LIMITATIONS = [
    "Environmental anomalies do not establish the cause of crop symptoms.",
    "This assessment is not a prediction of crop damage or yield loss.",
]


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quality_gate(features: dict[str, Any]) -> tuple[list[str], int, float]:
    reasons: list[str] = []
    model_inputs = features.get("model_inputs") or {}
    training = model_inputs.get("historical_vectors") or []
    if len(training) < MIN_BASELINE_YEARS:
        reasons.append(f"fewer than {MIN_BASELINE_YEARS} complete historical model windows")
    target = model_inputs.get("target_vector")
    if not isinstance(target, list) or len(target) != len(MODEL_FEATURES) or any(_finite_number(v) is None for v in target):
        reasons.append("one or more target standardized features are unavailable")

    feature_map = features.get("features") or {}
    for name in MODEL_FEATURES:
        entry = feature_map.get(name) or {}
        if not entry.get("statistically_valid") or _finite_number(entry.get("value")) is None:
            reasons.append(f"{name} is invalid")

    quality = features.get("data_quality") or {}
    complete = quality.get("target_window_complete_by_feature") or {}
    for key in ("rainfall", "mean_temperature", "maximum_temperature"):
        if complete.get(key) is False:
            reasons.append(f"target {key} window is incomplete")
    missing = quality.get("missing_value_fraction") or {}
    fractions = []
    for key in ("rainfall", "mean_temperature", "maximum_temperature"):
        pair = missing.get(key) or {}
        for label in ("target", "historical"):
            fraction = _finite_number(pair.get(label))
            if fraction is not None:
                fractions.append(fraction)
            elif pair:
                reasons.append(f"{key} {label} missingness metadata is invalid")
    worst_missing = max(fractions, default=1.0)
    if worst_missing > 0.25:
        reasons.append("missing-value fraction exceeds 25%")
    return list(dict.fromkeys(reasons)), len(training), worst_missing


def _explanations(features: dict[str, Any], z_values: list[float]) -> tuple[str, dict[str, Any], list[str]]:
    items = []
    feature_map = features["features"]
    for name, label in zip(MODEL_FEATURES, ("rainfall", "mean_temperature", "maximum_temperature")):
        entry = feature_map.get(name, {})
        z = _finite_number(entry.get("value"))
        anomaly = _finite_number(feature_map.get({
            "rainfall_z_score": "rainfall_anomaly_mm",
            "mean_temperature_z_score": "mean_temperature_anomaly_c",
            "maximum_temperature_z_score": "maximum_temperature_anomaly_c",
        }[name], {}).get("value"))
        items.append({"feature": name, "label": label, "z_score": z, "anomaly": anomaly,
                      "unit": "mm" if label == "rainfall" else "°C"})
    items.sort(key=lambda item: abs(item["z_score"] or 0), reverse=True)
    evidence = []
    rain_pct = _finite_number(feature_map.get("rainfall_anomaly_percent", {}).get("value"))
    if rain_pct is not None and abs(rain_pct) >= 5:
        direction = "below" if rain_pct < 0 else "above"
        evidence.append(f"rainfall {abs(rain_pct):.1f}% {direction} the seasonal baseline")
    for item in items:
        if abs(item["z_score"] or 0) < 0.5:
            continue
        direction = "below" if item["z_score"] < 0 else "above"
        if item["label"] == "rainfall":
            continue  # percentage anomaly above is clearer for rainfall
        if item["anomaly"] is not None:
            evidence.append(f"{item['label'].replace('_', ' ')} {abs(item['anomaly']):.1f}°C {direction} seasonal baseline")
        else:
            evidence.append(f"{item['label'].replace('_', ' ')} is unusually {('low' if item['z_score'] < 0 else 'high')} (z={item['z_score']:.2f})")

    rain_z = z_values[0]
    mean_z, max_z = z_values[1], z_values[2]
    if rain_z <= -1 and max(mean_z, max_z) >= 1:
        primary = "DRY_WARM_ANOMALY"
    elif rain_z <= -1:
        primary = "DRY_ANOMALY"
    elif rain_z >= 1:
        primary = "WET_ANOMALY"
    elif max(mean_z, max_z) >= 1:
        primary = "WARM_ANOMALY"
    elif min(mean_z, max_z) <= -1:
        primary = "COOL_ANOMALY"
    else:
        primary = "NO_STRONG_SINGLE_SIGNAL"
    signals = {
        item["feature"]: {"z_score": item["z_score"], "anomaly": item["anomaly"], "unit": item["unit"]}
        for item in items
    }
    return primary, signals, evidence[:3]


def infer_environmental_attention(features: dict[str, Any]) -> dict[str, Any]:
    """Score Task 2 output without recomputing any target feature values.

    Statistical magnitude is RMS of the three Task 2 z-scores. The score index
    combines capped RMS evidence (70%) and the target's empirical lower-tail
    IsolationForest rank (30%). Five equal-width bands map that index to scores
    1..5. IsolationForest uses 100 trees, ``max_samples='auto'``,
    ``contamination='auto'``, one CPU worker, and fixed seed 42.
    """
    reasons, n_baseline, missing_fraction = _quality_gate(features)
    model_inputs = features.get("model_inputs") or {}
    training_rows = model_inputs.get("historical_vectors") or []
    target = model_inputs.get("target_vector")
    feature_names = model_inputs.get("feature_names")
    if feature_names != MODEL_FEATURES:
        reasons.append("Task 2 model feature order is missing or unsupported")

    z_values = [_finite_number(value) for value in (target or [])]
    if len(z_values) != 3 or any(value is None for value in z_values):
        z_values = [0.0, 0.0, 0.0]
        reasons.append("target vector does not contain three finite values")
    z_values = [float(value) for value in z_values]

    stat_score = math.sqrt(sum(value * value for value in z_values) / len(z_values))
    statistical_valid = not any(
        not (features.get("features", {}).get(name) or {}).get("statistically_valid")
        for name in MODEL_FEATURES
    )

    raw_model_score: float | None = None
    forest_rank: float | None = None
    if not reasons:
        matrix = np.asarray([row.get("values", []) for row in training_rows], dtype=float)
        target_vector = np.asarray(target, dtype=float).reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(MODEL_FEATURES) or not np.isfinite(matrix).all():
            reasons.append("historical model vectors are malformed or non-finite")
        else:
            model = IsolationForest(
                n_estimators=100, max_samples="auto", contamination="auto",
                random_state=RANDOM_STATE, n_jobs=1,
            )
            model.fit(matrix)
            raw_model_score = float(model.score_samples(target_vector)[0])
            historical_scores = model.score_samples(matrix)
            less = int(np.sum(historical_scores < raw_model_score))
            ties = int(np.sum(historical_scores == raw_model_score))
            empirical_cdf = (less + 0.5 * ties) / len(historical_scores)
            forest_rank = 1.0 - empirical_cdf

    status = "INSUFFICIENT_EVIDENCE"
    attention_score: int | None = None
    confidence = "LOW"
    primary_signal, signals, evidence = _explanations(features, z_values)
    if not reasons and raw_model_score is not None and forest_rank is not None:
        normalized_stat = min(stat_score / 3.0, 1.0)
        combined_index = 0.70 * normalized_stat + 0.30 * forest_rank
        attention_score = min(5, int(combined_index * 5) + 1)
        status = "ATTENTION" if attention_score >= 3 else "NORMAL"
        if n_baseline >= 10 and missing_fraction <= 0.05:
            confidence = "HIGH"
        elif n_baseline >= 5 and missing_fraction <= 0.15:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
    else:
        evidence.extend(f"Insufficient evidence: {reason}" for reason in reasons)

    return {
        "status": status,
        "environmental_attention_score": attention_score,
        "confidence": confidence,
        "model": {
            "type": "isolation_forest",
            "anomaly_score": raw_model_score,
            "score_definition": "raw sklearn score_samples; lower values are more anomalous",
            "random_state": RANDOM_STATE,
            "training_sample_size": n_baseline,
        },
        "statistical_baseline_score": round(stat_score, 6) if statistical_valid else None,
        "primary_signal": primary_signal,
        "signals": signals,
        "evidence": evidence,
        "data_quality": {
            "missing_value_fraction_max": round(missing_fraction, 6),
            "baseline_sample_size": n_baseline,
            "critical_failure_reasons": reasons,
        },
        "limitations": LIMITATIONS.copy(),
    }


def _demo_features(unusual: bool = False, historical_years: list[int] | None = None) -> dict[str, Any]:
    """Small synthetic demo; it is illustrative, not observed field data."""
    from fieldsignal_features import build_environmental_features
    import pandas as pd

    rows = []
    historical_rain = {2020: 80, 2021: 95, 2022: 105, 2023: 120, 2024: 90, 2025: 110}
    historical_temp = {2020: 17, 2021: 18, 2022: 19, 2023: 20, 2024: 21, 2025: 22}
    if historical_years is not None:
        historical_rain = {year: historical_rain[year] for year in historical_years}
        historical_temp = {year: historical_temp[year] for year in historical_years}
    for year in [*historical_rain, 2026]:
        is_target = year == 2026
        rain_total = (10 if unusual else 100) if is_target else historical_rain[year]
        temp = (28 if unusual else 19.5) if is_target else historical_temp[year]
        for day in range(1, 8):
            rows.append({"date": f"{year}-06-{day:02d}", "PRECTOTCORR": rain_total / 7,
                         "T2M": temp, "T2M_MAX": temp + 5, "T2M_MIN": temp - 5})
    frame = pd.DataFrame(rows)
    frame.attrs["source"] = "Synthetic offline demo fixture"
    return build_environmental_features(frame, "2026-06-01", "2026-06-07", list(historical_rain))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run synthetic offline example through Tasks 2 and 3")
    parser.add_argument("--unusual", action="store_true", help="show a synthetic dry/warm demo")
    args = parser.parse_args()
    if not args.demo:
        parser.error("pass --demo to run the offline demonstration")
    print(json.dumps(infer_environmental_attention(_demo_features(args.unusual)), indent=2))


if __name__ == "__main__":
    main()
