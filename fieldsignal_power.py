"""Small NASA POWER daily-data client for the FieldSignal prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE = "NASA POWER Daily API"
ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN")
SENTINELS = {-999, -999.0, -9999, -9999.0}


class PowerDataError(ValueError):
    """Raised when a POWER response is missing or fails basic validation."""


def _iso_date(value: str | date) -> str:
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc
    return parsed.isoformat()


def _cache_path(cache_dir: str | Path, lat: float, lon: float, start: str, end: str) -> Path:
    key = json.dumps([round(lat, 5), round(lon, 5), start, end, PARAMETERS], separators=(",", ":"))
    return Path(cache_dir) / f"power_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def process_response(payload: dict[str, Any], *, allow_trailing_missing: bool = False) -> pd.DataFrame:
    """Validate a POWER payload and return tidy daily values.

    The default remains strict. ``allow_trailing_missing`` permits trimming
    only final dates where every requested parameter is a NASA sentinel, which
    supports APIs whose latest daily products are not yet published.
    """
    try:
        header = payload["header"]
        parameters = payload["parameters"]
        raw = payload["properties"]["parameter"]
        geometry = payload.get("geometry") or {}
        geometry_coordinates = geometry.get("coordinates", []) or []
        longitude = header["longitude"] if "longitude" in header else geometry_coordinates[0]
        latitude = header["latitude"] if "latitude" in header else geometry_coordinates[1]
        lon, lat = float(longitude), float(latitude)
        start, end = str(header["start"]), str(header["end"])
        source = header.get("title", SOURCE)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise PowerDataError("Malformed POWER response: required metadata or values are missing") from exc

    dates = sorted(set().union(*(raw.get(name, {}).keys() for name in PARAMETERS)))
    try:
        expected_dates = pd.date_range(
            pd.to_datetime(start, format="%Y%m%d"),
            pd.to_datetime(end, format="%Y%m%d"),
            freq="D",
        ).strftime("%Y%m%d").tolist()
    except (TypeError, ValueError) as exc:
        raise PowerDataError("POWER response contains invalid date-range metadata") from exc
    if dates != expected_dates:
        raise PowerDataError("POWER response has missing or out-of-range daily observations")

    process_dates = expected_dates.copy()
    trailing_missing_dates = []
    if allow_trailing_missing:
        while process_dates and all(
            raw.get(name, {}).get(process_dates[-1]) in SENTINELS for name in PARAMETERS
        ):
            trailing_missing_dates.append(process_dates.pop())
        if not process_dates:
            raise PowerDataError("POWER response contains no valid daily observations")

    rows = []
    for day in process_dates:
        row: dict[str, Any] = {"date": pd.to_datetime(day, format="%Y%m%d", errors="raise")}
        for name in PARAMETERS:
            val = raw.get(name, {}).get(day)
            if val is None or not isinstance(val, (int, float)) or not math.isfinite(val) or val in SENTINELS:
                raise PowerDataError(f"Missing, null, or sentinel value for {name} on {day}")
            row[name] = float(val)
        rows.append(row)

    if not rows:
        raise PowerDataError("POWER response contains no daily observations")
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if ((frame["PRECTOTCORR"] < 0) | (frame["PRECTOTCORR"] > 2000)).any():
        raise PowerDataError("PRECTOTCORR is outside the basic plausible range 0–2000 mm/day")
    for column in ("T2M", "T2M_MAX", "T2M_MIN"):
        if ((frame[column] < -100) | (frame[column] > 70)).any():
            raise PowerDataError(f"{column} is outside the basic plausible range -100–70 °C")
    if ((frame["T2M_MIN"] > frame["T2M"]) | (frame["T2M"] > frame["T2M_MAX"])).any():
        raise PowerDataError("Temperature ordering must satisfy T2M_MIN <= T2M <= T2M_MAX")

    frame.attrs.update({
        "source": source,
        "coordinates": {"latitude": lat, "longitude": lon},
        "date_range": {"start": start, "end": end},
        "available_date_range": {"start": process_dates[0], "end": process_dates[-1]},
        "latest_data_date": process_dates[-1],
        "trailing_missing_dates": list(reversed(trailing_missing_dates)),
        "retrieved_at": payload.get("retrieved_at"),
        "units": {name: parameters.get(name, {}).get("units") for name in PARAMETERS},
    })
    return frame


def fetch_daily(
    latitude: float,
    longitude: float,
    start_date: str | date,
    end_date: str | date,
    *,
    cache_dir: str | Path = "data/cache",
    timeout: int = 30,
    allow_trailing_missing: bool = False,
) -> pd.DataFrame:
    """Retrieve NASA POWER daily data, caching successful raw responses."""
    lat, lon = float(latitude), float(longitude)
    if not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("longitude must be between -180 and 180")
    start, end = _iso_date(start_date), _iso_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    cache_file = _cache_path(cache_dir, lat, lon, start, end)
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        payload.setdefault("retrieved_at", "cached")
        result = process_response(payload, allow_trailing_missing=allow_trailing_missing)
        result.attrs["cache"] = "hit"
        return result

    query = urllib.parse.urlencode({
        "parameters": ",".join(PARAMETERS), "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "JSON", "time-standard": "UTC",
    })
    request = urllib.request.Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "FieldSignal/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    result = process_response(payload, allow_trailing_missing=allow_trailing_missing)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result.attrs["cache"] = "miss"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, default=44.9778)
    parser.add_argument("--longitude", type=float, default=-93.2650)
    parser.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-07", help="YYYY-MM-DD")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--csv", help="Optional path to save the clean DataFrame as CSV")
    args = parser.parse_args()
    frame = fetch_daily(args.latitude, args.longitude, args.start, args.end, cache_dir=args.cache_dir)
    print(frame.to_string(index=False))
    print(f"\nSource: {frame.attrs['source']} | cache: {frame.attrs['cache']}")
    if args.csv:
        frame.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
