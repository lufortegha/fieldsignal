import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from fieldsignal_api import app


def sample_daily():
    rows = []
    rainfall = {2020: 80, 2021: 95, 2022: 105, 2023: 120, 2024: 90, 2025: 110, 2026: 100}
    temperatures = {year: 17 + year - 2020 for year in range(2020, 2026)}
    temperatures[2026] = 19.5
    for year in rainfall:
        for day in range(1, 8):
            temp = temperatures[year]
            rows.append({"date": f"{year}-06-{day:02d}", "PRECTOTCORR": rainfall[year] / 7,
                         "T2M": temp, "T2M_MAX": temp + 5, "T2M_MIN": temp - 5})
    frame = pd.DataFrame(rows)
    frame.attrs.update({"source": "NASA POWER Daily API", "coordinates": {"latitude": 45.0, "longitude": -93.0},
                        "retrieved_at": "2026-10-03T00:00:00+00:00", "cache": "miss"})
    return frame


def seasonal_daily(latest="2026-10-02", *, missing_target_day=None):
    rows = []
    start = pd.Timestamp("2026-09-19")
    end = pd.Timestamp(latest)
    for year in [*range(2016, 2026), 2026]:
        year_start = start.replace(year=year)
        year_end = (end if year == 2026 else end.replace(year=year))
        for day in pd.date_range(year_start, year_end):
            rain_total = 120 + (year - 2016) * 4 if year < 2026 else 135.75
            temp_mean = 22 + (year - 2016) * 0.1 if year < 2026 else 23.9
            rows.append({"date": day, "PRECTOTCORR": rain_total / 14,
                         "T2M": temp_mean, "T2M_MAX": temp_mean + 3.5,
                         "T2M_MIN": temp_mean - 3.5})
    frame = pd.DataFrame(rows)
    if missing_target_day:
        frame = frame[frame["date"] != pd.Timestamp(missing_target_day)].reset_index(drop=True)
    frame.attrs.update({"source": "NASA POWER Daily API", "coordinates": {"latitude": 7.41, "longitude": -7.55},
                        "retrieved_at": "2026-10-04T05:31:00+00:00", "cache": "miss",
                        "latest_data_date": end.strftime("%Y%m%d"),
                        "trailing_missing_dates": ["20261003", "20261004"] if latest == "2026-10-02" else []})
    return frame


class FieldSignalApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "version": "0.1.0"})

    def test_normal_demo(self):
        response = self.client.get("/demo/normal")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "NORMAL")
        self.assertEqual(response.json()["provenance"]["mode"], "fixture")

    def test_attention_demo(self):
        response = self.client.get("/demo/attention")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ATTENTION")
        self.assertIn("rainfall", response.json()["evidence"][0])

    def test_insufficient_demo(self):
        response = self.client.get("/demo/insufficient")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(response.json()["data_quality"]["critical_failure_reasons"])

    def test_demo_rejects_unsupported_scenario(self):
        response = self.client.get("/demo/unusual")
        self.assertEqual(response.status_code, 404)

    def test_valid_analyze_runs_pipeline_and_preserves_provenance(self):
        with patch("fieldsignal_api.fetch_daily", return_value=sample_daily()):
            response = self.client.post("/analyze", json={
                "latitude": 45, "longitude": -93, "crop": "coffee",
                "analysis_date": "2026-06-07", "window_days": 7,
                "historical_years": [2020, 2021, 2022, 2023, 2024, 2025],
            })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "NORMAL")
        self.assertEqual(result["provenance"]["mode"], "live")
        self.assertEqual(result["provenance"]["crop"], "coffee")
        self.assertEqual(result["provenance"]["analysis_period"]["start"], "2026-06-01")
        self.assertEqual(result["provenance"]["source"], "NASA POWER Daily API")
        self.assertEqual(result["provenance"]["freshness_status"], "CURRENT_DATA_AVAILABLE")
        self.assertEqual(result["provenance"]["data_age_days"], 0)
        self.assertIn("rainfall_z_score", result["environmental_features"]["features"])
        self.assertIn("missing_value_fraction", result["environmental_features"]["data_quality"])

    def test_trailing_sentinels_use_latest_complete_window_and_disclose_delay(self):
        with patch("fieldsignal_api.fetch_daily", return_value=seasonal_daily()):
            response = self.client.post("/analyze", json={
                "latitude": 7.41, "longitude": -7.55, "crop": "coffee",
                "analysis_date": "2026-10-04", "window_days": 14,
                "historical_years": list(range(2016, 2026)),
            })
        self.assertEqual(response.status_code, 200, response.text)
        provenance = response.json()["provenance"]
        self.assertEqual(provenance["requested_analysis_date"], "2026-10-04")
        self.assertEqual(provenance["actual_analysis_end_date"], "2026-10-02")
        self.assertEqual(provenance["latest_data_date"], "2026-10-02")
        self.assertEqual(provenance["data_age_days"], 2)
        self.assertEqual(provenance["freshness_status"], "LATEST_AVAILABLE_DATA_DELAYED")
        self.assertEqual(provenance["analysis_period"], {"start": "2026-09-19", "end": "2026-10-02"})

    def test_incomplete_latest_window_is_upstream_unavailable(self):
        with patch("fieldsignal_api.fetch_daily", return_value=seasonal_daily(missing_target_day="2026-09-20")):
            response = self.client.post("/analyze", json={
                "latitude": 7.41, "longitude": -7.55, "crop": "coffee",
                "analysis_date": "2026-10-04", "window_days": 14,
                "historical_years": list(range(2016, 2026)),
            })
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"]["freshness_status"], "UPSTREAM_UNAVAILABLE")

    def test_data_older_than_configured_freshness_limit_is_not_analyzed(self):
        with patch("fieldsignal_api.fetch_daily", return_value=seasonal_daily(latest="2026-09-26")):
            response = self.client.post("/analyze", json={
                "latitude": 7.41, "longitude": -7.55, "crop": "coffee",
                "analysis_date": "2026-10-04", "window_days": 14,
                "historical_years": list(range(2016, 2026)),
            })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["freshness_status"], "LATEST_AVAILABLE_DATA_DELAYED")
        self.assertEqual(response.json()["detail"]["data_age_days"], 8)
        self.assertEqual(response.json()["detail"]["max_data_age_days"], 7)

    def test_cached_analyze_discloses_cached_mode(self):
        frame = sample_daily()
        frame.attrs["cache"] = "hit"
        with patch("fieldsignal_api.fetch_daily", return_value=frame):
            response = self.client.post("/analyze", json={
                "latitude": 45, "longitude": -93, "crop": "coffee",
                "analysis_date": "2026-06-07", "window_days": 7,
                "historical_years": [2020, 2021, 2022, 2023, 2024, 2025],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provenance"]["mode"], "cached")

    def test_invalid_coordinates_or_input_are_rejected(self):
        invalid_coordinate = self.client.post("/analyze", json={
            "latitude": 91, "longitude": -93, "crop": "coffee", "analysis_date": "2026-06-07",
        })
        invalid_input = self.client.post("/analyze", json={
            "latitude": 45, "longitude": -93, "crop": "", "analysis_date": "2026-06-07",
        })
        cross_year = self.client.post("/analyze", json={
            "latitude": 45, "longitude": -93, "crop": "coffee", "analysis_date": "2026-01-03", "window_days": 7,
        })
        self.assertEqual(invalid_coordinate.status_code, 422)
        self.assertEqual(invalid_input.status_code, 422)
        self.assertEqual(cross_year.status_code, 422)

    def test_upstream_failure_is_explicit_error(self):
        with patch("fieldsignal_api.fetch_daily", side_effect=OSError("simulated upstream outage")):
            response = self.client.post("/analyze", json={
                "latitude": 45, "longitude": -93, "crop": "coffee",
                "analysis_date": "2026-06-07", "window_days": 7,
                "historical_years": [2020, 2021, 2022, 2023, 2024, 2025],
            })
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"]["code"], "UPSTREAM_DATA_UNAVAILABLE")
        self.assertEqual(response.json()["detail"]["freshness_status"], "UPSTREAM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
