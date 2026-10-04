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
        self.assertIn("rainfall_z_score", result["environmental_features"]["features"])
        self.assertIn("missing_value_fraction", result["environmental_features"]["data_quality"])

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


if __name__ == "__main__":
    unittest.main()
