import unittest

import pandas as pd

from fieldsignal_features import build_environmental_features


def daily_frame(target=(10, 10, 10), years=(2021, 2022, 2023), rainfall_by_year=None):
    rows = []
    rainfall_by_year = rainfall_by_year or {year: (10, 10, 10) for year in years}
    for year in years:
        for day_index, day in enumerate((1, 2, 3)):
            rain = rainfall_by_year[year][day_index]
            mean = 15 + (year - 2021) + day_index
            rows.append({"date": f"{year}-06-0{day}", "PRECTOTCORR": rain,
                         "T2M": mean, "T2M_MAX": mean + 5, "T2M_MIN": mean - 5})
    for day_index, day in enumerate((1, 2, 3)):
        mean = 15 + 3 + day_index
        rows.append({"date": f"2024-06-0{day}", "PRECTOTCORR": target[day_index],
                     "T2M": mean, "T2M_MAX": mean + 5, "T2M_MIN": mean - 5})
    frame = pd.DataFrame(rows)
    frame.attrs["source"] = "NASA POWER Daily API"
    frame.attrs["coordinates"] = {"latitude": 45.0, "longitude": -93.0}
    return frame


def analyze(frame, years=(2021, 2022, 2023)):
    return build_environmental_features(frame, "2024-06-01", "2024-06-03", years, recent_days=3)


class EnvironmentalFeatureTests(unittest.TestCase):
    def test_ordinary_conditions_match_seasonal_baseline(self):
        # Vary historical rain totals so the z-score has nonzero spread.
        historical = {2021: (8, 8, 8), 2022: (10, 10, 10), 2023: (12, 12, 12)}
        frame = daily_frame(target=(10, 10, 10), rainfall_by_year=historical)
        target_rows = frame["date"].astype(str).str.startswith("2024")
        frame.loc[target_rows, "T2M"] = [16, 17, 18]
        frame.loc[target_rows, "T2M_MAX"] = [21, 22, 23]
        frame.loc[target_rows, "T2M_MIN"] = [11, 12, 13]
        result = analyze(frame)
        features = result["features"]
        self.assertEqual(features["accumulated_rainfall"]["value"], 30.0)
        self.assertAlmostEqual(features["rainfall_anomaly_mm"]["value"], 0.0)
        self.assertAlmostEqual(features["rainfall_z_score"]["value"], 0.0)
        self.assertAlmostEqual(features["recent_rainfall_persistence"]["value"], 1.0)
        self.assertEqual(features["rainfall_z_score"]["baseline_sample_size"], 3)
        self.assertTrue(features["mean_temperature_z_score"]["statistically_valid"])
        self.assertEqual(result["target_window"]["start"], "2024-06-01")

    def test_unusually_dry_and_warm_target(self):
        historical = {2021: (8, 8, 8), 2022: (10, 10, 10), 2023: (12, 12, 12)}
        frame = daily_frame(target=(0, 0, 0), rainfall_by_year=historical)
        target_rows = frame["date"].astype(str).str.startswith("2024")
        frame.loc[target_rows, "T2M"] = [26, 27, 28]
        frame.loc[target_rows, "T2M_MAX"] = [31, 32, 33]
        result = analyze(frame)["features"]
        self.assertLess(result["rainfall_anomaly_percent"]["value"], -99)
        self.assertLess(result["rainfall_z_score"]["value"], 0)
        self.assertGreater(result["mean_temperature_anomaly_c"]["value"], 0)
        self.assertGreater(result["mean_temperature_z_score"]["value"], 0)
        self.assertGreater(result["maximum_temperature_z_score"]["value"], 0)

    def test_missing_target_and_baseline_values_are_reported(self):
        frame = daily_frame(rainfall_by_year={2021: (8, 8, 8), 2022: (10, 10, 10), 2023: (12, 12, 12)})
        dates = pd.to_datetime(frame["date"])
        frame.loc[dates == pd.Timestamp("2024-06-02"), "PRECTOTCORR"] = None
        frame.loc[dates == pd.Timestamp("2021-06-02"), "T2M"] = None
        result = analyze(frame)
        self.assertIsNone(result["features"]["accumulated_rainfall"]["value"])
        self.assertFalse(result["features"]["accumulated_rainfall"]["statistically_valid"])
        self.assertAlmostEqual(result["data_quality"]["missing_value_fraction"]["rainfall"]["target"], 1 / 3, places=5)
        self.assertEqual(result["features"]["mean_temperature_z_score"]["baseline_sample_size"], 2)
        self.assertAlmostEqual(result["data_quality"]["missing_value_fraction"]["mean_temperature"]["historical"], 1 / 9, places=5)

    def test_zero_variance_and_zero_rainfall_baseline_are_invalid(self):
        frame = daily_frame(target=(0, 0, 0), rainfall_by_year={
            2021: (0, 0, 0), 2022: (0, 0, 0), 2023: (0, 0, 0),
        })
        historical_rows = pd.to_datetime(frame["date"]).dt.year < 2024
        frame.loc[historical_rows, "T2M"] = 20
        frame.loc[historical_rows, "T2M_MAX"] = 25
        frame.loc[historical_rows, "T2M_MIN"] = 15
        result = analyze(frame)["features"]
        self.assertFalse(result["rainfall_anomaly_percent"]["statistically_valid"])
        self.assertIsNone(result["rainfall_anomaly_percent"]["value"])
        self.assertFalse(result["rainfall_z_score"]["statistically_valid"])
        self.assertFalse(result["mean_temperature_z_score"]["statistically_valid"])
        self.assertIn("zero or near zero", result["rainfall_z_score"]["reason"])

    def test_single_historical_year_cannot_produce_z_score(self):
        frame = daily_frame(years=(2023,), rainfall_by_year={2023: (5, 5, 5)})
        result = analyze(frame, years=(2023,))["features"]["rainfall_z_score"]
        self.assertFalse(result["statistically_valid"])
        self.assertIsNone(result["value"])
        self.assertEqual(result["baseline_sample_size"], 1)


if __name__ == "__main__":
    unittest.main()
