import unittest
import json
import tempfile
from unittest.mock import patch

from fieldsignal_power import PARAMETERS, PowerDataError, fetch_daily, process_response


def sample_payload():
    days = ["20240101", "20240102"]
    values = {
        "PRECTOTCORR": [0.2, 1.5],
        "T2M": [5.0, 4.0],
        "T2M_MAX": [8.0, 7.0],
        "T2M_MIN": [2.0, 1.0],
    }
    return {
        "header": {"title": "NASA POWER Daily API", "latitude": 44.98, "longitude": -93.27,
                   "start": "20240101", "end": "20240102"},
        "parameters": {name: {"units": "mm/day" if name == "PRECTOTCORR" else "°C"} for name in PARAMETERS},
        "properties": {"parameter": {name: dict(zip(days, numbers)) for name, numbers in values.items()}},
        "retrieved_at": "2026-10-03T00:00:00+00:00",
    }


class ProcessResponseTests(unittest.TestCase):
    def test_valid_payload_returns_clean_frame_and_metadata(self):
        frame = process_response(sample_payload())
        self.assertEqual(list(frame.columns), ["date", *PARAMETERS])
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.attrs["source"], "NASA POWER Daily API")
        self.assertEqual(frame.attrs["coordinates"]["latitude"], 44.98)
        self.assertEqual(frame.attrs["date_range"]["start"], "20240101")
        self.assertEqual(frame.attrs["retrieved_at"], "2026-10-03T00:00:00+00:00")

    def test_geometry_coordinates_are_fallback_when_header_omits_them(self):
        payload = sample_payload()
        del payload["header"]["latitude"]
        del payload["header"]["longitude"]
        payload["geometry"] = {"type": "Point", "coordinates": [-93.27, 44.98, 250.0]}
        frame = process_response(payload)
        self.assertEqual(frame.attrs["coordinates"], {"latitude": 44.98, "longitude": -93.27})

    def test_null_value_fails(self):
        payload = sample_payload()
        payload["properties"]["parameter"]["T2M"]["20240101"] = None
        with self.assertRaisesRegex(PowerDataError, "Missing, null, or sentinel"):
            process_response(payload)

    def test_sentinel_fails(self):
        payload = sample_payload()
        payload["properties"]["parameter"]["PRECTOTCORR"]["20240102"] = -999
        with self.assertRaises(PowerDataError):
            process_response(payload)

    def test_implausible_temperature_fails(self):
        payload = sample_payload()
        payload["properties"]["parameter"]["T2M_MAX"]["20240101"] = 999
        with self.assertRaisesRegex(PowerDataError, "plausible range"):
            process_response(payload)

    def test_inconsistent_temperature_order_fails(self):
        payload = sample_payload()
        payload["properties"]["parameter"]["T2M_MIN"]["20240101"] = 6
        with self.assertRaisesRegex(PowerDataError, "Temperature ordering"):
            process_response(payload)

    def test_successful_fetch_is_cached_and_reused(self):
        payload = sample_payload()
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        with tempfile.TemporaryDirectory(dir="work") as cache_dir:
            with patch("fieldsignal_power.urllib.request.urlopen", return_value=response) as open_url:
                first = fetch_daily(44.98, -93.27, "2024-01-01", "2024-01-02", cache_dir=cache_dir)
                second = fetch_daily(44.98, -93.27, "2024-01-01", "2024-01-02", cache_dir=cache_dir)
            self.assertEqual(len(first), 2)
            self.assertEqual(first.attrs["cache"], "miss")
            self.assertEqual(second.attrs["cache"], "hit")
            open_url.assert_called_once()

    def test_trailing_all_parameter_sentinels_can_be_trimmed_only_when_enabled(self):
        payload = sample_payload()
        payload["header"]["end"] = "20240103"
        for name in PARAMETERS:
            payload["properties"]["parameter"][name]["20240103"] = -999
        with self.assertRaisesRegex(PowerDataError, "Missing, null, or sentinel"):
            process_response(payload)
        frame = process_response(payload, allow_trailing_missing=True)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.attrs["date_range"]["end"], "20240103")
        self.assertEqual(frame.attrs["latest_data_date"], "20240102")
        self.assertEqual(frame.attrs["trailing_missing_dates"], ["20240103"])

    def test_internal_sentinel_is_not_trimmed(self):
        payload = sample_payload()
        payload["properties"]["parameter"]["T2M"]["20240101"] = -999
        with self.assertRaisesRegex(PowerDataError, "Missing, null, or sentinel"):
            process_response(payload, allow_trailing_missing=True)


if __name__ == "__main__":
    unittest.main()
