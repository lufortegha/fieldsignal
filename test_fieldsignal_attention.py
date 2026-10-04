import copy
import unittest

from fieldsignal_attention import _demo_features, infer_environmental_attention


class EnvironmentalAttentionTests(unittest.TestCase):
    def test_ordinary_season_is_normal(self):
        result = infer_environmental_attention(_demo_features())
        self.assertEqual(result["status"], "NORMAL")
        self.assertIn(result["environmental_attention_score"], range(1, 3))

    def test_unusual_dry_warm_combination_needs_attention(self):
        result = infer_environmental_attention(_demo_features(unusual=True))
        self.assertEqual(result["status"], "ATTENTION")
        self.assertEqual(result["primary_signal"], "DRY_WARM_ANOMALY")
        self.assertTrue(result["evidence"])
        self.assertIsInstance(result["model"]["anomaly_score"], float)

    def test_critical_invalid_data_overrides_model(self):
        features = copy.deepcopy(_demo_features())
        features["features"]["rainfall_z_score"]["statistically_valid"] = False
        features["features"]["rainfall_z_score"]["value"] = None
        features["model_inputs"]["target_vector"][0] = None
        result = infer_environmental_attention(features)
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["confidence"], "LOW")
        self.assertIsNone(result["environmental_attention_score"])

    def test_fixed_random_state_is_reproducible(self):
        features = _demo_features(unusual=True)
        first = infer_environmental_attention(features)
        second = infer_environmental_attention(features)
        self.assertEqual(first, second)

    def test_stronger_anomaly_has_equal_or_greater_score(self):
        ordinary = infer_environmental_attention(_demo_features())
        unusual = infer_environmental_attention(_demo_features(unusual=True))
        self.assertGreaterEqual(unusual["environmental_attention_score"], ordinary["environmental_attention_score"])


if __name__ == "__main__":
    unittest.main()
