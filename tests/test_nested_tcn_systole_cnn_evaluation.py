from __future__ import annotations

import unittest

import pandas as pd

from nested_tcn_systole_cnn.evaluation import add_oof_prediction_columns, format_threshold_key


class NestedEvaluationTests(unittest.TestCase):
    def test_format_threshold_key_is_filename_safe(self) -> None:
        self.assertEqual(format_threshold_key(0.5), "0p5")
        self.assertEqual(format_threshold_key(-0.25), "m0p25")

    def test_add_oof_prediction_columns(self) -> None:
        patient_oof = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "prob_present_raw": [0.4, 0.6],
                "prob_present_calibrated": [0.7, 0.2],
            }
        )

        result, key = add_oof_prediction_columns(patient_oof, 0.65)

        self.assertEqual(key, "0p65")
        self.assertEqual(result["pred_present_raw_threshold_05"].tolist(), [0, 1])
        self.assertEqual(result["pred_present_calibrated_threshold_05"].tolist(), [1, 0])
        self.assertEqual(result["pred_present_calibrated_threshold_0p65"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()

