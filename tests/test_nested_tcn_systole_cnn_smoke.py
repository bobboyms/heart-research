from __future__ import annotations

import subprocess
import sys
import unittest


class NestedSmokeTests(unittest.TestCase):
    def test_new_entrypoint_help_does_not_load_or_train_models(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "nested_tcn_systole_cnn.train", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--tcn-systole-weight-multiplier", result.stdout)
        self.assertIn("--no-cleanup-fold-artifacts", result.stdout)
        self.assertIn("--run-name", result.stdout)
        self.assertIn("--cnn-epochs", result.stdout)
        self.assertIn("--ltsrr-prob", result.stdout)
        self.assertIn("--spectrogram-type", result.stdout)
        self.assertIn("--smote-minority-augmentation", result.stdout)
        self.assertIn("--loss", result.stdout)
        self.assertIn("--auc-loss-weight", result.stdout)

    def test_dashboard_help_works(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "nested_tcn_systole_cnn.dashboard", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Generate static HTML dashboard", result.stdout)

    def test_legacy_entrypoint_help_still_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Nested TCN + systole CNN", result.stdout)


if __name__ == "__main__":
    unittest.main()
