from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nested_tcn_systole_cnn.artifacts import cleanup_fold_training_artifacts


class NestedArtifactCleanupTests(unittest.TestCase):
    def test_cleanup_removes_training_artifacts_and_keeps_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fold_dir = Path(tmp) / "fold_1"
            for relative_dir in [
                "tcn_dataset_train_patients",
                "predicted_tsvs",
                "spectrogram_cache",
                "tcn/cache",
            ]:
                path = fold_dir / relative_dir
                path.mkdir(parents=True)
                (path / "artifact.bin").write_text("temporary", encoding="utf-8")

            result_files = [
                fold_dir / "fold_config.json",
                fold_dir / "recording_metadata.csv",
                fold_dir / "tcn" / "best_model.pt",
                fold_dir / "cnn" / "fold_1_best_model.pt",
            ]
            for path in result_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("result", encoding="utf-8")

            removed = cleanup_fold_training_artifacts(fold_dir)

            self.assertEqual(
                {path.relative_to(fold_dir).as_posix() for path in removed},
                {"tcn_dataset_train_patients", "predicted_tsvs", "spectrogram_cache", "tcn/cache"},
            )
            self.assertFalse((fold_dir / "tcn_dataset_train_patients").exists())
            self.assertFalse((fold_dir / "predicted_tsvs").exists())
            self.assertFalse((fold_dir / "spectrogram_cache").exists())
            self.assertFalse((fold_dir / "tcn" / "cache").exists())
            for path in result_files:
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
