from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nested_tcn_systole_cnn.data import make_tcn_subset_dataset, select_patient_subset, split_cnn_fit_tune_patients


class NestedDataTests(unittest.TestCase):
    def test_split_cnn_fit_tune_patients_keeps_both_classes_in_each_split(self) -> None:
        meta = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
                "target": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        fit_ids, tune_ids = split_cnn_fit_tune_patients(meta, set(meta["patient_id"]), 0.25, 42, 1)

        self.assertTrue(fit_ids.isdisjoint(tune_ids))
        fit_targets = set(meta.loc[meta["patient_id"].isin(fit_ids), "target"])
        tune_targets = set(meta.loc[meta["patient_id"].isin(tune_ids), "target"])
        self.assertEqual(fit_targets, {0, 1})
        self.assertEqual(tune_targets, {0, 1})

    def test_select_patient_subset_limits_patients(self) -> None:
        meta = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p3", "p4", "p5"],
                "target": [1, 1, 1, 0, 0, 0],
            }
        )

        selected = select_patient_subset(meta, max_patients=3, seed=7)

        self.assertLessEqual(selected["patient_id"].nunique(), 3)
        self.assertTrue(set(selected["patient_id"]).issubset(set(meta["patient_id"])))

    def test_make_tcn_subset_dataset_writes_csv_and_links_recording_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            data_dir = source / "training_data"
            data_dir.mkdir(parents=True)
            pd.DataFrame({"Patient ID": ["1001", "1002"], "Murmur": ["Present", "Absent"]}).to_csv(
                source / "training_data.csv",
                index=False,
            )
            for name in ["1001_AV.wav", "1001_AV.tsv", "1002_PV.wav", "1002_PV.tsv"]:
                (data_dir / name).write_text("x", encoding="utf-8")

            subset = root / "subset"

            make_tcn_subset_dataset(
                source,
                subset,
                {"1001"},
                lambda path: path.stem.split("_", 1),
            )

            subset_table = pd.read_csv(subset / "training_data.csv", dtype={"Patient ID": str})
            self.assertEqual(subset_table["Patient ID"].tolist(), ["1001"])
            self.assertTrue((subset / "training_data" / "1001_AV.wav").is_symlink())
            self.assertTrue((subset / "training_data" / "1001_AV.tsv").is_symlink())
            self.assertFalse((subset / "training_data" / "1002_PV.wav").exists())


if __name__ == "__main__":
    unittest.main()
