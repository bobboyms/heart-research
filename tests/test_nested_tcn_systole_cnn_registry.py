from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nested_tcn_systole_cnn.cli import parse_args, validate_args
from nested_tcn_systole_cnn.experiment_registry import (
    build_metrics_summary,
    flatten_metrics_for_registry,
    resolve_experiment_paths,
    update_registry,
)
from nested_tcn_systole_cnn.scoring import parse_score_weights, weighted_mean_score


class NestedExperimentRegistryTests(unittest.TestCase):
    def test_named_run_gets_unique_output_dir_under_experiments_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--run-name", "TCN Weight 2", "--experiments-dir", tmp])

            validate_args(args)
            resolve_experiment_paths(args)

            self.assertEqual(args.run_name, "TCN Weight 2")
            self.assertTrue(args.run_id.startswith("tcn_weight_2__"))
            self.assertEqual(args.output_dir.parent, Path(tmp).resolve())

    def test_registry_writes_completed_row_and_metrics_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--run-name", "baseline", "--experiments-dir", tmp])
            validate_args(args)
            resolve_experiment_paths(args)
            args.output_dir.mkdir(parents=True)

            decision_metrics = {
                "threshold": 0.5,
                "auroc": 0.8,
                "auprc": 0.7,
                "balanced_accuracy": 0.75,
                "sensitivity": 0.9,
                "specificity": 0.6,
                "precision": 0.5,
                "f1": 0.64,
                "brier_score": 0.2,
                "tn": 6,
                "fp": 4,
                "fn": 1,
                "tp": 9,
            }
            metrics_summary = build_metrics_summary(args, {}, {}, {}, {}, decision_metrics)
            (args.output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary), encoding="utf-8")

            update_registry(args, "running")
            update_registry(args, "completed", flatten_metrics_for_registry(metrics_summary))

            registry = pd.read_csv(Path(tmp) / "registry.csv")
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry.loc[0, "status"], "completed")
            self.assertAlmostEqual(float(registry.loc[0, "sensitivity"]), 0.9)
            self.assertAlmostEqual(float(registry.loc[0, "precision_ppv"]), 0.5)
            self.assertAlmostEqual(float(registry.loc[0, "mean_score"]), (0.9 + 0.6 + 0.5 + 0.64) / 4)

    def test_score_weights_can_prioritize_sensitivity(self) -> None:
        weights = parse_score_weights("sensitivity=3,specificity=1,precision=1,f1=1")
        metrics = {"sensitivity": 1.0, "specificity": 0.0, "precision": 0.0, "f1": 0.0}

        self.assertAlmostEqual(weighted_mean_score(metrics, weights), 0.5)


if __name__ == "__main__":
    unittest.main()

