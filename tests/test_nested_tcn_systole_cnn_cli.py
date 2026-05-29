from __future__ import annotations

import unittest
from pathlib import Path

from nested_tcn_systole_cnn.cli import parse_args, validate_args
from nested_tcn_systole_cnn.models.tcn_segmenter import build_tcn_train_command, checkpoint_matches_args


class NestedCliTests(unittest.TestCase):
    def test_tcn_systole_weight_multiplier_is_parsed_and_validated(self) -> None:
        args = parse_args(["--tcn-systole-weight-multiplier", "2.0"])

        validate_args(args)

        self.assertEqual(args.tcn_systole_weight_multiplier, 2.0)

    def test_fold_artifact_cleanup_is_enabled_by_default_and_can_be_disabled(self) -> None:
        default_args = parse_args([])
        debug_args = parse_args(["--no-cleanup-fold-artifacts"])

        self.assertTrue(default_args.cleanup_fold_artifacts)
        self.assertFalse(debug_args.cleanup_fold_artifacts)

    def test_score_weights_are_validated(self) -> None:
        args = parse_args(["--score-weights", "sensitivity=2,specificity=1,precision=1,f1=1"])

        validate_args(args)

        invalid = parse_args(["--score-weights", "recall=2"])
        with self.assertRaisesRegex(ValueError, "Unsupported score weight"):
            validate_args(invalid)

    def test_ltsrr_options_are_parsed_and_validated(self) -> None:
        args = parse_args(
            ["--ltsrr-prob", "1.0", "--ltsrr-k", "4", "--ltsrr-frequency-ratio", "0.25", "--ltsrr-minority-only"]
        )

        validate_args(args)

        self.assertEqual(args.ltsrr_prob, 1.0)
        self.assertEqual(args.ltsrr_k, 4)
        self.assertEqual(args.ltsrr_frequency_ratio, 0.25)
        self.assertTrue(args.ltsrr_minority_only)

        invalid_prob = parse_args(["--ltsrr-prob", "1.5"])
        with self.assertRaisesRegex(ValueError, "ltsrr-prob"):
            validate_args(invalid_prob)

        invalid_k = parse_args(["--ltsrr-k", "0"])
        with self.assertRaisesRegex(ValueError, "ltsrr-k"):
            validate_args(invalid_k)

        invalid_frequency_ratio = parse_args(["--ltsrr-frequency-ratio", "0"])
        with self.assertRaisesRegex(ValueError, "ltsrr-frequency-ratio"):
            validate_args(invalid_frequency_ratio)

    def test_spectrogram_type_options_are_parsed_and_validated(self) -> None:
        args = parse_args(["--spectrogram-type", "log-mel", "--n-mels", "80"])

        validate_args(args)

        self.assertEqual(args.spectrogram_type, "log-mel")
        self.assertEqual(args.n_mels, 80)

        invalid_n_mels = parse_args(["--n-mels", "0"])
        with self.assertRaisesRegex(ValueError, "n-mels"):
            validate_args(invalid_n_mels)

    def test_smote_options_are_parsed_and_validated(self) -> None:
        args = parse_args(["--smote-minority-augmentation", "--smote-k-neighbors", "3", "--smote-target-ratio", "0.75"])

        validate_args(args)

        self.assertTrue(args.smote_minority_augmentation)
        self.assertEqual(args.smote_k_neighbors, 3)
        self.assertEqual(args.smote_target_ratio, 0.75)

        invalid_neighbors = parse_args(["--smote-k-neighbors", "0"])
        with self.assertRaisesRegex(ValueError, "smote-k-neighbors"):
            validate_args(invalid_neighbors)

        invalid_ratio = parse_args(["--smote-target-ratio", "1.5"])
        with self.assertRaisesRegex(ValueError, "smote-target-ratio"):
            validate_args(invalid_ratio)

        invalid_mil_combo = parse_args(["--smote-minority-augmentation", "--patient-mil-attention"])
        with self.assertRaisesRegex(ValueError, "smote-minority-augmentation"):
            validate_args(invalid_mil_combo)

    def test_focal_loss_options_are_parsed_and_validated(self) -> None:
        args = parse_args(["--loss", "focal", "--focal-gamma", "2.5", "--focal-alpha", "0.75"])

        validate_args(args)

        self.assertEqual(args.loss, "focal")
        self.assertEqual(args.focal_gamma, 2.5)
        self.assertEqual(args.focal_alpha, 0.75)

        invalid_gamma = parse_args(["--loss", "focal", "--focal-gamma", "-1"])
        with self.assertRaisesRegex(ValueError, "focal-gamma"):
            validate_args(invalid_gamma)

        invalid_alpha = parse_args(["--loss", "focal", "--focal-alpha", "1.5"])
        with self.assertRaisesRegex(ValueError, "focal-alpha"):
            validate_args(invalid_alpha)

    def test_auc_loss_options_are_parsed_and_validated(self) -> None:
        args = parse_args(["--auc-loss-weight", "0.2", "--auc-loss-margin", "0.5"])

        validate_args(args)

        self.assertEqual(args.auc_loss_weight, 0.2)
        self.assertEqual(args.auc_loss_margin, 0.5)

        invalid_weight = parse_args(["--auc-loss-weight", "-0.1"])
        with self.assertRaisesRegex(ValueError, "auc-loss-weight"):
            validate_args(invalid_weight)

        invalid_margin = parse_args(["--auc-loss-margin", "-1"])
        with self.assertRaisesRegex(ValueError, "auc-loss-margin"):
            validate_args(invalid_margin)

        invalid_batch_size = parse_args(["--auc-loss-weight", "0.1", "--cnn-batch-size", "1"])
        with self.assertRaisesRegex(ValueError, "cnn-batch-size"):
            validate_args(invalid_batch_size)

    def test_tcn_systole_weight_multiplier_must_be_positive(self) -> None:
        args = parse_args(["--tcn-systole-weight-multiplier", "0"])

        with self.assertRaisesRegex(ValueError, "tcn-systole-weight-multiplier"):
            validate_args(args)

    def test_tcn_command_forwards_weight_multiplier_to_legacy_tcn_script(self) -> None:
        args = parse_args(["--tcn-systole-weight-multiplier", "2.0", "--no-progress"])

        command = build_tcn_train_command(args, Path("/tmp/subset"), Path("/tmp/tcn"))

        self.assertIn("--systole-weight-multiplier", command)
        self.assertEqual(command[command.index("--systole-weight-multiplier") + 1], "2.0")
        self.assertIn("--no-progress", command)

    def test_checkpoint_reuse_requires_matching_weight_multiplier(self) -> None:
        args = parse_args(["--tcn-systole-weight-multiplier", "2.0"])
        matching_payload = {
            "feature_config": {
                "target_mode": "cardiac-phase",
                "other_mode": "keep",
                "boundary_ignore_ms": 0.0,
            },
            "args": {"systole_weight_multiplier": 2.0},
        }
        stale_payload = {
            "feature_config": {
                "target_mode": "cardiac-phase",
                "other_mode": "keep",
                "boundary_ignore_ms": 0.0,
            },
            "args": {"systole_weight_multiplier": 1.0},
        }

        self.assertTrue(checkpoint_matches_args(matching_payload, args))
        self.assertFalse(checkpoint_matches_args(stale_payload, args))


if __name__ == "__main__":
    unittest.main()
