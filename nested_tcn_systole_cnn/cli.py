"""Command-line interface for the nested TCN + systole CNN experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from .paths import DEFAULT_DATASET_DIR, DEFAULT_EXPERIMENTS_DIR, DEFAULT_OUTPUT_DIR
from .scoring import parse_score_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested TCN + systole CNN validation by patient.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for this run output. Defaults to the historical outputs_nested path when --run-name is not set, "
            "or to a unique directory under --experiments-dir when --run-name is set."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None, help="Human-readable experiment name.")
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument(
        "--score-weights",
        type=str,
        default="sensitivity=1,specificity=1,precision=1,f1=1",
        help=(
            "Comma-separated weights for mean_score. Supported keys: sensitivity, specificity, precision, f1. "
            "Example: sensitivity=2,specificity=1,precision=1,f1=1"
        ),
    )
    parser.add_argument("--locations", nargs="+", default=["AV", "PV", "TV", "MV"], choices=["AV", "PV", "TV", "MV"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--force-retrain-tcn", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--cleanup-fold-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove large fold-local training artifacts after each fold finishes. "
            "Keeps checkpoints, metrics, configs, plots, and CSV result files."
        ),
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--tcn-epochs", type=int, default=10)
    parser.add_argument("--tcn-batch-size", type=int, default=42)
    parser.add_argument("--tcn-device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument("--tcn-val-size", type=float, default=0.15)
    parser.add_argument("--tcn-test-size", type=float, default=0.15)
    parser.add_argument("--tcn-pooling", choices=["none", "attention"], default="none")
    parser.add_argument(
        "--tcn-boundary-ignore-ms",
        type=float,
        default=0.0,
        help="Mark TCN training frames within this distance from any cardiac-phase boundary as ignored.",
    )
    parser.add_argument(
        "--tcn-systole-weight-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier applied to the systole class weight while training each fold-specific TCN.",
    )
    parser.add_argument(
        "--tcn-target-mode",
        choices=["cardiac-phase", "systole-binary"],
        default="cardiac-phase",
        help="Target used to train the fold-specific TCN segmenter.",
    )
    parser.add_argument(
        "--other-mode",
        "--tcn-other-mode",
        dest="tcn_other_mode",
        choices=["keep", "ignore"],
        default="keep",
        help=(
            "How the fold-specific TCN handles original frame label 0. "
            "'ignore' maps label 0 frames to IGNORE_INDEX so training focuses on S1/systole/S2/diastole."
        ),
    )

    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--cnn-patience", type=int, default=8)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--cnn-inner-val-size", type=float, default=0.15)
    parser.add_argument("--cnn-device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument("--pooling", choices=["avg", "attention"], default="attention")
    parser.add_argument("--calibration", choices=["none", "platt"], default="platt")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Operational threshold reported for calibrated patient-level probabilities.",
    )
    parser.add_argument(
        "--weak-murmur-weight",
        type=float,
        default=1.0,
        help="Per-recording CNN loss multiplier for Present patients with systolic murmur grading I/VI.",
    )
    parser.add_argument(
        "--moderate-murmur-weight",
        type=float,
        default=1.0,
        help="Per-recording CNN loss multiplier for Present patients with systolic murmur grading II/VI.",
    )
    parser.add_argument(
        "--location-aware-calibration",
        action="store_true",
        help="Use cnn_tune to calibrate patient probabilities from per-location recording probabilities.",
    )
    parser.add_argument(
        "--ltsrr-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of applying LTSRR augmentation to each CNN training spectrogram. "
            "0 disables the augmentation."
        ),
    )
    parser.add_argument(
        "--ltsrr-k",
        type=int,
        default=4,
        help="Number of non-overlapping time segments used by LTSRR.",
    )
    parser.add_argument(
        "--ltsrr-frequency-ratio",
        type=float,
        default=0.25,
        help="Fraction of frequency bins replaced inside each LTSRR time segment.",
    )
    parser.add_argument(
        "--ltsrr-minority-only",
        action="store_true",
        help="Apply LTSRR only to positive/Present CNN training samples.",
    )
    parser.add_argument(
        "--smote-minority-augmentation",
        action="store_true",
        help="Use SMOTE to synthesize minority-class spectrograms for CNN training only.",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Number of nearest minority-class neighbors considered by SMOTE.",
    )
    parser.add_argument(
        "--smote-target-ratio",
        type=float,
        default=1.0,
        help="Target minority/majority ratio after SMOTE. 1.0 balances the CNN training split.",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help="CNN training loss. BCE keeps historical behavior; focal down-weights easy examples.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing parameter used when --loss focal.")
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional positive-class alpha used when --loss focal. If omitted, no alpha factor is applied.",
    )
    parser.add_argument(
        "--auc-loss-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary pairwise AUC ranking loss added to BCE/Focal. 0 disables it.",
    )
    parser.add_argument(
        "--auc-loss-margin",
        type=float,
        default=1.0,
        help="Margin used by the pairwise AUC ranking loss.",
    )
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dilations", type=str, default="1,2,4,8")
    parser.add_argument(
        "--encoder-block",
        choices=["residual", "multiscale"],
        default="residual",
        help="CNN encoder block used by both recording-level and MIL models.",
    )
    parser.add_argument(
        "--patient-mil-attention",
        action="store_true",
        help="Train the CNN classifier as a patient-level fixed-location fusion model over AV/PV/TV/MV embeddings.",
    )
    parser.add_argument("--mil-max-instances", type=int, default=8, help="Deprecated compatibility option.")
    parser.add_argument("--mil-location-embedding-dim", type=int, default=4, help="Deprecated compatibility option.")
    parser.add_argument("--mil-instance-loss-weight", type=float, default=0.25)

    parser.add_argument("--target-sample-rate", type=int, default=4000)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--low-hz", type=float, default=0.0)
    parser.add_argument("--high-hz", type=float, default=1000.0)
    parser.add_argument(
        "--spectrogram-type",
        choices=["stft", "log-mel"],
        default="stft",
        help="Spectrogram representation used by the CNN.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Number of Mel bins used when --spectrogram-type log-mel.",
    )
    parser.add_argument(
        "--stft-segment-mode",
        choices=["concat", "per-segment"],
        default="concat",
        help=(
            "How systole/diastole audio is fed into the STFT. "
            "'concat' concatenates segments in the time domain first (legacy). "
            "'per-segment' runs the STFT per segment and concatenates frames, avoiding spectral "
            "leakage at segment boundaries."
        ),
    )
    parser.add_argument(
        "--freq-norm",
        choices=["perbin", "global"],
        default="perbin",
        help=(
            "How spectrogram normalization statistics are computed from the training set. "
            "'perbin' (legacy/bc) z-scores each frequency bin independently over time+samples, "
            "which whitens the cross-band energy ratio that encodes murmur pitch. "
            "'global' uses a single scalar mean/std, preserving the spectral shape (better for "
            "low-pitch murmurs at the possible cost of harsh mid-frequency cases)."
        ),
    )
    parser.add_argument(
        "--phase-contrast",
        action="store_true",
        help=(
            "Re-reference the systole spectrogram by the same recording's diastole baseline per "
            "frequency (C[f,t] = systole_logmag - median_t diastole_logmag). Cancels sensor/patient "
            "coloration and exposes the systolic energy excess (the murmur); Absent recordings stay "
            "near zero. Fixed DSP, no params. Requires --cnn-phase-mode systole, --stft-segment-mode "
            "per-segment and --tcn-target-mode cardiac-phase (needs diastole segments)."
        ),
    )
    parser.add_argument(
        "--target",
        choices=["murmur", "outcome"],
        default="murmur",
        help=(
            "Prediction target. 'murmur' = Murmur Present vs Absent (location-aware, Unknown dropped). "
            "'outcome' = clinical Normal vs Abnormal (balanced ~50/50; includes Murmur=Unknown patients; "
            "per-recording label = patient Outcome, no location-aware)."
        ),
    )
    parser.add_argument(
        "--demographic",
        action="store_true",
        help=(
            "Add a parallel demographic branch: Age/Sex/Height/Weight/Pregnancy are encoded "
            "(one-hot + z-scored numerics with missing flags), embedded through a small MLP and "
            "ADDED to the pooled CNN features before the classification head. Requires --model-arch "
            "cnn; incompatible with SMOTE, mixup, --patient-mil-attention and --use-temporal-features."
        ),
    )
    parser.add_argument(
        "--phase-contrast-robust",
        action="store_true",
        help=(
            "Robust variant of --phase-contrast: divide the contrast by the diastole MAD per "
            "frequency (robust z-score), per Grupo B v3.1 — 'how many robust std's above the "
            "diastole baseline'. Amplifies bands with systolic excess (esp. low). Implies --phase-contrast."
        ),
    )
    parser.add_argument(
        "--phase-contrast-dual",
        action="store_true",
        help=(
            "Dual-channel variant of --phase-contrast: stack [systole, diastole-referenced contrast] "
            "along the frequency axis so the encoder sees both the raw systole texture and the "
            "contrast. Implies --phase-contrast."
        ),
    )
    parser.add_argument(
        "--aux-pitch-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Tier 2 multi-task: weight of an auxiliary cross-entropy head that predicts systolic "
            "murmur pitch (Low/Medium/High) from the pooled encoder features, supervised only on "
            "Present recordings. 0 disables. Forces the encoder to represent the low-band spectral "
            "shape that plain per-bin normalization whitens away. Requires --model-arch cnn and is "
            "incompatible with SMOTE, mixup, and --patient-mil-attention."
        ),
    )
    parser.add_argument(
        "--use-ground-truth-segments",
        action="store_true",
        help=(
            "Read cardiac-phase segments directly from each recording's .tsv (ground truth from the "
            "dataset) instead of training and using a TCN to predict them. Skips TCN training entirely."
        ),
    )
    parser.add_argument(
        "--exclude-present-grades",
        type=str,
        default="",
        help=(
            "Comma-separated systolic murmur gradings to drop from the Present class (e.g. 'I/VI'). "
            "Measures the achievable ceiling on audible murmurs (irreducible-floor analysis)."
        ),
    )
    parser.add_argument("--specaug-time-prob", type=float, default=0.0,
        help="Probability of applying each time mask in CNN SpecAugment. 0 disables time masking.")
    parser.add_argument("--specaug-time-width", type=int, default=24,
        help="Maximum width (in frames) of each time mask applied by CNN SpecAugment.")
    parser.add_argument("--specaug-time-num-masks", type=int, default=2,
        help="Number of independent time masks attempted per spectrogram in CNN SpecAugment.")
    parser.add_argument("--specaug-freq-prob", type=float, default=0.0,
        help="Probability of applying each frequency mask in CNN SpecAugment. 0 disables freq masking.")
    parser.add_argument("--specaug-freq-width", type=int, default=8,
        help="Maximum width (in frequency bins) of each frequency mask applied by CNN SpecAugment.")
    parser.add_argument("--specaug-freq-num-masks", type=int, default=2,
        help="Number of independent frequency masks attempted per spectrogram in CNN SpecAugment.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
        help="Beta(alpha, alpha) sampling parameter for mixup of CNN training batches. 0 disables mixup.")
    parser.add_argument("--use-temporal-features", action="store_true",
        help="Add a parallel MLP branch fed by per-beat temporal-dynamics features of the systole "
             "(fill fraction, envelope shape, spectral flux, sustained high band, temporal CV).")
    parser.add_argument("--model-arch", choices=["cnn", "rnn", "freq2d"], default="cnn",
        help="cnn: dilated 1D conv over time (default). rnn: bidirectional GRU/LSTM over the "
             "spectrogram read as a time sequence. freq2d: 2D conv keeping the frequency axis, "
             "hosting --freq-emphasis and --freq-attention.")
    parser.add_argument("--rnn-hidden", type=int, default=64, help="Hidden size per RNN direction.")
    parser.add_argument("--rnn-layers", type=int, default=2, help="Number of stacked RNN layers.")
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru", help="Recurrent cell type.")
    parser.add_argument("--freq-emphasis", action="store_true",
        help="(freq2d) Learnable soft emphasis of a frequency band on the input spectrogram. "
             "Neutral at init (alpha=0).")
    parser.add_argument("--freq-attention", action="store_true",
        help="(freq2d) Content-based per-frequency-band reweighting between the 2D conv blocks.")
    parser.add_argument("--freq-low-hz", type=float, default=100.0,
        help="(freq2d) Lower bound of the emphasized band for --freq-emphasis.")
    parser.add_argument("--freq-high-hz", type=float, default=600.0,
        help="(freq2d) Upper bound of the emphasized band for --freq-emphasis.")
    parser.add_argument("--freq-emphasis-alpha-init", type=float, default=0.0,
        help="(freq2d) Initial gain of the frequency-emphasis band. 0 = identity at start.")
    parser.add_argument("--freq-linear-branch", action="store_true",
        help="(cnn 1D) Add a parallel linear branch that sees only the murmur frequency band "
             "([--freq-low-hz, --freq-high-hz]) as per-bin mean+std over time, concatenated with "
             "the CNN features before the final classifier.")
    parser.add_argument("--freq-linear-hidden", type=int, default=32,
        help="Hidden/model size of the parallel frequency-band branch.")
    parser.add_argument("--freq-linear-arch", choices=["transformer", "mlp"], default="transformer",
        help="Frequency-band branch type: transformer (self-attention over band tokens) or mlp.")
    parser.add_argument("--freq-linear-heads", type=int, default=4,
        help="Attention heads when --freq-linear-arch transformer.")
    parser.add_argument("--freq-linear-layers", type=int, default=2,
        help="Number of transformer encoder layers when --freq-linear-arch transformer.")
    parser.add_argument("--window-mode", choices=["phase", "peak1s"], default="phase",
        help="phase: one spectrogram from the selected cardiac phase (default). "
             "peak1s: many fixed-length windows centered on each S1 onset (full cardiac cycle), "
             "one sample per window, aggregated to patient level.")
    parser.add_argument("--peak-window-seconds", type=float, default=1.0,
        help="Window length (seconds) centered on each onset when --window-mode peak1s.")
    parser.add_argument("--tcn-specaug-time-prob", type=float, default=0.0,
        help="Probability of applying each time mask in TCN SpecAugment. 0 disables time masking.")
    parser.add_argument("--tcn-specaug-time-width", type=int, default=20,
        help="Maximum width (in frames) of each time mask applied by TCN SpecAugment.")
    parser.add_argument("--tcn-specaug-time-num-masks", type=int, default=2,
        help="Number of independent time masks attempted per TCN window.")
    parser.add_argument("--tcn-specaug-freq-prob", type=float, default=0.0,
        help="Probability of applying each frequency mask in TCN SpecAugment. 0 disables freq masking.")
    parser.add_argument("--tcn-specaug-freq-width", type=int, default=6,
        help="Maximum width (in mel bins) of each frequency mask applied by TCN SpecAugment.")
    parser.add_argument("--tcn-specaug-freq-num-masks", type=int, default=2,
        help="Number of independent frequency masks attempted per TCN window.")
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument(
        "--cnn-phase-mode",
        choices=["systole", "diastole", "both"],
        default="systole",
        help="Cardiac phase audio used by the CNN: systole only, diastole only, or systole+diastole concatenated.",
    )
    parser.add_argument("--min-systole-seconds", type=float, default=0.10)
    parser.add_argument(
        "--systole-threshold",
        type=float,
        default=None,
        help=(
            "Optional TCN probability threshold for extracting systole into the CNN. "
            "For --cnn-phase-mode systole, only frames with p(systole) >= threshold are used. "
            "For diastole/both, argmax phase predictions are kept and low-confidence systole frames are suppressed."
        ),
    )
    parser.add_argument(
        "--systole-margin-ms",
        type=float,
        default=0.0,
        help="Expand each selected phase segment by this many milliseconds on both sides before CNN STFT extraction.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.cnn_inner_val_size < 0.5:
        raise ValueError("--cnn-inner-val-size must be greater than 0.0 and less than 0.5.")
    if args.systole_threshold is not None and not 0.0 <= args.systole_threshold <= 1.0:
        raise ValueError("--systole-threshold must be between 0.0 and 1.0.")
    if args.systole_margin_ms < 0.0:
        raise ValueError("--systole-margin-ms must be non-negative.")
    if args.low_hz < 0.0:
        raise ValueError("--low-hz must be non-negative.")
    if args.high_hz <= args.low_hz:
        raise ValueError("--high-hz must be greater than --low-hz.")
    if args.n_mels <= 0:
        raise ValueError("--n-mels must be greater than 0.")
    if args.tcn_boundary_ignore_ms < 0.0:
        raise ValueError("--tcn-boundary-ignore-ms must be non-negative.")
    if args.tcn_systole_weight_multiplier <= 0.0:
        raise ValueError("--tcn-systole-weight-multiplier must be greater than 0.")
    if args.tcn_target_mode == "systole-binary" and args.cnn_phase_mode in {"diastole", "both"}:
        raise ValueError("--cnn-phase-mode diastole/both requires --tcn-target-mode cardiac-phase.")
    if not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0.0 and 1.0.")
    if args.weak_murmur_weight <= 0.0:
        raise ValueError("--weak-murmur-weight must be greater than 0.")
    if args.moderate_murmur_weight <= 0.0:
        raise ValueError("--moderate-murmur-weight must be greater than 0.")
    if args.mil_max_instances <= 0:
        raise ValueError("--mil-max-instances must be greater than 0.")
    if args.mil_location_embedding_dim <= 0:
        raise ValueError("--mil-location-embedding-dim must be greater than 0.")
    if args.mil_instance_loss_weight < 0.0:
        raise ValueError("--mil-instance-loss-weight must be non-negative.")
    if not 0.0 <= args.ltsrr_prob <= 1.0:
        raise ValueError("--ltsrr-prob must be between 0.0 and 1.0.")
    if args.ltsrr_k <= 0:
        raise ValueError("--ltsrr-k must be greater than 0.")
    if not 0.0 < args.ltsrr_frequency_ratio <= 1.0:
        raise ValueError("--ltsrr-frequency-ratio must be greater than 0 and at most 1.")
    if args.smote_k_neighbors <= 0:
        raise ValueError("--smote-k-neighbors must be greater than 0.")
    if not 0.0 < args.smote_target_ratio <= 1.0:
        raise ValueError("--smote-target-ratio must be greater than 0 and at most 1.")
    if args.patient_mil_attention and args.smote_minority_augmentation:
        raise ValueError("--smote-minority-augmentation is not supported with --patient-mil-attention.")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative.")
    if args.focal_alpha is not None and not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be between 0.0 and 1.0.")
    if args.auc_loss_weight < 0.0:
        raise ValueError("--auc-loss-weight must be non-negative.")
    if args.auc_loss_margin < 0.0:
        raise ValueError("--auc-loss-margin must be non-negative.")
    if args.auc_loss_weight > 0.0 and args.cnn_batch_size < 2:
        raise ValueError("--cnn-batch-size must be at least 2 when --auc-loss-weight is enabled.")
    aux_pitch_loss_weight = float(getattr(args, "aux_pitch_loss_weight", 0.0))
    if aux_pitch_loss_weight < 0.0:
        raise ValueError("--aux-pitch-loss-weight must be non-negative.")
    if aux_pitch_loss_weight > 0.0:
        if str(getattr(args, "model_arch", "cnn")) != "cnn":
            raise ValueError("--aux-pitch-loss-weight requires --model-arch cnn.")
        if args.patient_mil_attention:
            raise ValueError("--aux-pitch-loss-weight is not supported with --patient-mil-attention.")
        if args.smote_minority_augmentation:
            raise ValueError("--aux-pitch-loss-weight is not supported with --smote-minority-augmentation.")
        if float(getattr(args, "mixup_alpha", 0.0)) > 0.0:
            raise ValueError("--aux-pitch-loss-weight is not supported with mixup.")
    # The aux head emits one logit per pitch class (Low/Medium/High); 0 disables it downstream.
    args.aux_pitch_classes = 3 if aux_pitch_loss_weight > 0.0 else 0
    if bool(getattr(args, "demographic", False)):
        if str(getattr(args, "model_arch", "cnn")) != "cnn":
            raise ValueError("--demographic requires --model-arch cnn.")
        if args.patient_mil_attention:
            raise ValueError("--demographic is not supported with --patient-mil-attention.")
        if args.smote_minority_augmentation:
            raise ValueError("--demographic is not supported with --smote-minority-augmentation.")
        if float(getattr(args, "mixup_alpha", 0.0)) > 0.0:
            raise ValueError("--demographic is not supported with mixup.")
        if bool(getattr(args, "use_temporal_features", False)):
            raise ValueError("--demographic is not supported together with --use-temporal-features.")
    if bool(getattr(args, "phase_contrast_dual", False)) or bool(getattr(args, "phase_contrast_robust", False)):
        args.phase_contrast = True  # dual/robust are variants of phase-contrast
    if bool(getattr(args, "phase_contrast", False)):
        if args.cnn_phase_mode != "systole":
            raise ValueError("--phase-contrast requires --cnn-phase-mode systole.")
        if getattr(args, "stft_segment_mode", "concat") != "per-segment":
            raise ValueError("--phase-contrast requires --stft-segment-mode per-segment.")
        if args.tcn_target_mode != "cardiac-phase":
            raise ValueError("--phase-contrast requires --tcn-target-mode cardiac-phase (needs diastole segments).")
    parse_score_weights(args.score_weights)
