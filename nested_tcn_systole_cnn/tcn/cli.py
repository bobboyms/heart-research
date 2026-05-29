from __future__ import annotations


import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch


from .audio import choose_device, set_seed
from .augment import build_tcn_specaug_augmenter
from .config import FeatureConfig, OTHER_MODES, TARGET_MODES, label_names_for_cfg
from .data import build_recording_index, split_by_patient
from .dataset import CirCorFrameDataset, compute_normalizer, compute_training_stats, count_train_labels, make_loaders
from .inference import checkpoint_payload, load_checkpoint_for_eval, predict_wav
from .losses import apply_systole_weight_multiplier, class_weights_from_counts, create_loss
from .model import TCNFrameSegmenter
from .report import load_cached_label_counts, load_cached_normalizer, maybe_discard_cached_label_counts, save_confusion_matrix, save_history_plot, save_json, save_split_manifest, write_summary
from .training import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_dataset = repo_root / "circor-heart-sound-1.0.3"
    default_output = script_dir / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised TCN to predict CirCor cardiac phase labels "
            "for every spectrogram frame: 0=other, 1=S1, 2=systole, 3=S2, 4=diastole."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--reuse-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse output normalization.json and train_label_counts.json when present. "
            "This avoids rereading every cached feature file after interrupted runs."
        ),
    )
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--exclude-murmur-unknown", action="store_true")

    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=1800.0)
    parser.add_argument("--no-deltas", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=["center", "overlap"],
        default="overlap",
        help="How frame labels are created from TSV intervals. 'overlap' uses the interval with largest frame overlap.",
    )
    parser.add_argument(
        "--boundary-ignore-ms",
        type=float,
        default=0.0,
        help="Mark frames within this distance from a phase boundary as IGNORE_INDEX during training.",
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="cardiac-phase",
        help=(
            "Prediction target. 'cardiac-phase' keeps 0=other, 1=S1, 2=systole, 3=S2, 4=diastole. "
            "'systole-binary' trains 0=non_systole and 1=systole, while exported TSV segments still use label 2."
        ),
    )
    parser.add_argument(
        "--other-mode",
        choices=OTHER_MODES,
        default="keep",
        help=(
            "How to handle TSV/background label 0 during training. "
            "'keep' trains it as the other/non_systole class; 'ignore' maps original label 0 frames to IGNORE_INDEX."
        ),
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build/reuse feature cache plus normalization/label statistics, then exit before training.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-window-seconds",
        type=float,
        default=6.0,
        help="Train on fixed-length windows instead of full recordings. Use 0 to train on full recordings.",
    )
    parser.add_argument(
        "--train-window-hop-seconds",
        type=float,
        default=3.0,
        help="Hop between training windows. Ignored when --train-window-seconds is 0.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--pooling",
        choices=["none", "attention"],
        default="none",
        help=(
            "Optional temporal attention pooling context inside the TCN. "
            "It preserves frame-level output length; it does not collapse time."
        ),
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a causal TCN. Default is non-causal, which is usually better for offline segmentation.",
    )
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--systole-weight-multiplier",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier applied to the systole class weight in the loss. "
            "Use values >1 to make systole false negatives more expensive."
        ),
    )
    parser.add_argument("--loss", choices=["ce", "ce_dice", "focal", "focal_dice"], default="ce_dice")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--median-filter-frames", type=int, default=5)
    parser.add_argument("--min-segment-frames", type=int, default=3)

    parser.add_argument("--specaug-time-prob", type=float, default=0.0,
        help="Probability of applying each time mask during TCN training. 0 disables.")
    parser.add_argument("--specaug-time-width", type=int, default=20,
        help="Maximum width (frames) of each time mask.")
    parser.add_argument("--specaug-time-num-masks", type=int, default=2,
        help="Number of time masks attempted per training window.")
    parser.add_argument("--specaug-freq-prob", type=float, default=0.0,
        help="Probability of applying each frequency mask during TCN training. 0 disables.")
    parser.add_argument("--specaug-freq-width", type=int, default=6,
        help="Maximum width (mel bins) of each frequency mask.")
    parser.add_argument("--specaug-freq-num-masks", type=int, default=2,
        help="Number of frequency masks attempted per training window.")

    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars during feature preparation, training, and evaluation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--predict-wav",
        type=Path,
        default=None,
        help="Run inference on a WAV file and write predicted cardiac phase segments.",
    )
    parser.add_argument(
        "--predict-output",
        type=Path,
        default=None,
        help="Output TSV for --predict-wav. Defaults to <wav>.predicted.tsv.",
    )
    parser.add_argument(
        "--predict-frame-output",
        type=Path,
        default=None,
        help="Optional CSV with per-frame probabilities for --predict-wav.",
    )
    parser.add_argument(
        "--allow-mps-predict",
        action="store_true",
        help="Allow single-file prediction on MPS. CPU is the safer default for this path.",
    )
    parser.add_argument(
        "--systole-threshold",
        type=float,
        default=None,
        help=(
            "Optional inference-only threshold for --predict-wav. "
            "When set, exports only frames with p(systole) >= threshold as systole segments."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.systole_threshold is not None and not 0.0 <= args.systole_threshold <= 1.0:
        raise ValueError("--systole-threshold must be between 0.0 and 1.0.")
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (output_dir / "cache")

    device = choose_device(args.device)
    print(f"Using device: {device}")

    if args.predict_wav is not None:
        if args.checkpoint is None:
            raise ValueError("--predict-wav requires --checkpoint")
        if args.epochs != 30 or args.batch_size != 8:
            print("Note: --predict-wav runs inference only; --epochs and --batch-size are ignored.")
        if device.type == "mps" and not args.allow_mps_predict:
            print("Single-file prediction is using CPU because MPS can be slow/unstable on this GroupNorm path.")
            print("Use --allow-mps-predict to force MPS for prediction.")
            device = torch.device("cpu")
        model, normalizer, cfg, _checkpoint = load_checkpoint_for_eval(args.checkpoint, device)
        output_tsv = args.predict_output or args.predict_wav.with_suffix(".predicted.tsv")
        predict_wav(
            model=model,
            normalizer=normalizer,
            cfg=cfg,
            wav_path=args.predict_wav,
            output_tsv=output_tsv,
            frame_output=args.predict_frame_output,
            device=device,
            postprocess=args.postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
            systole_threshold=args.systole_threshold,
        )
        return

    items = build_recording_index(args)
    splits = split_by_patient(items, args.dataset_dir, args.val_size, args.test_size, args.seed)
    save_split_manifest(output_dir, splits)

    cfg = FeatureConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_mels=args.n_mels,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        add_deltas=not args.no_deltas,
        label_mode=args.label_mode,
        boundary_ignore_ms=args.boundary_ignore_ms,
        target_mode=args.target_mode,
        other_mode=args.other_mode,
    )
    label_names = label_names_for_cfg(cfg)

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        model, normalizer, cfg, _checkpoint = load_checkpoint_for_eval(args.checkpoint, device)
        args.causal = bool(_checkpoint["model_config"].get("causal", True))
        args.target_mode = cfg.target_mode
        label_names = label_names_for_cfg(cfg)
        train_dataset = CirCorFrameDataset(splits["train"], cfg, cache_dir, args.overwrite_cache)
        label_counts = load_cached_label_counts(output_dir, label_names) if args.reuse_stats and not args.overwrite_cache else None
        label_counts = maybe_discard_cached_label_counts(label_counts, cfg)
        if label_counts is None:
            label_counts = count_train_labels(train_dataset, show_progress=args.progress)
    else:
        train_dataset_for_stats = CirCorFrameDataset(splits["train"], cfg, cache_dir, args.overwrite_cache)
        normalizer = load_cached_normalizer(output_dir) if args.reuse_stats and not args.overwrite_cache else None
        label_counts = load_cached_label_counts(output_dir, label_names) if args.reuse_stats and not args.overwrite_cache else None
        label_counts = maybe_discard_cached_label_counts(label_counts, cfg)

        if normalizer is None and label_counts is None:
            print("Computing/caching training features, normalization statistics, and label counts...")
            normalizer, label_counts = compute_training_stats(train_dataset_for_stats, show_progress=args.progress)
            save_json(output_dir / "normalization.json", asdict(normalizer))
            save_json(
                output_dir / "train_label_counts.json",
                {label_names[i]: int(count) for i, count in enumerate(label_counts)},
            )
        elif normalizer is None:
            print("Computing/caching training features and normalization statistics...")
            normalizer = compute_normalizer(train_dataset_for_stats, show_progress=args.progress)
            save_json(output_dir / "normalization.json", asdict(normalizer))
        else:
            print("Reusing normalization statistics from existing normalization.json.")

        if label_counts is not None:
            print("Train label counts are available.")
        else:
            label_counts = count_train_labels(train_dataset_for_stats, show_progress=args.progress)
            save_json(
                output_dir / "train_label_counts.json",
                {label_names[i]: int(count) for i, count in enumerate(label_counts)},
            )

        if args.prepare_only:
            print(f"Preparation complete. Stats and cache are ready under {output_dir}.")
            return

        in_channels = cfg.n_mels * (2 if cfg.add_deltas else 1)
        model = TCNFrameSegmenter(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            levels=args.levels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            num_classes=len(label_names),
            causal=args.causal,
            pooling=args.pooling,
        ).to(device)

    train_augmenter = build_tcn_specaug_augmenter(args)
    if train_augmenter is not None:
        print(
            f"TCN SpecAugment enabled — time(prob={args.specaug_time_prob}, "
            f"width<={args.specaug_time_width}, n={args.specaug_time_num_masks}); "
            f"freq(prob={args.specaug_freq_prob}, width<={args.specaug_freq_width}, "
            f"n={args.specaug_freq_num_masks})."
        )
    loaders = make_loaders(
        splits=splits,
        cfg=cfg,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
        normalizer=normalizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_window_seconds=args.train_window_seconds,
        train_window_hop_seconds=args.train_window_hop_seconds,
        show_progress=args.progress,
        train_augmenter=train_augmenter,
    )

    class_weights = class_weights_from_counts(label_counts).to(device) if args.use_class_weights else None
    if class_weights is not None:
        class_weights = apply_systole_weight_multiplier(class_weights, cfg.target_mode, args.systole_weight_multiplier)
        print(
            "Class weights:",
            {label_names[i]: float(class_weights[i].detach().cpu()) for i in range(len(label_names))},
        )
    loss_fn = create_loss(args, class_weights, len(label_names))

    best_epoch: int | None = None
    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_fn,
                device,
                args.grad_clip,
                show_progress=args.progress,
                epoch=epoch,
            )
            val_metrics = evaluate(
                model,
                loaders["val"],
                loss_fn,
                device,
                label_names,
                show_progress=args.progress,
                desc=f"Epoch {epoch:03d} val",
                postprocess=args.postprocess,
                median_filter_frames=args.median_filter_frames,
                min_segment_frames=args.min_segment_frames,
            )
            val_macro_f1 = float(val_metrics["macro_f1"])
            val_accuracy = float(val_metrics["accuracy"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "val_mean_iou": float(val_metrics["mean_iou"]),
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(model, args, cfg, normalizer, epoch, val_metrics),
                    output_dir / "best_model.pt",
                )

        print(f"Training finished in {(time.time() - start) / 60.0:.1f} minutes.")
        save_history_plot(output_dir, history)

        model, normalizer, cfg, _checkpoint = load_checkpoint_for_eval(output_dir / "best_model.pt", device)
        label_names = label_names_for_cfg(cfg)

    final_metrics = {
        "val": evaluate(
            model,
            loaders["val"],
            loss_fn,
            device,
            label_names,
            show_progress=args.progress,
            desc="Final val",
            postprocess=args.postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
        ),
        "test": evaluate(
            model,
            loaders["test"],
            loss_fn,
            device,
            label_names,
            show_progress=args.progress,
            desc="Final test",
            postprocess=args.postprocess,
            median_filter_frames=args.median_filter_frames,
            min_segment_frames=args.min_segment_frames,
        ),
    }
    save_json(output_dir / "metrics.json", final_metrics)
    save_confusion_matrix(output_dir, "val", final_metrics["val"]["confusion"], label_names)  # type: ignore[arg-type]
    save_confusion_matrix(output_dir, "test", final_metrics["test"]["confusion"], label_names)  # type: ignore[arg-type]
    write_summary(output_dir, args, cfg, splits, label_counts, label_names, final_metrics, best_epoch)
    print(f"Done. Summary written to {output_dir / 'summary.md'}")
