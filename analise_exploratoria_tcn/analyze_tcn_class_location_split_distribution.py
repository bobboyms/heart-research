#!/usr/bin/env python3
"""EDA for TCN label distribution by class, location, and split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "other",
    1: "s1",
    2: "systole",
    3: "s2",
    4: "diastole",
}
CLASS_ORDER = list(LABEL_NAMES.values())
SPLIT_ORDER = ["train", "val", "test"]
LOCATION_ORDER = ["AV", "PV", "TV", "MV", "Phc"]


def parse_args() -> argparse.Namespace:
    default_tcn_dir = (
        Path("modeling/Grupo H Nested TCN CNN systole")
        / "outputs_nested"
        / "fold_1"
        / "tcn"
    )
    default_output_dir = (
        Path("analise_exploratoria_tcn")
        / "outputs"
        / "fold_1"
        / "class_location_split_distribution"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TCN training labels by cardiac class, auscultation "
            "location, and train/val/test split."
        )
    )
    parser.add_argument("--tcn-dir", type=Path, default=default_tcn_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional direct path to the TCN feature cache directory.",
    )
    return parser.parse_args()


def load_manifest(tcn_dir: Path) -> dict[str, list[dict]]:
    manifest_path = tcn_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def build_cache_index(tcn_dir: Path, cache_dir: Path | None) -> dict[str, Path]:
    search_root = cache_dir if cache_dir is not None else tcn_dir / "cache"
    if not search_root.exists():
        raise FileNotFoundError(f"Missing TCN cache directory: {search_root}")
    index: dict[str, Path] = {}
    for path in search_root.rglob("*.npz"):
        index[path.stem] = path
    if not index:
        raise FileNotFoundError(f"No .npz files found under {search_root}")
    return index


def estimate_hop_seconds(starts: np.ndarray, ends: np.ndarray) -> float:
    if len(starts) > 1:
        diffs = np.diff(starts.astype(float))
        diffs = diffs[diffs > 0]
        if len(diffs):
            return float(np.median(diffs))
    durations = ends.astype(float) - starts.astype(float)
    durations = durations[durations > 0]
    if len(durations):
        return float(np.median(durations))
    return 0.0


def summarize_recording(item: dict, split: str, npz_path: Path) -> tuple[dict, list[dict]]:
    with np.load(npz_path, allow_pickle=False) as data:
        labels = data["y"].astype(int)
        starts = data["frame_starts_s"]
        ends = data["frame_ends_s"]

    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    total_frames = int(counts.sum())
    hop_seconds = estimate_hop_seconds(starts, ends)
    total_seconds = total_frames * hop_seconds

    base = {
        "split": split,
        "recording_id": item["recording_id"],
        "patient_id": str(item["patient_id"]),
        "location": item.get("location", ""),
        "murmur": item.get("murmur", ""),
        "outcome": item.get("outcome", ""),
        "total_frames": total_frames,
        "hop_seconds": hop_seconds,
        "total_seconds": total_seconds,
        "npz_path": str(npz_path),
    }

    recording_row = dict(base)
    class_rows: list[dict] = []
    for label_id, class_name in LABEL_NAMES.items():
        frames = int(counts[label_id])
        seconds = frames * hop_seconds
        pct = frames / total_frames if total_frames else 0.0
        recording_row[f"{class_name}_frames"] = frames
        recording_row[f"{class_name}_seconds"] = seconds
        recording_row[f"{class_name}_pct"] = pct
        class_rows.append(
            {
                **base,
                "class_id": label_id,
                "class_name": class_name,
                "frames": frames,
                "seconds": seconds,
                "pct_within_recording": pct,
            }
        )
    return recording_row, class_rows


def add_group_percentages(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    result["frame_pct_within_group"] = (
        result["frames"] / result.groupby(group_cols)["frames"].transform("sum")
    )
    result["seconds_pct_within_group"] = (
        result["seconds"] / result.groupby(group_cols)["seconds"].transform("sum")
    )
    return result


def aggregate_distributions(class_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split_class = (
        class_df.groupby(
            ["split", "class_id", "class_name"], as_index=False, observed=True
        )
        .agg(
            frames=("frames", "sum"),
            seconds=("seconds", "sum"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            recordings_with_class=("frames", lambda s: int((s > 0).sum())),
        )
        .pipe(add_group_percentages, ["split"])
    )

    split_location_class = (
        class_df.groupby(
            ["split", "location", "class_id", "class_name"],
            as_index=False,
            observed=True,
        )
        .agg(
            frames=("frames", "sum"),
            seconds=("seconds", "sum"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            recordings_with_class=("frames", lambda s: int((s > 0).sum())),
        )
        .pipe(add_group_percentages, ["split", "location"])
    )

    split_location = (
        class_df.drop_duplicates(["split", "location", "recording_id"])
        .groupby(["split", "location"], as_index=False, observed=True)
        .agg(
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            total_frames=("total_frames", "sum"),
            total_seconds=("total_seconds", "sum"),
        )
    )
    split_location["recording_pct_within_split"] = (
        split_location["recordings"]
        / split_location.groupby("split")["recordings"].transform("sum")
    )
    split_location["frame_pct_within_split"] = (
        split_location["total_frames"]
        / split_location.groupby("split")["total_frames"].transform("sum")
    )

    split_summary = (
        class_df.drop_duplicates(["split", "recording_id"])
        .groupby("split", as_index=False, observed=True)
        .agg(
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            total_frames=("total_frames", "sum"),
            total_seconds=("total_seconds", "sum"),
        )
    )

    return {
        "split_summary": split_summary,
        "split_class_distribution": split_class,
        "split_location_distribution": split_location,
        "split_location_class_distribution": split_location_class,
    }


def order_categories(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    sort_cols: list[str] = []
    if "split" in result.columns:
        result["_split_order"] = result["split"].map(
            {value: idx for idx, value in enumerate(SPLIT_ORDER)}
        )
        sort_cols.append("_split_order")
    if "location" in result.columns:
        result["_location_order"] = result["location"].map(
            {value: idx for idx, value in enumerate(LOCATION_ORDER)}
        )
        sort_cols.append("_location_order")
    if "class_name" in result.columns:
        result["_class_order"] = result["class_name"].map(
            {value: idx for idx, value in enumerate(CLASS_ORDER)}
        )
        sort_cols.append("_class_order")
    if sort_cols:
        result = result.sort_values(sort_cols)
        result = result.drop(columns=sort_cols)
    return result


def save_csvs(output_dir: Path, tables: dict[str, pd.DataFrame], recording_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_df.to_csv(output_dir / "recording_class_distribution.csv", index=False)
    for name, table in tables.items():
        order_categories(table).to_csv(output_dir / f"{name}.csv", index=False)


def plot_split_class(split_class: pd.DataFrame, output_dir: Path) -> None:
    table = split_class.pivot(
        index="split", columns="class_name", values="frame_pct_within_group"
    ).reindex(index=SPLIT_ORDER, columns=CLASS_ORDER)
    ax = (table * 100).plot(kind="bar", stacked=True, figsize=(9, 5), width=0.75)
    ax.set_title("Distribuicao de frames por classe e split")
    ax.set_xlabel("Split")
    ax.set_ylabel("% dos frames")
    ax.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "split_class_frame_pct.png", dpi=180)
    plt.close()


def plot_location_class(split_location_class: pd.DataFrame, output_dir: Path) -> None:
    for split in SPLIT_ORDER:
        subset = split_location_class[split_location_class["split"] == split]
        if subset.empty:
            continue
        table = subset.pivot(
            index="location",
            columns="class_name",
            values="frame_pct_within_group",
        ).reindex(index=LOCATION_ORDER, columns=CLASS_ORDER)
        ax = (table * 100).plot(kind="bar", stacked=True, figsize=(9, 5), width=0.75)
        ax.set_title(f"Distribuicao de frames por classe e local - {split}")
        ax.set_xlabel("Local")
        ax.set_ylabel("% dos frames")
        ax.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split}_location_class_frame_pct.png", dpi=180)
        plt.close()


def plot_location_volume(split_location: pd.DataFrame, output_dir: Path) -> None:
    table = split_location.pivot(
        index="location", columns="split", values="recordings"
    ).reindex(index=LOCATION_ORDER, columns=SPLIT_ORDER)
    ax = table.plot(kind="bar", figsize=(8, 5), width=0.75)
    ax.set_title("Gravacoes por local e split")
    ax.set_xlabel("Local")
    ax.set_ylabel("Gravacoes")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "location_recordings_by_split.png", dpi=180)
    plt.close()


def pct_fmt(series: pd.Series) -> pd.Series:
    return (series * 100).round(2)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    formatted = df.loc[:, columns].copy()
    for col in formatted.columns:
        if col.endswith("_pct") or "_pct_" in col:
            formatted[col] = pct_fmt(formatted[col])
        if col.endswith("seconds") or col == "total_seconds":
            formatted[col] = formatted[col].round(1)
    return formatted.to_markdown(index=False)


def write_summary(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    split_summary = order_categories(tables["split_summary"])
    split_class = order_categories(tables["split_class_distribution"])
    split_location = order_categories(tables["split_location_distribution"])
    split_location_class = order_categories(tables["split_location_class_distribution"])

    systole = split_class[split_class["class_name"] == "systole"].copy()
    systole["systole_frame_pct"] = systole["frame_pct_within_group"]
    systole_by_location = split_location_class[
        split_location_class["class_name"] == "systole"
    ].copy()
    systole_by_location["systole_frame_pct"] = systole_by_location[
        "frame_pct_within_group"
    ]

    lines = [
        "# Analise exploratoria TCN - distribuicao por classe, local e split",
        "",
        "Esta analise usa os labels `y` dos arquivos `.npz` do cache do TCN.",
        "Os percentuais sao calculados por contagem de frames; segundos usam `frames * hop` de cada gravacao.",
        "",
        "## Volume por split",
        "",
        markdown_table(
            split_summary,
            ["split", "recordings", "patients", "total_frames", "total_seconds"],
        ),
        "",
        "## Distribuicao por classe e split",
        "",
        markdown_table(
            split_class,
            [
                "split",
                "class_name",
                "frames",
                "seconds",
                "frame_pct_within_group",
                "recordings_with_class",
            ],
        ),
        "",
        "## Percentual de sistole por split",
        "",
        markdown_table(
            systole,
            ["split", "frames", "seconds", "systole_frame_pct", "recordings_with_class"],
        ),
        "",
        "## Volume por local e split",
        "",
        markdown_table(
            split_location,
            [
                "split",
                "location",
                "recordings",
                "patients",
                "total_frames",
                "total_seconds",
                "recording_pct_within_split",
                "frame_pct_within_split",
            ],
        ),
        "",
        "## Percentual de sistole por local dentro de cada split",
        "",
        markdown_table(
            systole_by_location,
            [
                "split",
                "location",
                "frames",
                "seconds",
                "systole_frame_pct",
                "recordings_with_class",
            ],
        ),
        "",
        "## Arquivos gerados",
        "",
        "- `recording_class_distribution.csv`",
        "- `split_summary.csv`",
        "- `split_class_distribution.csv`",
        "- `split_location_distribution.csv`",
        "- `split_location_class_distribution.csv`",
        "- `split_class_frame_pct.png`",
        "- `{train,val,test}_location_class_frame_pct.png`",
        "- `location_recordings_by_split.png`",
        "",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.tcn_dir)
    cache_index = build_cache_index(args.tcn_dir, args.cache_dir)

    recording_rows: list[dict] = []
    class_rows: list[dict] = []
    missing: list[str] = []
    for split, items in manifest.items():
        for item in items:
            recording_id = item["recording_id"]
            npz_path = cache_index.get(recording_id)
            if npz_path is None:
                missing.append(recording_id)
                continue
            recording_row, rows = summarize_recording(item, split, npz_path)
            recording_rows.append(recording_row)
            class_rows.extend(rows)

    if missing:
        missing_preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} cache files. First missing: {missing_preview}"
        )

    recording_df = order_categories(pd.DataFrame(recording_rows))
    class_df = order_categories(pd.DataFrame(class_rows))
    tables = aggregate_distributions(class_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csvs(args.output_dir, tables, recording_df)
    plot_split_class(order_categories(tables["split_class_distribution"]), args.output_dir)
    plot_location_class(
        order_categories(tables["split_location_class_distribution"]),
        args.output_dir,
    )
    plot_location_volume(order_categories(tables["split_location_distribution"]), args.output_dir)
    write_summary(args.output_dir, tables)
    print(f"Wrote EDA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
