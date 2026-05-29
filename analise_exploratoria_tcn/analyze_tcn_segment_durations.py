#!/usr/bin/env python3
"""EDA for real TSV segment durations used by the nested TCN fold."""

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
SHORT_SYSTOLE_THRESHOLDS_MS = [100, 150, 200]


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
        / "segment_durations"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Measure real TSV segment durations by class, split, location, "
            "recording, and patient."
        )
    )
    parser.add_argument("--tcn-dir", type=Path, default=default_tcn_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def load_manifest(tcn_dir: Path) -> dict[str, list[dict]]:
    manifest_path = tcn_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def read_tsv_segments(item: dict, split: str) -> pd.DataFrame:
    tsv_path = Path(item["tsv_path"])
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}")

    table = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["start_s", "end_s", "class_id"],
    )
    table["class_id"] = table["class_id"].astype(int)
    table["class_name"] = table["class_id"].map(LABEL_NAMES).fillna("unknown")
    table["duration_s"] = table["end_s"] - table["start_s"]
    table["duration_ms"] = table["duration_s"] * 1000
    table["split"] = split
    table["recording_id"] = item["recording_id"]
    table["patient_id"] = str(item["patient_id"])
    table["location"] = item.get("location", "")
    table["murmur"] = item.get("murmur", "")
    table["outcome"] = item.get("outcome", "")
    table["tsv_path"] = str(tsv_path)
    table["segment_index"] = np.arange(len(table))
    return table[
        [
            "split",
            "recording_id",
            "patient_id",
            "location",
            "murmur",
            "outcome",
            "segment_index",
            "start_s",
            "end_s",
            "duration_s",
            "duration_ms",
            "class_id",
            "class_name",
            "tsv_path",
        ]
    ]


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


def duration_quantiles(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    quantiles = (
        df.groupby(group_cols, as_index=False, observed=True)
        .agg(
            segments=("duration_ms", "size"),
            recordings=("recording_id", "nunique"),
            patients=("patient_id", "nunique"),
            mean_ms=("duration_ms", "mean"),
            std_ms=("duration_ms", "std"),
            min_ms=("duration_ms", "min"),
            p5_ms=("duration_ms", lambda s: s.quantile(0.05)),
            p25_ms=("duration_ms", lambda s: s.quantile(0.25)),
            median_ms=("duration_ms", "median"),
            p75_ms=("duration_ms", lambda s: s.quantile(0.75)),
            p95_ms=("duration_ms", lambda s: s.quantile(0.95)),
            max_ms=("duration_ms", "max"),
        )
        .fillna({"std_ms": 0.0})
    )
    return order_categories(quantiles)


def short_systole_summary(systole: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = systole.groupby(group_cols, observed=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key, strict=True))
        total = len(group)
        row = {
            **base,
            "systole_segments": total,
            "recordings": group["recording_id"].nunique(),
            "patients": group["patient_id"].nunique(),
            "median_ms": group["duration_ms"].median(),
            "p5_ms": group["duration_ms"].quantile(0.05),
            "p25_ms": group["duration_ms"].quantile(0.25),
            "p75_ms": group["duration_ms"].quantile(0.75),
            "p95_ms": group["duration_ms"].quantile(0.95),
        }
        for threshold in SHORT_SYSTOLE_THRESHOLDS_MS:
            count = int((group["duration_ms"] < threshold).sum())
            row[f"lt_{threshold}ms"] = count
            row[f"lt_{threshold}ms_pct"] = count / total if total else 0.0
        rows.append(row)
    return order_categories(pd.DataFrame(rows))


def aggregate_by_recording(systole: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        systole.groupby(
            ["split", "location", "patient_id", "recording_id", "murmur", "outcome"],
            as_index=False,
            observed=True,
        )
        .agg(
            systole_segments=("duration_ms", "size"),
            systole_total_ms=("duration_ms", "sum"),
            systole_mean_ms=("duration_ms", "mean"),
            systole_median_ms=("duration_ms", "median"),
            systole_min_ms=("duration_ms", "min"),
            systole_p5_ms=("duration_ms", lambda s: s.quantile(0.05)),
            systole_p25_ms=("duration_ms", lambda s: s.quantile(0.25)),
            systole_p75_ms=("duration_ms", lambda s: s.quantile(0.75)),
            systole_p95_ms=("duration_ms", lambda s: s.quantile(0.95)),
            systole_max_ms=("duration_ms", "max"),
        )
    )
    for threshold in SHORT_SYSTOLE_THRESHOLDS_MS:
        counts = (
            systole.assign(short=systole["duration_ms"] < threshold)
            .groupby(["recording_id"], observed=True)["short"]
            .sum()
            .rename(f"lt_{threshold}ms")
        )
        grouped = grouped.merge(counts, on="recording_id", how="left")
        grouped[f"lt_{threshold}ms_pct"] = (
            grouped[f"lt_{threshold}ms"] / grouped["systole_segments"]
        )
    return order_categories(grouped)


def aggregate_by_patient(systole: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        systole.groupby(["split", "patient_id", "murmur", "outcome"], as_index=False)
        .agg(
            locations=("location", "nunique"),
            recordings=("recording_id", "nunique"),
            systole_segments=("duration_ms", "size"),
            systole_total_ms=("duration_ms", "sum"),
            systole_mean_ms=("duration_ms", "mean"),
            systole_median_ms=("duration_ms", "median"),
            systole_min_ms=("duration_ms", "min"),
            systole_p5_ms=("duration_ms", lambda s: s.quantile(0.05)),
            systole_p25_ms=("duration_ms", lambda s: s.quantile(0.25)),
            systole_p75_ms=("duration_ms", lambda s: s.quantile(0.75)),
            systole_p95_ms=("duration_ms", lambda s: s.quantile(0.95)),
            systole_max_ms=("duration_ms", "max"),
        )
    )
    for threshold in SHORT_SYSTOLE_THRESHOLDS_MS:
        counts = (
            systole.assign(short=systole["duration_ms"] < threshold)
            .groupby(["split", "patient_id"], observed=True)["short"]
            .sum()
            .rename(f"lt_{threshold}ms")
            .reset_index()
        )
        grouped = grouped.merge(counts, on=["split", "patient_id"], how="left")
        grouped[f"lt_{threshold}ms_pct"] = (
            grouped[f"lt_{threshold}ms"] / grouped["systole_segments"]
        )
    return order_categories(grouped)


def plot_class_duration_box(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    cardiac = df[df["class_name"].isin(["s1", "systole", "s2", "diastole"])]
    for ax, split in zip(axes, SPLIT_ORDER, strict=True):
        split_df = cardiac[cardiac["split"] == split]
        values = [
            split_df[split_df["class_name"] == class_name]["duration_ms"].to_numpy()
            for class_name in ["s1", "systole", "s2", "diastole"]
        ]
        labels = ["s1", "systole", "s2", "diastole"]
        ax.boxplot(values, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title(split)
        ax.set_xlabel("Classe")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Duracao do segmento (ms)")
    fig.suptitle("Duracao dos segmentos cardiacos por split")
    plt.tight_layout()
    plt.savefig(output_dir / "cardiac_class_duration_boxplot_by_split.png", dpi=180)
    plt.close()


def plot_systole_histogram(systole: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, min(800, systole["duration_ms"].max() + 25), 25)
    for split in SPLIT_ORDER:
        values = systole[systole["split"] == split]["duration_ms"].to_numpy()
        ax.hist(values, bins=bins, alpha=0.45, label=split, density=True)
    for threshold in SHORT_SYSTOLE_THRESHOLDS_MS:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Distribuicao da duracao da sistole real")
    ax.set_xlabel("Duracao da sistole (ms)")
    ax.set_ylabel("Densidade")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "systole_duration_histogram_by_split.png", dpi=180)
    plt.close()


def plot_systole_by_location(systole: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER, strict=True):
        split_df = systole[systole["split"] == split]
        values = [
            split_df[split_df["location"] == location]["duration_ms"].to_numpy()
            for location in LOCATION_ORDER
            if not split_df[split_df["location"] == location].empty
        ]
        labels = [
            location
            for location in LOCATION_ORDER
            if not split_df[split_df["location"] == location].empty
        ]
        ax.boxplot(values, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title(split)
        ax.set_xlabel("Local")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Duracao da sistole (ms)")
    fig.suptitle("Duracao da sistole por local e split")
    plt.tight_layout()
    plt.savefig(output_dir / "systole_duration_boxplot_by_location_split.png", dpi=180)
    plt.close()


def pct_fmt(series: pd.Series) -> pd.Series:
    return (series * 100).round(2)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    formatted = df.loc[:, columns].copy()
    for col in formatted.columns:
        if col.endswith("_pct") or "_pct_" in col:
            formatted[col] = pct_fmt(formatted[col])
        elif col.endswith("_ms") or col in {"mean_ms", "median_ms"}:
            formatted[col] = formatted[col].round(1)
    return formatted.to_markdown(index=False)


def write_summary(
    output_dir: Path,
    split_class_quantiles: pd.DataFrame,
    split_location_quantiles: pd.DataFrame,
    short_by_split: pd.DataFrame,
    short_by_location: pd.DataFrame,
    patient_systole: pd.DataFrame,
    recording_systole: pd.DataFrame,
    invalid_count: int,
) -> None:
    cardiac_quantiles = split_class_quantiles[
        split_class_quantiles["class_name"].isin(["s1", "systole", "s2", "diastole"])
    ]
    systole_location_quantiles = split_location_quantiles[
        split_location_quantiles["class_name"] == "systole"
    ]
    patient_short_top = patient_systole.sort_values(
        ["lt_150ms_pct", "lt_150ms", "systole_segments"], ascending=False
    ).head(20)
    recording_short_top = recording_systole.sort_values(
        ["lt_150ms_pct", "lt_150ms", "systole_segments"], ascending=False
    ).head(20)

    lines = [
        "# Analise exploratoria TCN - duracao dos segmentos reais",
        "",
        "Esta analise usa diretamente os intervalos anotados nos arquivos `.tsv` do `fold_1`.",
        "Ela mede duracoes antes da discretizacao em frames e antes do pos-processamento do TCN.",
        f"Foram ignorados {invalid_count} segmentos com duracao nao positiva; eles foram salvos em `invalid_segment_durations.csv`.",
        "",
        "## Quantis por classe cardiaca e split",
        "",
        markdown_table(
            cardiac_quantiles,
            [
                "split",
                "class_name",
                "segments",
                "recordings",
                "patients",
                "mean_ms",
                "p5_ms",
                "p25_ms",
                "median_ms",
                "p75_ms",
                "p95_ms",
            ],
        ),
        "",
        "## Sistoles curtas por split",
        "",
        markdown_table(
            short_by_split,
            [
                "split",
                "systole_segments",
                "recordings",
                "patients",
                "median_ms",
                "p5_ms",
                "p25_ms",
                "lt_100ms",
                "lt_100ms_pct",
                "lt_150ms",
                "lt_150ms_pct",
                "lt_200ms",
                "lt_200ms_pct",
            ],
        ),
        "",
        "## Quantis de sistole por local",
        "",
        markdown_table(
            systole_location_quantiles,
            [
                "split",
                "location",
                "segments",
                "recordings",
                "patients",
                "mean_ms",
                "p5_ms",
                "p25_ms",
                "median_ms",
                "p75_ms",
                "p95_ms",
            ],
        ),
        "",
        "## Sistoles curtas por local",
        "",
        markdown_table(
            short_by_location,
            [
                "split",
                "location",
                "systole_segments",
                "recordings",
                "patients",
                "median_ms",
                "lt_100ms",
                "lt_100ms_pct",
                "lt_150ms",
                "lt_150ms_pct",
                "lt_200ms",
                "lt_200ms_pct",
            ],
        ),
        "",
        "## Pacientes com maior proporcao de sistole <150 ms",
        "",
        markdown_table(
            patient_short_top,
            [
                "split",
                "patient_id",
                "murmur",
                "recordings",
                "systole_segments",
                "systole_median_ms",
                "lt_150ms",
                "lt_150ms_pct",
            ],
        ),
        "",
        "## Gravacoes com maior proporcao de sistole <150 ms",
        "",
        markdown_table(
            recording_short_top,
            [
                "split",
                "location",
                "recording_id",
                "murmur",
                "systole_segments",
                "systole_median_ms",
                "lt_150ms",
                "lt_150ms_pct",
            ],
        ),
        "",
        "## Arquivos gerados",
        "",
        "- `segment_durations.csv`",
        "- `invalid_segment_durations.csv`",
        "- `class_duration_quantiles_by_split.csv`",
        "- `class_duration_quantiles_by_split_location.csv`",
        "- `systole_short_segments_by_split.csv`",
        "- `systole_short_segments_by_split_location.csv`",
        "- `systole_duration_by_patient.csv`",
        "- `systole_duration_by_recording.csv`",
        "- `cardiac_class_duration_boxplot_by_split.png`",
        "- `systole_duration_histogram_by_split.png`",
        "- `systole_duration_boxplot_by_location_split.png`",
        "",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.tcn_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    segment_tables: list[pd.DataFrame] = []
    for split, items in manifest.items():
        for item in items:
            segment_tables.append(read_tsv_segments(item, split))
    segments = order_categories(pd.concat(segment_tables, ignore_index=True))
    invalid = segments[segments["duration_ms"] <= 0]
    invalid.to_csv(args.output_dir / "invalid_segment_durations.csv", index=False)
    if not invalid.empty:
        segments = segments[segments["duration_ms"] > 0].copy()

    systole = segments[segments["class_name"] == "systole"].copy()
    split_class_quantiles = duration_quantiles(
        segments, ["split", "class_id", "class_name"]
    )
    split_location_quantiles = duration_quantiles(
        segments, ["split", "location", "class_id", "class_name"]
    )
    short_by_split = short_systole_summary(systole, ["split"])
    short_by_location = short_systole_summary(systole, ["split", "location"])
    patient_systole = aggregate_by_patient(systole)
    recording_systole = aggregate_by_recording(systole)

    segments.to_csv(args.output_dir / "segment_durations.csv", index=False)
    split_class_quantiles.to_csv(
        args.output_dir / "class_duration_quantiles_by_split.csv", index=False
    )
    split_location_quantiles.to_csv(
        args.output_dir / "class_duration_quantiles_by_split_location.csv",
        index=False,
    )
    short_by_split.to_csv(
        args.output_dir / "systole_short_segments_by_split.csv", index=False
    )
    short_by_location.to_csv(
        args.output_dir / "systole_short_segments_by_split_location.csv",
        index=False,
    )
    patient_systole.to_csv(args.output_dir / "systole_duration_by_patient.csv", index=False)
    recording_systole.to_csv(
        args.output_dir / "systole_duration_by_recording.csv", index=False
    )

    plot_class_duration_box(segments, args.output_dir)
    plot_systole_histogram(systole, args.output_dir)
    plot_systole_by_location(systole, args.output_dir)
    write_summary(
        args.output_dir,
        split_class_quantiles,
        split_location_quantiles,
        short_by_split,
        short_by_location,
        patient_systole,
        recording_systole,
        len(invalid),
    )
    print(f"Wrote segment duration EDA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
