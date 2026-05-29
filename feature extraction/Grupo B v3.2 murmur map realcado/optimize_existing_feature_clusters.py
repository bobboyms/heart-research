# /// script
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scikit-learn>=1.4",
#   "tabulate>=0.9",
# ]
# ///
"""Optimize clustering over existing v3.2 patient-level features.

This script does not recalculate audio. It reads:

    outputs/patient_enhanced_murmur_map_features.csv

Then sweeps feature views, scalers, PCA dimensions, and k-means cluster counts
to find cluster groupings that maximize Present capture at fixed purity levels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


META_COLUMNS = {
    "patient_id",
    "murmur",
    "outcome",
    "age",
    "sex",
    "campaign",
}

PURITY_THRESHOLDS = (0.50, 0.70, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Optimize clusters over existing v3.2 features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "outputs" / "patient_enhanced_murmur_map_features.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs" / "cluster_optimization")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--pca-components", type=str, default="2,3,5,8,10,15,20,30,50,80")
    parser.add_argument("--scalers", type=str, default="standard,robust")
    parser.add_argument(
        "--views",
        type=str,
        default="",
        help="Optional comma-separated feature views to run. Empty runs all views.",
    )
    parser.add_argument("--n-init", type=int, default=50)
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def contains_any(column: str, needles: tuple[str, ...]) -> bool:
    return any(needle in column for needle in needles)


def feature_views(feature_columns: list[str]) -> dict[str, list[str]]:
    low = tuple(["enh_low", "enh_very_low", "enh_low_mid", "lowres", "base_z_fraction"])
    mid = tuple(["enh_mid", "enh_upper_mid"])
    high = tuple(["enh_high", "enh_upper_mid"])
    map_low = tuple(["map_low_bin"])
    persistence = tuple(["active_fraction", "frame_active_fraction", "longest_run", "z_fraction", "z_gt", "top30"])
    energy = tuple(["energy", "p90", "max", "top3_mean", "map_total"])
    max_p90_prefixes = tuple(["max_", "p90_"])
    mean_prefixes = tuple(["mean_"])

    views = {
        "all": feature_columns,
        "all_no_heatmap": [c for c in feature_columns if "map_low_bin" not in c],
        "low": [c for c in feature_columns if contains_any(c, low) and "map_low_bin" not in c],
        "mid": [c for c in feature_columns if contains_any(c, mid)],
        "high": [c for c in feature_columns if contains_any(c, high)],
        "map_low": [c for c in feature_columns if contains_any(c, map_low)],
        "low_mid": [c for c in feature_columns if (contains_any(c, low) or contains_any(c, mid)) and "map_low_bin" not in c],
        "low_map": [c for c in feature_columns if contains_any(c, low + map_low)],
        "low_mid_map": [c for c in feature_columns if contains_any(c, low + mid + map_low)],
        "low_persistence": [
            c for c in feature_columns if contains_any(c, low) and contains_any(c, persistence) and "map_low_bin" not in c
        ],
        "low_energy": [
            c for c in feature_columns if contains_any(c, low) and contains_any(c, energy) and "map_low_bin" not in c
        ],
        "low_max_p90": [
            c for c in feature_columns if contains_any(c, low) and c.startswith(max_p90_prefixes) and "map_low_bin" not in c
        ],
        "low_mean": [
            c for c in feature_columns if contains_any(c, low) and c.startswith(mean_prefixes) and "map_low_bin" not in c
        ],
        "map_low_max_p90": [c for c in feature_columns if "map_low_bin" in c and c.startswith(max_p90_prefixes)],
        "map_low_mean": [c for c in feature_columns if "map_low_bin" in c and c.startswith(mean_prefixes)],
    }
    return {name: cols for name, cols in views.items() if cols}


def cluster_stats(labels: np.ndarray, murmur: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"cluster": labels, "murmur": murmur.to_numpy()})
    stats = frame.groupby("cluster")["murmur"].agg(
        rows="size",
        present_count=lambda values: int((values == "Present").sum()),
        present_rate=lambda values: float((values == "Present").mean()),
    )
    return stats.reset_index()


def summarize(labels: np.ndarray, murmur: pd.Series) -> dict[str, float | int]:
    stats = cluster_stats(labels, murmur)
    total_present = int((murmur == "Present").sum())
    best_idx = int(stats["present_rate"].idxmax())
    best = stats.loc[best_idx]
    out: dict[str, float | int] = {
        "best_cluster": int(best["cluster"]),
        "best_cluster_rows": int(best["rows"]),
        "best_cluster_present_count": int(best["present_count"]),
        "best_cluster_present_rate": float(best["present_rate"]),
        "best_cluster_present_capture": float(best["present_count"] / total_present),
    }
    for threshold in PURITY_THRESHOLDS:
        selected = stats.loc[stats["present_rate"] >= threshold]
        key = f"purity_{int(threshold * 100)}"
        rows = int(selected["rows"].sum()) if not selected.empty else 0
        present = int(selected["present_count"].sum()) if not selected.empty else 0
        out[f"{key}_rows"] = rows
        out[f"{key}_present_count"] = present
        out[f"{key}_present_rate"] = float(present / rows) if rows else 0.0
        out[f"{key}_present_capture"] = float(present / total_present) if total_present else 0.0
    return out


def run_sweep(
    df: pd.DataFrame,
    views: dict[str, list[str]],
    k_values: range,
    pca_components: list[int],
    scalers: list[str],
    n_init: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    murmur = df["murmur"]
    for view_name, columns in views.items():
        print(f"Running view={view_name} features={len(columns)}", flush=True)
        matrix = df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        if matrix.shape[1] == 0:
            continue
        max_components = min(max(pca_components), matrix.shape[0], matrix.shape[1])
        for scaler_name in scalers:
            scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
            scaled = scaler.fit_transform(matrix)
            pca = PCA(n_components=max_components, random_state=42)
            embedding_full = pca.fit_transform(scaled)
            explained = np.cumsum(pca.explained_variance_ratio_)
            for n_components in pca_components:
                if n_components > max_components:
                    continue
                embedding = embedding_full[:, :n_components]
                for k in k_values:
                    labels = KMeans(n_clusters=k, n_init=n_init, random_state=42).fit_predict(embedding)
                    row: dict[str, float | str | int] = {
                        "view": view_name,
                        "scaler": scaler_name,
                        "pca_components": int(n_components),
                        "k": int(k),
                        "feature_count": int(len(columns)),
                        "pca_variance": float(explained[n_components - 1]),
                    }
                    row.update(summarize(labels, murmur))
                    rows.append(row)
    return pd.DataFrame(rows)


def write_summary(output_path: Path, results: pd.DataFrame) -> None:
    def top_table(metric: str, min_rate_col: str | None = None) -> pd.DataFrame:
        df = results.copy()
        if min_rate_col is not None:
            df = df.loc[df[min_rate_col] > 0.0]
        return df.sort_values([metric, "purity_90_present_rate", "best_cluster_present_rate"], ascending=False).head(20)

    cols = [
        "view",
        "scaler",
        "pca_components",
        "k",
        "feature_count",
        "pca_variance",
        "best_cluster_rows",
        "best_cluster_present_count",
        "best_cluster_present_rate",
        "best_cluster_present_capture",
        "purity_90_rows",
        "purity_90_present_count",
        "purity_90_present_rate",
        "purity_90_present_capture",
        "purity_95_rows",
        "purity_95_present_count",
        "purity_95_present_rate",
        "purity_95_present_capture",
    ]
    lines = [
        "# Otimizacao de clusters sobre features v3.2",
        "",
        "## Melhor captura com pureza >=90%",
        "",
        top_table("purity_90_present_capture", "purity_90_present_rate")[cols].to_markdown(index=False),
        "",
        "## Melhor captura com pureza >=95%",
        "",
        top_table("purity_95_present_capture", "purity_95_present_rate")[cols].to_markdown(index=False),
        "",
        "## Melhores clusters unicos",
        "",
        results.sort_values(["best_cluster_present_capture", "best_cluster_present_rate"], ascending=False)
        .head(20)[cols]
        .to_markdown(index=False),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    feature_columns = numeric_feature_columns(df)
    views = feature_views(feature_columns)
    if args.views.strip():
        requested = {value.strip() for value in args.views.split(",") if value.strip()}
        views = {name: cols for name, cols in views.items() if name in requested}
        missing = sorted(requested - set(views))
        if missing:
            raise ValueError(f"Unknown or empty views requested: {missing}")
    k_values = range(args.k_min, args.k_max + 1)
    pca_components = [int(value.strip()) for value in args.pca_components.split(",") if value.strip()]
    scalers = [value.strip() for value in args.scalers.split(",") if value.strip()]
    results = run_sweep(df, views, k_values, pca_components, scalers, args.n_init)
    results.to_csv(output_dir / "optimized_cluster_sweep.csv", index=False)
    write_summary(output_dir / "summary.md", results)
    print(f"Rows: {len(results)}")
    print(f"Views: {len(views)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
