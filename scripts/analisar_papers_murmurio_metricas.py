#!/usr/bin/env python3
"""Analyze murmur paper metrics extracted into a Markdown table."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TABLE = Path("tabela_papers_murmurio_metricas.md")

HIGHER_IS_BETTER_METRICS = [
    "AUROC",
    "AUPRC",
    "BA / weighted accuracy",
    "UAR",
    "Sensibilidade",
    "Especificidade",
    "Precisao / PPV",
    "F1-score",
    "Acuracia",
]

LOWER_IS_BETTER_METRICS = ["Brier"]
COUNT_METRICS = ["TP", "FP", "FN", "TN"]

HEADER_NORMALIZATION = {
    "Precisão / PPV": "Precisao / PPV",
    "Acurácia": "Acuracia",
}


@dataclass(frozen=True)
class PaperMetric:
    year: int
    paper: str
    metrics: dict[str, str]


def normalize_header(header: str) -> str:
    header = header.strip()
    return HEADER_NORMALIZATION.get(header, header)


def parse_markdown_table(path: Path) -> list[PaperMetric]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"No Markdown table found in {path}")

    headers = [normalize_header(cell) for cell in split_table_row(table_lines[0])]
    rows: list[PaperMetric] = []

    for line in table_lines[2:]:
        cells = split_table_row(line)
        if len(cells) != len(headers):
            raise ValueError(f"Invalid table row with {len(cells)} cells: {line}")

        row = dict(zip(headers, cells, strict=True))
        metrics = {header: row[header] for header in headers[2:]}
        rows.append(PaperMetric(year=int(row["Ano"]), paper=row["Paper"], metrics=metrics))

    return rows


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def metric_value(raw: str) -> float | None:
    """Return a comparable number from a metric cell.

    If a cell contains more than one value, the highest value is used. This
    handles cells like "Present: 0,827; Unknown: 0,312" conservatively for
    best-score detection.
    """

    if not raw or raw.upper() == "N/A":
        return None

    normalized = raw.replace(",", ".")
    values = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if not values:
        return None
    return max(values)


def best_by_metric(rows: list[PaperMetric], metric: str, *, lower_is_better: bool = False) -> PaperMetric | None:
    candidates = [row for row in rows if metric_value(row.metrics.get(metric, "")) is not None]
    if not candidates:
        return None

    return min(candidates, key=lambda row: metric_value(row.metrics[metric])) if lower_is_better else max(
        candidates, key=lambda row: metric_value(row.metrics[metric])
    )


def overall_score(row: PaperMetric) -> float | None:
    values = [
        metric_value(row.metrics.get(metric, ""))
        for metric in HIGHER_IS_BETTER_METRICS
        if metric_value(row.metrics.get(metric, "")) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def print_metric_winners(rows: list[PaperMetric]) -> None:
    print("Melhor resultado por metrica")
    print("-" * 31)
    for metric in HIGHER_IS_BETTER_METRICS:
        winner = best_by_metric(rows, metric)
        if winner is None:
            continue
        print(f"{metric}: {winner.metrics[metric]} | {winner.year} | {winner.paper}")

    for metric in LOWER_IS_BETTER_METRICS:
        winner = best_by_metric(rows, metric, lower_is_better=True)
        if winner is None:
            continue
        print(f"{metric} menor: {winner.metrics[metric]} | {winner.year} | {winner.paper}")


def print_overall_winner(rows: list[PaperMetric]) -> None:
    scored_rows = [(row, overall_score(row)) for row in rows]
    scored_rows = [(row, score) for row, score in scored_rows if score is not None]
    if not scored_rows:
        print("\nNao foi possivel calcular vencedor geral.")
        return

    winner, score = max(scored_rows, key=lambda item: item[1])
    available_metrics = [
        metric
        for metric in HIGHER_IS_BETTER_METRICS
        if metric_value(winner.metrics.get(metric, "")) is not None
    ]

    print("\nMelhor resultado geral")
    print("-" * 22)
    print(f"Paper: {winner.paper}")
    print(f"Ano: {winner.year}")
    print(f"Score medio: {score:.4f}")
    print(f"Metricas usadas: {', '.join(available_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the Markdown table of heart murmur paper metrics."
    )
    parser.add_argument(
        "table",
        nargs="?",
        default=DEFAULT_TABLE,
        type=Path,
        help=f"Markdown table path. Default: {DEFAULT_TABLE}",
    )
    args = parser.parse_args()

    rows = parse_markdown_table(args.table)
    print_metric_winners(rows)
    print_overall_winner(rows)


if __name__ == "__main__":
    main()
