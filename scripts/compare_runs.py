"""Compare two nested_tcn_systole_cnn runs side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "nested_tcn_systole_cnn"


def _load(run_name: str) -> dict:
    path = EXPERIMENTS_DIR / run_name / "metrics_summary.json"
    return json.loads(path.read_text())


def _calibrated(summary: dict) -> dict:
    return summary.get("calibrated_metrics_threshold_05", {})


def _calibrated_pf_youden(summary: dict) -> dict:
    return summary.get("calibrated_tuned_metrics_by_fold", {})


def _raw(summary: dict) -> dict:
    return summary.get("raw_metrics_threshold_05", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    base = _load(args.baseline)
    new = _load(args.new)

    print(f"Baseline: {args.baseline}")
    print(f"Novo:     {args.new}\n")

    def row(name: str, get):
        b = get(base) or {}
        n = get(new) or {}
        keys = ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1"]
        head = "  " + name
        print(head)
        print("  " + "-" * (len(head) - 2))
        print(f"  {'metric':<22} {'baseline':>10} {'novo':>10} {'delta':>10}")
        for k in keys:
            bv = float(b.get(k, 0.0))
            nv = float(n.get(k, 0.0))
            print(f"  {k:<22} {bv:>10.4f} {nv:>10.4f} {nv - bv:>+10.4f}")
        print()

    row("Probabilidade bruta @0.5",       _raw)
    row("Calibrado @0.5 (paciente OOF)",  _calibrated)
    row("Calibrado por-fold Youden",      _calibrated_pf_youden)


if __name__ == "__main__":
    main()
