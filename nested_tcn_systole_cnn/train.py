"""Entrypoint for nested TCN segmentation + systole CNN training."""

from __future__ import annotations

from collections.abc import Sequence

from .cli import parse_args
from .pipeline import run_nested_experiment


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_nested_experiment(args)


if __name__ == "__main__":
    main()

