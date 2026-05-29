"""Compatibility shim. The systole CNN/RNN now lives in `nested_tcn_systole_cnn.cnn`.

Kept so existing references (standalone CLI, the exploratory scripts that load this file via
`importlib`, and `nested_tcn_systole_cnn.paths.CNN_SCRIPT`) keep working. New code should import
from `nested_tcn_systole_cnn.cnn` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this file is executed/loaded directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nested_tcn_systole_cnn.cnn import *  # noqa: F401,F403
from nested_tcn_systole_cnn.cnn.cli import main, parse_args  # noqa: F401


if __name__ == "__main__":
    main()
