"""Compatibility shim. The TCN segmenter now lives in `nested_tcn_systole_cnn.tcn`.

Kept so existing references (standalone CLI, any `importlib`/path-based loaders) keep working.
New code should import from `nested_tcn_systole_cnn.tcn` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this file is executed directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nested_tcn_systole_cnn.tcn import *  # noqa: F401,F403
from nested_tcn_systole_cnn.tcn.cli import main, parse_args  # noqa: F401


if __name__ == "__main__":
    main()
