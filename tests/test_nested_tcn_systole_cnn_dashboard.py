from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nested_tcn_systole_cnn.dashboard import load_registry, render_dashboard


class NestedDashboardTests(unittest.TestCase):
    def test_dashboard_embeds_registry_data_in_static_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "run_name": "baseline",
                        "status": "completed",
                        "mean_score": 0.72,
                        "sensitivity": 0.8,
                        "specificity": 0.7,
                        "precision_ppv": 0.68,
                        "f1": 0.7,
                        "summary_path": str(experiments_dir / "baseline" / "summary.md"),
                        "output_dir": str(experiments_dir / "baseline"),
                    }
                ]
            ).to_csv(experiments_dir / "registry.csv", index=False)

            table = load_registry(experiments_dir)
            output_path = experiments_dir / "dashboard.html"
            render_dashboard(table, output_path)

            html = output_path.read_text(encoding="utf-8")
            self.assertIn("const rows =", html)
            self.assertIn("baseline", html)
            self.assertIn("mean_score", html)


if __name__ == "__main__":
    unittest.main()

