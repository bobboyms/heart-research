"""Generate a static HTML dashboard for registered experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from .paths import DEFAULT_EXPERIMENTS_DIR


DISPLAY_COLUMNS = [
    "run_name",
    "status",
    "mean_score",
    "sensitivity",
    "specificity",
    "precision_ppv",
    "f1",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "tp",
    "fp",
    "tn",
    "fn",
    "tcn_systole_weight_multiplier",
    "cnn_phase_mode",
    "encoder_block",
    "patient_mil_attention",
    "updated_at",
]


def load_registry(experiments_dir: Path) -> pd.DataFrame:
    registry_csv = experiments_dir / "registry.csv"
    if not registry_csv.exists():
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    table = pd.read_csv(registry_csv)
    for column in DISPLAY_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    return table


def render_dashboard(table: pd.DataFrame, output_path: Path) -> None:
    records = table.fillna("").to_dict(orient="records")
    data_json = json.dumps(records, ensure_ascii=False)
    columns_json = json.dumps(DISPLAY_COLUMNS)
    title = "Nested TCN + Systole CNN Experiments"
    html_text = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #617080;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --bad: #b42318;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 720;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 24px 32px 40px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      font: inherit;
      padding: 9px 10px;
    }}
    input {{
      min-width: 300px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .metric-value {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 720;
    }}
    .table-wrap {{
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 1280px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f5;
      cursor: pointer;
      user-select: none;
      font-weight: 680;
    }}
    tr.best {{
      background: var(--accent-soft);
    }}
    .status-failed {{
      color: var(--bad);
      font-weight: 680;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 620;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="subtle">Dashboard estatico gerado a partir de registry.csv. Regere este HTML apos novos treinos.</div>
  </header>
  <main>
    <section class="metric-grid" id="metrics"></section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Filtrar por nome, status, parametros ou pasta">
      <select id="status">
        <option value="">Todos os status</option>
        <option value="completed">completed</option>
        <option value="running">running</option>
        <option value="failed">failed</option>
      </select>
    </div>
    <div class="table-wrap">
      <table id="experiments"></table>
    </div>
  </main>
  <script>
    const rows = {data_json};
    const columns = {columns_json};
    let sortKey = "mean_score";
    let sortDirection = -1;

    function numericValue(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }}

    function formatValue(key, value) {{
      const numeric = numericValue(value);
      if (numeric !== null && ["mean_score", "sensitivity", "specificity", "precision_ppv", "f1", "balanced_accuracy", "auroc", "auprc"].includes(key)) {{
        return numeric.toFixed(3);
      }}
      return String(value ?? "");
    }}

    function filteredRows() {{
      const query = document.getElementById("search").value.toLowerCase();
      const status = document.getElementById("status").value;
      return rows.filter(row => {{
        if (status && row.status !== status) return false;
        if (!query) return true;
        return JSON.stringify(row).toLowerCase().includes(query);
      }}).sort((a, b) => {{
        const av = numericValue(a[sortKey]);
        const bv = numericValue(b[sortKey]);
        if (av !== null && bv !== null) return (av - bv) * sortDirection;
        return String(a[sortKey] ?? "").localeCompare(String(b[sortKey] ?? "")) * sortDirection;
      }});
    }}

    function renderMetrics(currentRows) {{
      const completed = currentRows.filter(row => row.status === "completed");
      const best = completed.reduce((acc, row) => {{
        const score = numericValue(row.mean_score) ?? -1;
        return score > acc.score ? {{ row, score }} : acc;
      }}, {{ row: null, score: -1 }}).row;
      const metrics = [
        ["Experimentos", currentRows.length],
        ["Completos", completed.length],
        ["Melhor score", best ? formatValue("mean_score", best.mean_score) : "-"],
        ["Melhor run", best ? best.run_name : "-"],
      ];
      document.getElementById("metrics").innerHTML = metrics.map(([label, value]) => `
        <div class="metric"><div class="metric-label">${{label}}</div><div class="metric-value">${{value}}</div></div>
      `).join("");
    }}

    function renderTable() {{
      const currentRows = filteredRows();
      renderMetrics(currentRows);
      const bestScore = Math.max(...currentRows.filter(row => row.status === "completed").map(row => numericValue(row.mean_score) ?? -1));
      const head = `<thead><tr>${{columns.map(column => `<th data-key="${{column}}">${{column}}</th>`).join("")}}<th>links</th></tr></thead>`;
      const body = currentRows.map(row => {{
        const isBest = row.status === "completed" && (numericValue(row.mean_score) ?? -2) === bestScore;
        const statusClass = row.status === "failed" ? "status-failed" : "";
        const cells = columns.map(column => `<td class="${{column === "status" ? statusClass : ""}}">${{formatValue(column, row[column])}}</td>`).join("");
        const links = `<td><a href="${{row.summary_path || ""}}">summary</a> · <a href="${{row.output_dir || ""}}">pasta</a></td>`;
        return `<tr class="${{isBest ? "best" : ""}}">${{cells}}${{links}}</tr>`;
      }}).join("");
      document.getElementById("experiments").innerHTML = `${{head}}<tbody>${{body}}</tbody>`;
      document.querySelectorAll("th[data-key]").forEach(th => {{
        th.addEventListener("click", () => {{
          const key = th.dataset.key;
          if (sortKey === key) sortDirection *= -1;
          else {{
            sortKey = key;
            sortDirection = -1;
          }}
          renderTable();
        }});
      }});
    }}

    document.getElementById("search").addEventListener("input", renderTable);
    document.getElementById("status").addEventListener("change", renderTable);
    renderTable();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static HTML dashboard for nested TCN + systole CNN experiments.")
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    experiments_dir = args.experiments_dir.resolve()
    output_path = args.output or (experiments_dir / "dashboard.html")
    table = load_registry(experiments_dir)
    render_dashboard(table, output_path)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
