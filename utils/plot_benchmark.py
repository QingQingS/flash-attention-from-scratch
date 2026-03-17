"""
plot_benchmark.py
=================
从 flash_benchmark_*.json 生成独立的 HTML 可视化报告。

Usage:
    python plot_benchmark.py results/flash_benchmark_1773729042.json
    python plot_benchmark.py results/flash_benchmark_1773729042.json --out report.html
    python plot_benchmark.py results/flash_benchmark_1773729042.json --open
"""

import json
import argparse
import sys
import webbrowser
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract(payload: dict):
    """Pull the arrays Chart.js needs from the raw JSON payload."""
    rows = payload["results"]
    return {
        "seq_lens":        [r["seq_len"]              for r in rows],
        "naive_lat_ms":    [r["naive_latency_ms"] for r in rows],
        "sdpa_lat_ms":     [r["sdpa_latency_ms"]  for r in rows],
        "flash_lat_ms":    [r["flash_latency_ms"] for r in rows],
        "naive_mem":       [r["naive_mem_extra"]        for r in rows],
        "sdpa_mem":        [r["sdpa_mem_extra"]         for r in rows],
        "flash_mem":       [r["flash_mem_extra"]        for r in rows],
        "speed_vs_naive":  [r["flash_speedup_vs_naive"] for r in rows],
        "speed_vs_sdpa":   [r["flash_speedup_vs_sdpa"]  for r in rows],
        "naive_tput":      [r["naive_throughput_tokens_s"] / 1e9 for r in rows],
        "sdpa_tput":       [r["sdpa_throughput_tokens_s"]  / 1e9 for r in rows],
        "flash_tput":      [r["flash_throughput_tokens_s"] / 1e9 for r in rows],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Flash Attention Benchmark Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:      #0d0f12;
    --bg2:     #13161b;
    --bg3:     #1a1e25;
    --border:  rgba(255,255,255,0.07);
    --text:    #d4d6db;
    --muted:   #6b7280;
    --red:     #e05252;
    --blue:    #4e8ef7;
    --green:   #3ecf6e;
    --amber:   #f0a030;
    --font:    'IBM Plex Sans', system-ui, sans-serif;
    --mono:    'IBM Plex Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* ── header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    align-items: flex-end;
    gap: 32px;
    flex-wrap: wrap;
  }
  .report-title {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  header h1 {
    font-size: 26px;
    font-weight: 300;
    letter-spacing: -.02em;
    color: #f0f1f3;
  }
  header h1 span { color: var(--blue); font-weight: 500; }
  .meta-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-left: auto;
  }
  .pill {
    font-family: var(--mono);
    font-size: 11px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    color: var(--muted);
  }
  .pill strong { color: var(--text); font-weight: 500; }

  /* ── layout ── */
  main { padding: 32px 40px 60px; max-width: 1280px; }

  /* ── metric cards ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 40px;
  }
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    transition: border-color .2s;
  }
  .card:hover { border-color: rgba(255,255,255,0.15); }
  .card-label {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 8px;
  }
  .card-value {
    font-size: 24px;
    font-weight: 300;
    letter-spacing: -.02em;
  }
  .card-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
    font-family: var(--mono);
  }
  .c-red   { color: var(--red);   }
  .c-blue  { color: var(--blue);  }
  .c-green { color: var(--green); }
  .c-amber { color: var(--amber); }

  /* ── section ── */
  .section { margin-bottom: 40px; }
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
  }
  .section-title {
    font-size: 12px;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
  }
  .section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── legend ── */
  .legend {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 10px;
    font-family: var(--mono);
  }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-dot {
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  /* ── charts ── */
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  @media (max-width: 860px) { .chart-grid { grid-template-columns: 1fr; } }

  .chart-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }
  .chart-box-full {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .chart-box-label {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 12px;
  }
  .canvas-wrap { position: relative; width: 100%; }

  /* ── table ── */
  .tbl-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
  }
  thead th {
    background: var(--bg3);
    padding: 10px 16px;
    text-align: right;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .05em;
    border-bottom: 1px solid var(--border);
  }
  thead th:first-child { text-align: left; }
  tbody tr { border-bottom: 1px solid var(--border); transition: background .15s; }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--bg3); }
  tbody td {
    padding: 9px 16px;
    text-align: right;
    color: var(--text);
  }
  tbody td:first-child { text-align: left; color: var(--muted); }
  .best { color: var(--green); font-weight: 500; }
  .worst { color: var(--red); }

  /* ── footer ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 20px 40px;
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
</style>
</head>
<body>

<header>
  <div>
    <div class="report-title">Benchmark Report</div>
    <h1>Flash<span>Attention</span> — Performance Analysis</h1>
  </div>
  <div class="meta-pills">
    {{META_PILLS}}
  </div>
</header>

<main>

  <!-- ── Metric cards ── -->
  <div class="cards">
    {{METRIC_CARDS}}
  </div>

  <!-- ── Latency ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-title">Latency (ms, log scale)</span>
      <span class="section-line"></span>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#e05252"></span>Naive</span>
      <span class="legend-item"><span class="legend-dot" style="background:#4e8ef7"></span>SDPA (PyTorch)</span>
      <span class="legend-item"><span class="legend-dot" style="background:#3ecf6e"></span>Flash Attention (Triton)</span>
    </div>
    <div class="chart-box-full">
      <div class="canvas-wrap" style="height:260px"><canvas id="latChart"></canvas></div>
    </div>
  </div>

  <!-- ── Memory + Speedup ── -->
  <div class="chart-grid section">
    <div>
      <div class="section-header">
        <span class="section-title">Extra Memory (MB, log)</span>
        <span class="section-line"></span>
      </div>
      <div class="chart-box">
        <div class="canvas-wrap" style="height:240px"><canvas id="memChart"></canvas></div>
      </div>
    </div>
    <div>
      <div class="section-header">
        <span class="section-title">Flash Speedup Ratio</span>
        <span class="section-line"></span>
      </div>
      <div class="chart-box">
        <div class="canvas-wrap" style="height:240px"><canvas id="speedChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- ── Throughput ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-title">Throughput (G tokens/s)</span>
      <span class="section-line"></span>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#e05252"></span>Naive</span>
      <span class="legend-item"><span class="legend-dot" style="background:#4e8ef7"></span>SDPA</span>
      <span class="legend-item"><span class="legend-dot" style="background:#3ecf6e"></span>Flash (Triton)</span>
    </div>
    <div class="chart-box-full">
      <div class="canvas-wrap" style="height:220px"><canvas id="tputChart"></canvas></div>
    </div>
  </div>

  <!-- ── Raw Data Table ── -->
  <div class="section">
    <div class="section-header">
      <span class="section-title">Raw Results</span>
      <span class="section-line"></span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>seq_len</th>
            <th>naive (ms)</th>
            <th>SDPA (ms)</th>
            <th>Flash (ms)</th>
            <th>naive mem (MB)</th>
            <th>SDPA mem (MB)</th>
            <th>flash mem (MB)</th>
            <th>×naive</th>
            <th>×SDPA</th>
          </tr>
        </thead>
        <tbody>
          {{TABLE_ROWS}}
        </tbody>
      </table>
    </div>
  </div>

</main>

<footer>
  <span>Generated {{TIMESTAMP}}</span>
  <span>plot_benchmark.py · Chart.js 4.4.1</span>
</footer>

<script>
const DATA = {{DATA_JSON}};

const GRID  = 'rgba(255,255,255,0.05)';
const TICK  = '#6b7280';
const baseOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: TICK, font: { family: "'IBM Plex Mono'" } }, grid: { color: GRID } },
    y: { ticks: { color: TICK, font: { family: "'IBM Plex Mono'" } }, grid: { color: GRID } },
  }
};

// ── Latency ──
new Chart(document.getElementById('latChart'), {
  type: 'line',
  data: {
    labels: DATA.seq_lens,
    datasets: [
      { label: 'Naive', data: DATA.naive_lat_ms, borderColor: '#e05252', backgroundColor: 'rgba(224,82,82,0.08)',  tension: 0.3, fill: true, pointRadius: 5, pointHoverRadius: 7 },
      { label: 'SDPA',  data: DATA.sdpa_lat_ms,  borderColor: '#4e8ef7', backgroundColor: 'rgba(78,142,247,0.08)', tension: 0.3, fill: true, pointRadius: 5, pointHoverRadius: 7 },
      { label: 'Flash', data: DATA.flash_lat_ms, borderColor: '#3ecf6e', backgroundColor: 'rgba(62,207,110,0.08)', tension: 0.3, fill: true, pointRadius: 5, pointHoverRadius: 7 },
    ]
  },
  options: {
    ...baseOpts,
    scales: {
      x: { ...baseOpts.scales.x, title: { display: true, text: 'sequence length', color: TICK, font: { family: "'IBM Plex Mono'", size: 11 } } },
      y: { ...baseOpts.scales.y, type: 'logarithmic', title: { display: true, text: 'ms (log)', color: TICK, font: { family: "'IBM Plex Mono'", size: 11 } } },
    }
  }
});

// ── Memory ──
new Chart(document.getElementById('memChart'), {
  type: 'bar',
  data: {
    labels: DATA.seq_lens,
    datasets: [
      { label: 'Naive', data: DATA.naive_mem, backgroundColor: 'rgba(224,82,82,0.7)',  borderRadius: 3 },
      { label: 'SDPA',  data: DATA.sdpa_mem,  backgroundColor: 'rgba(78,142,247,0.7)', borderRadius: 3 },
      { label: 'Flash', data: DATA.flash_mem, backgroundColor: 'rgba(62,207,110,0.7)', borderRadius: 3 },
    ]
  },
  options: {
    ...baseOpts,
    scales: {
      x: { ...baseOpts.scales.x },
      y: { ...baseOpts.scales.y, type: 'logarithmic', title: { display: true, text: 'MB (log)', color: TICK, font: { family: "'IBM Plex Mono'", size: 11 } } },
    }
  }
});

// ── Speedup ──
new Chart(document.getElementById('speedChart'), {
  type: 'bar',
  data: {
    labels: DATA.seq_lens,
    datasets: [
      { label: '×naive', data: DATA.speed_vs_naive, backgroundColor: 'rgba(78,142,247,0.75)', borderRadius: 3 },
      { label: '×SDPA',  data: DATA.speed_vs_sdpa,  backgroundColor: 'rgba(240,160,48,0.75)', borderRadius: 3 },
    ]
  },
  options: {
    ...baseOpts,
    scales: {
      x: { ...baseOpts.scales.x },
      y: {
        ...baseOpts.scales.y,
        title: { display: true, text: 'speedup ×', color: TICK, font: { family: "'IBM Plex Mono'", size: 11 } },
        min: 0,
      }
    },
    plugins: {
      ...baseOpts.plugins,
      annotation: {},
    }
  }
});

// ── Throughput ──
new Chart(document.getElementById('tputChart'), {
  type: 'bar',
  data: {
    labels: DATA.seq_lens,
    datasets: [
      { label: 'Naive', data: DATA.naive_tput, backgroundColor: 'rgba(224,82,82,0.7)',  borderRadius: 3 },
      { label: 'SDPA',  data: DATA.sdpa_tput,  backgroundColor: 'rgba(78,142,247,0.7)', borderRadius: 3 },
      { label: 'Flash', data: DATA.flash_tput, backgroundColor: 'rgba(62,207,110,0.7)', borderRadius: 3 },
    ]
  },
  options: {
    ...baseOpts,
    scales: {
      x: { ...baseOpts.scales.x },
      y: { ...baseOpts.scales.y, title: { display: true, text: 'G tokens/s', color: TICK, font: { family: "'IBM Plex Mono'", size: 11 } } },
    }
  }
});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def meta_pills(payload: dict) -> str:
    meta = payload.get("meta", {})
    gpu  = meta.get("gpu", {})
    cfg  = meta.get("config", {})
    items = [
        ("GPU",    gpu.get("name", "—")),
        ("VRAM",   f"{gpu.get('memory_gb', '—')} GB"),
        ("CUDA",   gpu.get("cuda", "—")),
        ("PyTorch",gpu.get("torch", "—")),
        ("dtype",  cfg.get("dtype", "—")),
        ("batch",  str(cfg.get("batch_size", "—"))),
        ("heads",  str(cfg.get("num_heads", "—"))),
        ("dim",    str(cfg.get("dim", "—"))),
        ("runs",   str(cfg.get("runs", "—"))),
    ]
    return "\n    ".join(
        f'<div class="pill"><strong>{k}</strong> {v}</div>' for k, v in items
    )


def metric_cards(d: dict) -> str:
    rows = []
    # max speedup vs naive
    max_sv_naive = max(d["speed_vs_naive"])
    idx_max = d["speed_vs_naive"].index(max_sv_naive)
    seq_max = d["seq_lens"][idx_max]

    # flash vs sdpa at longest seq
    last_sv_sdpa = d["speed_vs_sdpa"][-1]
    last_seq      = d["seq_lens"][-1]

    # flash mem at longest
    flash_mem_last = d["flash_mem"][-1]
    naive_mem_last = d["naive_mem"][-1]

    cards = [
        ("max speedup vs naive", f'<span class="c-blue">{max_sv_naive:.2f}×</span>',
         f"at seq={seq_max:,}"),
        ("flash vs sdpa at peak",
         f'<span class="c-{"red" if last_sv_sdpa < 1 else "green"}">{last_sv_sdpa:.3f}×</span>',
         f"seq={last_seq:,} · {'Flash slower' if last_sv_sdpa < 1 else 'Flash faster'}"),
        ("flash mem at peak seq",
         f'<span class="c-green">{flash_mem_last:.0f} MB</span>',
         f"naive {naive_mem_last:.0f} MB · {naive_mem_last/flash_mem_last:.0f}× reduction"),
        ("naive mem at peak seq",
         f'<span class="c-red">{naive_mem_last/1024:.1f} GB</span>',
         "O(N²) attention score matrix"),
    ]
    parts = []
    for label, val, sub in cards:
        parts.append(
            f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value">{val}</div>'
            f'<div class="card-sub">{sub}</div>'
            f'</div>'
        )
    return "\n    ".join(parts)


def table_rows(d: dict) -> str:
    rows = []
    for i, seq in enumerate(d["seq_lens"]):
        nl = d["naive_lat_ms"][i]
        sl = d["sdpa_lat_ms"][i]
        fl = d["flash_lat_ms"][i]
        nm = d["naive_mem"][i]
        sm = d["sdpa_mem"][i]
        fm = d["flash_mem"][i]
        sn = d["speed_vs_naive"][i]
        ss = d["speed_vs_sdpa"][i]

        def lat_cls(v, vals):
            if v == min(vals): return ' class="best"'
            if v == max(vals): return ' class="worst"'
            return ''

        lvals = [nl, sl, fl]
        rows.append(
            f"<tr>"
            f"<td>{seq:,}</td>"
            f"<td{lat_cls(nl, lvals)}>{nl:.5f}</td>"
            f"<td{lat_cls(sl, lvals)}>{sl:.5f}</td>"
            f"<td{lat_cls(fl, lvals)}>{fl:.5f}</td>"
            f"<td>{nm:.2f}</td>"
            f"<td>{sm:.2f}</td>"
            f"<td>{fm:.2f}</td>"
            f"<td {'class=\"best\"' if sn >= 1 else 'class=\"worst\"'}>{sn:.3f}</td>"
            f"<td {'class=\"best\"' if ss >= 1 else 'class=\"worst\"'}>{ss:.3f}</td>"
            f"</tr>"
        )
    return "\n          ".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_report(json_path: str, out_path: str | None = None) -> Path:
    payload = load(json_path)
    d       = extract(payload)

    # Choose output path
    if out_path is None:
        src  = Path(json_path)
        out  = src.with_name(src.stem + "_report.html")
    else:
        out = Path(out_path)

    html = (
        HTML
        .replace("{{META_PILLS}}",   meta_pills(payload))
        .replace("{{METRIC_CARDS}}", metric_cards(d))
        .replace("{{TABLE_ROWS}}",   table_rows(d))
        .replace("{{DATA_JSON}}",    json.dumps(d))
        .replace("{{TIMESTAMP}}",    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    out.write_text(html, encoding="utf-8")
    print(f"✓  Report saved → {out.resolve()}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML benchmark report from flash_benchmark JSON"
    )
    parser.add_argument("json", help="Path to flash_benchmark_*.json")
    parser.add_argument("--out",  default=None, help="Output HTML path (default: <json_stem>_report.html)")
    parser.add_argument("--open", action="store_true", help="Open in default browser after generating")
    args = parser.parse_args()

    out = build_report(args.json, args.out)

    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
