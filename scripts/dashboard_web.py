#!/usr/bin/env python3
"""Pipeline monitoring web dashboard. Run: .venv/bin/python3 scripts/dashboard_web.py"""
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from flask import Flask, jsonify
from sqlalchemy import create_engine, text

app = Flask(__name__)

DB_DSN = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
QUEUE_DSN = os.getenv("LLM_QUEUE_DSN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["docker", "exec", "ollama", "nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,fan.speed",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        temp, util, pwr, pwr_cap, mem_used, mem_total, fan = [x.strip() for x in out.split(",")]
        return {
            "temp": int(temp), "util": int(util),
            "power": float(pwr), "power_cap": float(pwr_cap),
            "vram_used": int(mem_used), "vram_total": int(mem_total),
            "fan": int(fan), "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ollama_model():
    try:
        out = subprocess.check_output(
            ["curl", "-s", f"{OLLAMA_URL}/api/ps"], timeout=5, text=True,
        )
        data = json.loads(out)
        models = data.get("models", [])
        if not models:
            return {"name": "none", "vram_gb": 0, "ctx": 0}
        m = models[0]
        return {
            "name": m["name"],
            "vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
            "ctx": m.get("context_length", 0),
        }
    except Exception:
        return {"name": "unavailable", "vram_gb": 0, "ctx": 0}


def queue_stats():
    engine = create_engine(QUEUE_DSN)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT topic, status, count(*)
            FROM llm_queue.tasks
            WHERE status IN ('pending', 'processing', 'done', 'failed')
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2
        """)).fetchall()

        speed = conn.execute(text("""
            SELECT topic, count(*),
                   EXTRACT(EPOCH FROM max(done_at) - min(done_at)) / NULLIF(count(*)-1, 0) as avg_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND done_at >= NOW() - INTERVAL '30 minutes'
            GROUP BY 1
        """)).fetchall()

        processing = conn.execute(text("""
            SELECT topic, started_at::text
            FROM llm_queue.tasks WHERE status = 'processing'
        """)).fetchall()

        # Per profile done + pending
        totals = conn.execute(text("""
            SELECT payload->>'sys_profile', topic, status, count(*)
            FROM llm_queue.tasks
            WHERE status IN ('done', 'pending', 'processing')
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2, 3
        """)).fetchall()

    engine.dispose()

    topics = {}
    for topic, status, count in rows:
        topics.setdefault(topic, {})[status] = count

    speeds = {}
    for topic, count, avg in speed:
        speeds[topic] = {"count": count, "avg": round(avg, 1) if avg else 0}

    procs = [{"topic": t, "started": s[:19] if s else "?"} for t, s in processing]

    profiles = {}
    for profile, topic, status, count in totals:
        p = profiles.setdefault(profile or "unknown", {})
        key = topic + "_" + ("done" if status == "done" else "pending")
        p[key] = p.get(key, 0) + count

    result = {}
    for topic in ("job_extract", "job_eval"):
        s = topics.get(topic, {})
        sp = speeds.get(topic, {})
        pend = s.get("pending", 0) + s.get("processing", 0)
        avg = sp.get("avg", 0)
        eta_sec = avg * pend if avg and pend else 0
        result[topic] = {
            "pending": s.get("pending", 0),
            "processing": s.get("processing", 0),
            "done": s.get("done", 0),
            "failed": s.get("failed", 0),
            "speed": avg,
            "eta_sec": round(eta_sec),
        }

    return result, procs, profiles


# Token estimates (measured from actual prompts/payloads)
# Extract: ~2300 input (job description) + ~130 output (title+summary)
# Eval: ~600 prompt + ~550 resume + ~1300 payload ≈ 2450 input + ~150 output
# Haiku 4.5: $1/MTok input, $5/MTok output
EXTRACT_IN, EXTRACT_OUT = 2300, 130
EVAL_IN, EVAL_OUT = 2450, 150
HAIKU_INPUT_RATE = 1.0 / 1_000_000   # $/token
HAIKU_OUTPUT_RATE = 5.0 / 1_000_000  # $/token
GPU_WATTS = 0.200  # kW
ELECTRICITY_RATE = 0.13  # $/kWh


def cost_stats():
    engine = create_engine(QUEUE_DSN)
    with engine.connect() as conn:
        # All time totals — sum actual per-task durations
        alltime = conn.execute(text("""
            SELECT topic, count(*),
                   SUM(EXTRACT(EPOCH FROM done_at - started_at)) as total_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND started_at IS NOT NULL AND done_at IS NOT NULL
            GROUP BY 1
        """)).fetchall()
        # Today
        today = conn.execute(text("""
            SELECT topic, count(*),
                   SUM(EXTRACT(EPOCH FROM done_at - started_at)) as total_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND started_at IS NOT NULL AND done_at IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY 1
        """)).fetchall()
    engine.dispose()

    def calc(rows):
        counts = {r[0]: {"count": r[1], "gpu_sec": float(r[2] or 0)} for r in rows}
        ext_n = counts.get("job_extract", {}).get("count", 0)
        eval_n = counts.get("job_eval", {}).get("count", 0)
        ext_sec = counts.get("job_extract", {}).get("gpu_sec", 0)
        eval_sec = counts.get("job_eval", {}).get("gpu_sec", 0)

        haiku = (
            ext_n * (EXTRACT_IN * HAIKU_INPUT_RATE + EXTRACT_OUT * HAIKU_OUTPUT_RATE)
            + eval_n * (EVAL_IN * HAIKU_INPUT_RATE + EVAL_OUT * HAIKU_OUTPUT_RATE)
        )
        gpu_hrs = (ext_sec + eval_sec) / 3600
        electricity = gpu_hrs * GPU_WATTS * ELECTRICITY_RATE
        return {
            "extracts": ext_n, "evals": eval_n,
            "haiku_cost": round(haiku, 2),
            "gpu_hours": round(gpu_hrs, 1),
            "electricity": round(electricity, 2),
            "savings": round(haiku - electricity, 2),
        }

    return {"alltime": calc(alltime), "today": calc(today)}


def prefect_schedule():
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", """
import asyncio
from prefect.client.orchestration import get_client
from datetime import datetime, timezone, timedelta
async def main():
    async with get_client() as c:
        flows = await c.read_flows()
        fm = {f.id: f.name for f in flows}
        runs = await c.read_flow_runs(limit=8, sort="EXPECTED_START_TIME_DESC")
        now = datetime.now(timezone.utc)
        for r in sorted(runs, key=lambda r: r.expected_start_time or now):
            if r.expected_start_time and r.expected_start_time > now - timedelta(hours=12):
                print(f"{fm.get(r.flow_id,'?')}|{r.state_name}|{r.expected_start_time.strftime('%b %d %H:%M UTC')}")
asyncio.run(main())
"""], timeout=15, text=True,
        ).strip()
        return [{"name": n, "state": s, "time": t} for n, s, t in
                (line.split("|") for line in out.splitlines() if line)]
    except Exception:
        return []


@app.route("/")
def index():
    return HTML


@app.route("/api/status")
def api_status():
    gpu = gpu_stats()
    model = ollama_model()
    queue, procs, profiles = queue_stats()
    schedule = prefect_schedule()
    costs = cost_stats()
    return jsonify({
        "gpu": gpu, "model": model, "queue": queue,
        "processing": procs, "profiles": profiles, "schedule": schedule,
        "costs": costs,
    })


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pipeline Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { font-size: 1.4em; margin-bottom: 16px; color: #58a6ff; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1000px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.9em; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .card.full { grid-column: 1 / -1; }
  .metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: #8b949e; }
  .metric .value { color: #c9d1d9; font-weight: 600; }
  .bar-wrap { background: #21262d; border-radius: 4px; height: 20px; margin-top: 4px; overflow: hidden; position: relative; }
  .bar { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .bar.temp { background: linear-gradient(90deg, #3fb950, #d29922, #f85149); }
  .bar.util { background: #58a6ff; }
  .bar.vram { background: #bc8cff; }
  .bar-label { position: absolute; right: 8px; top: 1px; font-size: 0.75em; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
  .badge.done { background: #238636; color: #fff; }
  .badge.pending { background: #d29922; color: #000; }
  .badge.running { background: #1f6feb; color: #fff; }
  .badge.failed { background: #da3633; color: #fff; }
  .badge.scheduled { background: #30363d; color: #8b949e; }
  .badge.completed { background: #238636; color: #fff; }
  .badge.gpu-hot { background: #f85149; color: #fff; }
  .badge.gpu-warm { background: #d29922; color: #000; }
  .badge.gpu-cool { background: #238636; color: #fff; }
  .eta { color: #58a6ff; font-weight: 600; }
  .toolbar { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
  .btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
         padding: 6px 16px; cursor: pointer; font-family: inherit; font-size: 0.85em; }
  .btn:hover { background: #30363d; }
  .btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .status-dot.ok { background: #3fb950; }
  .status-dot.err { background: #f85149; }
  #updated { color: #484f58; font-size: 0.8em; }
  .error-msg { color: #f85149; padding: 12px; text-align: center; }
</style>
</head>
<body>

<div class="toolbar">
  <h1>📊 Pipeline Dashboard</h1>
  <button class="btn" onclick="refresh()">↻ Refresh</button>
  <button class="btn" id="autoBtn" onclick="toggleAuto()">Auto: OFF</button>
  <span id="updated"></span>
</div>

<div class="grid">
  <div class="card" id="gpuCard">
    <h2>🖥️ GPU</h2>
    <div id="gpuContent">Loading...</div>
  </div>

  <div class="card">
    <h2>🤖 Model</h2>
    <div id="modelContent">Loading...</div>
  </div>

  <div class="card full">
    <h2>📋 Queue (24h)</h2>
    <div id="queueContent">Loading...</div>
  </div>

  <div class="card">
    <h2>👤 By Profile (24h)</h2>
    <div id="profileContent">Loading...</div>
  </div>

  <div class="card">
    <h2>📅 Schedule</h2>
    <div id="scheduleContent">Loading...</div>
  </div>

  <div class="card full">
    <h2>💰 Cost Savings vs Haiku 4.5</h2>
    <div id="costContent">Loading...</div>
  </div>
</div>

<script>
let autoInterval = null;

function toggleAuto() {
  const btn = document.getElementById('autoBtn');
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    btn.textContent = 'Auto: OFF';
    btn.classList.remove('active');
  } else {
    autoInterval = setInterval(refresh, 30000);
    btn.textContent = 'Auto: 30s';
    btn.classList.add('active');
  }
}

function fmtEta(sec) {
  if (!sec) return '—';
  if (sec > 3600) return (sec / 3600).toFixed(1) + 'h';
  return Math.round(sec / 60) + 'min';
}

function tempBadge(t) {
  if (t >= 85) return '<span class="badge gpu-hot">HOT</span>';
  if (t >= 70) return '<span class="badge gpu-warm">WARM</span>';
  return '<span class="badge gpu-cool">OK</span>';
}

function barHtml(pct, cls) {
  return `<div class="bar-wrap"><div class="bar ${cls}" style="width:${pct}%"></div><div class="bar-label">${pct}%</div></div>`;
}

function stateBadge(state) {
  const s = state.toLowerCase();
  return `<span class="badge ${s}">${state}</span>`;
}

function refresh() {
  fetch('/api/status')
    .then(r => r.json())
    .then(d => render(d))
    .catch(e => {
      document.getElementById('updated').textContent = 'Error: ' + e.message;
    });
}

function render(d) {
  // GPU
  const g = d.gpu;
  if (g.ok) {
    const vramPct = Math.round(g.vram_used / g.vram_total * 100);
    document.getElementById('gpuContent').innerHTML = `
      <div class="metric"><span class="label">Temp</span><span class="value">${g.temp}°C ${tempBadge(g.temp)}</span></div>
      ${barHtml(Math.min(g.temp, 100), 'temp')}
      <div class="metric"><span class="label">Utilization</span><span class="value">${g.util}%</span></div>
      ${barHtml(g.util, 'util')}
      <div class="metric"><span class="label">VRAM</span><span class="value">${g.vram_used}/${g.vram_total} MiB</span></div>
      ${barHtml(vramPct, 'vram')}
      <div class="metric"><span class="label">Power</span><span class="value">${g.power.toFixed(0)}W / ${g.power_cap.toFixed(0)}W</span></div>
      <div class="metric"><span class="label">Fan</span><span class="value">${g.fan}%</span></div>
    `;
  } else {
    document.getElementById('gpuContent').innerHTML =
      `<div class="error-msg"><span class="status-dot err"></span>GPU unavailable: ${g.error}</div>`;
  }

  // Model
  const m = d.model;
  document.getElementById('modelContent').innerHTML = m.name === 'none' || m.name === 'unavailable'
    ? `<div class="metric"><span class="label">Status</span><span class="value">${m.name}</span></div>`
    : `<div class="metric"><span class="label">Name</span><span class="value">${m.name}</span></div>
       <div class="metric"><span class="label">VRAM</span><span class="value">${m.vram_gb} GB</span></div>
       <div class="metric"><span class="label">Context</span><span class="value">${m.ctx.toLocaleString()}</span></div>`;

  // Queue
  const q = d.queue;
  let qhtml = `<table><tr><th>Stage</th><th class="num">Pending</th><th class="num">Done</th><th class="num">Failed</th><th class="num">Speed</th><th class="num">ETA</th></tr>`;
  for (const [key, label] of [['job_extract', 'Extract (8B)'], ['job_eval', 'Eval (14B)']]) {
    const s = q[key] || {};
    const speed = s.speed ? s.speed + 's' : '—';
    qhtml += `<tr>
      <td>${label}</td>
      <td class="num">${s.pending || 0}${s.processing ? ' <span class="badge running">+' + s.processing + '</span>' : ''}</td>
      <td class="num"><span class="badge done">${s.done || 0}</span></td>
      <td class="num">${s.failed ? '<span class="badge failed">' + s.failed + '</span>' : '0'}</td>
      <td class="num">${speed}</td>
      <td class="num eta">${fmtEta(s.eta_sec)}</td>
    </tr>`;
  }
  qhtml += '</table>';
  if (d.processing && d.processing.length) {
    qhtml += '<div style="margin-top:10px;color:#8b949e;font-size:0.8em">⚙️ Processing: ';
    qhtml += d.processing.map(p => (p.topic.includes('extract') ? 'Extract' : 'Eval') + ' since ' + p.started).join(', ');
    qhtml += '</div>';
  }
  document.getElementById('queueContent').innerHTML = qhtml;

  // Profiles
  const pr = d.profiles || {};
  let phtml = '';
  for (const [name, t] of Object.entries(pr)) {
    const exPend = t.job_extract_pending || 0;
    const exDone = t.job_extract_done || 0;
    const evPend = t.job_eval_pending || 0;
    const evDone = t.job_eval_done || 0;
    const total = exPend + exDone + evPend + evDone || 1;
    const exPendPct = Math.round(exPend / total * 100);
    const exDonePct = Math.round(exDone / total * 100);
    const evPendPct = Math.round(evPend / total * 100);
    const evDonePct = Math.round(evDone / total * 100);
    phtml += `<div style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-weight:600">${name}</span>
        <span style="color:#8b949e;font-size:0.8em">${exPend+exDone+evPend+evDone} total</span>
      </div>
      <div style="display:flex;height:28px;border-radius:4px;overflow:hidden;font-size:0.7em;line-height:28px;text-align:center">
        ${exPend ? `<div style="background:#d29922;width:${exPendPct}%;color:#000" title="Extract pending">${exPend}</div>` : ''}
        ${exDone ? `<div style="background:#1f6feb;width:${exDonePct}%;color:#fff" title="Extracted">${exDone}</div>` : ''}
        ${evPend ? `<div style="background:#bc8cff;width:${evPendPct}%;color:#000" title="Eval pending">${evPend}</div>` : ''}
        ${evDone ? `<div style="background:#238636;width:${evDonePct}%;color:#fff" title="Evaluated">${evDone}</div>` : ''}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:0.7em;color:#8b949e">
        <span>⬛ Extract pend: ${exPend}</span>
        <span>🔵 Extracted: ${exDone}</span>
        <span>🟣 Eval pend: ${evPend}</span>
        <span>🟢 Evaluated: ${evDone}</span>
      </div>
    </div>`;
  }
  document.getElementById('profileContent').innerHTML = phtml;
  document.getElementById('profileContent').innerHTML = phtml;

  // Schedule
  const sc = d.schedule || [];
  let shtml = '';
  for (const s of sc) {
    shtml += `<div class="metric"><span class="label">${s.name}</span><span class="value">${stateBadge(s.state)} ${s.time}</span></div>`;
  }
  document.getElementById('scheduleContent').innerHTML = shtml || '<div style="color:#484f58">No upcoming runs</div>';

  // Costs
  const c = d.costs || {};
  const at = c.alltime || {};
  const td = c.today || {};
  let chtml = `
    <table>
      <tr><th></th><th class="num">Jobs</th><th class="num">Haiku Cost</th><th class="num">GPU Hours</th><th class="num">Electricity</th><th class="num">Saved</th></tr>
      <tr>
        <td>Today (24h)</td>
        <td class="num">${(td.extracts||0) + (td.evals||0)}</td>
        <td class="num">$${(td.haiku_cost||0).toFixed(2)}</td>
        <td class="num">${(td.gpu_hours||0).toFixed(1)}h</td>
        <td class="num">$${(td.electricity||0).toFixed(2)}</td>
        <td class="num" style="color:#3fb950;font-weight:700">$${(td.savings||0).toFixed(2)}</td>
      </tr>
      <tr style="font-weight:600">
        <td>All Time</td>
        <td class="num">${(at.extracts||0) + (at.evals||0)}</td>
        <td class="num">$${(at.haiku_cost||0).toFixed(2)}</td>
        <td class="num">${(at.gpu_hours||0).toFixed(1)}h</td>
        <td class="num">$${(at.electricity||0).toFixed(2)}</td>
        <td class="num" style="color:#3fb950;font-weight:700;font-size:1.1em">$${(at.savings||0).toFixed(2)}</td>
      </tr>
    </table>
    <div style="margin-top:8px;font-size:0.7em;color:#484f58">
      Based on Haiku 4.5 ($1/MTok in, $5/MTok out) vs local qwen3 on RTX 3070 @ $0.13/kWh
    </div>`;
  document.getElementById('costContent').innerHTML = chtml;

  document.getElementById('updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

refresh();
</script>
</body>
</html>""";

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    print(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
