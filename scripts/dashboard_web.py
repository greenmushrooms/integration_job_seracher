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
        # All tasks active in the last 24h (use done_at if available, else created_at)
        rows = conn.execute(text("""
            SELECT topic, status, count(*)
            FROM llm_queue.tasks
            WHERE status IN ('pending', 'processing', 'done', 'failed')
              AND COALESCE(done_at, created_at) >= NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2
        """)).fetchall()

        speed = conn.execute(text("""
            SELECT topic, count(*),
                   EXTRACT(EPOCH FROM max(done_at) - min(done_at)) / NULLIF(count(*)-1, 0) as avg_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND done_at >= NOW() - INTERVAL '60 minutes'
            GROUP BY 1
        """)).fetchall()

        processing = conn.execute(text("""
            SELECT topic, started_at::text
            FROM llm_queue.tasks WHERE status = 'processing'
        """)).fetchall()

        # Per profile: all tasks active in last 24h
        totals = conn.execute(text("""
            SELECT payload->>'sys_profile', topic, status, count(*)
            FROM llm_queue.tasks
            WHERE status IN ('pending', 'processing', 'done')
              AND COALESCE(done_at, created_at) >= NOW() - INTERVAL '24 hours'
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
        # Last 24h (use done_at if available, else created_at)
        today = conn.execute(text("""
            SELECT topic, count(*),
                   SUM(EXTRACT(EPOCH FROM done_at - started_at)) as total_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND started_at IS NOT NULL AND done_at IS NOT NULL
              AND COALESCE(done_at, created_at) >= NOW() - INTERVAL '24 hours'
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


def jobs_stats():
    engine = create_engine(DB_DSN)
    with engine.connect() as conn:
        weekly_matches = conn.execute(text("""
            SELECT
                DATE_TRUNC('week', e.created_at)::date AS week,
                COUNT(*) AS total_evals,
                COUNT(*) FILTER (WHERE e.avg_score >= 6.9) AS good_matches
            FROM public.evaluated_jobs e
            WHERE e.sys_profile = 'Slava'
              AND e.created_at >= NOW() - INTERVAL '13 weeks'
            GROUP BY 1
            ORDER BY 1
        """)).fetchall()

        weekly_comp = conn.execute(text("""
            SELECT
                DATE_TRUNC('week', e.created_at)::date AS week,
                AVG((j.min_amount::numeric + j.max_amount::numeric) / 2.0) FILTER (WHERE e.avg_score >= 6.9) AS avg_comp_all,
                AVG((j.min_amount::numeric + j.max_amount::numeric) / 2.0) FILTER (WHERE e.avg_score >= 8.0) AS avg_comp_top,
                MAX((j.min_amount::numeric + j.max_amount::numeric) / 2.0) FILTER (WHERE e.avg_score >= 6.9) AS max_comp
            FROM public.evaluated_jobs e
            JOIN public.jobspy_jobs j ON e.job_id = j.id
            WHERE e.sys_profile = 'Slava'
              AND e.avg_score >= 6.9
              AND j.min_amount IS NOT NULL
              AND j.max_amount IS NOT NULL
              AND j.min_amount ~ '^[0-9]+(\\.[0-9]+)?$'
              AND j.max_amount ~ '^[0-9]+(\\.[0-9]+)?$'
              AND (j.interval = 'yearly' OR j.interval = 'hourly')
              AND e.created_at >= NOW() - INTERVAL '13 weeks'
            GROUP BY 1
            ORDER BY 1
        """)).fetchall()

        # All compensation data for matched jobs — yearly normalized, no date filter
        comp_jobs = conn.execute(text("""
            SELECT
                j.min_amount::numeric,
                j.max_amount::numeric,
                j.interval,
                j.currency,
                e.avg_score,
                j.title,
                j.company,
                e.created_at::date AS eval_date
            FROM public.evaluated_jobs e
            JOIN public.jobspy_jobs j ON e.job_id = j.id
            WHERE e.sys_profile = 'Slava'
              AND e.avg_score >= 6.9
              AND j.min_amount IS NOT NULL
              AND j.max_amount IS NOT NULL
              AND j.min_amount ~ '^[0-9]+(\\.[0-9]+)?$'
              AND j.max_amount ~ '^[0-9]+(\\.[0-9]+)?$'
              AND j.min_amount::numeric > 0
        """)).fetchall()

        top_jobs = conn.execute(text("""
            SELECT
                j.title,
                j.company,
                j.location,
                e.avg_score,
                CASE WHEN j.min_amount ~ '^[0-9]+(\\.[0-9]+)?$' THEN j.min_amount::numeric ELSE NULL END,
                CASE WHEN j.max_amount ~ '^[0-9]+(\\.[0-9]+)?$' THEN j.max_amount::numeric ELSE NULL END,
                j.interval,
                j.currency,
                e.reasoning,
                COALESCE(j.job_url_direct, j.job_url) AS url,
                e.created_at::date AS eval_date
            FROM public.evaluated_jobs e
            JOIN public.jobspy_jobs j ON e.job_id = j.id
            WHERE e.sys_profile = 'Slava'
              AND e.avg_score >= 6.9
            ORDER BY e.avg_score DESC, e.created_at DESC
            LIMIT 300
        """)).fetchall()

    engine.dispose()

    def yearly(amount, interval):
        v = float(amount)
        return v * 2080 if interval == 'hourly' else v

    wm = [{"week": str(r[0]), "total": int(r[1]), "matches": int(r[2])} for r in weekly_matches]

    wc = []
    for r in weekly_comp:
        wc.append({
            "week": str(r[0]),
            "avg_all": round(float(r[1])) if r[1] is not None else None,
            "avg_top": round(float(r[2])) if r[2] is not None else None,
            "max": round(float(r[3])) if r[3] is not None else None,
        })

    comp_data = []
    for r in comp_jobs:
        mid = yearly((float(r[0]) + float(r[1])) / 2.0, r[2])
        comp_data.append({
            "mid": round(mid),
            "min": round(yearly(r[0], r[2])),
            "max": round(yearly(r[1], r[2])),
            "currency": r[3] or "?",
            "score": float(r[4]),
            "title": r[5],
            "company": r[6],
            "date": str(r[7]),
        })

    jobs = []
    for r in top_jobs:
        try:
            reasoning = r[8] if isinstance(r[8], dict) else json.loads(r[8]) if r[8] else {}
        except Exception:
            reasoning = {}
        comp_str = ""
        if r[4] and r[5]:
            lo = yearly(r[4], r[6])
            hi = yearly(r[5], r[6])
            comp_str = f"{r[7] or '?'} {lo/1000:.0f}k–{hi/1000:.0f}k"
        jobs.append({
            "title": r[0],
            "company": r[1],
            "location": r[2],
            "score": float(r[3]),
            "comp": comp_str,
            "verdict": reasoning.get("verdict", ""),
            "summary": reasoning.get("summary", ""),
            "url": r[9],
            "date": str(r[10]),
        })

    return {"weekly_matches": wm, "weekly_comp": wc, "comp_data": comp_data, "top_jobs": jobs}


@app.route("/")
def index():
    return HTML


@app.route("/api/jobs")
def api_jobs():
    try:
        return jsonify(jobs_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { font-size: 1.4em; color: #58a6ff; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1100px; }
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
  td.wrap { max-width: 260px; white-space: normal; word-break: break-word; font-size: 0.78em; color: #8b949e; }
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
  .badge.step-up { background: #1f6feb; color: #fff; }
  .badge.lateral { background: #238636; color: #fff; }
  .badge.pivot { background: #bc8cff; color: #000; }
  .badge.title-regression { background: #484f58; color: #c9d1d9; }
  .eta { color: #58a6ff; font-weight: 600; }
  .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
         padding: 6px 16px; cursor: pointer; font-family: inherit; font-size: 0.85em; }
  .btn:hover { background: #30363d; }
  .btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .tab-btn { background: #161b22; color: #8b949e; border: 1px solid #30363d; border-radius: 6px 6px 0 0;
             padding: 8px 20px; cursor: pointer; font-family: inherit; font-size: 0.9em; border-bottom: none; }
  .tab-btn.active { background: #0d1117; color: #58a6ff; border-bottom: 2px solid #0d1117; }
  .tab-bar { display: flex; gap: 4px; margin-bottom: 0; border-bottom: 1px solid #30363d; }
  .tab-content { display: none; padding-top: 16px; }
  .tab-content.active { display: block; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .status-dot.ok { background: #3fb950; }
  .status-dot.err { background: #f85149; }
  #updated { color: #484f58; font-size: 0.8em; }
  .error-msg { color: #f85149; padding: 12px; text-align: center; }
  .score-bar { display: inline-block; background: #238636; height: 8px; border-radius: 4px; vertical-align: middle; margin-right: 4px; }
  .slider-wrap { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .slider-wrap input[type=range] { flex: 1; accent-color: #58a6ff; }
  .slider-label { color: #8b949e; font-size: 0.85em; white-space: nowrap; }
  .slider-val { color: #58a6ff; font-weight: 700; min-width: 40px; }
  canvas { max-width: 100%; }
</style>
</head>
<body>

<div class="toolbar">
  <h1>📊 Pipeline Dashboard</h1>
  <button class="btn" onclick="refreshCurrent()">↻ Refresh</button>
  <button class="btn" id="autoBtn" onclick="toggleAuto()">Auto: OFF</button>
  <span id="updated"></span>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('pipeline')">⚙️ Pipeline</button>
  <button class="tab-btn" onclick="switchTab('matches')">🎯 Matches — Slava</button>
</div>

<div id="tab-pipeline" class="tab-content active">
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
    <h2>📋 Queue</h2>
    <div id="queueContent">Loading...</div>
  </div>

  <div class="card">
    <h2>👤 By Profile</h2>
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
</div>

<div id="tab-matches" class="tab-content">
<div class="grid" style="max-width:1100px">

  <div class="card full">
    <h2>📈 Weekly Match Volume — jobs scored ≥ 6.9</h2>
    <canvas id="matchesChart" height="120"></canvas>
  </div>

  <div class="card full">
    <h2>💰 Compensation Distribution</h2>
    <div class="slider-wrap">
      <span class="slider-label">Min score filter:</span>
      <input type="range" id="scoreSlider" min="6.9" max="9.5" step="0.1" value="6.9" oninput="activeCompBin=null; updateCompHist(); document.getElementById('jobsFilterLabel').textContent=''; renderTopJobs(recentJobs(jobsData.top_jobs||[]));">
      <span class="slider-val" id="sliderVal">6.9</span>
      <span class="slider-label" id="compCount"></span>
    </div>
    <canvas id="compHistChart" height="130" style="cursor:pointer"></canvas>
    <div style="margin-top:6px;font-size:0.75em;color:#484f58">Click a bar to filter the jobs table below · click again to clear</div>
  </div>

  <div class="card full">
    <h2>🏆 Top Matches <span id="jobsFilterLabel" style="color:#58a6ff;font-size:0.85em;font-weight:400">— last 4 weeks</span></h2>
    <div id="topJobsContent">Loading...</div>
  </div>

</div>
</div>

<script>
let autoInterval = null;
let currentTab = 'pipeline';
let jobsData = null;
let matchesChartInst = null;
let compHistChartInst = null;
let activeCompBin = null;  // {lo, hi} when a bar is selected, null = show all

const CHART_DEFAULTS = {
  color: '#c9d1d9',
  plugins: { legend: { labels: { color: '#8b949e', font: { family: 'SF Mono, Fira Code, Consolas, monospace', size: 11 } } } },
  scales: {
    x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
  },
};

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`.tab-btn[onclick*="${tab}"]`).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'matches' && !jobsData) loadJobs();
}

function toggleAuto() {
  const btn = document.getElementById('autoBtn');
  if (autoInterval) {
    clearInterval(autoInterval);
    autoInterval = null;
    btn.textContent = 'Auto: OFF';
    btn.classList.remove('active');
  } else {
    autoInterval = setInterval(refreshCurrent, 30000);
    btn.textContent = 'Auto: 30s';
    btn.classList.add('active');
  }
}

function refreshCurrent() {
  if (currentTab === 'pipeline') refreshPipeline();
  else { jobsData = null; loadJobs(); }
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

function verdictBadge(v) {
  if (!v) return '';
  const cls = v.toLowerCase().replace(/\\s+/g, '-');
  return `<span class="badge ${cls}" style="margin-left:4px">${v}</span>`;
}

function scoreColor(s) {
  if (s >= 8.5) return '#3fb950';
  if (s >= 7.5) return '#58a6ff';
  if (s >= 7.0) return '#d29922';
  return '#8b949e';
}

// ── Pipeline tab ─────────────────────────────────────────────────────────────

function refreshPipeline() {
  fetch('/api/status')
    .then(r => r.json())
    .then(d => renderPipeline(d))
    .catch(e => {
      document.getElementById('updated').textContent = 'Error: ' + e.message;
    });
}

function renderPipeline(d) {
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

  const m = d.model;
  document.getElementById('modelContent').innerHTML = m.name === 'none' || m.name === 'unavailable'
    ? `<div class="metric"><span class="label">Status</span><span class="value">${m.name}</span></div>`
    : `<div class="metric"><span class="label">Name</span><span class="value">${m.name}</span></div>
       <div class="metric"><span class="label">VRAM</span><span class="value">${m.vram_gb} GB</span></div>
       <div class="metric"><span class="label">Context</span><span class="value">${m.ctx.toLocaleString()}</span></div>`;

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

  const sc = d.schedule || [];
  let shtml = '';
  for (const s of sc) {
    shtml += `<div class="metric"><span class="label">${s.name}</span><span class="value">${stateBadge(s.state)} ${s.time}</span></div>`;
  }
  document.getElementById('scheduleContent').innerHTML = shtml || '<div style="color:#484f58">No upcoming runs</div>';

  const c = d.costs || {};
  const at = c.alltime || {};
  const td = c.today || {};
  document.getElementById('costContent').innerHTML = `
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

  document.getElementById('updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

// ── Matches tab ───────────────────────────────────────────────────────────────

function loadJobs() {
  document.getElementById('topJobsContent').textContent = 'Loading…';
  fetch('/api/jobs')
    .then(r => r.json())
    .then(d => {
      if (d.error) throw new Error(d.error);
      jobsData = d;
      renderMatches(d);
      document.getElementById('updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
    })
    .catch(e => {
      document.getElementById('topJobsContent').innerHTML = `<div class="error-msg">Error: ${e.message}</div>`;
    });
}

function recentJobs(allJobs, weeks=4) {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - weeks * 7);
  return allJobs.filter(j => new Date(j.date) >= cutoff);
}

function renderMatches(d) {
  renderMatchesChart(d.weekly_matches || []);
  updateCompHist();
  renderTopJobs(recentJobs(d.top_jobs || []));
}

function renderMatchesChart(rows) {
  const labels = rows.map(r => r.week.slice(5));  // MM-DD
  const total  = rows.map(r => r.total);
  const good   = rows.map(r => r.matches);

  if (matchesChartInst) matchesChartInst.destroy();
  matchesChartInst = new Chart(document.getElementById('matchesChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Total evals', data: total, backgroundColor: '#21262d', borderColor: '#30363d', borderWidth: 1 },
        { label: 'Matches ≥6.9', data: good,  backgroundColor: '#238636', borderColor: '#3fb950', borderWidth: 1 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, stacked: false },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true, title: { display: true, text: 'Jobs', color: '#484f58' } },
      },
    },
  });
}

function updateCompHist() {
  if (!jobsData) return;
  const minScore = parseFloat(document.getElementById('scoreSlider').value);
  document.getElementById('sliderVal').textContent = minScore.toFixed(1);

  const filtered = (jobsData.comp_data || []).filter(j => j.score >= minScore);
  document.getElementById('compCount').textContent = `(${filtered.length} jobs with salary data)`;

  if (!filtered.length) {
    if (compHistChartInst) { compHistChartInst.destroy(); compHistChartInst = null; }
    return;
  }

  const binSize = 10000;
  const vals = filtered.map(j => j.mid);
  const minV = Math.floor(Math.min(...vals) / binSize) * binSize;
  const maxV = Math.ceil(Math.max(...vals) / binSize) * binSize;
  const binKeys = [];
  for (let b = minV; b < maxV; b += binSize) binKeys.push(b);
  const counts = binKeys.map(() => 0);
  vals.forEach(v => {
    const i = Math.floor((v - minV) / binSize);
    if (i >= 0 && i < counts.length) counts[i]++;
  });
  const labels = binKeys.map(b => (b / 1000).toFixed(0) + 'k');

  const nBins = binKeys.length;
  function binColor(i, selected) {
    const t = nBins > 1 ? i / (nBins - 1) : 0.5;
    const r = Math.round(31 + t * (88 - 31));
    const g = Math.round(111 + t * (185 - 111));
    const b = Math.round(235 + t * (80 - 235));
    return selected ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},0.28)`;
  }

  const selIdx = activeCompBin
    ? binKeys.findIndex(b => b === activeCompBin.lo)
    : -1;
  const colors = binKeys.map((_, i) =>
    selIdx === -1 ? binColor(i, true) : binColor(i, i === selIdx)
  );

  if (compHistChartInst) compHistChartInst.destroy();
  compHistChartInst = new Chart(document.getElementById('compHistChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Jobs', data: counts, backgroundColor: colors, borderWidth: 0, barPercentage: 0.95, categoryPercentage: 1.0 }],
    },
    options: {
      ...CHART_DEFAULTS,
      onClick(evt) {
        const pts = compHistChartInst.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, false);
        if (!pts.length) return;
        const idx = pts[0].index;
        const lo = binKeys[idx];
        const hi = lo + binSize;
        if (activeCompBin && activeCompBin.lo === lo) {
          activeCompBin = null;
        } else {
          activeCompBin = { lo, hi };
        }
        updateCompHist();
        filterJobsByBin();
      },
      plugins: { legend: { display: false } },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Annual Salary (midpoint)', color: '#484f58' } },
        y: { ...CHART_DEFAULTS.scales.y, beginAtZero: true, ticks: { ...CHART_DEFAULTS.scales.y.ticks, precision: 0 } },
      },
    },
  });
}

function filterJobsByBin() {
  if (!jobsData) return;
  const minScore = parseFloat(document.getElementById('scoreSlider').value);
  const label = document.getElementById('jobsFilterLabel');
  if (!activeCompBin) {
    label.textContent = '';
    renderTopJobs(recentJobs(jobsData.top_jobs || []));
    return;
  }
  const { lo, hi } = activeCompBin;
  label.textContent = `— filtered: ${(lo/1000).toFixed(0)}k–${(hi/1000).toFixed(0)}k salary range (all time)`;

  const compFiltered = new Set(
    (jobsData.comp_data || [])
      .filter(j => j.score >= minScore && j.mid >= lo && j.mid < hi)
      .map(j => j.title + '|' + j.company)
  );
  const jobs = (jobsData.top_jobs || []).filter(j => compFiltered.has(j.title + '|' + j.company));
  renderTopJobs(jobs);
}

function renderTopJobs(jobs) {
  if (!jobs.length) {
    document.getElementById('topJobsContent').innerHTML = '<div style="color:#484f58;padding:12px">No matches found.</div>';
    return;
  }
  let html = `<table>
    <tr>
      <th>Score</th><th>Title</th><th>Company</th><th>Location</th>
      <th>Comp</th><th>Verdict</th><th>Summary</th><th>Date</th>
    </tr>`;
  for (const j of jobs) {
    const sc = j.score.toFixed(1);
    const scorePct = Math.round((j.score - 6.9) / (10 - 6.9) * 100);
    const scoreHtml = `<span style="color:${scoreColor(j.score)};font-weight:700">${sc}</span>
      <div class="bar-wrap" style="width:60px;margin-top:2px">
        <div style="background:${scoreColor(j.score)};width:${scorePct}%;height:100%;border-radius:4px"></div>
      </div>`;
    const titleHtml = j.url
      ? `<a href="${j.url}" target="_blank" style="color:#58a6ff;text-decoration:none">${j.title}</a>`
      : j.title;
    html += `<tr>
      <td class="num" style="min-width:70px">${scoreHtml}</td>
      <td style="font-weight:600">${titleHtml}</td>
      <td style="color:#8b949e">${j.company}</td>
      <td style="color:#8b949e;font-size:0.8em">${j.location || '—'}</td>
      <td class="num" style="font-size:0.8em;white-space:nowrap">${j.comp || '—'}</td>
      <td>${verdictBadge(j.verdict)}</td>
      <td class="wrap">${j.summary || ''}</td>
      <td style="color:#484f58;font-size:0.75em;white-space:nowrap">${j.date}</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('topJobsContent').innerHTML = html;
}

// ── Init ──────────────────────────────────────────────────────────────────────
refreshPipeline();
</script>
</body>
</html>""";

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    print(f"Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
