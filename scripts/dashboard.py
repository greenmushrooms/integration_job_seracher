#!/usr/bin/env python3
"""Pipeline monitoring dashboard. Run: .venv/bin/python3 scripts/dashboard.py"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import create_engine, text

DB_DSN = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
QUEUE_DSN = os.getenv("LLM_QUEUE_DSN")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
REFRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 30  # seconds


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
            "temp": f"{temp}°C", "util": f"{util}%", "power": f"{pwr}W/{pwr_cap}W",
            "vram": f"{mem_used}/{mem_total} MiB", "fan": f"{fan}%",
        }
    except Exception as e:
        return {"error": str(e)}


def ollama_model():
    try:
        out = subprocess.check_output(
            ["curl", "-s", f"{OLLAMA_URL}/api/ps"], timeout=5, text=True,
        )
        data = json.loads(out)
        models = data.get("models", [])
        if not models:
            return "(none loaded)"
        m = models[0]
        vram_gb = m.get("size_vram", 0) / 1e9
        ctx = m.get("context_length", 0)
        return f"{m['name']} | VRAM {vram_gb:.1f}GB | ctx {ctx}"
    except Exception:
        return "(unavailable)"


def queue_stats():
    engine = create_engine(QUEUE_DSN)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT topic, status, count(*)
            FROM llm_queue.tasks
            WHERE status IN ('pending', 'processing', 'done', 'failed')
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)).fetchall()

        # Throughput last 30 min
        speed = conn.execute(text("""
            SELECT topic, count(*),
                   EXTRACT(EPOCH FROM max(done_at) - min(done_at)) / NULLIF(count(*)-1, 0) as avg_sec
            FROM llm_queue.tasks
            WHERE status = 'done' AND done_at >= NOW() - INTERVAL '30 minutes'
            GROUP BY 1
        """)).fetchall()

        # Currently processing
        processing = conn.execute(text("""
            SELECT topic, started_at::text
            FROM llm_queue.tasks WHERE status = 'processing'
        """)).fetchall()

    engine.dispose()

    stats = {}
    for topic, status, count in rows:
        stats.setdefault(topic, {})[status] = count

    speeds = {}
    for topic, count, avg in speed:
        speeds[topic] = {"count": count, "avg": avg or 0}

    procs = []
    for topic, started in processing:
        procs.append({"topic": topic, "started": started})

    return stats, speeds, procs


def prefect_schedule():
    try:
        out = subprocess.check_output(
            [".venv/bin/python3", "-c", """
import asyncio
from prefect.client.orchestration import get_client
from datetime import datetime, timezone
async def main():
    async with get_client() as c:
        flows = await c.read_flows()
        fm = {f.id: f.name for f in flows}
        runs = await c.read_flow_runs(limit=6, sort="EXPECTED_START_TIME_DESC")
        now = datetime.now(timezone.utc)
        for r in sorted(runs, key=lambda r: r.expected_start_time or now):
            if r.expected_start_time and r.expected_start_time > now - __import__('datetime').timedelta(hours=2):
                print(f"{fm.get(r.flow_id,'?')}|{r.state_name}|{r.expected_start_time.strftime('%b %d %H:%M UTC')}")
asyncio.run(main())
"""], timeout=15, text=True,
        ).strip()
        return [line.split("|") for line in out.splitlines() if line]
    except Exception:
        return []


def render(gpu, model, stats, speeds, procs, schedule):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w = 62

    print(f"\033[2J\033[H", end="")  # clear screen
    print(f"{'═' * w}")
    print(f"  📊 PIPELINE DASHBOARD          {now}")
    print(f"{'═' * w}")

    # GPU
    print(f"\n  🖥️  GPU")
    if "error" in gpu:
        print(f"     ❌ {gpu['error']}")
    else:
        print(f"     Temp: {gpu['temp']:>6}   Fan: {gpu['fan']:>5}   Power: {gpu['power']}")
        print(f"     Util: {gpu['util']:>6}   VRAM: {gpu['vram']}")

    # Model
    print(f"\n  🤖 Model: {model}")

    # Queue
    print(f"\n  📋 QUEUE (last 24h)")
    print(f"     {'Topic':<15} {'Pend':>5} {'Proc':>5} {'Done':>5} {'Fail':>5}  {'Speed':>10}  {'ETA':>10}")
    print(f"     {'─' * 65}")
    for topic in ("job_extract", "job_eval"):
        s = stats.get(topic, {})
        pend = s.get("pending", 0)
        proc = s.get("processing", 0)
        done = s.get("done", 0)
        fail = s.get("failed", 0)
        sp = speeds.get(topic, {})
        avg = sp.get("avg", 0)
        speed_str = f"{avg:.0f}s/job" if avg else "—"
        if avg and (pend + proc) > 0:
            eta_sec = avg * (pend + proc)
            if eta_sec > 3600:
                eta_str = f"{eta_sec / 3600:.1f}h"
            else:
                eta_str = f"{eta_sec / 60:.0f}min"
        else:
            eta_str = "—"
        label = "Extract" if "extract" in topic else "Eval"
        print(f"     {label:<15} {pend:>5} {proc:>5} {done:>5} {fail:>5}  {speed_str:>10}  {eta_str:>10}")

    # Currently processing
    if procs:
        print(f"\n  ⚙️  Processing:")
        for p in procs:
            label = "Extract" if "extract" in p["topic"] else "Eval"
            started = p["started"][:19] if p["started"] else "?"
            print(f"     {label} — started {started}")

    # Prefect schedule
    if schedule:
        print(f"\n  📅 SCHEDULE")
        for name, state, when in schedule:
            icon = "✅" if state == "Completed" else "🔄" if state == "Running" else "⏳" if state == "Scheduled" else "❌"
            print(f"     {icon} {name:<25} {state:<12} {when}")

    print(f"\n{'═' * w}")
    print(f"  Refreshing every {REFRESH}s  |  Ctrl+C to exit")
    print(f"{'═' * w}")


def main():
    while True:
        try:
            gpu = gpu_stats()
            model = ollama_model()
            stats, speeds, procs = queue_stats()
            schedule = prefect_schedule()
            render(gpu, model, stats, speeds, procs, schedule)
            time.sleep(REFRESH)
        except KeyboardInterrupt:
            print("\n")
            break


if __name__ == "__main__":
    main()
