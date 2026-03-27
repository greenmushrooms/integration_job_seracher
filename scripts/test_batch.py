"""
End-to-end test: existing unevaluated jobs → extract → eval, no Telegram.
Run from the job_searcher_2 directory with env vars set.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import (
    load_search_configs,
    load_resume,
    load_jobs,
    _push_profile_to_queue,
    _drain_queue_results,
    get_queue_engine,
)
from sqlalchemy import text
import requests

RUN_NAME = "test-v4fix-2026-03-21"
BATCH_SIZE = 5  # per profile
POLL_INTERVAL = 10
POLL_TIMEOUT = 3600
WORKER_URL = os.getenv("LLM_QUEUE_WORKER_URL", "http://llm-queue-worker:8080")


def pause_topic(topic: str):
    requests.post(f"{WORKER_URL}/topics/{topic}/pause", timeout=5)
    print(f"  Paused {topic}")


def resume_topic(topic: str):
    requests.post(f"{WORKER_URL}/topics/{topic}/resume", timeout=5)
    print(f"  Resumed {topic}")


def wait_for_extract(run_name: str) -> bool:
    """Wait for all job_extract tasks whose payload.pack_id matches run_name."""
    engine = get_queue_engine()
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE status IN ('done','failed','cancelled')) AS finished,
                    COUNT(*) AS total
                FROM llm_queue.tasks
                WHERE topic = 'job_extract'
                  AND payload->>'pack_id' = :run_name
            """), {"run_name": run_name}).fetchone()
        finished, total = row
        print(f"  job_extract [{run_name}]: {finished}/{total} done", flush=True)
        if total > 0 and finished == total:
            return True
        time.sleep(POLL_INTERVAL)
    print("  Timed out waiting for job_extract.")
    return False


def wait_for_eval(run_name: str) -> bool:
    """Wait for all job_eval tasks with pack_id column = run_name."""
    engine = get_queue_engine()
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE status IN ('done','failed','cancelled')) AS finished,
                    COUNT(*) AS total
                FROM llm_queue.tasks
                WHERE topic = 'job_eval'
                  AND pack_id = :run_name
            """), {"run_name": run_name}).fetchone()
        finished, total = row
        print(f"  job_eval [{run_name}]: {finished}/{total} done", flush=True)
        if total > 0 and finished == total:
            return True
        time.sleep(POLL_INTERVAL)
    print("  Timed out waiting for job_eval.")
    return False


def print_results(run_name: str):
    engine = get_queue_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                ex.payload->>'title'          AS title,
                ex.payload->>'company'        AS company,
                ex.payload->>'sys_profile'    AS profile,
                ev.result->>'verdict'         AS verdict,
                ev.result->'match_scores'     AS scores,
                ev.result->>'one_line_summary' AS summary,
                ev.status                     AS eval_status
            FROM llm_queue.tasks ev
            JOIN llm_queue.tasks ex ON ex.id = ANY(ev.depends_on)
            WHERE ev.pack_id = :run_name
              AND ev.topic = 'job_eval'
            ORDER BY ev.id
        """), {"run_name": run_name}).fetchall()

    if not rows:
        print("  No results found.")
        return

    for profile in ("Slava", "Kezia"):
        profile_rows = [r for r in rows if r[2] == profile]
        print(f"\n{'='*70}")
        print(f"  {profile} ({len(profile_rows)} jobs)")
        print(f"{'='*70}")
        for title, company, _, verdict, scores, summary, status in profile_rows:
            if status != "done":
                print(f"  [{status:16}]  {title} @ {company}")
                continue
            avg = "?"
            if scores:
                s = scores if isinstance(scores, dict) else {}
                vals = [v for v in s.values() if isinstance(v, (int, float))]
                avg = round(sum(vals) / len(vals), 1) if vals else "?"
            verdict_str = (verdict or status or "?")[:16]
            print(f"  [{verdict_str:16}] avg={avg}  {title} @ {company}")
            if summary:
                print(f"    {summary[:100]}")
    print()


def main():
    configs = load_search_configs()
    print(f"Profiles: {[c['profile'] for c in configs]}\n")

    # Push a small batch per profile
    for config in configs:
        profile = config["profile"]
        resume, _ = load_resume(profile)
        jobs_df = load_jobs(profile, limit=BATCH_SIZE)
        if jobs_df.empty:
            print(f"{profile}: no unevaluated jobs found")
            continue
        print(f"{profile}: pushing {len(jobs_df)} jobs (run={RUN_NAME})")
        _push_profile_to_queue(profile, resume, jobs_df, RUN_NAME)

    # Temporarily resume job_extract so these tasks can run
    print("\nResuming job_extract for test run...")
    resume_topic("job_extract")

    try:
        print("\nWaiting for job_extract...")
        wait_for_extract(RUN_NAME)

        print("\nWaiting for job_eval...")
        wait_for_eval(RUN_NAME)
    finally:
        # Always re-pause job_extract when done
        print("\nRe-pausing job_extract...")
        pause_topic("job_extract")

    print("\nDraining results to evaluated_jobs...")
    for config in configs:
        profile = config["profile"]
        written = _drain_queue_results(profile, RUN_NAME)
        print(f"  {profile}: {len(written)} written")

    print("\nResults (no Telegram sent):")
    print_results(RUN_NAME)


if __name__ == "__main__":
    main()
