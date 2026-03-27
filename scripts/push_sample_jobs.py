"""
Push the locked sample jobs from cache_test_run.md through the pipeline.
Used to compare extract model speed/quality.
"""
import json, os, sys, time
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN    = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
QUEUE_DSN = os.getenv("LLM_QUEUE_DSN")
PACK_ID   = os.getenv("PACK_ID", "test-8b-extract-2026-03-21")

SAMPLE = [
    ("li-4384349390", "Slava"),
    ("li-4384349389", "Slava"),
    ("li-4384094286", "Slava"),
    ("li-4388574605", "Kezia"),
    ("li-4388573486", "Kezia"),
    ("li-4388569873", "Kezia"),
    ("li-4388567674", "Kezia"),
    ("li-4388566964", "Kezia"),
]

def main():
    db = psycopg2.connect(DB_DSN)
    q  = psycopg2.connect(QUEUE_DSN)
    q.autocommit = True

    # Load resumes
    resumes = {}
    with db.cursor() as cur:
        cur.execute("SELECT profile, resume_json FROM adm.resume WHERE is_active = TRUE")
        for profile, rj in cur.fetchall():
            resumes[profile] = json.dumps(rj)

    # Load job descriptions
    job_ids = [j for j, _ in SAMPLE]
    with db.cursor() as cur:
        cur.execute("SELECT id, title, company, description FROM jobspy_jobs WHERE id = ANY(%s)", (job_ids,))
        jobs = {r[0]: r for r in cur.fetchall()}

    pushed = 0
    with q.cursor() as cur:
        for job_id, profile in SAMPLE:
            job = jobs.get(job_id)
            if not job:
                print(f"  SKIP {job_id} — not found"); continue
            _, title, company, description = job
            resume = resumes.get(profile, "{}")
            payload = json.dumps({
                "job_id": job_id,
                "title": title,
                "company": company,
                "sys_profile": profile,
                "sys_run_name": PACK_ID,
                "pack_id": PACK_ID,
                "inputs": {"description": description or ""},
                "next_step": {
                    "topic": "job_eval",
                    "step": "eval",
                    "inputs": {"candidate_json": resume},
                },
            })
            cur.execute(
                "INSERT INTO llm_queue.tasks (topic, payload, pack_id, priority) VALUES ('job_extract', %s, %s, 10) RETURNING id",
                (payload, PACK_ID)
            )
            tid = cur.fetchone()[0]
            print(f"  pushed {job_id} ({profile}) → task {tid}")
            pushed += 1

    print(f"\n{pushed} jobs pushed to pack_id={PACK_ID}")
    db.close(); q.close()

if __name__ == "__main__":
    main()
