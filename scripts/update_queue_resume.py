"""
Update pending llm_queue.tasks payloads to replace raw resume text
with compact JSON resume from adm.resume.resume_json.

Reduces per-task prompt size ~5x for all pending job_eval tasks.
"""
import json
import os
import psycopg2
import psycopg2.extras

DB_DSN = os.environ["DATABASE_URL"]
QUEUE_DSN = os.environ["LLM_QUEUE_DSN"]


def main():
    main_db = psycopg2.connect(DB_DSN)
    queue_db = psycopg2.connect(QUEUE_DSN)
    main_db.autocommit = True
    queue_db.autocommit = True

    # Load resume JSON for all active profiles
    with main_db.cursor() as cur:
        cur.execute(
            "SELECT profile, resume_json FROM adm.resume WHERE is_active = TRUE AND resume_json IS NOT NULL"
        )
        profiles = {row[0]: json.dumps(row[1]) for row in cur.fetchall()}

    print(f"Loaded resume JSON for profiles: {list(profiles.keys())}")

    total_updated = 0
    for profile, resume_json_str in profiles.items():
        with queue_db.cursor() as cur:
            cur.execute(
                """
                UPDATE llm_queue.tasks
                SET payload = jsonb_set(payload, '{resume}', to_jsonb(%s::text))
                WHERE topic = 'job_eval'
                  AND status = 'pending'
                  AND payload->>'sys_profile' = %s
                """,
                (resume_json_str, profile),
            )
            updated = cur.rowcount
            print(f"  {profile}: updated {updated} pending tasks")
            total_updated += updated

    print(f"\nTotal updated: {total_updated} tasks")

    # Show remaining pending count
    with queue_db.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM llm_queue.tasks WHERE topic = 'job_eval' AND status = 'pending'"
        )
        remaining = cur.fetchone()[0]
    print(f"Pending tasks remaining: {remaining}")


if __name__ == "__main__":
    main()
