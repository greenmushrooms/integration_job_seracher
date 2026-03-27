"""
Push the worst-offender job through 10 extract prompt variations.
Waits for all to complete, prints what each returned.
"""
import json, os, time, sys, psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN       = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
QUEUE_DSN    = os.getenv("LLM_QUEUE_DSN")
WORKER_URL   = os.getenv("LLM_QUEUE_WORKER_URL")
JOB_ID       = "li-4384349390"   # Excellence Manager-Field Services @ AtkinsRéalis
PACK_ID      = "prompt-variants-2026-03-21"

VARIANTS = {
    "v1_current": """\
Extract a compact structured summary from the job posting below.
Return only a JSON object with no markdown, no explanation.

Job posting:
{{description}}

Return JSON with exactly these keys:
{
  "job_json": {
    "title": "exact job title",
    "level": "junior|mid|senior|lead|director|vp|individual contributor",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1"],
    "domain": "primary domain e.g. data engineering, ml, finance, etc",
    "location": "city or remote",
    "compensation": "range if mentioned, else null",
    "key_requirements": ["concise bullet 1", "concise bullet 2"]
  }
}""",

    "v2_plain_summary": """\
Read this job posting and write a 3-5 sentence plain text summary.
Cover: what the person actually does day-to-day, what industry/domain this is in, the seniority level, and pay if mentioned.
Do not infer anything not stated. Return plain text only, no JSON.

Job posting:
{{description}}""",

    "v3_minimal_json": """\
Extract the key facts from this job posting. Return JSON only, no markdown.

Job posting:
{{description}}

Return:
{
  "title": "exact title",
  "domain": "what industry/function is this (e.g. nuclear engineering, finance, software, healthcare, logistics)",
  "what_you_do": "one sentence describing the actual day-to-day work",
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}""",

    "v4_domain_first": """\
First identify what domain/industry this job is in, then summarise it.
Return JSON only.

Job posting:
{{description}}

Return:
{
  "domain": "be specific - e.g. nuclear field services, retail operations, data engineering, investment banking",
  "title": "exact job title",
  "level": "junior|mid|senior|lead|director",
  "what_you_do": "1-2 sentences on actual responsibilities",
  "required_background": "what kind of professional would be hired for this",
  "pay": "range if mentioned, else null"
}""",

    "v5_role_clarify": """\
Read this job posting carefully. Describe what kind of person would be hired for it.
Return JSON only, no markdown.

Job posting:
{{description}}

Return:
{
  "title": "exact title",
  "industry": "the actual industry (e.g. nuclear, banking, healthcare, tech)",
  "function": "the job function (e.g. operations management, software engineering, data analysis)",
  "day_to_day": "what does this person spend their day doing",
  "requires_technical_skills": true or false,
  "key_technical_skills": ["only list if explicitly mentioned in posting"],
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}""",

    "v6_anti_hallucination": """\
Summarise this job posting. Only include facts explicitly stated in the posting — do not infer or add anything.
Return JSON only, no markdown.

Job posting:
{{description}}

Return:
{
  "title": "exact title from posting",
  "what_this_job_is": "describe the role in plain English, based only on what is written",
  "explicitly_required_skills": ["only skills/tools/languages named in the posting"],
  "level": "junior|mid|senior|lead|director",
  "pay": "exact text from posting or null"
}""",

    "v7_two_sentence": """\
Summarise this job in exactly 2 sentences: one describing what the role does, one describing what background is required.
Return as JSON: {"summary": "..."}

Job posting:
{{description}}""",

    "v8_checklist": """\
Classify and summarise this job posting. Return JSON only.

Job posting:
{{description}}

Return:
{
  "title": "exact title",
  "category": "one of: software engineering | data engineering | data science | business analysis | operations management | finance | sales | marketing | hr | legal | healthcare | engineering (non-software) | other",
  "sub_domain": "more specific e.g. nuclear operations, CRM implementation, cloud infrastructure",
  "level": "junior|mid|senior|lead|director",
  "summary": "2-3 sentences on what the role actually does",
  "pay": "range if mentioned, else null"
}""",

    "v9_bullets": """\
Extract the key facts from this job posting as plain text bullets. No JSON, no headers.
Cover: actual job function, industry, required background, seniority, pay.

Job posting:
{{description}}""",

    "v10_what_is_this": """\
Answer these questions about the job posting below. Return JSON only.

Job posting:
{{description}}

Questions:
{
  "what_does_this_company_do": "one sentence",
  "what_does_this_person_do": "one sentence on actual responsibilities",
  "what_department_is_this": "e.g. engineering, operations, finance, IT, HR",
  "is_this_a_technical_role": true or false,
  "technical_tools_mentioned": ["list only tools/languages explicitly named"],
  "title": "exact title",
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}"""
}


def get_description():
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT description FROM jobspy_jobs WHERE id = %s", (JOB_ID,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def push_tasks(description):
    conn = psycopg2.connect(QUEUE_DSN)
    conn.autocommit = True
    ids = {}
    with conn.cursor() as cur:
        for name, template in VARIANTS.items():
            prompt = template.replace("{{description}}", description)
            payload = json.dumps({
                "job_id": JOB_ID,
                "variant": name,
                "prompt": prompt,
            })
            cur.execute(
                "INSERT INTO llm_queue.tasks (topic, payload, pack_id, priority) VALUES ('job_extract', %s, %s, 10) RETURNING id",
                (payload, PACK_ID)
            )
            ids[name] = cur.fetchone()[0]
            print(f"  Pushed {name} → task {ids[name]}")
    conn.close()
    return ids


def wait_and_collect(task_ids: dict):
    conn = psycopg2.connect(QUEUE_DSN)
    deadline = time.time() + 600
    id_list = list(task_ids.values())

    while time.time() < deadline:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, status, result, EXTRACT(EPOCH FROM (done_at - started_at))::int FROM llm_queue.tasks WHERE id = ANY(%s)",
                (id_list,)
            )
            rows = {r[0]: r for r in cur.fetchall()}

        done = sum(1 for r in rows.values() if r[1] in ('done', 'failed'))
        print(f"  {done}/{len(id_list)} done...", flush=True)
        if done == len(id_list):
            break
        time.sleep(10)

    conn.close()
    return rows


def main():
    print(f"Job: {JOB_ID}  pack: {PACK_ID}\n")

    desc = get_description()
    if not desc:
        print("Job not found"); return

    print("Pushing 10 variants...")
    task_ids = push_tasks(desc)

    print("\nResuming job_extract...")
    import requests
    requests.post(f"{WORKER_URL}/topics/job_extract/resume", timeout=5)

    print("\nWaiting for results...")
    rows = wait_and_collect(task_ids)

    print("\n" + "="*80)
    for name, tid in task_ids.items():
        row = rows.get(tid)
        status = row[1] if row else "?"
        result = row[2] if row else None
        secs   = row[3] if row else "?"
        print(f"\n{'='*80}")
        print(f"  {name}  [task {tid}]  status={status}  secs={secs}")
        print(f"{'='*80}")
        if result:
            try:
                print(json.dumps(result, indent=2))
            except Exception:
                print(result)
        else:
            print("  (no result)")

    print("\nRe-pausing job_extract...")
    requests.post(f"{WORKER_URL}/topics/job_extract/pause", timeout=5)


if __name__ == "__main__":
    main()
