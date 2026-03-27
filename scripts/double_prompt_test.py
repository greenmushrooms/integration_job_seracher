"""
Double-prompt test: extract job JSON (7B) → eval candidate vs job JSON (14B).
Runs directly against Ollama, no queue needed.
Compares scores to original single-prompt results.
"""
import json
import os
import re
import time
import requests
import psycopg2

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
SMALL_MODEL  = os.getenv("SMALL_MODEL", "qwen2.5:7b")
LARGE_MODEL  = os.getenv("LARGE_MODEL", "qwen3:14b")
DB_PASSWORD  = os.getenv("DB_PASSWORD", "")
DB_DSN       = f"postgresql://user_job_searcher:{DB_PASSWORD}@localhost:5432/job_searcher"
QUEUE_DSN    = "postgresql://llm_queue_worker:lq-a437d02922eae4ba942d7299@172.23.0.4:5432/llm_queue"

EXTRACT_PROMPT = open("/mnt/ContainerHub/DockerConfig/llm-queue/prompts/job_extract.txt").read()

EVAL_PROMPT = """You are a skeptical recruiter who has seen thousands of applications. Score conservatively.

Scoring anchors:
- 10 = near-perfect fit, rare
- 7  = genuinely worth applying
- 5  = possible but a stretch
- 3  = wrong domain or level
- 1  = completely irrelevant

The job summary below was extracted from the posting. Use compensation, level, and domain to calibrate.
If the job domain differs from the candidate's primary domain, cap skills_match at 5.
If the job is contract/temp, reduce career_level_alignment by 2.
If the listed compensation is below the candidate's target_compensation, reduce culture_fit by 2.

Candidate:
<candidate>
{{candidate_json}}
</candidate>

Job (structured summary):
<job>
{{job_json}}
</job>

Return JSON only (no markdown):
{
  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",
  "match_scores": {
    "skills_match": <1-10>,
    "career_level_alignment": <1-10>,
    "experience_relevance": <1-10>,
    "culture_fit": <1-10>
  },
  "one_line_summary": "one sentence max 20 words"
}
"""

CANONICAL = {"skills_match", "career_level_alignment", "experience_relevance", "culture_fit"}


def ollama(model, prompt):
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return {}


def avg_score(scores):
    c = {k: v for k, v in scores.items() if k in CANONICAL}
    return round(sum(c.values()) / len(c), 2) if c else 0


def main():
    db   = psycopg2.connect(DB_DSN)
    qdb  = psycopg2.connect(QUEUE_DSN)

    # Load Slava resume JSON
    with db.cursor() as cur:
        cur.execute("SELECT resume_json FROM adm.resume WHERE profile = 'Slava' AND is_active = TRUE LIMIT 1")
        candidate_json = json.dumps(cur.fetchone()[0])

    # Load highscore test jobs from queue
    with qdb.cursor() as cur:
        cur.execute("SELECT payload->>'job_id', payload->>'title', payload->>'company' FROM llm_queue.tasks WHERE pack_id='llm-highscore-test' ORDER BY id")
        queue_rows = cur.fetchall()

    job_ids = [r[0] for r in queue_rows]
    with db.cursor() as cur:
        cur.execute("SELECT id, title, company, description FROM public.jobspy_jobs WHERE id = ANY(%s)", (job_ids,))
        jobs_map = {r[0]: r for r in cur.fetchall()}

    # Load original scores
    with db.cursor() as cur:
        cur.execute("""
            SELECT j.id, e.avg_score
            FROM public.evaluated_jobs e
            JOIN public.jobspy_jobs j ON e.job_id = j.id
            WHERE e.sys_profile = 'Slava' AND j.id = ANY(%s)
            ORDER BY e.created_at DESC
        """, (job_ids,))
        orig_scores = {}
        for job_id, score in cur.fetchall():
            orig_scores.setdefault(job_id, score)

    print(f"{'Job':<45} {'Orig':>5} {'New':>5}  {'Verdict':<20} Extracted domain/comp")
    print("-" * 120)

    for job_id, title, company in queue_rows:
        job = jobs_map.get(job_id)
        if not job:
            continue
        description = (job[3] or "")[:6000]

        # Stage 1: extract with small model
        t0 = time.time()
        extract_raw = ollama(SMALL_MODEL, EXTRACT_PROMPT.replace("{{description}}", description))
        extracted = extract_json(extract_raw)
        job_json = extracted.get("job_json", extracted)
        t1 = time.time()

        # Stage 2: eval with large model
        eval_p = EVAL_PROMPT.replace("{{candidate_json}}", candidate_json).replace("{{job_json}}", json.dumps(job_json))
        eval_raw = ollama(LARGE_MODEL, eval_p)
        result = extract_json(eval_raw)
        t2 = time.time()

        score = avg_score(result.get("match_scores", {}))
        orig  = orig_scores.get(job_id, "?")
        domain = job_json.get("domain", "?")
        comp   = job_json.get("compensation", "null")
        level  = job_json.get("level", "?")

        print(f"{title[:45]:<45} {str(orig):>5} {score:>5.1f}  {result.get('verdict','?'):<20} {domain} / {level} / {comp}")
        print(f"  summary: {result.get('one_line_summary','')[:100]}")
        print(f"  extract={t1-t0:.0f}s  eval={t2-t1:.0f}s")
        print()


if __name__ == "__main__":
    main()
