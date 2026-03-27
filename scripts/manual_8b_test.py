"""
Directly test qwen3:8b extraction on the 8 locked sample jobs.
No queue involved — calls Ollama API directly and times each call.
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

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

PROMPT_TEMPLATE = open("/mnt/ContainerHub/DockerConfig/llm-queue/prompts/job_extract.txt").read()

def call_ollama(prompt: str) -> tuple[str, float]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    elapsed = time.time() - t0
    return resp["message"]["content"], elapsed


def main():
    db = psycopg2.connect(DB_DSN)
    job_ids = [j for j, _ in SAMPLE]
    with db.cursor() as cur:
        cur.execute("SELECT id, title, company, description FROM jobspy_jobs WHERE id = ANY(%s)", (job_ids,))
        jobs = {r[0]: r for r in cur.fetchall()}
    db.close()

    results = []
    total_time = 0.0
    for job_id, profile in SAMPLE:
        job = jobs.get(job_id)
        if not job:
            print(f"  SKIP {job_id} — not found"); continue
        _, title, company, description = job
        prompt = PROMPT_TEMPLATE.replace("{{description}}", description or "")

        print(f"\n{'='*60}")
        print(f"  {title} @ {company}  [{profile}]")
        print(f"  job_id: {job_id}")

        content, elapsed = call_ollama(prompt)
        total_time += elapsed
        print(f"  time: {elapsed:.1f}s")

        # Parse JSON
        try:
            # strip thinking tags if present
            text = content
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            # strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            print(f"  domain:    {parsed.get('domain')}")
            print(f"  title:     {parsed.get('title')}")
            print(f"  level:     {parsed.get('level')}")
            print(f"  what:      {parsed.get('what_you_do','')[:80]}")
            print(f"  background:{parsed.get('required_background','')[:80]}")
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            print(f"  raw: {content[:200]}")
            parsed = None

        results.append({"job_id": job_id, "profile": profile, "title": title,
                         "company": company, "elapsed": elapsed, "parsed": parsed})

    print(f"\n{'='*60}")
    print(f"  8b extract: {len(results)} jobs, total={total_time:.1f}s, avg={total_time/len(results):.1f}s")
    print(f"  (14b baseline with num_ctx:16384 was ~60-87s/job)")


if __name__ == "__main__":
    main()
