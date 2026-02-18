#!/usr/bin/env python3
"""
Run 5 real jobs from the DB through multiple models and compare results.
Models: qwen3:14b, deepseek-r1:14b, claude-haiku
"""

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import anthropic
import psycopg2
import requests

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"
PROFILE = "Slava"
N_JOBS = 5

MODELS = ["qwen3:14b", "deepseek-r1:14b", "claude-haiku-4-5-20251001"]
NO_TOOL_MODELS = {"deepseek-r1:14b"}
ANTHROPIC_MODELS = {"claude-haiku-4-5-20251001"}

# ── Env ───────────────────────────────────────────────────────────────────────
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

load_env()

# ── DB ────────────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

def load_jobs(profile: str, n: int) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, company, description
            FROM public.jobspy_jobs
            WHERE sys_profile = %s AND description IS NOT NULL AND length(description) > 100
            ORDER BY id DESC
            LIMIT %s
        """, (profile, n))
        cols = ["id", "title", "company", "description"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def load_resume(profile: str) -> str:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT resume_body FROM adm.resume
            WHERE profile = %s AND is_active = TRUE
            ORDER BY updated_at DESC LIMIT 1
        """, (profile,))
        row = cur.fetchone()
        return row[0] if row else None

# ── Ollama unload ─────────────────────────────────────────────────────────────
def unload_ollama():
    try:
        ps = requests.get("http://localhost:11434/api/ps", timeout=5).json()
        for m in ps.get("models", []):
            name = m["name"]
            requests.post(OLLAMA_NATIVE_URL, json={
                "model": name, "keep_alive": 0,
                "messages": [{"role": "user", "content": "x"}]
            }, timeout=15)
        time.sleep(3)
    except Exception:
        pass

# ── Tool schema ───────────────────────────────────────────────────────────────
TOOL_SCHEMA_OLLAMA = {
    "type": "function",
    "function": {
        "name": "submit_job_evaluation",
        "description": "Submit the evaluation of a job posting against a candidate resume.",
        "parameters": {
            "type": "object",
            "properties": {
                "tech_stack_analysis": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string"},
                        "matches": {"type": "array", "items": {"type": "string"}},
                        "gaps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["verdict", "matches", "gaps"],
                },
                "verdict": {"type": "string", "enum": ["Step Up", "Lateral", "Title Regression", "Pivot"]},
                "match_scores": {
                    "type": "object",
                    "properties": {
                        "skills_match": {"type": "integer", "minimum": 1, "maximum": 10},
                        "career_level_alignment": {"type": "integer", "minimum": 1, "maximum": 10},
                        "experience_relevance": {"type": "integer", "minimum": 1, "maximum": 10},
                        "culture_fit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["skills_match", "career_level_alignment", "experience_relevance", "culture_fit"],
                },
                "one_line_summary": {"type": "string"},
            },
            "required": ["tech_stack_analysis", "verdict", "match_scores", "one_line_summary"],
        },
    },
}

TOOL_SCHEMA_ANTHROPIC = {
    "name": "submit_job_evaluation",
    "description": "Submit the evaluation of a job posting against a candidate resume.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tech_stack_analysis": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "matches": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["verdict", "matches", "gaps"],
            },
            "verdict": {"type": "string", "enum": ["Step Up", "Lateral", "Title Regression", "Pivot"]},
            "match_scores": {
                "type": "object",
                "properties": {
                    "skills_match": {"type": "integer", "minimum": 1, "maximum": 10},
                    "career_level_alignment": {"type": "integer", "minimum": 1, "maximum": 10},
                    "experience_relevance": {"type": "integer", "minimum": 1, "maximum": 10},
                    "culture_fit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["skills_match", "career_level_alignment", "experience_relevance", "culture_fit"],
            },
            "one_line_summary": {"type": "string"},
        },
        "required": ["tech_stack_analysis", "verdict", "match_scores", "one_line_summary"],
    },
}

PROMPT_FALLBACK = """You are an expert Talent Evaluator.
Your goal is to identify growth opportunities.
1. Analyze the candidate's RESUME to establish a baseline.
2. Compare strictly against the JOB DESCRIPTION.

Output ONLY a valid JSON object with this exact structure:
{
  "verdict": "<Step Up | Lateral | Title Regression | Pivot>",
  "one_line_summary": "<single sentence>",
  "match_scores": {"skills_match": <1-10>, "career_level_alignment": <1-10>, "experience_relevance": <1-10>, "culture_fit": <1-10>},
  "tech_stack_analysis": {"verdict": "<brief>", "matches": ["..."], "gaps": ["..."]}
}"""

# ── Evaluate one job ──────────────────────────────────────────────────────────
def eval_job(model: str, resume: str, job: dict) -> tuple[dict | None, float]:
    system_prompt = f"You are an expert Talent Evaluator.\n1. Analyze the RESUME.\n2. Compare against the JOB DESCRIPTION.\n\n<candidate_resume>\n{resume}\n</candidate_resume>"
    user_msg = f"Please evaluate this position:\n\nCOMPANY: {job['company']}\nTITLE: {job['title']}\n\nDESCRIPTION:\n{job['description'][:3000]}"

    start = time.time()

    if model in ANTHROPIC_MODELS:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model, max_tokens=1500, temperature=0,
            system=[
                {"type": "text", "text": "You are an expert Talent Evaluator.\n1. Analyze the RESUME.\n2. Compare against the JOB DESCRIPTION."},
                {"type": "text", "text": f"<candidate_resume>\n{resume}\n</candidate_resume>", "cache_control": {"type": "ephemeral"}},
            ],
            tools=[TOOL_SCHEMA_ANTHROPIC],
            tool_choice={"type": "tool", "name": "submit_job_evaluation"},
            messages=[{"role": "user", "content": user_msg}],
        )
        elapsed = time.time() - start
        tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
        return (tool_block.input if tool_block else None), elapsed

    elif model in NO_TOOL_MODELS:
        payload = {
            "model": model, "temperature": 0,
            "messages": [
                {"role": "system", "content": PROMPT_FALLBACK + f"\n\n<candidate_resume>\n{resume}\n</candidate_resume>"},
                {"role": "user", "content": user_msg},
            ],
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        elapsed = time.time() - start
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        return json.loads(content), elapsed

    else:
        payload = {
            "model": model, "temperature": 0,
            "tools": [TOOL_SCHEMA_OLLAMA],
            "tool_choice": {"type": "function", "function": {"name": "submit_job_evaluation"}},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        elapsed = time.time() - start
        tool_calls = resp.json()["choices"][0]["message"].get("tool_calls")
        if not tool_calls:
            return None, elapsed
        return json.loads(tool_calls[0]["function"]["arguments"]), elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading jobs for profile: {PROFILE}")
    jobs = load_jobs(PROFILE, N_JOBS)
    print(f"Loaded {len(jobs)} jobs")

    resume = load_resume(PROFILE)
    if not resume:
        print("ERROR: No resume found")
        return
    print(f"Resume loaded ({len(resume)} chars)\n")

    # Results: {job_id: {model: result}}
    all_results = {job["id"]: {"job": job} for job in jobs}

    for model in MODELS:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model}")
        print(f"{'#'*60}")

        if model not in ANTHROPIC_MODELS:
            print("  Clearing Ollama memory...")
            unload_ollama()

        for job in jobs:
            print(f"  → {job['company']:30s} | {job['title'][:40]}")
            try:
                result, elapsed = eval_job(model, resume, job)
                scores = result.get("match_scores", {}) if result else {}
                avg = sum(scores.values()) / len(scores) if scores else 0
                verdict = result.get("verdict", "ERR") if result else "ERR"
                print(f"    {verdict:<18} avg={avg:.1f}  ({elapsed:.0f}s)")
                all_results[job["id"]][model] = {"result": result, "elapsed": elapsed, "avg": avg}
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[job["id"]][model] = {"result": None, "elapsed": 0, "avg": 0}

    # ── Summary table ──────────────────────────────────────────────────────────
    col = 18
    short = [m.split("/")[-1].split(":")[0][:col] for m in MODELS]

    print(f"\n\n{'='*80}")
    print("  FINAL COMPARISON")
    print(f"{'='*80}")
    header = f"  {'Company':<22} {'Title':<30}" + "".join(f"{n:>{col}}" for n in short)
    print(header)
    print("  " + "-" * (52 + col * len(MODELS)))

    for job_id, data in all_results.items():
        job = data["job"]
        row = f"  {job['company'][:20]:<22} {job['title'][:28]:<30}"
        for model in MODELS:
            md = data.get(model, {})
            r = md.get("result")
            if r:
                avg = md["avg"]
                verdict = r.get("verdict", "?")[:6]
                row += f"  {verdict:<8} {avg:>4.1f}  "
            else:
                row += f"{'ERR':>{col}}"
        print(row)

    # ── Per-job detail ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  PER-JOB DETAIL")
    print(f"{'='*80}")

    for job_id, data in all_results.items():
        job = data["job"]
        print(f"\n  {job['company']} — {job['title']}")
        print(f"  {'─'*60}")
        for model in MODELS:
            md = data.get(model, {})
            r = md.get("result")
            if not r:
                print(f"  [{model}] ERROR")
                continue
            scores = r.get("match_scores", {})
            tech = r.get("tech_stack_analysis", {})
            print(f"  [{model.split(':')[0]}]")
            print(f"    Verdict : {r.get('verdict')}  avg={md['avg']:.1f}  ({md['elapsed']:.0f}s)")
            print(f"    Summary : {r.get('one_line_summary', '')[:100]}")
            print(f"    Gaps    : {', '.join(tech.get('gaps', []))[:120]}")

    # Save full results
    out = {jid: {k: v for k, v in d.items() if k != "job"} for jid, d in all_results.items()}
    # make results serializable
    for jid in out:
        for model in out[jid]:
            if isinstance(out[jid][model].get("result"), dict):
                pass  # already serializable
    with open("batch_results.json", "w") as f:
        json.dump({jid: {**{"job": all_results[jid]["job"]}, **{m: {"avg": all_results[jid].get(m, {}).get("avg", 0), "result": all_results[jid].get(m, {}).get("result")} for m in MODELS}} for jid in all_results}, f, indent=2)
    print(f"\n  Saved → batch_results.json")


if __name__ == "__main__":
    main()
