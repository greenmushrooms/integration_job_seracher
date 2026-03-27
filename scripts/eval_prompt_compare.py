"""
Phase 2: Eval prompt comparison.
Fixed extract: v11_bare (8b, ~7s).
Variable: 3 eval prompt variants.
All 20 jobs. Batched: all extracts first (8b), then all evals (14b).
Output: score grid vs claude_control + flagged Telegram previews.
"""
import json, os, sys, time, textwrap
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_8B   = "qwen3:8b"
MODEL_14B  = "qwen3:14b"
PROFILE    = "Slava"

JOB_IDS = [
    "li-4377018253",        # Data Engineer @ TekStaff             → Lateral
    "in-290633ee2965c8a2",  # Lead Product Analyst @ Scribd         → Pivot
    "li-4386307997",        # Advisor BI & DE @ City of Brampton   → Title Regression
    "in-3fb9e93fdd02d41d",  # DevOps @ TCS                         → Pivot
    "in-4bcf3fc0483efcc4",  # BI Specialist @ TD                   → Title Regression
    "li-4385018687",        # Technical BA O2C @ Astir             → Pivot
    "in-1312cba6f3d46c1c",  # System Analyst @ TCS                 → Pivot
    "in-5b29876b95d790b8",  # Business Analyst @ Key Food          → Pivot
    "in-50c8748c768e2f50",  # Cloud Infrastructure @ AWS           → Pivot (false pos risk)
    "li-4370606419",        # Sr Analyst Strategy & Ops @ Scotia   → Pivot
    "in-1783da45863c66c0",  # AI Data Engineer @ Deloitte          → Step Up
    "in-73f45ab65515bc5a",  # Computer Systems Analyst @ Craftner  → Pivot
    "in-a0c5c60f9ff09fde",  # Pharmacy Assistant                   → Pivot (extreme)
    "li-4387991366",        # BA Fraud Ops AI @ Optomi             → Pivot
    "in-592e5841f9a16d5f",  # Sr Partner Consultant @ AWS          → Pivot
    "li-4387748514",        # BA IT III @ Robertson                → Pivot
    "li-4377515148",        # BA Store Technology @ Primark        → Pivot
    "in-4f4b2d67d4f0190f",  # Sr Ops Research Analyst @ Noblis     → Pivot
    "in-5786658763ac8049",  # Data Centre Chief Engineer @ Amazon  → Pivot (false pos risk)
    "in-8abe8673798de6fd",  # Director GenAI & ML @ RBC            → Step Up
]

# ---------- Fixed extract prompt (v11_bare winner from Phase 1) ----------
EXTRACT_PROMPT = """\
Extract job details. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact title",
  "domain": "specific domain",
  "level": "junior|mid|senior|lead|director",
  "summary": "2 sentences max"
}}"""

# ---------- Eval prompt variants ----------
EVAL_SCHEMA = """\
{{
  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",
  "match_scores": {{
    "skills_match": <1-10>,
    "career_level_alignment": <1-10>,
    "experience_relevance": <1-10>,
    "culture_fit": <1-10>
  }},
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"
}}"""

EVAL_VARIANTS = {
    "eval_v1_schema": f"""\
You are a skeptical recruiter. Score conservatively. A 7 means worth applying. A 10 is rare.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "eval_v2_domain_gate": f"""\
You are a skeptical recruiter. Score conservatively. A 7 means worth applying. A 10 is rare.

Domain check: if this job requires a fundamentally different professional background from the candidate \
(e.g. facilities engineering, pharmacy, nuclear operations, retail store operations, healthcare, \
legal, HR, physical infrastructure), set experience_relevance ≤ 3 and verdict = Pivot — \
regardless of any keyword or tool overlap.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "eval_v3_advisor": f"""\
You are a career advisor helping a senior data engineer find the right next role. \
Be honest about gaps but surface genuine opportunities. A 7 means worth pursuing.

Domain check: if this job requires a fundamentally different professional background \
(facilities, pharmacy, nuclear ops, retail BA, physical infrastructure), \
set experience_relevance ≤ 3 and verdict = Pivot.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",
}


def ollama(model: str, prompt: str, num_ctx: int = 8192) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp["message"]["content"], round(time.time() - t0, 1)


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def parse_json(text: str):
    text = strip_think(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {"_raw": text[:300]}


def avg_score(scores: dict) -> float:
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def load_data():
    db = psycopg2.connect(DB_DSN)
    with db.cursor() as cur:
        cur.execute("SELECT id, title, company, description FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        jobs = {r[0]: {"title": r[1], "company": r[2], "description": r[3]} for r in cur.fetchall()}
        cur.execute("SELECT resume_json FROM adm.resume WHERE profile = %s AND is_active = TRUE", (PROFILE,))
        row = cur.fetchone()
        resume = json.dumps(row[0]) if row else "{}"
    db.close()
    return jobs, resume


def telegram_preview(title, company, avg, verdict, scores, job_in_one_line, why_you_fit, key_gap, job_id) -> str:
    sk = scores.get("skills_match", "-")
    lv = scores.get("career_level_alignment", "-")
    ex = scores.get("experience_relevance", "-")
    cu = scores.get("culture_fit", "-")
    verdict_icon = {"Step Up": "⬆️", "Lateral": "↔️", "Title Regression": "⬇️", "Pivot": "↩️"}.get(verdict, "?")
    return (
        f"📊 {avg:.1f}/10  {verdict_icon} {verdict}\n"
        f"🏢 {title} — {company}\n"
        f"\n"
        f"📋 {job_in_one_line}\n"
        f"✅ {why_you_fit}\n"
        f"⚠️  {key_gap}\n"
        f"\n"
        f"Tech:{sk} Lvl:{lv} Exp:{ex} Cult:{cu}"
    )


def main():
    print("Loading jobs and resume...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs, resume loaded ({len(resume)} chars)\n")

    # Load claude_control ground truth
    claude_path = os.path.join(os.path.dirname(__file__), "claude_control.json")
    with open(claude_path) as f:
        claude_rows = json.load(f)
    claude = {r["job_id"]: r for r in claude_rows}

    # Load extract cache if exists (avoid re-running 8b)
    cache_path = os.path.join(os.path.dirname(__file__), "eval_compare_extract_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            extract_cache = json.load(f)
        print(f"  Extract cache loaded ({len(extract_cache)} entries)\n")
    else:
        extract_cache = {}

    # ---- Phase 1: v11_bare extracts (8b stays loaded) ----
    print("=== PHASE 1: v11_bare extracts (8b) ===\n")
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job:
            print(f"  {job_id} — not in DB, skipping")
            continue
        if job_id in extract_cache:
            print(f"  {job_id} — cached ({extract_cache[job_id]['domain']})")
            continue
        description = job["description"] or ""
        prompt = EXTRACT_PROMPT.replace("{description}", description)
        ext_text, ext_s = ollama(MODEL_8B, prompt, num_ctx=8192)
        ext_parsed = parse_json(strip_think(ext_text))
        domain = ext_parsed.get("domain") or ext_parsed.get("summary", "?")
        extract_cache[job_id] = {
            "text": ext_text,
            "parsed": ext_parsed,
            "domain": str(domain)[:60],
            "ext_s": ext_s,
        }
        print(f"  {job_id}  {ext_s}s  domain={domain!r:.50}", flush=True)
        # Save cache incrementally
        with open(cache_path, "w") as f:
            json.dump(extract_cache, f, indent=2)

    print(f"\n=== PHASE 2: evals (14b stays loaded) ===\n")

    results = {v: [] for v in EVAL_VARIANTS}

    # Load existing results to checkpoint
    out_path = os.path.join(os.path.dirname(__file__), "eval_compare_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded existing eval results\n")

    total = len(EVAL_VARIANTS) * len([j for j in JOB_IDS if j in jobs])
    done = sum(len(results[v]) for v in EVAL_VARIANTS)

    for v_name, prompt_template in EVAL_VARIANTS.items():
        done_ids = {r["job_id"] for r in results.get(v_name, [])}
        results.setdefault(v_name, [])

        for job_id in JOB_IDS:
            job = jobs.get(job_id)
            if not job or job_id not in extract_cache:
                continue
            if job_id in done_ids:
                print(f"  [{v_name}] {job_id} — cached", flush=True)
                continue

            ext_text = extract_cache[job_id]["text"]
            prompt = prompt_template.format(candidate_json=resume, job_json=ext_text)
            raw, eval_s = ollama(MODEL_14B, prompt, num_ctx=8192)
            eval_res = parse_json(strip_think(raw))

            verdict = eval_res.get("verdict", "?")
            scores = eval_res.get("match_scores", {})
            avg = avg_score(scores)
            job_in_one_line = eval_res.get("job_in_one_line", "")
            why_you_fit = eval_res.get("why_you_fit", "")
            key_gap = eval_res.get("key_gap", "")

            sk = scores.get("skills_match", "?")
            lv = scores.get("career_level_alignment", "?")
            ex = scores.get("experience_relevance", "?")
            cu = scores.get("culture_fit", "?")
            print(f"  [{v_name}] {job['title'][:40]}")
            print(f"    {verdict}  avg={avg}  skill={sk} lvl={lv} exp={ex} cult={cu}")
            print(f"    {job_in_one_line[:80]}", flush=True)

            results[v_name].append({
                "job_id": job_id,
                "title": job["title"],
                "company": job["company"],
                "eval_s": eval_s,
                "domain": extract_cache[job_id]["domain"],
                "verdict": verdict,
                "scores": scores,
                "avg": avg,
                "job_in_one_line": job_in_one_line,
                "why_you_fit": why_you_fit,
                "key_gap": key_gap,
            })
            done += 1
            print(f"  Progress: {done}/{total}", flush=True)

            # Save incrementally
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # ---- Summary: score grid vs claude_control ----
    print(f"\n\n{'='*80}")
    print("SCORE GRID vs claude_control  (delta in parens)")
    print(f"{'='*80}")
    v_names = list(EVAL_VARIANTS.keys())
    hdr = f"{'Job':<35} {'claude':>7}" + "".join(f"  {v[-4:]:>12}" for v in v_names)
    print(hdr)
    print("-" * len(hdr))

    maes = {v: 0.0 for v in v_names}
    verdict_matches = {v: 0 for v in v_names}
    n_jobs = 0

    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job:
            continue
        c = claude.get(job_id)
        if not c:
            continue
        label = job["title"][:32]
        row = f"{label:<35} {c['avg']:>7.1f}"
        n_jobs += 1
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if r:
                delta = r["avg"] - c["avg"]
                maes[v] += abs(delta)
                if r["verdict"] == c["verdict"]:
                    verdict_matches[v] += 1
                row += f"  {r['avg']:>4.1f}({delta:>+4.1f})"
            else:
                row += "       N/A"
        print(row)

    print("-" * len(hdr))
    mae_row = f"{'MAE':<35} {'':>7}"
    for v in v_names:
        mae_row += f"  {maes[v]/n_jobs if n_jobs else 0:>9.2f}  "
    print(mae_row)
    print(f"\nVerdict accuracy vs claude_control:")
    for v in v_names:
        print(f"  {v:<30} {verdict_matches[v]}/{n_jobs}")

    # ---- Flag critical jobs (biggest delta or verdict flip) ----
    print(f"\n\n{'='*80}")
    print("FLAGGED JOBS — biggest disagreements + false positive risks")
    print(f"{'='*80}")

    flagged = []
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c = claude.get(job_id)
        if not job or not c:
            continue
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if not r:
                continue
            delta = abs(r["avg"] - c["avg"])
            verdict_flip = r["verdict"] != c["verdict"]
            # Flag if: large delta, or verdict flip on a non-Pivot job, or false positive (Pivot should be low but isn't)
            is_fp = c["verdict"] == "Pivot" and r["avg"] > 6.0
            if delta >= 1.5 or (verdict_flip and c["verdict"] != "Pivot") or is_fp:
                flagged.append((delta, job_id, v, job, c, r))

    # Sort by delta descending, deduplicate per job
    flagged.sort(key=lambda x: -x[0])
    seen_jobs = set()
    top_flagged = []
    for item in flagged:
        jid = item[1]
        if jid not in seen_jobs:
            seen_jobs.add(jid)
            top_flagged.append(item)
        if len(top_flagged) >= 5:
            break

    for delta, job_id, best_v, job, c, r in top_flagged:
        print(f"\n{'─'*70}")
        print(f"  ⚠️  {job['title']} @ {job['company']}")
        print(f"  claude_control: {c['verdict']} {c['avg']} | {best_v}: {r['verdict']} {r['avg']} (Δ{delta:+.1f})")
        print(f"  claude summary: {c['summary']}")
        print(f"\n  --- TELEGRAM PREVIEW ({best_v}) ---")
        preview = telegram_preview(
            job["title"], job["company"], r["avg"], r["verdict"],
            r["scores"], r["job_in_one_line"], r["why_you_fit"], r["key_gap"], job_id
        )
        for line in preview.split("\n"):
            print(f"  {line}")

    # ---- Side-by-side Telegram previews for top flagged jobs ----
    print(f"\n\n{'='*80}")
    print("SIDE-BY-SIDE TELEGRAM PREVIEWS  (top 5 flagged jobs × all 3 variants)")
    print(f"{'='*80}")

    for delta, job_id, _, job, c, _ in top_flagged:
        print(f"\n{'━'*70}")
        print(f"  {job['title']} @ {job['company']}")
        print(f"  claude_control: {c['verdict']} {c['avg']} | \"{c['summary']}\"")
        print(f"{'━'*70}")
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if not r:
                continue
            print(f"\n  [{v}]")
            preview = telegram_preview(
                job["title"], job["company"], r["avg"], r["verdict"],
                r["scores"], r["job_in_one_line"], r["why_you_fit"], r["key_gap"], job_id
            )
            for line in preview.split("\n"):
                print(f"    {line}")

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
