"""
Scientific comparison: 11 extract prompt variants (8b) + 14b one-shot
on 20 diverse jobs. Each extract result is fed into the eval model.
Output: per-method timing, domain specificity, eval verdict + scores.
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
    "in-1783da45863c66c0",  # AI Data Engineer @ Deloitte        → Step Up
    "li-4377018253",        # Data Engineer @ TekStaff           → Lateral
    "li-4386307997",        # Advisor BI & DE @ City of Brampton → Title Regression
    "li-4370606419",        # Sr Analyst Strategy & Ops @ Scotia → Pivot
    "in-a0c5c60f9ff09fde",  # Pharmacy Assistant                 → Pivot (extreme)
]

# ---------- Extract prompt variants ----------
VARIANTS = {
    "v1_schema": """\
Extract a compact structured summary from the job posting below.
Return only a JSON object with no markdown, no explanation.

Job posting:
{description}

Return JSON with exactly these keys:
{{
  "job_json": {{
    "title": "exact job title",
    "level": "junior|mid|senior|lead|director|vp|individual contributor",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1"],
    "domain": "primary domain e.g. data engineering, ml, finance, etc",
    "location": "city or remote",
    "compensation": "range if mentioned, else null",
    "key_requirements": ["concise bullet 1", "concise bullet 2"]
  }}
}}""",

    "v2_plain": """\
Read this job posting and write a 3-5 sentence plain text summary.
Cover: what the person actually does day-to-day, what industry/domain this is in, the seniority level, and pay if mentioned.
Do not infer anything not stated. Return plain text only, no JSON.

Job posting:
{description}""",

    "v3_minimal": """\
Extract the key facts from this job posting. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact title",
  "domain": "what industry/function is this (e.g. nuclear engineering, finance, software, healthcare, logistics)",
  "what_you_do": "one sentence describing the actual day-to-day work",
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}}""",

    "v4_domain_first": """\
First identify what domain/industry this job is in, then summarise it.
Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "domain": "be specific - e.g. nuclear field services, retail operations, data engineering, investment banking",
  "title": "exact job title",
  "level": "junior|mid|senior|lead|director",
  "what_you_do": "1-2 sentences on actual responsibilities",
  "required_background": "what kind of professional would be hired for this",
  "pay": "range if mentioned, else null"
}}""",

    "v5_role_clarify": """\
Read this job posting carefully. Describe what kind of person would be hired for it.
Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact title",
  "industry": "the actual industry (e.g. nuclear, banking, healthcare, tech)",
  "function": "the job function (e.g. operations management, software engineering, data analysis)",
  "day_to_day": "what does this person spend their day doing",
  "requires_technical_skills": true,
  "key_technical_skills": ["only list if explicitly mentioned in posting"],
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}}""",

    "v6_anti_hallucination": """\
Summarise this job posting. Only include facts explicitly stated in the posting — do not infer or add anything.
Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact title from posting",
  "what_this_job_is": "describe the role in plain English, based only on what is written",
  "explicitly_required_skills": ["only skills/tools/languages named in the posting"],
  "level": "junior|mid|senior|lead|director",
  "pay": "exact text from posting or null"
}}""",

    "v7_two_sentence": """\
Summarise this job in exactly 2 sentences: one describing what the role does, one describing what background is required.
Return as JSON: {{"summary": "..."}}

Job posting:
{description}""",

    "v8_checklist": """\
Classify and summarise this job posting. Return JSON only.

Job posting:
{description}

Return:
{{
  "title": "exact title",
  "category": "one of: software engineering | data engineering | data science | business analysis | operations management | finance | sales | marketing | hr | legal | healthcare | engineering (non-software) | other",
  "sub_domain": "more specific e.g. nuclear operations, CRM implementation, cloud infrastructure",
  "level": "junior|mid|senior|lead|director",
  "summary": "2-3 sentences on what the role actually does",
  "pay": "range if mentioned, else null"
}}""",

    "v9_bullets": """\
Extract the key facts from this job posting as plain text bullets. No JSON, no headers.
Cover: actual job function, industry, required background, seniority, pay.

Job posting:
{description}""",

    "v10_questions": """\
Answer these questions about the job posting below. Return JSON only.

Job posting:
{description}

Questions:
{{
  "what_does_this_company_do": "one sentence",
  "what_does_this_person_do": "one sentence on actual responsibilities",
  "what_department_is_this": "e.g. engineering, operations, finance, IT, HR",
  "is_this_a_technical_role": true,
  "technical_tools_mentioned": ["list only tools/languages explicitly named"],
  "title": "exact title",
  "level": "junior|mid|senior|lead|director",
  "pay": "range if mentioned, else null"
}}""",

    "v11_bare": """\
Extract job details. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact title",
  "domain": "specific domain",
  "level": "junior|mid|senior|lead|director",
  "summary": "2 sentences max"
}}""",
}

EVAL_PROMPT_TEMPLATE = """\
You are a skeptical recruiter. Score conservatively. A 7 means worth applying. A 10 is rare.

Candidate:
{candidate_json}

Job:
{job_json}

Return JSON only:
{{
  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",
  "match_scores": {{
    "skills_match": <1-10>,
    "career_level_alignment": <1-10>,
    "experience_relevance": <1-10>,
    "culture_fit": <1-10>
  }},
  "one_line_summary": "one sentence"
}}"""


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


def run_extract(variant_name: str, template: str, description: str, model: str) -> tuple[str, float]:
    prompt = template.replace("{description}", description)
    # 14b uses 16k to handle long descriptions
    num_ctx = 16384 if model == MODEL_14B else 8192
    return ollama(model, prompt, num_ctx)


def run_eval(job_json_text: str, candidate_json: str) -> tuple[dict, float]:
    prompt = EVAL_PROMPT_TEMPLATE.format(
        candidate_json=candidate_json,
        job_json=job_json_text,
    )
    text, elapsed = ollama(MODEL_14B, prompt, num_ctx=8192)
    return parse_json(strip_think(text)), elapsed


def main():
    print("Loading jobs and resume...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs, resume loaded ({len(resume)} chars)\n")

    # Survivors from 3-job pilot: plain summaries only, no 8b classification
    all_variants = {
        "v0_control":      None,
        "v7_two_sentence": VARIANTS["v7_two_sentence"],
        "v11_bare":        VARIANTS["v11_bare"],
        "v12_14b_oneshot": VARIANTS["v4_domain_first"],
    }
    MODELS = {name: MODEL_14B if name == "v12_14b_oneshot" else MODEL_8B for name in all_variants}

    # Load claude manual evaluations (no model call — human judgment as ground truth)
    claude_control_path = os.path.join(os.path.dirname(__file__), "claude_control.json")
    with open(claude_control_path) as f:
        claude_rows = json.load(f)

    # Load existing results to skip already-done (job, variant) pairs
    out_path = os.path.join(os.path.dirname(__file__), "extract_eval_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded existing results from {out_path}")
    else:
        results = {}

    results["claude_control"] = claude_rows
    for v_name in all_variants:
        results.setdefault(v_name, [])

    def already_done(v_name, job_id):
        return any(r["job_id"] == job_id for r in results[v_name])

    # ---- Phase 1: batch all extracts, grouped by model to minimise swaps ----
    # extract_cache[(job_id, v_name)] = (text, ext_s, ext_parsed, domain)
    extract_cache = {}

    # v0_control — no model call
    for job_id, job in jobs.items():
        description = job["description"] or ""
        extract_cache[(job_id, "v0_control")] = (description, 0.0, {"_control": "raw description"}, "raw")

    # 8b extracts (v7_two_sentence, v11_bare) — keep 8b loaded across all jobs
    for v_name in ["v7_two_sentence", "v11_bare"]:
        template = all_variants[v_name]
        for job_id, job in jobs.items():
            if already_done(v_name, job_id):
                # Pull extract text from existing result so eval doesn't re-run
                existing = next(r for r in results[v_name] if r["job_id"] == job_id)
                extract_cache[(job_id, v_name)] = (
                    json.dumps(existing.get("extract_parsed", {})),
                    existing["extract_s"],
                    existing.get("extract_parsed", {}),
                    existing.get("domain", "?"),
                )
                print(f"  [{v_name}] {job_id} — skipping extract (already done)", flush=True)
                continue
            description = job["description"] or ""
            ext_text, ext_s = run_extract(v_name, template, description, MODEL_8B)
            ext_parsed = parse_json(strip_think(ext_text))
            domain = (ext_parsed.get("domain") or ext_parsed.get("summary") or "?")[:60]
            extract_cache[(job_id, v_name)] = (ext_text, ext_s, ext_parsed, domain)
            print(f"  [{v_name}] {job_id}  extract={ext_s}s  domain={domain!r}", flush=True)

    # 14b extract (v12_14b_oneshot) — keep 14b loaded across all jobs
    for job_id, job in jobs.items():
        v_name = "v12_14b_oneshot"
        if already_done(v_name, job_id):
            existing = next(r for r in results[v_name] if r["job_id"] == job_id)
            extract_cache[(job_id, v_name)] = (
                json.dumps(existing.get("extract_parsed", {})),
                existing["extract_s"],
                existing.get("extract_parsed", {}),
                existing.get("domain", "?"),
            )
            print(f"  [{v_name}] {job_id} — skipping extract (already done)", flush=True)
            continue
        description = job["description"] or ""
        template = all_variants[v_name]
        ext_text, ext_s = run_extract(v_name, template, description, MODEL_14B)
        ext_parsed = parse_json(strip_think(ext_text))
        domain = (ext_parsed.get("domain") or "?")[:60]
        extract_cache[(job_id, v_name)] = (ext_text, ext_s, ext_parsed, domain)
        print(f"  [{v_name}] {job_id}  extract={ext_s}s  domain={domain!r}", flush=True)

    print("\n---- All extracts done. Starting evals (14b stays loaded) ----\n", flush=True)

    # ---- Phase 2: batch all evals — 14b stays loaded ----
    total_evals = sum(1 for v_name in all_variants for job_id in jobs if not already_done(v_name, job_id))
    done = 0

    for job_id, job in jobs.items():
        title = job["title"]
        company = job["company"]
        print(f"\n{'='*70}")
        print(f"  {title[:50]} @ {company[:25]}")
        print(f"{'='*70}")

        for v_name in all_variants:
            if already_done(v_name, job_id):
                existing = next(r for r in results[v_name] if r["job_id"] == job_id)
                sk = existing["scores"].get("skills_match", "?")
                lv = existing["scores"].get("career_level_alignment", "?")
                ex = existing["scores"].get("experience_relevance", "?")
                cu = existing["scores"].get("culture_fit", "?")
                print(f"  [{v_name}]  CACHED  {existing['verdict']}  avg={existing['avg']}  skill={sk} lvl={lv} exp={ex} cult={cu}", flush=True)
                continue

            ext_text, ext_s, ext_parsed, domain = extract_cache[(job_id, v_name)]
            eval_res, eval_s = run_eval(ext_text, resume)
            verdict = eval_res.get("verdict", "?")
            scores = eval_res.get("match_scores", {})
            avg = avg_score(scores)
            summary = eval_res.get("one_line_summary", "")
            sk = scores.get("skills_match", "?")
            lv = scores.get("career_level_alignment", "?")
            ex = scores.get("experience_relevance", "?")
            cu = scores.get("culture_fit", "?")
            print(f"  [{v_name}]  extract={ext_s}s  domain={domain[:40]!r}")
            print(f"    {verdict}  avg={avg}  skill={sk} lvl={lv} exp={ex} cult={cu}")
            print(f"    {summary}", flush=True)

            results[v_name].append({
                "job_id": job_id, "title": title, "company": company,
                "extract_s": ext_s, "eval_s": eval_s, "domain": domain,
                "extract_parsed": ext_parsed, "verdict": verdict,
                "scores": scores, "avg": avg, "summary": summary,
            })
            done += 1
            print(f"  Evals done: {done}/{total_evals}", flush=True)

    # ---- Summary table ----
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'AvgExtract':>10} {'AvgEval':>8} {'AvgScore':>9} {'Lateral%':>9} {'Pivot%':>7}")
    print("-"*70)

    for v_name, rows in results.items():
        if not rows: continue
        avg_ext = round(sum(r["extract_s"] for r in rows) / len(rows), 1)
        avg_ev  = round(sum(r["eval_s"] for r in rows) / len(rows), 1)
        avg_sc  = round(sum(r["avg"] for r in rows) / len(rows), 1)
        n_lat   = sum(1 for r in rows if r["verdict"] == "Lateral")
        n_piv   = sum(1 for r in rows if r["verdict"] == "Pivot")
        n       = len(rows)
        print(f"{v_name:<22} {avg_ext:>10.1f}s {avg_ev:>7.1f}s {avg_sc:>9.1f} {n_lat/n*100:>8.0f}% {n_piv/n*100:>6.0f}%")

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), "extract_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # ---- Main table: rows=jobs, cols=variants, cells=avg score ----
    # claude_control is first so it anchors the comparison
    variant_names = ["claude_control"] + list(all_variants.keys())
    short = {v: v[:6] for v in variant_names}

    SCORE_KEYS = ["skills_match", "career_level_alignment", "experience_relevance", "culture_fit"]
    SHORT_SCORE = ["skill", "level", "exp", "cult"]

    for v_name in variant_names:
        rows_v = results[v_name]
        if not rows_v:
            continue
        print(f"\n{'='*80}")
        print(f"  {v_name}  (extract avg {round(sum(r['extract_s'] for r in rows_v)/len(rows_v),1)}s)")
        print(f"{'='*80}")
        hdr = f"{'Job':<38} {'avg':>5}  " + "  ".join(f"{s:>5}" for s in SHORT_SCORE) + "  verdict"
        print(hdr)
        print("-" * len(hdr))
        for job_id in JOB_IDS:
            job = jobs.get(job_id)
            if not job:
                continue
            r = next((x for x in rows_v if x["job_id"] == job_id), None)
            if not r:
                continue
            label = f"{job['title'][:35]}"
            sc = r["scores"]
            vals = [sc.get(k, 0) for k in SCORE_KEYS]
            cells = "  ".join(f"{v:>5}" for v in vals)
            vcode = {"Step Up": "S+", "Lateral": "L", "Title Regression": "TR", "Pivot": "P"}.get(r["verdict"], "?")
            print(f"{label:<38} {r['avg']:>5.1f}  {cells}  {vcode}")

    # ---- Cross-variant avg score grid: rows=jobs, cols=variants ----
    print(f"\n\n{'='*80}")
    print("AVG SCORE GRID  (rows=jobs, cols=variants)")
    print(f"{'='*80}")
    col_w = 6
    header = f"{'Job':<38}" + "".join(f"{v[:5]:>{col_w}}" for v in variant_names)
    print(header)
    print("-" * len(header))
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job:
            continue
        label = job["title"][:35]
        row_str = f"{label:<38}"
        for v_name in variant_names:
            r = next((x for x in results[v_name] if x["job_id"] == job_id), None)
            cell = f"{r['avg']:>5.1f}" if r else "  N/A"
            row_str += f"{cell:>{col_w}}"
        print(row_str)

    # ---- CSV export: rows=jobs, cols=variant avg + verdict ----
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "extract_eval_results.csv")
    SCORE_KEYS = ["skills_match", "career_level_alignment", "experience_relevance", "culture_fit"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Header: job_id, title, company, then per-variant: v_avg, v_verdict, v_skill, v_lvl, v_exp, v_cult, v_domain, v_summary
        col_headers = ["job_id", "title", "company"]
        for v in variant_names:
            col_headers += [f"{v}_avg", f"{v}_verdict", f"{v}_skill", f"{v}_lvl", f"{v}_exp", f"{v}_cult", f"{v}_domain", f"{v}_summary"]
        w.writerow(col_headers)
        for job_id in JOB_IDS:
            job = jobs.get(job_id)
            if not job:
                continue
            row = [job_id, job["title"], job["company"]]
            for v_name in variant_names:
                r = next((x for x in results[v_name] if x["job_id"] == job_id), None)
                if r:
                    sc = r["scores"]
                    row += [r["avg"], r["verdict"],
                            sc.get("skills_match", ""), sc.get("career_level_alignment", ""),
                            sc.get("experience_relevance", ""), sc.get("culture_fit", ""),
                            r.get("domain", ""), r.get("summary", "")]
                else:
                    row += [""] * 8
            w.writerow(row)
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
