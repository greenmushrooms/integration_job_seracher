"""
Phase 2b: Evolutionary eval prompt variants.
6 new variants (3×v1_skeptic, 3×v3_advisor) on 10 diagnostic jobs.
Key problems to fix: score inflation on clear Pivots, missing Step-Up stretch roles.

Reuses extract cache from Phase 2 (eval_compare_extract_cache.json).
Saves to eval_compare2_results.json with incremental checkpointing.
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_14B  = "qwen3:14b"
PROFILE    = "Slava"

# 10 diagnostic jobs — all 4 verdicts + known inflation cases + false-positive risks
JOB_IDS = [
    "in-1783da45863c66c0",  # AI Data Engineer @ Deloitte          → Step Up  8.0  (borderline, all v1/v3 missed it)
    "in-8abe8673798de6fd",  # Director GenAI & ML @ RBC            → Step Up  7.3  (stretch, v1 missed it)
    "li-4377018253",        # Data Engineer @ TekStaff             → Lateral  7.8  (easy true positive)
    "li-4386307997",        # Advisor BI & DE @ City of Brampton   → TR       6.5  (borderline, v1/v3 mostly correct)
    "in-4bcf3fc0483efcc4",  # BI Specialist @ TD                   → TR       5.8  (v1/v3 both missed)
    "li-4370606419",        # Sr Analyst Strategy & Ops @ Scotia   → Pivot    3.5  (clear pivot, v1/v3 correct)
    "in-a0c5c60f9ff09fde",  # Pharmacy Assistant                   → Pivot    1.0  (extreme, all correct)
    "in-73f45ab65515bc5a",  # Computer Systems Analyst @ Craftner  → Pivot    2.0  (inflation: v1=5.5, v3=5.0)
    "in-5b29876b95d790b8",  # Business Analyst @ Key Food          → Pivot    2.8  (inflation: v1=5.3, v3=4.8)
    "in-50c8748c768e2f50",  # Cloud Infrastructure @ AWS           → Pivot    5.8  (false positive risk)
]

# ---------- Output schema (same as Phase 2) ----------
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

# ---------- 6 evolutionary eval variants ----------
EVAL_VARIANTS = {
    # ── v1 branch: skeptic framing ─────────────────────────────────────────
    "v1a_bare": f"""\
Evaluate this job-candidate match. Score conservatively. \
1-3=wrong domain or function, 4-6=adjacent or weak match, 7-8=worth applying, 9-10=rare.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "v1b_anchors": f"""\
You are a skeptical recruiter. Score each dimension with this rubric:
  1-3: wrong professional domain/function, or fundamental skill mismatch
  4-6: adjacent field or partial overlap — significant gaps remain
  7-8: strong match — worth applying
  9-10: near-perfect fit — rare

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "v1c_honest": f"""\
You are a skeptical recruiter. A 7 means worth applying. A 10 is rare.

why_you_fit: genuine technical/functional overlap only. \
If the job is a different professional domain, write "no genuine overlap — different domain".

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    # ── v3 branch: advisor framing ─────────────────────────────────────────
    "v3a_bare": f"""\
Career advisor. Evaluate fit for a senior data engineer. Be honest. \
A 7 means worth pursuing. A 10 is rare.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "v3b_anchors": f"""\
You are a career advisor helping a senior data engineer find the right next role. \
A 7 means worth pursuing.

Score each dimension with this rubric:
  1-3: wrong domain or function — data engineering skills don't transfer
  4-6: adjacent or partial — notable gaps, weak match
  7-8: solid match — encourage applying
  9-10: exceptional fit — rare

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{EVAL_SCHEMA}""",

    "v3c_honest": f"""\
You are a career advisor helping a senior data engineer. \
Be honest about gaps but surface genuine opportunities. A 7 means worth pursuing.

why_you_fit: genuine functional overlap only. \
If this is a different professional domain, write "no genuine overlap — different domain".

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
        cur.execute("SELECT id, title, company FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        jobs = {r[0]: {"title": r[1], "company": r[2]} for r in cur.fetchall()}
        cur.execute("SELECT resume_json FROM adm.resume WHERE profile = %s AND is_active = TRUE", (PROFILE,))
        row = cur.fetchone()
        resume = json.dumps(row[0]) if row else "{}"
    db.close()
    return jobs, resume


def telegram_preview(title, company, avg, verdict, scores, job_in_one_line, why_you_fit, key_gap) -> str:
    sk = scores.get("skills_match", "-")
    lv = scores.get("career_level_alignment", "-")
    ex = scores.get("experience_relevance", "-")
    cu = scores.get("culture_fit", "-")
    icon = {"Step Up": "⬆️", "Lateral": "↔️", "Title Regression": "⬇️", "Pivot": "↩️"}.get(verdict, "?")
    return (
        f"📊 {avg:.1f}/10  {icon} {verdict}\n"
        f"🏢 {title} — {company}\n"
        f"\n"
        f"📋 {job_in_one_line}\n"
        f"✅ {why_you_fit}\n"
        f"⚠️  {key_gap}\n"
        f"\n"
        f"Tech:{sk} Lvl:{lv} Exp:{ex} Cult:{cu}"
    )


def main():
    scripts_dir = os.path.dirname(__file__)

    # Load jobs metadata
    print("Loading job metadata and resume...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs loaded\n")

    # Load claude_control ground truth
    with open(os.path.join(scripts_dir, "claude_control.json")) as f:
        claude = {r["job_id"]: r for r in json.load(f)}

    # Load Phase 2 extract cache (all 20 already cached — do not re-run)
    cache_path = os.path.join(scripts_dir, "eval_compare_extract_cache.json")
    if not os.path.exists(cache_path):
        print("ERROR: eval_compare_extract_cache.json not found. Run eval_prompt_compare.py first.")
        sys.exit(1)
    with open(cache_path) as f:
        extract_cache = json.load(f)
    print(f"  Extract cache: {len(extract_cache)} entries\n")

    # Check all diagnostic jobs are cached
    missing = [j for j in JOB_IDS if j not in extract_cache]
    if missing:
        print(f"WARNING: {len(missing)} jobs not in extract cache: {missing}")

    # Load existing results (checkpoint)
    out_path = os.path.join(scripts_dir, "eval_compare2_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded checkpoint ({sum(len(v) for v in results.values())} existing evals)\n")
    else:
        results = {v: [] for v in EVAL_VARIANTS}

    # Ensure all variant keys exist
    for v in EVAL_VARIANTS:
        results.setdefault(v, [])

    # Count progress
    avail_jobs = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail_jobs)
    done = sum(len(results[v]) for v in EVAL_VARIANTS)
    print(f"=== PHASE 2b: 6 eval variants × {len(avail_jobs)} jobs = {total} evals ===")
    print(f"    Already done: {done}/{total}\n")

    # ---- Run evals (14b stays loaded, inner loop = variants per job) ----
    # Outer loop: job → inner loop: variant (minimizes model swaps)
    for job_id in avail_jobs:
        job = jobs[job_id]
        c = claude.get(job_id)
        ext_text = extract_cache[job_id]["text"]

        for v_name, prompt_template in EVAL_VARIANTS.items():
            done_ids = {r["job_id"] for r in results[v_name]}
            if job_id in done_ids:
                continue

            prompt = prompt_template.format(candidate_json=resume, job_json=ext_text)
            raw, eval_s = ollama(MODEL_14B, prompt, num_ctx=8192)
            ev = parse_json(strip_think(raw))

            verdict = ev.get("verdict", "?")
            scores  = ev.get("match_scores", {})
            avg     = avg_score(scores)
            jol     = ev.get("job_in_one_line", "")
            wyf     = ev.get("why_you_fit", "")
            kg      = ev.get("key_gap", "")

            ctrl_verdict = c["verdict"] if c else "?"
            ctrl_avg     = c["avg"] if c else 0.0
            delta        = avg - ctrl_avg
            match_mark   = "✓" if verdict == ctrl_verdict else "✗"

            print(f"  [{v_name}] {job['title'][:35]} @ {job['company'][:20]}")
            print(f"    {match_mark} {verdict:16s}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{delta:+.1f})  {eval_s}s", flush=True)

            results[v_name].append({
                "job_id": job_id,
                "title":  job["title"],
                "company": job["company"],
                "eval_s": eval_s,
                "domain": extract_cache[job_id].get("domain", ""),
                "verdict": verdict,
                "scores": scores,
                "avg": avg,
                "job_in_one_line": jol,
                "why_you_fit": wyf,
                "key_gap": kg,
            })
            done += 1

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"  [{job_id}] done — {done}/{total} total", flush=True)

    # ---- Summary: score grid vs claude_control ----
    v_names = list(EVAL_VARIANTS.keys())
    print(f"\n\n{'='*100}")
    print("SCORE GRID vs claude_control  (delta shown in parens)")
    print(f"{'='*100}")

    # Header
    col_w = 13
    hdr = f"{'Job':<38} {'ctrl':>5}"
    for v in v_names:
        hdr += f"  {v[:col_w]:>{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    maes = {v: 0.0 for v in v_names}
    verdict_matches = {v: 0 for v in v_names}
    n_jobs = 0

    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c = claude.get(job_id)
        if not job or not c:
            continue
        label = f"{job['title'][:20]}@{job['company'][:15]}"
        row = f"{label:<38} {c['avg']:>5.1f}"
        n_jobs += 1
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if r:
                delta = r["avg"] - c["avg"]
                maes[v] += abs(delta)
                if r["verdict"] == c["verdict"]:
                    verdict_matches[v] += 1
                mark = "✓" if r["verdict"] == c["verdict"] else "✗"
                row += f"  {r['avg']:>4.1f}({delta:>+4.1f}){mark}"
            else:
                row += f"{'N/A':>{col_w+2}}"
        print(row)

    print("-" * len(hdr))
    mae_row = f"{'MAE':<38} {'':>5}"
    for v in v_names:
        val = maes[v] / n_jobs if n_jobs else 0
        mae_row += f"  {val:>9.2f}    "
    print(mae_row)

    print(f"\nVerdict accuracy vs claude_control  ({n_jobs} jobs):")
    for v in v_names:
        acc = verdict_matches[v]
        bar = "█" * acc + "░" * (n_jobs - acc)
        print(f"  {v:<20} {acc:>2}/{n_jobs}  {bar}")

    # ---- Verdict comparison table ----
    print(f"\n\n{'='*100}")
    print("VERDICT TABLE  (✓=match  ✗=wrong)")
    print(f"{'='*100}")
    vhdr = f"{'Job':<38} {'ctrl':>14}"
    for v in v_names:
        vhdr += f"  {v[:10]:>10}"
    print(vhdr)
    print("-" * len(vhdr))

    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c = claude.get(job_id)
        if not job or not c:
            continue
        label = f"{job['title'][:20]}@{job['company'][:15]}"
        vrow = f"{label:<38} {c['verdict']:>14}"
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if r:
                mark = "✓" if r["verdict"] == c["verdict"] else "✗"
                vrow += f"  {r['verdict'][:9]:>9}{mark}"
            else:
                vrow += f"{'N/A':>11}"
        print(vrow)

    # ---- Flag critical jobs ----
    print(f"\n\n{'='*100}")
    print("FLAGGED JOBS — biggest deltas + false positives + missed Step Ups")
    print(f"{'='*100}")

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
            is_fp  = c["verdict"] == "Pivot" and r["avg"] > 5.5
            is_fn  = c["verdict"] == "Step Up" and r["verdict"] == "Pivot"
            if delta >= 1.5 or is_fp or is_fn:
                flagged.append((delta, job_id, v, job, c, r))

    flagged.sort(key=lambda x: -x[0])
    seen = set()
    top5 = []
    for item in flagged:
        if item[1] not in seen:
            seen.add(item[1])
            top5.append(item)
        if len(top5) >= 5:
            break

    for delta, job_id, best_v, job, c, r in top5:
        print(f"\n{'─'*80}")
        print(f"  ⚠️  {job['title']} @ {job['company']}")
        print(f"  claude_control: {c['verdict']} avg={c['avg']} | worst variant {best_v}: {r['verdict']} avg={r['avg']} (Δ{delta:+.1f})")
        print(f"  claude summary: {c['summary']}")

    # ---- Side-by-side Telegram previews ----
    print(f"\n\n{'='*100}")
    print("TELEGRAM PREVIEWS  (top 5 flagged × all 6 variants)")
    print(f"{'='*100}")

    for _, job_id, _, job, c, _ in top5:
        print(f"\n{'━'*80}")
        ctrl_v = c["verdict"]
        print(f"  {job['title']} @ {job['company']}")
        print(f"  claude_control: {ctrl_v} {c['avg']} | \"{c['summary']}\"")
        print(f"{'━'*80}")

        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if not r:
                continue
            match = "✓" if r["verdict"] == ctrl_v else "✗"
            print(f"\n  [{v}] {match}")
            preview = telegram_preview(
                job["title"], job["company"], r["avg"], r["verdict"],
                r["scores"], r["job_in_one_line"], r["why_you_fit"], r["key_gap"]
            )
            for line in preview.split("\n"):
                print(f"    {line}")

    print(f"\n\nResults saved to {out_path}")
    print(f"\n\nMY RECOMMENDATION:")
    # Pick winner: lowest MAE, best verdict accuracy, not tied
    ranked = sorted(v_names, key=lambda v: (maes[v] / n_jobs if n_jobs else 0, -verdict_matches[v]))
    print(f"  Best MAE:             {ranked[0]}  ({maes[ranked[0]]/n_jobs:.2f})")
    best_verdicts = max(v_names, key=lambda v: verdict_matches[v])
    print(f"  Most verdict matches: {best_verdicts}  ({verdict_matches[best_verdicts]}/{n_jobs})")
    # Check Pharmacy + Craftner (inflation test)
    inflation_jobs = {
        "in-a0c5c60f9ff09fde": "Pharmacy (ctrl 1.0)",
        "in-73f45ab65515bc5a": "Craftner (ctrl 2.0)",
        "in-5b29876b95d790b8": "Key Food (ctrl 2.8)",
    }
    print(f"\n  Inflation check (lower=better):")
    for v in v_names:
        scores_str = ""
        for jid, label in inflation_jobs.items():
            r = next((x for x in results[v] if x["job_id"] == jid), None)
            if r:
                scores_str += f"  {label}={r['avg']:.1f}"
        print(f"    {v:<20} {scores_str}")

    print(f"\n  Step Up detection (both should be ≥6.5):")
    stepup_jobs = {"in-1783da45863c66c0": "Deloitte(8.0)", "in-8abe8673798de6fd": "RBC(7.3)"}
    for v in v_names:
        scores_str = ""
        for jid, label in stepup_jobs.items():
            r = next((x for x in results[v] if x["job_id"] == jid), None)
            if r:
                match = "✓" if r["verdict"] == "Step Up" else "✗"
                scores_str += f"  {label}={r['avg']:.1f}{match}"
        print(f"    {v:<20} {scores_str}")


if __name__ == "__main__":
    main()
