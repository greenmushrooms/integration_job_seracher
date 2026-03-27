"""
Phase 2e: Remove culture_fit — does 3-field avg calibrate better?

Hypothesis: culture_fit is unassessable from a job posting and adds noise.
Removing it gives a cleaner 3-field average from objective dimensions only.

Base: advisor_orig (Phase 2c winner — lowest MAE, experience_relevance schema).
advisor_orig results injected from Phase 2c checkpoint — not re-run.

Variants:
  advisor_orig   — baseline: skills + level + exp_relevance + culture (from 2c)
  advisor_3field — remove culture_fit, keep experience_relevance

Same 10 diagnostic jobs. Reuses extract cache.
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_14B  = "qwen3:14b"
PROFILE    = "Slava"

JOB_IDS = [
    "in-1783da45863c66c0",  # AI Data Engineer @ Deloitte          → Step Up  8.0
    "in-8abe8673798de6fd",  # Director GenAI & ML @ RBC            → Step Up  7.3
    "li-4377018253",        # Data Engineer @ TekStaff             → Lateral  7.8
    "li-4386307997",        # Advisor BI & DE @ City of Brampton   → TR       6.5
    "in-4bcf3fc0483efcc4",  # BI Specialist @ TD                   → TR       5.8
    "li-4370606419",        # Sr Analyst Strategy & Ops @ Scotia   → Pivot    3.5
    "in-a0c5c60f9ff09fde",  # Pharmacy Assistant                   → Pivot    1.0
    "in-73f45ab65515bc5a",  # Computer Systems Analyst @ Craftner  → Pivot    2.0
    "in-5b29876b95d790b8",  # Business Analyst @ Key Food          → Pivot    2.8
    "in-50c8748c768e2f50",  # Cloud Infrastructure @ AWS           → Pivot    5.8
]

EVAL_VARIANTS = {
    # advisor_orig baseline (Phase 2c winner) — results injected from 2c, not re-run
    "advisor_orig": None,

    # 3-field: culture_fit removed, experience_relevance kept
    "advisor_3field": """\
You are an honest career advisor.

Candidate:
{candidate_json}

Job:
{job_json}

Return JSON only:
{
  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",
  "match_scores": {
    "skills_match": <1-10>,
    "career_level_alignment": <1-10>,
    "experience_relevance": <1-10>
  },
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"
}""",
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


def telegram_preview(r) -> str:
    s = r["scores"]
    icon = {"Step Up": "⬆️", "Lateral": "↔️", "Title Regression": "⬇️", "Pivot": "↩️"}.get(r["verdict"], "?")
    sk = s.get("skills_match", "-")
    lv = s.get("career_level_alignment", "-")
    ex = s.get("experience_relevance") or s.get("growth_potential", "-")
    cu = s.get("culture_fit")
    score_line = f"Tech:{sk} Lvl:{lv} Exp:{ex}"
    if cu is not None:
        score_line += f" Cult:{cu}"
    return (
        f"📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
        f"🏢 {r['title']} — {r['company']}\n\n"
        f"📋 {r['job_in_one_line']}\n"
        f"✅ {r['why_you_fit']}\n"
        f"⚠️  {r['key_gap']}\n\n"
        f"{score_line}"
    )


def main():
    scripts_dir = os.path.dirname(__file__)

    print("Loading job metadata and resume...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs loaded\n")

    with open(os.path.join(scripts_dir, "claude_control.json")) as f:
        claude = {r["job_id"]: r for r in json.load(f)}

    cache_path = os.path.join(scripts_dir, "eval_compare_extract_cache.json")
    if not os.path.exists(cache_path):
        print("ERROR: extract cache not found. Run eval_prompt_compare.py first.")
        sys.exit(1)
    with open(cache_path) as f:
        extract_cache = json.load(f)
    print(f"  Extract cache: {len(extract_cache)} entries\n")

    out_path = os.path.join(scripts_dir, "eval_compare5_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        existing = sum(len(v) for v in results.values())
        print(f"  Loaded checkpoint ({existing} existing evals)\n")
    else:
        results = {v: [] for v in EVAL_VARIANTS}

    for v in EVAL_VARIANTS:
        results.setdefault(v, [])

    # Inject advisor_orig from Phase 2c (no re-run needed)
    if not results["advisor_orig"]:
        p2c_path = os.path.join(scripts_dir, "eval_compare3_results.json")
        if os.path.exists(p2c_path):
            with open(p2c_path) as f:
                p2c = json.load(f)
            results["advisor_orig"] = p2c.get("advisor_orig", [])
            print(f"  Injected advisor_orig from Phase 2c ({len(results['advisor_orig'])} jobs)\n")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    avail_jobs = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail_jobs)
    done  = sum(len(results[v]) for v in EVAL_VARIANTS)
    print(f"=== PHASE 2e: 2 variants × {len(avail_jobs)} jobs = {total} evals ===")
    print(f"    Already done: {done}/{total}\n")

    for job_id in avail_jobs:
        job = jobs[job_id]
        c   = claude.get(job_id)
        ext_text = extract_cache[job_id]["text"]

        for v_name, prompt_template in EVAL_VARIANTS.items():
            if prompt_template is None:
                continue  # injected from prior run, skip
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

            ctrl_v   = c["verdict"] if c else "?"
            ctrl_avg = c["avg"] if c else 0.0
            delta    = avg - ctrl_avg
            mark     = "✓" if verdict == ctrl_v else "✗"

            print(f"  [{v_name}] {job['title'][:35]} @ {job['company'][:20]}")
            print(f"    {mark} {verdict:16s}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{delta:+.1f})  {eval_s}s", flush=True)

            results[v_name].append({
                "job_id":          job_id,
                "title":           job["title"],
                "company":         job["company"],
                "eval_s":          eval_s,
                "domain":          extract_cache[job_id].get("domain", ""),
                "verdict":         verdict,
                "scores":          scores,
                "avg":             avg,
                "job_in_one_line": jol,
                "why_you_fit":     wyf,
                "key_gap":         kg,
            })
            done += 1
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"  [{job_id}] done — {done}/{total} total", flush=True)

    # ---- Summary ----
    v_names = list(EVAL_VARIANTS.keys())
    print(f"\n\n{'='*90}")
    print("SCORE GRID vs claude_control")
    print(f"{'='*90}")

    col_w = 16
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
        c   = claude.get(job_id)
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
                row += f"  {r['avg']:>5.1f}({delta:>+4.1f}){mark}"
            else:
                row += f"{'N/A':>{col_w+2}}"
        print(row)

    print("-" * len(hdr))
    mae_row = f"{'MAE':<38} {'':>5}"
    for v in v_names:
        val = maes[v] / n_jobs if n_jobs else 0
        mae_row += f"  {val:>12.2f}    "
    print(mae_row)

    print(f"\nVerdict accuracy ({n_jobs} jobs):")
    for v in v_names:
        bar = "█" * verdict_matches[v] + "░" * (n_jobs - verdict_matches[v])
        print(f"  {v:<20} {verdict_matches[v]:>2}/{n_jobs}  {bar}")

    # ---- Key diagnostics ----
    print(f"\n{'='*90}")
    print("KEY DIAGNOSTICS")
    print(f"{'='*90}")
    checks = {
        "Step Up (verdict=Step Up, avg≥6.5)": {
            "in-1783da45863c66c0": ("Deloitte", 8.0, "Step Up", 6.5),
            "in-8abe8673798de6fd": ("RBC",     7.3, "Step Up", 6.5),
        },
        "Inflation suppressed (avg≤threshold)": {
            "in-73f45ab65515bc5a": ("Craftner", 2.0, "Pivot", 3.5),
            "in-5b29876b95d790b8": ("Key Food", 2.8, "Pivot", 4.0),
            "in-a0c5c60f9ff09fde": ("Pharmacy", 1.0, "Pivot", 2.5),
        },
        "Lateral preserved": {
            "li-4377018253": ("TekStaff", 7.8, "Lateral", 7.0),
        },
    }
    for section, jobs_check in checks.items():
        print(f"\n  {section}:")
        for jid, (label, ctrl_avg, ctrl_v, threshold) in jobs_check.items():
            row = f"    {label:12} ctrl={ctrl_v} {ctrl_avg:.1f}:"
            for v in v_names:
                r = next((x for x in results[v] if x["job_id"] == jid), None)
                if r:
                    if ctrl_v in ("Step Up", "Lateral"):
                        ok = r["verdict"] == ctrl_v and r["avg"] >= threshold
                    else:
                        ok = r["avg"] <= threshold
                    mark = "✓" if ok else "✗"
                    row += f"  {v}={r['avg']:.1f}({r['verdict'][:5]}){mark}"
            print(row)

    # ---- Side-by-side Telegram previews for key jobs ----
    print(f"\n{'='*90}")
    print("TELEGRAM PREVIEWS — 4-field vs 3-field side by side")
    print(f"{'='*90}")
    preview_jobs = [
        "in-1783da45863c66c0",  # Deloitte Step Up
        "in-73f45ab65515bc5a",  # Craftner inflation
        "li-4377018253",        # TekStaff Lateral
    ]
    for jid in preview_jobs:
        job = jobs.get(jid)
        c   = claude.get(jid)
        if not job or not c:
            continue
        print(f"\n{'━'*60}")
        print(f"  {job['title']} @ {job['company']}  (ctrl: {c['verdict']} {c['avg']})")
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == jid), None)
            if not r:
                continue
            mark = "✓" if r["verdict"] == c["verdict"] else "✗"
            print(f"\n  [{v}] {mark}")
            for line in telegram_preview(r).split("\n"):
                print(f"    {line}")

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
