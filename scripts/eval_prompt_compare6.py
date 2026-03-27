"""
Phase 2f: Raw description vs extract — does the extract lose critical signal?

Root cause hypothesis: v11_bare labels Deloitte as domain="AI/ML Engineering",
which anchors the eval model toward Pivot before it even reads the requirements.
Passing the raw description lets the model see the actual skill overlap directly.

"Without About section": strip company intro boilerplate (first paragraph if it's
marketing copy) and the legal tail (salary, locations, hashtags). Keep the meat:
responsibilities + requirements.

Variants:
  advisor_orig    — baseline from Phase 2c (uses v11_bare extract output)
  raw_full        — raw description, first 3000 chars (captures requirements)
  raw_no_about    — raw description, skip first paragraph, first 3000 chars from there

Same 10 diagnostic jobs. advisor_orig injected from Phase 2c.
"""
import json, os, sys, time, re
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

# Section headers that signal start of real job content
CONTENT_HEADERS = re.compile(
    r"(responsibilities|requirements|what you.ll do|you will|key responsibilities"
    r"|role overview|about the role|the role|qualifications|about this (role|job|position)"
    r"|what we.re looking for|work you.ll do)",
    re.IGNORECASE
)

# Tail boilerplate to cut at
TAIL_MARKERS = re.compile(
    r"(salary|compensation|wage|pay range|benefits|equal opportunity|eeo|"
    r"accommodation|disability|sponsorship|#[A-Z]{2,}|about deloitte|about amazon"
    r"|about rbc|about td |about the company|about us\b)",
    re.IGNORECASE
)


def clean_description(raw: str, skip_about: bool = False) -> str:
    """Strip About section and legal tail from job description."""
    if not raw:
        return ""

    lines = raw.split("\n")
    clean_lines = []
    started = not skip_about

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if clean_lines:
                clean_lines.append("")
            continue

        # If skipping about: look for a SHORT header line (≤80 chars) that matches
        # content markers. Long lines are paragraphs, not section headers.
        if not started:
            is_header_line = len(stripped) <= 80 and CONTENT_HEADERS.search(stripped)
            if is_header_line:
                started = True
            else:
                continue

        # Stop at tail boilerplate
        if TAIL_MARKERS.search(stripped):
            break

        clean_lines.append(stripped)

    text = "\n".join(clean_lines).strip()
    return text[:3000]


# Eval prompt template — same advisor framing as Phase 2c winner
EVAL_PROMPT = """\
You are an honest career advisor.

Candidate:
{candidate_json}

Job posting:
{job_text}

Return JSON only:
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
    "advisor_orig":   None,   # injected from Phase 2c
    "raw_full":       "full",
    "raw_no_about":   "no_about",
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
        cur.execute(
            "SELECT id, title, company, description FROM jobspy_jobs WHERE id = ANY(%s)",
            (JOB_IDS,)
        )
        jobs = {r[0]: {"title": r[1], "company": r[2], "description": r[3] or ""} for r in cur.fetchall()}
        cur.execute(
            "SELECT resume_json FROM adm.resume WHERE profile = %s AND is_active = TRUE",
            (PROFILE,)
        )
        row = cur.fetchone()
        resume = json.dumps(row[0]) if row else "{}"
    db.close()
    return jobs, resume


def telegram_preview(r) -> str:
    s = r["scores"]
    icon = {"Step Up": "⬆️", "Lateral": "↔️", "Title Regression": "⬇️", "Pivot": "↩️"}.get(r["verdict"], "?")
    sk = s.get("skills_match", "-")
    lv = s.get("career_level_alignment", "-")
    ex = s.get("experience_relevance", "-")
    cu = s.get("culture_fit", "-")
    return (
        f"📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
        f"🏢 {r['title']} — {r['company']}\n\n"
        f"📋 {r['job_in_one_line']}\n"
        f"✅ {r['why_you_fit']}\n"
        f"⚠️  {r['key_gap']}\n\n"
        f"Tech:{sk} Lvl:{lv} Exp:{ex} Cult:{cu}"
    )


def main():
    scripts_dir = os.path.dirname(__file__)

    print("Loading jobs and resume...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs loaded\n")

    with open(os.path.join(scripts_dir, "claude_control.json")) as f:
        claude = {r["job_id"]: r for r in json.load(f)}

    out_path = os.path.join(scripts_dir, "eval_compare6_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded checkpoint ({sum(len(v) for v in results.values())} existing)\n")
    else:
        results = {v: [] for v in EVAL_VARIANTS}

    for v in EVAL_VARIANTS:
        results.setdefault(v, [])

    # Inject advisor_orig from Phase 2c
    if not results["advisor_orig"]:
        p2c_path = os.path.join(scripts_dir, "eval_compare3_results.json")
        if os.path.exists(p2c_path):
            with open(p2c_path) as f:
                p2c = json.load(f)
            results["advisor_orig"] = p2c.get("advisor_orig", [])
            print(f"  Injected advisor_orig from 2c ({len(results['advisor_orig'])} jobs)\n")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    avail_jobs = [j for j in JOB_IDS if j in jobs]
    total = len([v for v in EVAL_VARIANTS if EVAL_VARIANTS[v] is not None]) * len(avail_jobs)
    done  = sum(len(results[v]) for v in EVAL_VARIANTS if EVAL_VARIANTS[v] is not None)
    print(f"=== PHASE 2f: 2 raw variants × {len(avail_jobs)} jobs = {total} new evals ===")
    print(f"    Already done: {done}/{total}\n")

    # Show what the cleaned descriptions look like for Deloitte
    deloitte = jobs.get("in-1783da45863c66c0")
    if deloitte:
        print("--- Sample: Deloitte raw_full (first 300 chars) ---")
        print(clean_description(deloitte["description"], skip_about=False)[:300])
        print("\n--- Sample: Deloitte raw_no_about (first 300 chars) ---")
        print(clean_description(deloitte["description"], skip_about=True)[:300])
        print()

    for job_id in avail_jobs:
        job = jobs[job_id]
        c   = claude.get(job_id)

        for v_name, mode in EVAL_VARIANTS.items():
            if mode is None:
                continue
            done_ids = {r["job_id"] for r in results[v_name]}
            if job_id in done_ids:
                continue

            job_text = clean_description(job["description"], skip_about=(mode == "no_about"))
            prompt   = EVAL_PROMPT.format(candidate_json=resume, job_text=job_text)

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
    print("KEY DIAGNOSTICS — does raw description fix Step Up detection?")
    print(f"{'='*90}")
    checks = {
        "Step Up (must pass)": {
            "in-1783da45863c66c0": ("Deloitte", 8.0, "Step Up", 6.5),
            "in-8abe8673798de6fd": ("RBC",     7.3, "Step Up", 6.5),
        },
        "Inflation (avg must stay low)": {
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

    # ---- Telegram previews for Step Up jobs ----
    print(f"\n{'='*90}")
    print("TELEGRAM PREVIEWS — Step Up jobs × all variants")
    print(f"{'='*90}")
    for jid in ["in-1783da45863c66c0", "in-8abe8673798de6fd"]:
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
