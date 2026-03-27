"""
Phase 3a: New minimal extract — title + summary only, no domain, no level, no length cap.

Hypothesis: v11_bare's domain label ("AI/ML Engineering") was poisoning the eval.
Removing it forces the eval model to judge domain fit from actual content.

New extract prompt:
  {title, summary} — no synthetic fields, no length constraint

Eval variants run on new extract:
  v1_skeptic    — original skeptic framing (Phase 2 baseline)
  v3_advisor    — original advisor framing (Phase 2 baseline)
  advisor_3field — advisor, no culture_fit (Phase 2e variant)

Steps:
  1. Re-extract all 10 diagnostic jobs with new prompt (new cache)
  2. Run all 3 eval variants on new extracts
  3. Compare vs claude_control + vs Phase 2 originals on old extract
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_8B   = "qwen3:8b"
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

# New minimal extract — no domain, no level, no length cap
EXTRACT_PROMPT = """\
Extract job details. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact job title",
  "summary": "key responsibilities and requirements"
}}"""

# Eval variants to run on new extract
EVAL_VARIANTS = {
    "v1_skeptic": """\
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
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"
}}""",

    "v3_advisor": """\
You are an honest career advisor.

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
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"
}}""",

    "advisor_3field": """\
You are an honest career advisor.

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
    "experience_relevance": <1-10>
  }},
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"
}}""",
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
        jobs = {r[0]: {"title": r[1], "company": r[2], "description": r[3] or "", "url": ""}
                for r in cur.fetchall()}
        # fetch URLs separately
        cur.execute("SELECT id, job_url FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        for jid, url in cur.fetchall():
            if jid in jobs:
                jobs[jid]["url"] = url or ""
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

    # ── Phase 1: new extracts ──────────────────────────────────────────────
    cache_path = os.path.join(scripts_dir, "new_extract_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            extract_cache = json.load(f)
        print(f"  New extract cache loaded ({len(extract_cache)} entries)\n")
    else:
        extract_cache = {}

    print("=== PHASE 1: new minimal extracts (8b) ===\n")
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job:
            print(f"  {job_id} — not in DB")
            continue
        if job_id in extract_cache:
            print(f"  {job_id} — cached  summary={extract_cache[job_id]['summary'][:60]!r}")
            continue

        prompt = EXTRACT_PROMPT.format(description=job["description"])
        raw, ext_s = ollama(MODEL_8B, prompt, num_ctx=8192)
        parsed = parse_json(strip_think(raw))
        summary = parsed.get("summary", "")
        title   = parsed.get("title", job["title"])

        extract_cache[job_id] = {
            "text":    raw,
            "parsed":  parsed,
            "title":   title,
            "summary": summary,
            "ext_s":   ext_s,
        }
        print(f"  {job_id}  {ext_s}s")
        print(f"    title:   {title}")
        print(f"    summary: {summary[:120]}", flush=True)

        with open(cache_path, "w") as f:
            json.dump(extract_cache, f, indent=2)

    # Show extract comparison: old domain label vs new summary
    print(f"\n=== EXTRACT COMPARISON (old domain label vs new summary) ===\n")
    old_cache_path = os.path.join(scripts_dir, "eval_compare_extract_cache.json")
    if os.path.exists(old_cache_path):
        with open(old_cache_path) as f:
            old_cache = json.load(f)
        for job_id in JOB_IDS:
            job = jobs.get(job_id)
            if not job:
                continue
            old = old_cache.get(job_id, {})
            new = extract_cache.get(job_id, {})
            print(f"  {job['title'][:40]} @ {job['company'][:20]}")
            print(f"    OLD domain: {old.get('domain', 'N/A')}")
            print(f"    NEW summary: {new.get('summary', 'N/A')[:100]}")
            print()

    # ── Phase 2: eval with new extracts (3 variants) ──────────────────────
    print("=== PHASE 2: eval (14b) with new extracts — 3 variants ===\n")

    out_path = os.path.join(scripts_dir, "new_extract_eval_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            eval_results = json.load(f)
        existing = sum(len(v) for v in eval_results.values())
        print(f"  Loaded checkpoint ({existing} existing evals)\n")
    else:
        eval_results = {v: [] for v in EVAL_VARIANTS}

    for v in EVAL_VARIANTS:
        eval_results.setdefault(v, [])

    avail = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail)
    done  = sum(len(eval_results[v]) for v in EVAL_VARIANTS)
    print(f"  {done}/{total} already done\n")

    for job_id in avail:
        job      = jobs[job_id]
        c        = claude.get(job_id)
        ext_text = extract_cache[job_id]["text"]

        for v_name, prompt_template in EVAL_VARIANTS.items():
            done_ids = {r["job_id"] for r in eval_results[v_name]}
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

            eval_results[v_name].append({
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
                json.dump(eval_results, f, indent=2)

        print(f"  [{job_id}] done — {done}/{total} total", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    # Load Phase 2 original baselines for comparison
    p2_path = os.path.join(scripts_dir, "eval_compare_results.json")
    p2_skeptic = p2_advisor = []
    if os.path.exists(p2_path):
        with open(p2_path) as f:
            p2 = json.load(f)
        p2_skeptic = p2.get("eval_v1_schema", [])
        p2_advisor = p2.get("eval_v3_advisor", [])

    cols = [
        ("p2_skeptic",        p2_skeptic),
        ("p2_advisor",        p2_advisor),
        ("3a_v1_skeptic",     eval_results.get("v1_skeptic", [])),
        ("3a_v3_advisor",     eval_results.get("v3_advisor", [])),
        ("3a_advisor_3field", eval_results.get("advisor_3field", [])),
    ]

    print(f"\n\n{'='*100}")
    print("SCORE GRID  — v1_skeptic | v3_advisor | 3a new extract")
    print(f"{'='*100}")
    hdr = f"{'Job':<38} {'ctrl':>5}"
    for label, _ in cols:
        hdr += f"  {label:>15}"
    print(hdr)
    print("-" * len(hdr))

    maes = {l: 0.0 for l, _ in cols}
    vms  = {l: 0   for l, _ in cols}
    n = 0

    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c   = claude.get(job_id)
        if not job or not c:
            continue
        label_str = f"{job['title'][:20]}@{job['company'][:15]}"
        row = f"{label_str:<38} {c['avg']:>5.1f}"
        n += 1
        for col_label, col_data in cols:
            r = next((x for x in col_data if x["job_id"] == job_id), None)
            if r:
                d = r["avg"] - c["avg"]
                maes[col_label] += abs(d)
                if r["verdict"] == c["verdict"]: vms[col_label] += 1
                mk = "✓" if r["verdict"] == c["verdict"] else "✗"
                row += f"  {r['avg']:>4.1f}({d:>+4.1f}){mk}  "
            else:
                row += f"  {'N/A':>15}"
        print(row)

    print("-" * len(hdr))
    mae_row = f"{'MAE':<44}"
    for col_label, _ in cols:
        mae_row += f"  {maes[col_label]/n:>11.2f}    "
    print(mae_row)

    print(f"\nVerdict accuracy ({n} jobs):")
    for col_label, _ in cols:
        bar = "█" * vms[col_label] + "░" * (n - vms[col_label])
        print(f"  {col_label:<20} {vms[col_label]:>2}/{n}  {bar}")

    # ── Verdict table ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("VERDICT TABLE")
    print(f"{'='*100}")
    vhdr = f"{'Job':<38} {'ctrl':>14}"
    for col_label, _ in cols:
        vhdr += f"  {col_label:>15}"
    print(vhdr)
    print("-" * len(vhdr))
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c   = claude.get(job_id)
        if not job or not c:
            continue
        label_str = f"{job['title'][:20]}@{job['company'][:15]}"
        vrow = f"{label_str:<38} {c['verdict']:>14}"
        for col_label, col_data in cols:
            r = next((x for x in col_data if x["job_id"] == job_id), None)
            if r:
                mk = "✓" if r["verdict"] == c["verdict"] else "✗"
                vrow += f"  {r['verdict'][:13]:>13}{mk} "
            else:
                vrow += f"  {'N/A':>15}"
        print(vrow)

    # ── Key diagnostics ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("KEY DIAGNOSTICS — did removing domain label fix Step Up detection?")
    print(f"{'='*100}")
    checks = {
        "Step Up (must pass)": {
            "in-1783da45863c66c0": ("Deloitte", 8.0, "Step Up"),
            "in-8abe8673798de6fd": ("RBC",      7.3, "Step Up"),
        },
        "Inflation suppressed (avg must stay low)": {
            "in-73f45ab65515bc5a": ("Craftner", 2.0, "Pivot"),
            "in-5b29876b95d790b8": ("Key Food", 2.8, "Pivot"),
            "in-a0c5c60f9ff09fde": ("Pharmacy", 1.0, "Pivot"),
        },
        "Lateral preserved": {
            "li-4377018253": ("TekStaff", 7.8, "Lateral"),
        },
    }
    for section, jobs_check in checks.items():
        print(f"\n  {section}:")
        for jid, (lbl, ctrl_avg, ctrl_v) in jobs_check.items():
            row = f"    {lbl:12} ctrl={ctrl_v} {ctrl_avg}:"
            for col_label, col_data in cols:
                r = next((x for x in col_data if x["job_id"] == jid), None)
                if r:
                    ok = (r["verdict"] == ctrl_v) if ctrl_v in ("Step Up","Lateral") else (r["avg"] <= ctrl_avg + 1.5)
                    mk = "✓" if ok else "✗"
                    row += f"  {col_label}={r['avg']:.1f}({r['verdict'][:4]}){mk}"
            print(row)

    # ── Telegram sends ────────────────────────────────────────────────────
    import os as _os, requests as _req
    BOT   = _os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
    ICON  = {"Step Up":"⬆️","Lateral":"↔️","Title Regression":"⬇️","Pivot":"↩️"}

    def tg_send(text):
        r = _req.post(
            f"https://api.telegram.org/bot{BOT}/sendMessage",
            data={"chat_id": CHAT, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True}
        )
        if not r.json().get("ok"):
            print("TG FAIL:", r.json().get("description"))

    def tg_preview(r, col_label):
        s    = r["scores"]
        icon = ICON.get(r["verdict"], "?")
        ctrl_r = claude.get(r["job_id"], {})
        mk   = "✓" if r["verdict"] == ctrl_r.get("verdict") else "✗"
        sk   = s.get("skills_match","-"); lv = s.get("career_level_alignment","-")
        ex   = s.get("experience_relevance","-"); cu = s.get("culture_fit","-")
        return (
            f"<b>[{col_label}] {mk}</b>\n"
            f"📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
            f"📋 {r['job_in_one_line']}\n"
            f"✅ {r['why_you_fit']}\n"
            f"⚠️ {r['key_gap']}\n"
            f"Tech:{sk} Lvl:{lv} Exp:{ex} Cult:{cu}"
        )

    SHOW = [
        ("in-1783da45863c66c0", "Deloitte AI DE"),
        ("in-8abe8673798de6fd", "RBC Director GenAI"),
        ("li-4377018253",       "TekStaff DE"),
        ("in-73f45ab65515bc5a", "Craftner COBOL"),
        ("in-5b29876b95d790b8", "Key Food BA"),
    ]

    tg_send(
        "🧪 <b>Phase 3a — New extract (no domain label)</b>\n"
        "p2_skeptic | p2_advisor | 3a_v1 | 3a_v3 | 3a_3field\n"
        "——————————————"
    )

    for jid, name in SHOW:
        c   = claude.get(jid, {})
        job = jobs.get(jid, {})
        url = job.get("url", "")
        link = f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        tg_send(
            f"<b>── {name} ──</b>\n"
            f"ctrl: {c.get('verdict')} {c.get('avg')} | <i>{c.get('summary','')}</i>"
            f"{link}"
        )
        for col_label, col_data in cols:
            r = next((x for x in col_data if x["job_id"] == jid), None)
            if r:
                tg_send(tg_preview(r, col_label))

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
