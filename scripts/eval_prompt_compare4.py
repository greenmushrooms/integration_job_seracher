"""
Phase 2d: Research-backed structural variants.
Base: mentor + growth_potential (Phase 2c hypothesis winner).
3 targeted changes from research, one per variant:

  v_reason_first  — fit_analysis field before verdict (forces reasoning before scoring)
  v_binary_gate   — same_domain + same_function binary flags before verdict
  v_deficit_anchor — gaps-first instruction + minimal 3-level score anchor

Plus mentor_growth baseline for direct comparison.

Research basis:
- Field ordering: reasoning before verdict tokens improves accuracy 48%→61% (Tam 2024)
- Binary sub-criteria: improves inter-evaluator agreement +0.45 pts (CheckEval EMNLP 2025)
- Deficit framing + rubric anchors: ICC3 0.560→0.819 (ACM ICER 2025)
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

# ── Shared schema blocks ──────────────────────────────────────────────────────

# growth_potential replaces experience_relevance (Phase 2c finding)
SCORES_GROWTH = """\
  "match_scores": {{
    "skills_match": <1-10>,
    "career_level_alignment": <1-10>,
    "growth_potential": <1-10>,
    "culture_fit": <1-10>
  }}"""

TAIL = """\
  "job_in_one_line": "what this role does and what domain/industry it is",
  "why_you_fit": "strongest overlap between candidate and this role in one sentence",
  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence\""""

VERDICT = '"verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot"'

# ── 4 variants ────────────────────────────────────────────────────────────────
EVAL_VARIANTS = {

    # Baseline — mentor + growth_potential, no extra structure
    "mentor_growth": f"""\
You are an honest mentor.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{{{{
  {VERDICT},
  {SCORES_GROWTH},
  {TAIL}
}}}}""",

    # Research finding #1 — reasoning field BEFORE verdict forces model to think first.
    # Autoregressive: tokens to the left influence tokens to the right.
    # fit_analysis generated before verdict = verdict informed by reasoning.
    "v_reason_first": f"""\
You are an honest mentor.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{{{{
  "fit_analysis": "assess industry overlap, function match, seniority, and key gaps in 2-3 sentences",
  {VERDICT},
  {SCORES_GROWTH},
  {TAIL}
}}}}""",

    # Research finding #2 — binary gates before verdict.
    # Forces explicit domain/function commit before scoring.
    # same_domain=false should anchor experience_relevance low and push toward Pivot.
    "v_binary_gate": f"""\
You are an honest mentor.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{{{{
  "same_domain": true | false,
  "same_function": true | false,
  {VERDICT},
  {SCORES_GROWTH},
  {TAIL}
}}}}""",

    # Research finding #3 — deficit framing + minimal 3-level anchor.
    # "note gaps first" fights RLHF agreeableness.
    # 3-level anchor (not a full rubric) recalibrates without over-constraining.
    "v_deficit_anchor": f"""\
You are an honest mentor. Note the candidate's gaps before their strengths. \
Score 1-3 if wrong domain or function, 4-6 if adjacent with gaps, 7+ if aligned.

Candidate:
{{candidate_json}}

Job:
{{job_json}}

Return JSON only:
{{{{
  {VERDICT},
  {SCORES_GROWTH},
  {TAIL}
}}}}""",
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
    ex = s.get("growth_potential") or s.get("experience_relevance", "-")
    ex_lbl = "Grow" if "growth_potential" in s else "Exp"
    cu = s.get("culture_fit", "-")
    return (
        f"📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
        f"🏢 {r['title']} — {r['company']}\n\n"
        f"📋 {r['job_in_one_line']}\n"
        f"✅ {r['why_you_fit']}\n"
        f"⚠️  {r['key_gap']}\n\n"
        f"Tech:{sk} Lvl:{lv} {ex_lbl}:{ex} Cult:{cu}"
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

    out_path = os.path.join(scripts_dir, "eval_compare4_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        existing = sum(len(v) for v in results.values())
        print(f"  Loaded checkpoint ({existing} existing evals)\n")
    else:
        results = {v: [] for v in EVAL_VARIANTS}

    for v in EVAL_VARIANTS:
        results.setdefault(v, [])

    avail_jobs = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail_jobs)
    done  = sum(len(results[v]) for v in EVAL_VARIANTS)
    print(f"=== PHASE 2d: 4 variants × {len(avail_jobs)} jobs = {total} evals ===")
    print(f"    Already done: {done}/{total}\n")

    for job_id in avail_jobs:
        job = jobs[job_id]
        c   = claude.get(job_id)
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

            ctrl_v   = c["verdict"] if c else "?"
            ctrl_avg = c["avg"] if c else 0.0
            delta    = avg - ctrl_avg
            mark     = "✓" if verdict == ctrl_v else "✗"

            # Extra: capture binary gate fields if present
            extras = ""
            if "same_domain" in ev:
                extras = f"  domain={ev['same_domain']} func={ev.get('same_function','?')}"
            if "fit_analysis" in ev:
                extras = f"  reasoning: {str(ev['fit_analysis'])[:60]}"

            print(f"  [{v_name}] {job['title'][:35]} @ {job['company'][:20]}")
            print(f"    {mark} {verdict:16s}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{delta:+.1f})  {eval_s}s{extras}", flush=True)

            results[v_name].append({
                "job_id":         job_id,
                "title":          job["title"],
                "company":        job["company"],
                "eval_s":         eval_s,
                "domain":         extract_cache[job_id].get("domain", ""),
                "verdict":        verdict,
                "scores":         scores,
                "avg":            avg,
                "job_in_one_line": jol,
                "why_you_fit":    wyf,
                "key_gap":        kg,
                # preserve extra fields for analysis
                "same_domain":    ev.get("same_domain"),
                "same_function":  ev.get("same_function"),
                "fit_analysis":   ev.get("fit_analysis"),
            })
            done += 1
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"  [{job_id}] done — {done}/{total} total", flush=True)

    # ---- Summary ----
    v_names = list(EVAL_VARIANTS.keys())
    print(f"\n\n{'='*100}")
    print("SCORE GRID vs claude_control")
    print(f"{'='*100}")

    col_w = 15
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
                row += f"  {r['avg']:>4.1f}({delta:>+4.1f}){mark}"
            else:
                row += f"{'N/A':>{col_w+2}}"
        print(row)

    print("-" * len(hdr))
    mae_row = f"{'MAE':<38} {'':>5}"
    for v in v_names:
        val = maes[v] / n_jobs if n_jobs else 0
        mae_row += f"  {val:>11.2f}    "
    print(mae_row)

    print(f"\nVerdict accuracy ({n_jobs} jobs):")
    for v in v_names:
        bar = "█" * verdict_matches[v] + "░" * (n_jobs - verdict_matches[v])
        print(f"  {v:<22} {verdict_matches[v]:>2}/{n_jobs}  {bar}")

    # ---- Verdict table ----
    print(f"\n{'='*100}")
    print("VERDICT TABLE")
    print(f"{'='*100}")
    vhdr = f"{'Job':<38} {'ctrl':>14}"
    for v in v_names:
        vhdr += f"  {v[:13]:>13}"
    print(vhdr)
    print("-" * len(vhdr))
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        c   = claude.get(job_id)
        if not job or not c:
            continue
        label = f"{job['title'][:20]}@{job['company'][:15]}"
        vrow = f"{label:<38} {c['verdict']:>14}"
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == job_id), None)
            if r:
                mark = "✓" if r["verdict"] == c["verdict"] else "✗"
                vrow += f"  {r['verdict'][:12]:>12}{mark}"
            else:
                vrow += f"{'N/A':>14}"
        print(vrow)

    # ---- Binary gate analysis (v_binary_gate only) ----
    gate_results = results.get("v_binary_gate", [])
    if any(r.get("same_domain") is not None for r in gate_results):
        print(f"\n{'='*100}")
        print("BINARY GATE ANALYSIS — v_binary_gate domain/function flags")
        print(f"{'='*100}")
        print(f"  {'Job':<38} {'ctrl_v':>14}  domain  func  verdict")
        print("  " + "-"*70)
        for job_id in JOB_IDS:
            job = jobs.get(job_id)
            c   = claude.get(job_id)
            r   = next((x for x in gate_results if x["job_id"] == job_id), None)
            if not job or not c or not r:
                continue
            mark = "✓" if r["verdict"] == c["verdict"] else "✗"
            label = f"{job['title'][:20]}@{job['company'][:15]}"
            print(f"  {label:<38} {c['verdict']:>14}  {str(r.get('same_domain','?')):>6}  {str(r.get('same_function','?')):>4}  {r['verdict']} {mark}")

    # ---- Key diagnostics ----
    print(f"\n{'='*100}")
    print("KEY PASS/FAIL DIAGNOSTICS")
    print(f"{'='*100}")
    checks = {
        "Step Up must pass (verdict=Step Up, avg≥6.5)": {
            "in-1783da45863c66c0": ("Deloitte", 8.0, "Step Up", 6.5),
            "in-8abe8673798de6fd": ("RBC", 7.3, "Step Up", 6.5),
        },
        "Inflation must be suppressed (avg≤threshold)": {
            "in-73f45ab65515bc5a": ("Craftner", 2.0, "Pivot", 3.5),
            "in-5b29876b95d790b8": ("Key Food", 2.8, "Pivot", 4.0),
            "in-a0c5c60f9ff09fde": ("Pharmacy", 1.0, "Pivot", 2.5),
        },
        "Lateral must be preserved": {
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
                    row += f"  {v}={r['avg']:.1f}({r['verdict'][:4]}){mark}"
            print(row)

    # ---- Telegram previews for Step Up jobs ----
    print(f"\n{'='*100}")
    print("TELEGRAM PREVIEWS — Step Up jobs × all 4 variants")
    print(f"{'='*100}")
    for jid in ["in-1783da45863c66c0", "in-8abe8673798de6fd"]:
        job = jobs.get(jid)
        c   = claude.get(jid)
        if not job or not c:
            continue
        print(f"\n{'━'*70}")
        print(f"  {job['title']} @ {job['company']}  (ctrl: {c['verdict']} {c['avg']})")
        for v in v_names:
            r = next((x for x in results[v] if x["job_id"] == jid), None)
            if not r:
                continue
            mark = "✓" if r["verdict"] == c["verdict"] else "✗"
            print(f"\n  [{v}] {mark}")
            if r.get("fit_analysis"):
                print(f"    reasoning: {r['fit_analysis'][:120]}")
            for line in telegram_preview(r).split("\n"):
                print(f"    {line}")

    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
