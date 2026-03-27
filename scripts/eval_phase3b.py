"""
Phase 3b: Full 20 jobs with new minimal extract.
Validates Phase 3a findings (10 jobs) on the complete ground truth set.
Same 3 eval variants: v1_skeptic, v3_advisor, advisor_3field.
New extract cache extended to all 20 jobs if needed.
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

# All 20 diagnostic jobs
JOB_IDS = [
    "li-4377018253",        # Data Engineer @ TekStaff             → Lateral  7.8
    "in-290633ee2965c8a2",  # Lead Product Analyst @ Scribd         → Pivot    6.0
    "li-4386307997",        # Advisor BI & DE @ City of Brampton   → TR       6.5
    "in-3fb9e93fdd02d41d",  # DevOps @ TCS                         → Pivot    4.5
    "in-4bcf3fc0483efcc4",  # BI Specialist @ TD                   → TR       5.8
    "li-4385018687",        # Technical BA O2C @ Astir             → Pivot    3.0
    "in-1312cba6f3d46c1c",  # System Analyst @ TCS                 → Pivot    3.8
    "in-5b29876b95d790b8",  # Business Analyst @ Key Food          → Pivot    2.8
    "in-50c8748c768e2f50",  # Cloud Infrastructure @ AWS           → Pivot    5.8
    "li-4370606419",        # Sr Analyst Strategy & Ops @ Scotia   → Pivot    3.5
    "in-1783da45863c66c0",  # AI Data Engineer @ Deloitte          → Step Up  8.0
    "in-73f45ab65515bc5a",  # Computer Systems Analyst @ Craftner  → Pivot    2.0
    "in-a0c5c60f9ff09fde",  # Pharmacy Assistant                   → Pivot    1.0
    "li-4387991366",        # BA Fraud Ops AI @ Optomi             → Pivot    4.5
    "in-592e5841f9a16d5f",  # Sr Partner Consultant @ AWS          → Pivot    4.8
    "li-4387748514",        # BA IT III @ Robertson                → Pivot    4.0
    "li-4377515148",        # BA Store Technology @ Primark        → Pivot    3.0
    "in-4f4b2d67d4f0190f",  # Sr Ops Research Analyst @ Noblis     → Pivot    2.3
    "in-5786658763ac8049",  # Data Centre Chief Engineer @ Amazon  → Pivot    2.5
    "in-8abe8673798de6fd",  # Director GenAI & ML @ RBC            → Step Up  7.3
]

EXTRACT_PROMPT = """\
Extract job details. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact job title",
  "summary": "key responsibilities and requirements"
}}"""

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


def ollama(model, prompt, num_ctx=8192):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"num_ctx": num_ctx}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp["message"]["content"], round(time.time() - t0, 1)

def strip_think(t):
    return t.split("</think>", 1)[1].strip() if "</think>" in t else t

def parse_json(text):
    text = strip_think(text).strip()
    if text.startswith("```"): text = text.split("\n",1)[1].rsplit("```",1)[0].strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        try: return json.loads(text[s:e+1])
        except: pass
    return {"_raw": text[:300]}

def avg_score(scores):
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals)/len(vals), 1) if vals else 0.0

def load_data():
    db = psycopg2.connect(DB_DSN)
    with db.cursor() as cur:
        cur.execute("SELECT id, title, company, description FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        jobs = {r[0]: {"title": r[1], "company": r[2], "description": r[3] or "", "url": ""} for r in cur.fetchall()}
        cur.execute("SELECT id, job_url FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        for jid, url in cur.fetchall():
            if jid in jobs: jobs[jid]["url"] = url or ""
        cur.execute("SELECT resume_json FROM adm.resume WHERE profile = %s AND is_active = TRUE", (PROFILE,))
        row = cur.fetchone()
        resume = json.dumps(row[0]) if row else "{}"
    db.close()
    return jobs, resume

def tg_send(text, bot, chat):
    import requests
    r = requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})
    if not r.json().get("ok"): print("TG FAIL:", r.json().get("description"))


def main():
    scripts_dir = os.path.dirname(__file__)
    print("Loading data...")
    jobs, resume = load_data()
    print(f"  {len(jobs)} jobs\n")

    with open(os.path.join(scripts_dir, "claude_control.json")) as f:
        claude = {r["job_id"]: r for r in json.load(f)}

    # Extend extract cache (reuse Phase 3a cache, add missing jobs)
    cache_path = os.path.join(scripts_dir, "new_extract_cache.json")
    extract_cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    print("=== PHASE 1: extracts (8b) — new jobs only ===\n")
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job: continue
        if job_id in extract_cache:
            print(f"  {job_id} — cached"); continue
        prompt = EXTRACT_PROMPT.format(description=job["description"])
        raw, ext_s = ollama(MODEL_8B, prompt, num_ctx=8192)
        parsed = parse_json(strip_think(raw))
        extract_cache[job_id] = {"text": raw, "parsed": parsed,
                                  "summary": parsed.get("summary",""), "ext_s": ext_s}
        print(f"  {job_id}  {ext_s}s  {parsed.get('title','')[:50]}", flush=True)
        with open(cache_path, "w") as f: json.dump(extract_cache, f, indent=2)

    # Evals
    out_path = os.path.join(scripts_dir, "phase3b_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {v: [] for v in EVAL_VARIANTS}
    for v in EVAL_VARIANTS: results.setdefault(v, [])

    avail = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail)
    done  = sum(len(results[v]) for v in EVAL_VARIANTS)
    print(f"\n=== PHASE 2: evals (14b) — {total} total, {done} done ===\n")

    for job_id in avail:
        job = jobs[job_id]; c = claude.get(job_id)
        ext_text = extract_cache[job_id]["text"]
        for v_name, tmpl in EVAL_VARIANTS.items():
            if job_id in {r["job_id"] for r in results[v_name]}: continue
            raw, eval_s = ollama(MODEL_14B, tmpl.format(candidate_json=resume, job_json=ext_text))
            ev = parse_json(strip_think(raw))
            verdict = ev.get("verdict","?"); scores = ev.get("match_scores",{})
            avg = avg_score(scores)
            ctrl_v = c["verdict"] if c else "?"; ctrl_avg = c["avg"] if c else 0.0
            mark = "✓" if verdict==ctrl_v else "✗"
            print(f"  [{v_name}] {job['title'][:30]} @ {job['company'][:18]}")
            print(f"    {mark} {verdict:16}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{avg-ctrl_avg:+.1f})  {eval_s}s", flush=True)
            results[v_name].append({"job_id": job_id, "title": job["title"], "company": job["company"],
                "eval_s": eval_s, "verdict": verdict, "scores": scores, "avg": avg,
                "job_in_one_line": ev.get("job_in_one_line",""), "why_you_fit": ev.get("why_you_fit",""),
                "key_gap": ev.get("key_gap","")})
            done += 1
            with open(out_path, "w") as f: json.dump(results, f, indent=2)
        print(f"  [{job_id}] done — {done}/{total}", flush=True)

    # Summary
    v_names = list(EVAL_VARIANTS.keys())
    print(f"\n\n{'='*90}\nSCORE GRID vs claude_control\n{'='*90}")
    hdr = f"{'Job':<36} {'ctrl':>5}" + "".join(f"  {v[:13]:>13}" for v in v_names)
    print(hdr); print("-"*len(hdr))
    maes = {v:0.0 for v in v_names}; vms = {v:0 for v in v_names}; n=0
    for jid in JOB_IDS:
        job=jobs.get(jid); c=claude.get(jid)
        if not job or not c: continue
        label=f"{job['title'][:18]}@{job['company'][:15]}"
        row=f"{label:<36} {c['avg']:>5.1f}"; n+=1
        for v in v_names:
            r=next((x for x in results[v] if x["job_id"]==jid),None)
            if r:
                d=r["avg"]-c["avg"]; maes[v]+=abs(d)
                if r["verdict"]==c["verdict"]: vms[v]+=1
                mk="✓" if r["verdict"]==c["verdict"] else "✗"
                row+=f"  {r['avg']:>4.1f}({d:>+4.1f}){mk}"
            else: row+=f"  {'N/A':>13}"
        print(row)
    print("-"*len(hdr))
    print(f"{'MAE':<42}"+"".join(f"  {maes[v]/n:>10.2f}   " for v in v_names))
    print(f"\nVerdict accuracy ({n} jobs):")
    for v in v_names:
        print(f"  {v:<22} {vms[v]:>2}/{n}  {'█'*vms[v]+'░'*(n-vms[v])}")

    # Telegram
    import os as _os, requests as _req
    BOT = _os.getenv("TELEGRAM_BOT_TOKEN"); CHAT = _os.getenv("TELEGRAM_CHAT_ID", "")
    ICON = {"Step Up":"⬆️","Lateral":"↔️","Title Regression":"⬇️","Pivot":"↩️"}

    def send(t): tg_send(t, BOT, CHAT)
    def preview(r, lbl):
        s=r["scores"]; icon=ICON.get(r["verdict"],"?")
        c=claude.get(r["job_id"],{}); mk="✓" if r["verdict"]==c.get("verdict") else "✗"
        sk=s.get("skills_match","-"); lv=s.get("career_level_alignment","-")
        ex=s.get("experience_relevance","-"); cu=s.get("culture_fit","—")
        scores_line = f"Tech:{sk} Lvl:{lv} Exp:{ex}" + (f" Cult:{cu}" if cu!="—" else "")
        url=jobs.get(r["job_id"],{}).get("url","")
        link=f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        return (f"<b>[{lbl}] {mk}</b>\n📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
                f"📋 {r['job_in_one_line']}\n✅ {r['why_you_fit']}\n⚠️ {r['key_gap']}\n{scores_line}{link}")

    SHOW = [
        ("in-1783da45863c66c0","Deloitte AI DE"),
        ("in-8abe8673798de6fd","RBC Director GenAI"),
        ("li-4377018253","TekStaff DE"),
        ("in-73f45ab65515bc5a","Craftner COBOL"),
        ("in-5b29876b95d790b8","Key Food BA"),
    ]
    send("📊 <b>Phase 3b — Full 20 jobs, new extract</b>\nv1_skeptic | v3_advisor | advisor_3field\n——————————")
    for jid, name in SHOW:
        c=claude.get(jid,{}); job=jobs.get(jid,{})
        url=job.get("url",""); link=f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        send(f"<b>── {name} ──</b>\nctrl: {c.get('verdict')} {c.get('avg')} | <i>{c.get('summary','')}</i>{link}")
        for v in v_names:
            r=next((x for x in results[v] if x["job_id"]==jid),None)
            if r: send(preview(r, v))

    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
