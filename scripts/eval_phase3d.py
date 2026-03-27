"""
Phase 3d: Clean validation set — 20 jobs never seen during prompt tuning.

These 20 jobs were NOT part of the diagnostic set used in Phases 2-3c.
Ground truth in new_control_slava.json was established independently by Claude Sonnet 4.6.

Distribution: Lateral×5, Step Up×2, Title Regression×3, Pivot×10

Tests same 3 variants as Phase 3b (no few-shot) to get uncontaminated baseline:
  v1_skeptic         — skeptic persona, 4 fields
  v3_advisor         — advisor persona, 4 fields
  advisor_3field     — advisor, no culture_fit

Also tests Phase 3c few-shot variants with synthetic examples on clean data:
  v3_fewshot         — advisor + 3 synthetic examples, 4 fields
  advisor_3f_fewshot — advisor + 3 synthetic examples, no culture_fit
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
    "li-4338989335",        # Analytics Engineer @ Alliants          → Lateral   7.3
    "in-ddcbcfec708f9ce1",  # AI & Data Engineer @ RBC               → Lateral   7.0
    "li-4375956508",        # Analytics Engineer @ CDAI              → Lateral   8.0
    "in-ae1b8b7885cae87b",  # Analyst Portfolio DE @ TD              → TR        6.3
    "li-4252714741",        # AI Data Manager Sol Arch @ EY          → Step Up   7.0
    "in-a0d15fa967e7aee3",  # Applied AI Data Engineer @ RBC         → Step Up   7.5
    "li-4381907300",        # AI Data Engineer @ BMO                 → Lateral   7.5
    "in-235965b432706652",  # Apache Spark lead @ Wipro              → Lateral   6.0
    "in-764712110bcfc60c",  # 2026 Summer DE (8mo) @ RBC             → TR        4.5
    "in-4ffddaa92af67e4d",  # 2026 Summer Data Analyst @ RBC         → TR        4.0
    "in-3cf939de087826a1",  # Account Executive @ DocuSign           → Pivot     2.3
    "in-9277040ca5f684b7",  # 3rd Class Power Engineer @ OxyChem     → Pivot     1.8
    "in-c86890c9d54ede48",  # Accounts Payable & BA @ FELLFAB        → Pivot     2.8
    "in-0e34187d5dfcfd5c",  # Agentic AI Developer @ CGI             → Pivot     5.5
    "li-4385082905",        # AI Data Scientist @ EY                 → Pivot     4.8
    "in-0753574d1a5c935a",  # Actuarial DE @ Lincoln Financial       → Pivot     5.3
    "li-4385604575",        # Analyst @ Insight Global               → Pivot     2.0
    "in-2a146301dd54a3c0",  # AI DSP Sr UI Engineer @ Qualcomm       → Pivot     3.5
    "in-997f8fc141e0ac40",  # AI Product Manager @ Groupon           → Pivot     3.5
    "in-f22a6f890366d5c0",  # Agile Product Coach @ NTT DATA         → Pivot     3.5
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

# Synthetic few-shot examples (not from DB — zero contamination)
SYNTHETIC_EXAMPLES_4F = [
    {
        "job": '{"title": "Pharmacy Technician", "summary": "Assist licensed pharmacists dispensing prescription medications. Process insurance claims, maintain drug inventory, provide customer service in a retail pharmacy."}',
        "verdict": "Pivot",
        "scores": {"skills_match": 1, "career_level_alignment": 3, "experience_relevance": 1, "culture_fit": 2},
        "jol": "Retail pharmacy role dispensing medications and processing insurance — healthcare domain",
        "wyf": "No genuine overlap — completely different domain, tools, and responsibilities from data engineering",
        "kg": "Entirely wrong domain: healthcare/pharmacy vs data engineering; no transferable technical skills",
    },
    {
        "job": '{"title": "Senior Data Engineer", "summary": "Build and maintain ETL pipelines in Python and dbt on AWS. Own Airflow orchestration, Snowflake warehousing, and Spark batch jobs. Collaborate with analytics teams to deliver data products."}',
        "verdict": "Lateral",
        "scores": {"skills_match": 9, "career_level_alignment": 8, "experience_relevance": 9, "culture_fit": 7},
        "jol": "Data engineering role building ETL pipelines with Python, dbt, AWS, Airflow, Snowflake",
        "wyf": "Direct stack match — Python, dbt, AWS, Spark, Airflow all align with candidate's core expertise",
        "kg": "Similar scope and level to current role — not a step up in responsibility or compensation",
    },
    {
        "job": '{"title": "Staff Data Engineer / Tech Lead", "summary": "Lead a team of 4 engineers building the company\'s data platform. Define architecture for streaming and batch systems. Drive adoption of dbt, Spark, and Kafka. Present roadmap to VP Engineering. Requires 7+ years experience and prior tech lead experience."}',
        "verdict": "Step Up",
        "scores": {"skills_match": 8, "career_level_alignment": 7, "experience_relevance": 8, "culture_fit": 7},
        "jol": "Staff/lead data engineering role with team leadership and platform architecture ownership",
        "wyf": "Same technical stack (dbt, Spark) plus leadership scope — natural next step from senior IC",
        "kg": "Requires formal tech lead experience and people management; candidate may need to demonstrate leadership track record",
    },
]

SYNTHETIC_EXAMPLES_3F = [
    {**e, "scores": {k: v for k, v in e["scores"].items() if k != "culture_fit"}}
    for e in SYNTHETIC_EXAMPLES_4F
]


def build_fewshot_prompt(examples: list, with_culture: bool) -> str:
    culture_score = '\n    "culture_fit": <1-10>,' if with_culture else ""
    ex_blocks = []
    for i, ex in enumerate(examples, 1):
        ex_blocks.append(
            f"--- EXAMPLE {i}: {ex['verdict']} ---\n"
            f"Job: {ex['job']}\n"
            f"Output:\n"
            + json.dumps({
                "verdict": ex["verdict"],
                "match_scores": ex["scores"],
                "job_in_one_line": ex["jol"],
                "why_you_fit": ex["wyf"],
                "key_gap": ex["kg"],
            }, indent=2)
        )
    # Escape literal braces in examples so they survive .format() later
    examples_str = "\n\n".join(ex_blocks).replace("{", "{{").replace("}", "}}")
    return (
        "You are an honest career advisor.\n\n"
        + examples_str
        + "\n\n--- NOW EVALUATE ---\n"
        "Candidate:\n{candidate_json}\n\n"
        "Job:\n{job_json}\n\n"
        "Return JSON only:\n"
        "{{\n"
        '  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",\n'
        '  "match_scores": {{\n'
        '    "skills_match": <1-10>,\n'
        '    "career_level_alignment": <1-10>,\n'
        f'    "experience_relevance": <1-10>{culture_score}\n'
        "  }},\n"
        '  "job_in_one_line": "what this role does and what domain/industry it is",\n'
        '  "why_you_fit": "strongest overlap between candidate and this role in one sentence",\n'
        '  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"\n'
        "}}"
    )


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

    "v3_fewshot":         build_fewshot_prompt(SYNTHETIC_EXAMPLES_4F, with_culture=True),
    "advisor_3f_fewshot": build_fewshot_prompt(SYNTHETIC_EXAMPLES_3F, with_culture=False),
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
        cur.execute("SELECT id, title, company, job_url, description FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        jobs = {r[0]: {"title": r[1], "company": r[2], "url": r[3] or "", "desc": r[4] or ""} for r in cur.fetchall()}
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

    with open(os.path.join(scripts_dir, "new_control_slava.json")) as f:
        control = {r["job_id"]: r for r in json.load(f)}

    # Extract cache — separate from the diagnostic 20 cache
    cache_path = os.path.join(scripts_dir, "new_extract_cache_v2.json")
    extract_cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    print("=== PHASE 1: extract (8b) — new jobs only ===\n")
    for job_id in JOB_IDS:
        job = jobs.get(job_id)
        if not job: continue
        if job_id in extract_cache:
            print(f"  {job_id} — cached"); continue
        prompt = EXTRACT_PROMPT.format(description=job["desc"])
        raw, ext_s = ollama(MODEL_8B, prompt, num_ctx=8192)
        parsed = parse_json(strip_think(raw))
        extract_cache[job_id] = {"text": raw, "parsed": parsed,
                                  "summary": parsed.get("summary",""), "ext_s": ext_s}
        print(f"  {job_id}  {ext_s}s  {parsed.get('title','')[:50]}", flush=True)
        with open(cache_path, "w") as f: json.dump(extract_cache, f, indent=2)

    # Evals
    out_path = os.path.join(scripts_dir, "phase3d_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {v: [] for v in EVAL_VARIANTS}
    for v in EVAL_VARIANTS: results.setdefault(v, [])

    avail = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    v_names = list(EVAL_VARIANTS.keys())
    total = len(v_names) * len(avail)
    done  = sum(len(results[v]) for v in v_names)
    print(f"\n=== PHASE 2: evals (14b) — {len(v_names)} variants × {len(avail)} jobs = {total} ({done} done) ===")
    print(f"    Clean test set: never used in Phase 2/3 prompt tuning\n")

    for job_id in avail:
        job = jobs[job_id]; c = control.get(job_id)
        ext_text = extract_cache[job_id]["text"]
        for v_name, tmpl in EVAL_VARIANTS.items():
            if job_id in {r["job_id"] for r in results[v_name]}: continue
            num_ctx = 12288 if "fewshot" in v_name else 8192
            prompt = tmpl.format(candidate_json=resume, job_json=ext_text)
            raw, eval_s = ollama(MODEL_14B, prompt, num_ctx=num_ctx)
            ev = parse_json(strip_think(raw))
            verdict = ev.get("verdict","?"); scores = ev.get("match_scores",{})
            avg = avg_score(scores)
            ctrl_v = c["verdict"] if c else "?"; ctrl_avg = c["avg"] if c else 0.0
            mark = "✓" if verdict==ctrl_v else "✗"
            print(f"  [{v_name}] {job['title'][:28]} @ {job['company'][:16]}")
            print(f"    {mark} {verdict:16}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{avg-ctrl_avg:+.1f})  {eval_s}s", flush=True)
            results[v_name].append({"job_id": job_id, "title": job["title"], "company": job["company"],
                "eval_s": eval_s, "verdict": verdict, "scores": scores, "avg": avg,
                "job_in_one_line": ev.get("job_in_one_line",""),
                "why_you_fit": ev.get("why_you_fit",""), "key_gap": ev.get("key_gap","")})
            done += 1
            with open(out_path, "w") as f: json.dump(results, f, indent=2)
        print(f"  [{job_id}] done — {done}/{total}", flush=True)

    # Summary
    print(f"\n\n{'='*90}\nSCORE GRID vs new_control_slava (20 clean jobs)\n{'='*90}")
    maes = {v:0.0 for v in v_names}; vms = {v:0 for v in v_names}; n=0
    hdr = f"{'Job':<36} {'ctrl':>5}" + "".join(f"  {v[:14]:>14}" for v in v_names)
    print(hdr); print("-"*len(hdr))
    for jid in JOB_IDS:
        job=jobs.get(jid); c=control.get(jid)
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
            else: row+=f"  {'N/A':>12}"
        print(row)
    print("-"*len(hdr))
    print(f"{'MAE':<42}"+"".join(f"  {maes[v]/n:>10.2f}   " for v in v_names))
    print(f"\nVerdict accuracy ({n} clean jobs):")
    for v in v_names:
        print(f"  {v:<26} {vms[v]:>2}/{n}  {'█'*vms[v]+'░'*(n-vms[v])}")

    # Telegram
    import os as _os
    BOT = _os.getenv("TELEGRAM_BOT_TOKEN"); CHAT = _os.getenv("TELEGRAM_CHAT_ID", "")
    ICON = {"Step Up":"⬆️","Lateral":"↔️","Title Regression":"⬇️","Pivot":"↩️"}
    def send(t): tg_send(t, BOT, CHAT)
    def preview(r, lbl):
        s=r["scores"]; icon=ICON.get(r["verdict"],"?")
        c=control.get(r["job_id"],{}); mk="✓" if r["verdict"]==c.get("verdict") else "✗"
        sk=s.get("skills_match","-"); lv=s.get("career_level_alignment","-")
        ex_s=s.get("experience_relevance","-"); cu=s.get("culture_fit","—")
        sc_line=f"Tech:{sk} Lvl:{lv} Exp:{ex_s}"+(f" Cult:{cu}" if cu!="—" else "")
        url=jobs.get(r["job_id"],{}).get("url","")
        link=f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        return (f"<b>[{lbl}] {mk}</b>\n📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
                f"📋 {r['job_in_one_line']}\n✅ {r['why_you_fit']}\n⚠️ {r['key_gap']}\n{sc_line}{link}")

    # Show key boundary cases
    SHOW = [
        ("in-a0d15fa967e7aee3", "RBC Applied AI DE"),   # Step Up 7.5
        ("li-4252714741",       "EY Manager Sol Arch"),  # Step Up 7.0
        ("li-4381907300",       "BMO AI DE"),            # Lateral 7.5
        ("in-0e34187d5dfcfd5c", "CGI Agentic AI Dev"),   # Pivot 5.5
        ("li-4385082905",       "EY Data Scientist"),    # Pivot 4.8
    ]
    mae_summary = " | ".join(f"{v}={maes[v]/n:.2f}" for v in v_names)
    vacc_summary = " | ".join(f"{v}={vms[v]}/{n}" for v in v_names)
    send(f"🎯 <b>Phase 3d — Clean 20 (no contamination)</b>\n"
         f"MAE: {mae_summary}\nVerdict: {vacc_summary}\n——————————")
    for jid, name in SHOW:
        c=control.get(jid,{}); job=jobs.get(jid,{})
        url=job.get("url",""); link=f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        send(f"<b>── {name} ──</b>\nctrl: {c.get('verdict')} {c.get('avg')} | <i>{c.get('summary','')}</i>{link}")
        for v in v_names:
            r=next((x for x in results[v] if x["job_id"]==jid),None)
            if r: send(preview(r, v))

    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
