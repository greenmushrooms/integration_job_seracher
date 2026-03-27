"""
Phase 3c: Few-shot contrastive examples on new extract — all 20 jobs.

Synthetic few-shot examples (not from the 20 diagnostic jobs — zero contamination).
All 20 jobs remain in the test set.

3 synthetic examples teach the Pivot / Lateral / Step Up boundary:
  - Pharmacy Tech     → Pivot    (completely wrong domain)
  - Senior DE         → Lateral  (same stack, same level)
  - Staff DE TechLead → Step Up  (same domain, stretch level + leadership)

Variants:
  v3_fewshot         — advisor + 3 synthetic examples (4-field scores)
  advisor_3f_fewshot — advisor, no culture_fit + 3 synthetic examples
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_14B  = "qwen3:14b"
PROFILE    = "Slava"

# All 20 diagnostic jobs — no exclusions
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

# Synthetic examples — not from the DB, zero contamination with test set
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


def build_prompt(examples: list, with_culture: bool) -> str:
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
    return (
        "You are an honest career advisor.\n\n"
        + "\n\n".join(ex_blocks)
        + "\n\n--- NOW EVALUATE ---\n"
        "Candidate:\n{candidate_json}\n\n"
        "Job:\n{job_json}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",\n'
        '  "match_scores": {\n'
        '    "skills_match": <1-10>,\n'
        '    "career_level_alignment": <1-10>,\n'
        f'    "experience_relevance": <1-10>{culture_score}\n'
        "  },\n"
        '  "job_in_one_line": "what this role does and what domain/industry it is",\n'
        '  "why_you_fit": "strongest overlap between candidate and this role in one sentence",\n'
        '  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"\n'
        "}"
    )


EVAL_VARIANTS = {
    "v3_fewshot":         build_prompt(SYNTHETIC_EXAMPLES_4F, with_culture=True),
    "advisor_3f_fewshot": build_prompt(SYNTHETIC_EXAMPLES_3F, with_culture=False),
}


def ollama(model, prompt, num_ctx=12288):
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
        cur.execute("SELECT id, title, company FROM jobspy_jobs WHERE id = ANY(%s)", (JOB_IDS,))
        jobs = {r[0]: {"title": r[1], "company": r[2], "url": ""} for r in cur.fetchall()}
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

    cache_path = os.path.join(scripts_dir, "new_extract_cache.json")
    with open(cache_path) as f:
        extract_cache = json.load(f)

    out_path = os.path.join(scripts_dir, "phase3c_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {v: [] for v in EVAL_VARIANTS}
    for v in EVAL_VARIANTS: results.setdefault(v, [])

    avail = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    total = len(EVAL_VARIANTS) * len(avail)
    done  = sum(len(results[v]) for v in EVAL_VARIANTS)
    print(f"=== PHASE 3c: {len(EVAL_VARIANTS)} variants × {len(avail)} jobs = {total} evals ({done} done) ===")
    print(f"    Synthetic few-shot examples — all 20 jobs in test set\n")

    for job_id in avail:
        job = jobs[job_id]; c = claude.get(job_id)
        ext_text = extract_cache[job_id]["text"]
        for v_name, tmpl in EVAL_VARIANTS.items():
            if job_id in {r["job_id"] for r in results[v_name]}: continue
            prompt = tmpl.format(candidate_json=resume, job_json=ext_text)
            raw, eval_s = ollama(MODEL_14B, prompt)
            ev = parse_json(strip_think(raw))
            verdict = ev.get("verdict","?"); scores = ev.get("match_scores",{})
            avg = avg_score(scores)
            ctrl_v = c["verdict"] if c else "?"; ctrl_avg = c["avg"] if c else 0.0
            mark = "✓" if verdict==ctrl_v else "✗"
            print(f"  [{v_name}] {job['title'][:30]} @ {job['company'][:18]}")
            print(f"    {mark} {verdict:16}  avg={avg:.1f} (ctrl {ctrl_avg:.1f} Δ{avg-ctrl_avg:+.1f})  {eval_s}s", flush=True)
            results[v_name].append({"job_id": job_id, "title": job["title"], "company": job["company"],
                "eval_s": eval_s, "verdict": verdict, "scores": scores, "avg": avg,
                "job_in_one_line": ev.get("job_in_one_line",""),
                "why_you_fit": ev.get("why_you_fit",""), "key_gap": ev.get("key_gap","")})
            done += 1
            with open(out_path, "w") as f: json.dump(results, f, indent=2)
        print(f"  [{job_id}] done — {done}/{total}", flush=True)

    # Summary
    v_names = list(EVAL_VARIANTS.keys())
    print(f"\n\n{'='*80}\nSCORE GRID vs claude_control (20 jobs)\n{'='*80}")
    maes = {v:0.0 for v in v_names}; vms = {v:0 for v in v_names}; n=0
    hdr = f"{'Job':<36} {'ctrl':>5}" + "".join(f"  {v[:18]:>18}" for v in v_names)
    print(hdr); print("-"*len(hdr))
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
                row+=f"  {r['avg']:>5.1f}({d:>+4.1f}){mk}  "
            else: row+=f"  {'N/A':>20}"
        print(row)
    print("-"*len(hdr))
    print(f"{'MAE':<42}"+"".join(f"  {maes[v]/n:>14.2f}    " for v in v_names))
    print(f"\nVerdict accuracy ({n} test jobs):")
    for v in v_names:
        print(f"  {v:<26} {vms[v]:>2}/{n}  {'█'*vms[v]+'░'*(n-vms[v])}")

    # Telegram
    import os as _os
    BOT = _os.getenv("TELEGRAM_BOT_TOKEN"); CHAT = _os.getenv("TELEGRAM_CHAT_ID", "")
    ICON = {"Step Up":"⬆️","Lateral":"↔️","Title Regression":"⬇️","Pivot":"↩️"}
    def send(t): tg_send(t, BOT, CHAT)
    def preview(r, lbl):
        s=r["scores"]; icon=ICON.get(r["verdict"],"?")
        c=claude.get(r["job_id"],{}); mk="✓" if r["verdict"]==c.get("verdict") else "✗"
        sk=s.get("skills_match","-"); lv=s.get("career_level_alignment","-")
        ex_s=s.get("experience_relevance","-"); cu=s.get("culture_fit","—")
        sc_line=f"Tech:{sk} Lvl:{lv} Exp:{ex_s}"+(f" Cult:{cu}" if cu!="—" else "")
        url=jobs.get(r["job_id"],{}).get("url","")
        link=f'\n🔗 <a href="{url}">{url[:55]}</a>' if url else ""
        return (f"<b>[{lbl}] {mk}</b>\n📊 {r['avg']:.1f}/10  {icon} {r['verdict']}\n"
                f"📋 {r['job_in_one_line']}\n✅ {r['why_you_fit']}\n⚠️ {r['key_gap']}\n{sc_line}{link}")

    SHOW = [
        ("in-1783da45863c66c0","Deloitte AI DE"),
        ("in-8abe8673798de6fd","RBC Director GenAI"),
        ("li-4377018253","TekStaff DE"),
        ("in-73f45ab65515bc5a","Craftner COBOL"),
        ("in-5b29876b95d790b8","Key Food BA"),
    ]
    send("🎯 <b>Phase 3c — Synthetic few-shot on new extract</b>\n"
         "3 synthetic examples: Pivot+Lateral+StepUp (zero contamination)\n"
         "Test set: all 20 jobs\n——————————")
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
