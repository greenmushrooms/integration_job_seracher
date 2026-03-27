"""
Phase 3e: Clean validation set for Kezia — 20 jobs, no contamination.

Ground truth in new_control_kezia.json established by Claude Sonnet 4.6.
Kezia profile: Technical Business Analyst, Salesforce/CRM, Agile, mid-level, Toronto.

Distribution: Lateral×4, Step Up×1, Title Regression×3, Pivot×12

Same 5 variants as Phase 3d to enable direct cross-profile comparison:
  v1_skeptic, v3_advisor, advisor_3field, v3_fewshot, advisor_3f_fewshot

Synthetic few-shot examples are adapted for a BA profile (not DE).
"""
import json, os, sys, time
import psycopg2
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_DSN     = f"postgresql://user_job_searcher:{os.getenv('DB_PASSWORD')}@localhost:5432/job_searcher"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_8B   = "qwen3:8b"
MODEL_14B  = "qwen3:14b"
PROFILE    = "Kezia"

JOB_IDS = [
    "in-7fcfb930800d1c59",   # Sr. Technical BA @ Scotiabank          → Step Up   7.8
    "in-383459c78ee8a783",   # IT BA (AI Programme) @ States of Jersey → Lateral   6.5
    "in-eb3783eda39282fa",   # System Documentation Analyst @ Geo Sol  → TR        6.3
    "in-5f6198db850d0cb2",   # Sr. Biz Ops Analyst (Salesforce) @ BTI  → Lateral   7.8
    "in-43cfd11eab1cac0e",   # Sr. Manager Finance Analytics @ Verily  → Pivot     5.3
    "in-68551f9123fc9e1f",   # Power Platform CoE Lead @ Chevo         → Pivot     4.8
    "in-3cccb8c0dbb0dcad",   # Senior Solutions Consultant @ Heartland → Pivot     5.5
    "in-cabb2e385192916d",   # 2026 IT Analyst Apprenticeship @ Mott   → TR        2.8
    "in-80012a7bb37bc35f",   # Senior Yield Data Analyst @ Hearst      → Pivot     4.3
    "in-7bfdb13b19c76fb1",   # Business Intelligence Analyst @ Heritage → Lateral   6.0
    "in-4be39eaac95bbc6c",   # Data Analyst @ Flint Hills Resources    → Pivot     3.5
    "in-3fa5cf8e605b24e4",   # Sr. SharePoint & Power BI Dev @ Diaconia → Pivot    4.5
    "in-630ae7899d80de30",   # Software Engineer (Python) @ Scotiabank  → Pivot    4.5
    "in-69b35bcfc1bfea7a",   # Management Analyst II @ Synectic         → Pivot    5.0
    "in-478dd13cb7cfd65d",   # Fire Project Manager @ Johnson Controls  → Pivot    3.5
    "in-3c916eb2aaeb64fa",   # Consultant Psychiatrist @ Great Ormond   → Pivot    2.3
    "in-141b559138ebfb7c",   # Home Construction PM @ Woodcastle        → Pivot    2.5
    "in-09f1bd1318b657bf",   # Executive Assistant to CFO @ Foundation  → TR       4.0
    "in-21c41ef36a571ea8",   # Design Coordinator @ Charcoalblue        → Pivot    3.5
    "in-5c8c1a18d0bba825",   # Interim Biz Data Insight Analyst @ Tri  → Lateral  5.8
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

# Synthetic few-shot examples — BA profile (not contaminated from Kezia test set)
SYNTHETIC_EXAMPLES_4F = [
    {
        "job": '{"title": "Electrician", "summary": "Install and maintain electrical systems in commercial and residential buildings. Read blueprints, wire panels, troubleshoot circuits. Requires journeyman license and 3+ years field experience."}',
        "verdict": "Pivot",
        "scores": {"skills_match": 1, "career_level_alignment": 3, "experience_relevance": 1, "culture_fit": 2},
        "jol": "Trades/electrical installation role in construction — completely different domain from business analysis",
        "wyf": "No genuine overlap — hands-on trades work vs business systems and CRM analysis",
        "kg": "Entirely wrong domain and skill set: electrical trades vs business analysis/CRM/Agile delivery",
    },
    {
        "job": '{"title": "Business Systems Analyst", "summary": "Gather and document business requirements for CRM and ERP system implementations. Facilitate stakeholder workshops, write user stories, manage UAT. Work in Agile/Scrum environment with JIRA and Confluence."}',
        "verdict": "Lateral",
        "scores": {"skills_match": 9, "career_level_alignment": 8, "experience_relevance": 9, "culture_fit": 7},
        "jol": "Business systems analyst role implementing CRM/ERP with Agile delivery methodology",
        "wyf": "Direct match — CRM implementations, user stories, UAT, JIRA/Confluence all align with candidate's core expertise",
        "kg": "Same scope and level as current role — not a step up in title or responsibility",
    },
    {
        "job": '{"title": "Lead Business Analyst / Product Owner", "summary": "Own the product backlog for a Salesforce CRM transformation program. Lead a team of 3 BAs. Define roadmap, prioritize features with business sponsors, drive cross-functional delivery. Requires 6+ years BA experience and CRM programme leadership."}',
        "verdict": "Step Up",
        "scores": {"skills_match": 8, "career_level_alignment": 7, "experience_relevance": 8, "culture_fit": 7},
        "jol": "Lead BA / Product Owner for Salesforce CRM transformation with team and backlog ownership",
        "wyf": "Salesforce CRM expertise plus backlog management experience positions candidate well for this leadership step",
        "kg": "Requires CRM programme leadership and managing a team of BAs — stretch beyond current individual contributor scope",
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
    print(f"  {len(jobs)} jobs (Kezia profile)\n")

    with open(os.path.join(scripts_dir, "new_control_kezia.json")) as f:
        control = {r["job_id"]: r for r in json.load(f)}

    cache_path = os.path.join(scripts_dir, "new_extract_cache_kezia.json")
    extract_cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    print("=== PHASE 1: extract (8b) — Kezia jobs ===\n")
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

    out_path = os.path.join(scripts_dir, "phase3e_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {v: [] for v in EVAL_VARIANTS}
    for v in EVAL_VARIANTS: results.setdefault(v, [])

    avail = [j for j in JOB_IDS if j in extract_cache and j in jobs]
    v_names = list(EVAL_VARIANTS.keys())
    total = len(v_names) * len(avail)
    done  = sum(len(results[v]) for v in v_names)
    print(f"\n=== PHASE 2: evals (14b) — {len(v_names)} variants × {len(avail)} jobs = {total} ({done} done) ===")
    print(f"    Profile: Kezia | Clean test set\n")

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
    print(f"\n\n{'='*90}\nSCORE GRID vs new_control_kezia (20 clean jobs)\n{'='*90}")
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
    print(f"\nVerdict accuracy ({n} clean Kezia jobs):")
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

    SHOW = [
        ("in-7fcfb930800d1c59", "Scotiabank Sr. TBA"),    # Step Up 7.8
        ("in-5f6198db850d0cb2", "BTI Salesforce BA"),     # Lateral 7.8
        ("in-43cfd11eab1cac0e", "Verily Finance Mgr"),    # Pivot 5.3
        ("in-3cccb8c0dbb0dcad", "Heartland Solutions"),   # Pivot 5.5
        ("in-69b35bcfc1bfea7a", "Synectic Mgmt Analyst"), # Pivot 5.0
    ]
    mae_summary = " | ".join(f"{v}={maes[v]/n:.2f}" for v in v_names)
    vacc_summary = " | ".join(f"{v}={vms[v]}/{n}" for v in v_names)
    send(f"🎯 <b>Phase 3e — Kezia Clean 20</b>\n"
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
