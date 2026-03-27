#!/usr/bin/env python3
"""
Eval runner for job-fit scoring pipeline.

Runs eval prompt variants against ground truth, computes all metrics:
  - Spearman's rho (rank correlation) -- PRIMARY
  - Pairwise concordance
  - Match recall (Step Up + Lateral above threshold)
  - Verdict accuracy
  - MAE

Usage:
  .venv/bin/python3 scripts/eval/agent_eval.py --profile Slava
  .venv/bin/python3 scripts/eval/agent_eval.py --profile Kezia
  .venv/bin/python3 scripts/eval/agent_eval.py --profile Slava --variant v3_fewshot
  .venv/bin/python3 scripts/eval/agent_eval.py --profile Slava --score-only results.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL_8B = "qwen3:8b"
MODEL_14B = "qwen3:14b"

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# ---------------------------------------------------------------------------
# Fewshot examples (profile-specific, synthetic, NOT from test set)
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES = {
    "Slava": [
        {
            "job": '{"title": "Pharmacy Technician", "summary": "Dispense medications, manage inventory, assist pharmacists in a retail pharmacy setting. Requires pharmacy technician certification."}',
            "verdict": "Pivot",
            "scores": {"skills_match": 1, "career_level_alignment": 4, "experience_relevance": 1, "culture_fit": 3},
            "jol": "Retail pharmacy technician role dispensing medications",
            "wyf": "No relevant overlap with data engineering background",
            "kg": "Completely different domain and function — requires pharmacy certification",
        },
        {
            "job": '{"title": "Senior Data Engineer", "summary": "Build and maintain cloud data pipelines using Spark, Python, and dbt. Design data models, mentor junior engineers, collaborate with analytics teams."}',
            "verdict": "Lateral",
            "scores": {"skills_match": 9, "career_level_alignment": 7, "experience_relevance": 9, "culture_fit": 7},
            "jol": "Senior DE role building data pipelines with dbt and Spark",
            "wyf": "Direct stack match — Python, dbt, cloud pipelines, mentoring all align with current experience",
            "kg": "Senior title is one level below current Lead DE role",
        },
        {
            "job": '{"title": "Staff Data Engineer / Tech Lead", "summary": "Technical lead for data platform engineering team. Own architecture decisions, lead 5-8 engineers, set engineering standards across the org, interface with VP-level stakeholders."}',
            "verdict": "Step Up",
            "scores": {"skills_match": 8, "career_level_alignment": 9, "experience_relevance": 8, "culture_fit": 7},
            "jol": "Staff-level DE tech lead owning platform architecture and team leadership",
            "wyf": "Current Lead DE experience with mentoring and architecture directly prepares for Staff-level ownership",
            "kg": "Larger team scope and VP-level stakeholder management is a stretch",
        },
    ],
    "Kezia": [
        {
            "job": '{"title": "Electrician Apprentice", "summary": "Install and maintain electrical systems in residential and commercial buildings. Requires electrical apprenticeship certification and physical site work."}',
            "verdict": "Pivot",
            "scores": {"skills_match": 1, "career_level_alignment": 3, "experience_relevance": 1, "culture_fit": 2},
            "jol": "Trades apprenticeship in electrical installation and maintenance",
            "wyf": "No relevant overlap with business analysis or Salesforce background",
            "kg": "Completely different domain — requires trades certification and physical site work",
        },
        {
            "job": '{"title": "Business Systems Analyst", "summary": "Gather requirements from stakeholders, document processes, manage Salesforce CRM configuration, support Agile delivery teams with user stories and UAT testing."}',
            "verdict": "Lateral",
            "scores": {"skills_match": 9, "career_level_alignment": 7, "experience_relevance": 9, "culture_fit": 7},
            "jol": "BSA role with Salesforce, Agile delivery and requirements gathering",
            "wyf": "Direct match — Salesforce, Agile, user stories, UAT are core to current role at MLSE",
            "kg": "Same seniority level with no meaningful career step up",
        },
        {
            "job": '{"title": "Lead Business Analyst / Product Owner", "summary": "Lead a team of BAs, own the product backlog, drive roadmap decisions with C-suite stakeholders, accountable for delivery across multiple workstreams."}',
            "verdict": "Step Up",
            "scores": {"skills_match": 8, "career_level_alignment": 9, "experience_relevance": 8, "culture_fit": 7},
            "jol": "Lead BA/PO role with team ownership and executive stakeholder management",
            "wyf": "Strong BA foundation with Agile and SDLC experience positions well for team lead ownership",
            "kg": "Managing a team of BAs and C-suite accountability is a meaningful stretch beyond current scope",
        },
    ],
}

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract job details. Return JSON only, no markdown.

Job posting:
{description}

Return:
{{
  "title": "exact job title",
  "summary": "key responsibilities and requirements"
}}"""


def _build_fewshot_block(examples):
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
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
    return "\n\n".join(blocks)


def build_eval_variants(profile):
    examples = FEWSHOT_EXAMPLES.get(profile, FEWSHOT_EXAMPLES["Slava"])
    fewshot_block = _build_fewshot_block(examples)

    base_schema = (
        '  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",\n'
        '  "match_scores": {{\n'
        '    "skills_match": <1-10>,\n'
        '    "career_level_alignment": <1-10>,\n'
        '    "experience_relevance": <1-10>,\n'
        '    "culture_fit": <1-10>\n'
        '  }},\n'
        '  "job_in_one_line": "what this role does and its domain/industry",\n'
        '  "why_you_fit": "strongest overlap between candidate and this role in one sentence",\n'
        '  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"'
    )

    return {
        "v1_skeptic": (
            "You are a skeptical recruiter. Score conservatively. A 7 means worth applying. A 10 is rare.\n\n"
            "Candidate:\n{candidate_json}\n\nJob:\n{job_json}\n\n"
            f"Return JSON only:\n{{{{\n{base_schema}\n}}}}"
        ),
        "v3_advisor": (
            "You are an honest career advisor.\n\n"
            "Candidate:\n{candidate_json}\n\nJob:\n{job_json}\n\n"
            f"Return JSON only:\n{{{{\n{base_schema}\n}}}}"
        ),
        "v3_fewshot": (
            "You are an honest career advisor.\n\n"
            + fewshot_block
            + "\n\n--- NOW EVALUATE ---\n"
            "Candidate:\n{candidate_json}\n\nJob:\n{job_json}\n\n"
            f"Return JSON only:\n{{{{\n{base_schema}\n}}}}"
        ),
    }


# ---------------------------------------------------------------------------
# Ollama / parsing helpers
# ---------------------------------------------------------------------------

def ollama(model, prompt, num_ctx=8192):
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


def strip_think(t):
    return t.split("</think>", 1)[1].strip() if "</think>" in t else t


def parse_json(text):
    text = strip_think(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            pass
    return {"_raw": text[:300]}


def avg_score(scores):
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman_rho(gt_scores, pred_scores):
    """Spearman's rank correlation between two score lists."""
    n = len(gt_scores)
    if n < 3:
        return float("nan")

    def ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[indexed[k][0]] = avg_rank
            i = j + 1
        return r

    r_gt = ranks(gt_scores)
    r_pred = ranks(pred_scores)
    d_sq = sum((a - b) ** 2 for a, b in zip(r_gt, r_pred))
    return 1 - (6 * d_sq) / (n * (n * n - 1))


def pairwise_concordance(gt_scores, pred_scores):
    """% of pairs where pred agrees with gt on ordering."""
    n = len(gt_scores)
    concordant = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = gt_scores[i] - gt_scores[j]
            pred_diff = pred_scores[i] - pred_scores[j]
            if gt_diff == 0:
                continue
            total += 1
            if (gt_diff > 0 and pred_diff > 0) or (gt_diff < 0 and pred_diff < 0):
                concordant += 1
            elif pred_diff == 0:
                concordant += 0.5  # tie counts as half
    return concordant / total if total else float("nan")


def match_recall(gt_entries, pred_entries, threshold=5.5):
    """What fraction of GT matches (Step Up / Lateral) did the model surface above threshold?"""
    gt_matches = {e["job_id"] for e in gt_entries if e["verdict"] in ("Step Up", "Lateral")}
    if not gt_matches:
        return float("nan")
    pred_surfaced = {
        e["job_id"] for e in pred_entries
        if e["avg"] >= threshold and e["job_id"] in gt_matches
    }
    return len(pred_surfaced) / len(gt_matches)


def verdict_accuracy(gt_entries, pred_entries):
    """Fraction of matching verdicts."""
    gt_map = {e["job_id"]: e["verdict"] for e in gt_entries}
    correct = total = 0
    for p in pred_entries:
        if p["job_id"] in gt_map:
            total += 1
            if p["verdict"] == gt_map[p["job_id"]]:
                correct += 1
    return correct / total if total else float("nan")


def mae(gt_entries, pred_entries):
    gt_map = {e["job_id"]: e["avg"] for e in gt_entries}
    errors = []
    for p in pred_entries:
        if p["job_id"] in gt_map:
            errors.append(abs(p["avg"] - gt_map[p["job_id"]]))
    return sum(errors) / len(errors) if errors else float("nan")


def compute_all_metrics(gt_entries, pred_entries):
    """Compute all metrics given aligned gt and pred entry lists."""
    # Align by job_id
    gt_map = {e["job_id"]: e for e in gt_entries}
    common_ids = [p["job_id"] for p in pred_entries if p["job_id"] in gt_map]
    gt_aligned = [gt_map[jid] for jid in common_ids]
    pred_aligned = [next(p for p in pred_entries if p["job_id"] == jid) for jid in common_ids]

    gt_scores = [e["avg"] for e in gt_aligned]
    pred_scores = [e["avg"] for e in pred_aligned]

    return {
        "n": len(common_ids),
        "spearman_rho": round(spearman_rho(gt_scores, pred_scores), 3),
        "pairwise_concordance": round(pairwise_concordance(gt_scores, pred_scores), 3),
        "match_recall": round(match_recall(gt_aligned, pred_aligned), 3),
        "verdict_accuracy": round(verdict_accuracy(gt_aligned, pred_aligned), 3),
        "mae": round(mae(gt_aligned, pred_aligned), 2),
        "mean_bias": round(
            sum(p - g for p, g in zip(pred_scores, gt_scores)) / len(gt_scores), 2
        ) if gt_scores else 0,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(profile):
    path = os.path.join(SCRIPT_DIR, f"ground_truth_{profile.lower()}.json")
    if not os.path.exists(path):
        print(f"Ground truth not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_jobs_from_db(job_ids):
    db = psycopg2.connect(DB_DSN)
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, title, company, job_url, description FROM public.jobspy_jobs WHERE id = ANY(%s)",
            (job_ids,),
        )
        jobs = {
            r[0]: {"title": r[1], "company": r[2], "url": r[3] or "", "desc": r[4] or ""}
            for r in cur.fetchall()
        }
    db.close()
    return jobs


def load_resume(profile):
    db = psycopg2.connect(DB_DSN)
    with db.cursor() as cur:
        cur.execute(
            "SELECT resume_json FROM adm.resume WHERE profile = %s AND is_active = TRUE",
            (profile,),
        )
        row = cur.fetchone()
        resume = json.dumps(row[0]) if row else "{}"
    db.close()
    return resume


# ---------------------------------------------------------------------------
# Run eval
# ---------------------------------------------------------------------------

def run_eval(profile, variant_filter=None):
    gt = load_ground_truth(profile)
    job_ids = [e["job_id"] for e in gt]
    gt_map = {e["job_id"]: e for e in gt}

    print(f"Profile: {profile} | {len(gt)} ground truth jobs")
    jobs = load_jobs_from_db(job_ids)
    resume = load_resume(profile)
    print(f"Loaded {len(jobs)} jobs from DB\n")

    variants = build_eval_variants(profile)
    if variant_filter:
        variants = {k: v for k, v in variants.items() if k in variant_filter}

    # Extract phase
    cache_path = os.path.join(SCRIPT_DIR, f"extract_cache_{profile.lower()}.json")
    extract_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            extract_cache = json.load(f)

    print(f"=== PHASE 1: extract ({MODEL_8B}) ===\n")
    for job_id in job_ids:
        job = jobs.get(job_id)
        if not job:
            print(f"  {job_id} — not in DB, skipping")
            continue
        if job_id in extract_cache:
            print(f"  {job_id} — cached")
            continue
        prompt = EXTRACT_PROMPT.format(description=job["desc"])
        raw, ext_s = ollama(MODEL_8B, prompt, num_ctx=8192)
        parsed = parse_json(strip_think(raw))
        extract_cache[job_id] = {
            "text": raw,
            "parsed": parsed,
            "summary": parsed.get("summary", ""),
            "ext_s": ext_s,
        }
        print(f"  {job_id}  {ext_s}s  {parsed.get('title', '')[:50]}", flush=True)
        with open(cache_path, "w") as f:
            json.dump(extract_cache, f, indent=2)

    # Eval phase
    results_path = os.path.join(SCRIPT_DIR, f"results_{profile.lower()}.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    for v in variants:
        results.setdefault(v, [])

    avail = [j for j in job_ids if j in extract_cache and j in jobs]
    total = len(variants) * len(avail)
    done = sum(len(results.get(v, [])) for v in variants)
    print(f"\n=== PHASE 2: eval ({MODEL_14B}) — {len(variants)} variants x {len(avail)} jobs = {total} ({done} done) ===\n")

    for job_id in avail:
        job = jobs[job_id]
        ctrl = gt_map.get(job_id)
        ext_text = extract_cache[job_id]["text"]
        for v_name, tmpl in variants.items():
            if job_id in {r["job_id"] for r in results[v_name]}:
                continue
            num_ctx = 12288 if "fewshot" in v_name else 8192
            prompt = tmpl.format(candidate_json=resume, job_json=ext_text)
            raw, eval_s = ollama(MODEL_14B, prompt, num_ctx=num_ctx)
            ev = parse_json(strip_think(raw))
            verdict = ev.get("verdict", "?")
            scores = ev.get("match_scores", {})
            avg = avg_score(scores)
            ctrl_v = ctrl["verdict"] if ctrl else "?"
            ctrl_avg = ctrl["avg"] if ctrl else 0.0
            mark = "+" if verdict == ctrl_v else "x"
            print(
                f"  [{v_name}] {job['title'][:28]} @ {job['company'][:16]}"
                f"  {mark} {verdict:16} avg={avg:.1f} (gt {ctrl_avg:.1f} d={avg - ctrl_avg:+.1f})  {eval_s}s",
                flush=True,
            )
            results[v_name].append({
                "job_id": job_id,
                "title": job["title"],
                "company": job["company"],
                "eval_s": eval_s,
                "verdict": verdict,
                "scores": scores,
                "avg": avg,
                "job_in_one_line": ev.get("job_in_one_line", ""),
                "why_you_fit": ev.get("why_you_fit", ""),
                "key_gap": ev.get("key_gap", ""),
            })
            done += 1
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
        print(f"  [{job_id}] done — {done}/{total}", flush=True)

    print_report(profile, gt, results, variants)


# ---------------------------------------------------------------------------
# Score-only mode (re-compute metrics from existing results)
# ---------------------------------------------------------------------------

def score_only(profile, results_path):
    gt = load_ground_truth(profile)
    with open(results_path) as f:
        results = json.load(f)
    variants = {v: None for v in results}
    print_report(profile, gt, results, variants)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(profile, gt, results, variants):
    gt_map = {e["job_id"]: e for e in gt}
    v_names = list(variants.keys())

    print(f"\n\n{'=' * 90}")
    print(f"METRICS REPORT — {profile} ({len(gt)} ground truth jobs)")
    print(f"{'=' * 90}\n")

    # Score grid
    hdr = f"{'Job':<36} {'GT':>5}" + "".join(f"  {v[:14]:>14}" for v in v_names)
    print(hdr)
    print("-" * len(hdr))
    for gt_entry in gt:
        jid = gt_entry["job_id"]
        label = f"{gt_entry['title'][:18]}@{gt_entry.get('company', '')[:15]}"
        row = f"{label:<36} {gt_entry['avg']:>5.1f}"
        for v in v_names:
            r = next((x for x in results.get(v, []) if x["job_id"] == jid), None)
            if r:
                d = r["avg"] - gt_entry["avg"]
                mk = "+" if r["verdict"] == gt_entry["verdict"] else "x"
                row += f"  {r['avg']:>4.1f}({d:>+4.1f}){mk}"
            else:
                row += f"  {'N/A':>12}"
        print(row)
    print("-" * len(hdr))

    # Metrics table
    print(f"\n{'Metric':<25}" + "".join(f"  {v[:14]:>14}" for v in v_names))
    print("-" * (25 + 16 * len(v_names)))
    for metric_name in ["spearman_rho", "pairwise_concordance", "match_recall", "verdict_accuracy", "mae", "mean_bias"]:
        row = f"{metric_name:<25}"
        for v in v_names:
            pred_entries = []
            for r in results.get(v, []):
                pred_entries.append(r)
            m = compute_all_metrics(gt, pred_entries)
            val = m.get(metric_name, float("nan"))
            if isinstance(val, float):
                row += f"  {val:>14.3f}"
            else:
                row += f"  {val:>14}"
        print(row)

    # Verdict breakdown
    print(f"\nVerdict accuracy breakdown ({len(gt)} jobs):")
    for v in v_names:
        pred_map = {r["job_id"]: r["verdict"] for r in results.get(v, [])}
        correct = sum(1 for e in gt if pred_map.get(e["job_id"]) == e["verdict"])
        n = sum(1 for e in gt if e["job_id"] in pred_map)
        bar = "=" * correct + "-" * (n - correct)
        print(f"  {v:<26} {correct:>2}/{n}  [{bar}]")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval runner for job-fit pipeline")
    parser.add_argument("--profile", required=True, help="Profile name (Slava, Kezia)")
    parser.add_argument("--variant", nargs="*", help="Run only these variants (default: all)")
    parser.add_argument("--score-only", metavar="FILE", help="Skip eval, compute metrics from existing results JSON")
    args = parser.parse_args()

    if args.score_only:
        score_only(args.profile, args.score_only)
    else:
        run_eval(args.profile, variant_filter=args.variant)


if __name__ == "__main__":
    main()
