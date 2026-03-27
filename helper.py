import json
from html import escape


def format_job_message_telegram(job, index: int, total: int) -> str:
    """Format a single job posting for Telegram"""
    title, company, location, avg_score, match_scores_raw, reasoning_raw, job_url, *_ = job
    title = escape(str(title or ""))
    company = escape(str(company or ""))

    try:
        data = json.loads(reasoning_raw)
        scores = (
            json.loads(match_scores_raw)
            if isinstance(match_scores_raw, str)
            else match_scores_raw
        )
        verdict = data.get("verdict", "")
        job_desc = data.get("summary") or "Check details below."
        why_fit = data.get("why_you_fit") or ""
        key_gap = data.get("key_gap") or ""
    except:
        scores = {}
        verdict = ""
        job_desc = str(reasoning_raw)[:100] + "..."
        why_fit = ""
        key_gap = ""

    s_tech = scores.get("skills_match", "-")
    s_lvl = scores.get("career_level_alignment", "-")
    s_exp = scores.get("experience_relevance", "-")
    s_cult = scores.get("culture_fit", "-")

    VERDICT_ICON = {"Step Up": "⬆️", "Lateral": "↔️", "Title Regression": "⬇️", "Pivot": "↩️"}
    verdict_line = f"{VERDICT_ICON.get(verdict, '❓')} {verdict}" if verdict else ""
    score_line = f"Tech: {s_tech} | Lvl: {s_lvl} | Exp: {s_exp} | Cult: {s_cult}"

    fit_section = ""
    if why_fit:
        fit_section += f"✅ <i>{escape(why_fit)}</i>\n"
    if key_gap:
        fit_section += f"⚠️ <i>{escape(key_gap)}</i>\n"

    message = f"""
📊 <b>JOB {index}/{total} | {avg_score:.1f}/10</b>  {verdict_line}

🏢 <b>{title}</b>
🏭 {company}

📋 {escape(job_desc)}

{fit_section}🎯 {score_line}

🔗 <a href="{job_url}">Apply Now</a>
{"=" * 30}
""".strip()

    return message


REGION_EMOJI = {"canada": "🇨🇦", "uk": "🇬🇧", "usa": "🇺🇸"}
REGION_LABEL = {"canada": "Canada", "uk": "United Kingdom", "usa": "United States"}


def format_summary_message_telegram(region_jobs, run_name: str, profile: str = "", total_evaluated: int = 0) -> str:
    # region_jobs is list of (region, job_row)
    jobs = [job for _, job in region_jobs]
    avg_of_avgs = sum(job[3] for job in jobs) / len(jobs) if jobs else 0
    profile_line = f"👤 <b>{profile}</b>\n" if profile else ""
    presented = len(jobs)

    # Per-region counts
    region_counts: dict[str, int] = {}
    for region, _ in region_jobs:
        region_counts[region] = region_counts.get(region, 0) + 1

    region_lines = ""
    for region in ("canada", "uk", "usa"):
        count = region_counts.get(region, 0)
        if count:
            region_lines += f"\n{REGION_EMOJI[region]} {REGION_LABEL[region]}: <b>{count}</b>"

    message = f"""
🎯 <b>JOB SEARCH RESULTS</b>
{profile_line}Run: <code>{run_name}</code>
Evaluated: <b>{total_evaluated}</b> | Presented: <b>{presented}</b> | Avg: <b>{avg_of_avgs:.1f}/10</b>
{region_lines}

📬 Details by region below...
""".strip()
    return message
