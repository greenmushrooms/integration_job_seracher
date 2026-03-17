import json


def format_job_message_telegram(job, index: int, total: int) -> str:
    """Format a single job posting for Telegram"""
    title, company, location, avg_score, match_scores_raw, reasoning_raw, job_url = job

    try:
        data = json.loads(reasoning_raw)
        scores = (
            json.loads(match_scores_raw)
            if isinstance(match_scores_raw, str)
            else match_scores_raw
        )
        comparisons = data.get("comparisons", [])
        tech = data.get("tech_stack", {})
        summary = data.get("summary", "Check details below.")
    except:
        scores = {}
        comparisons = []
        tech = {}
        summary = str(reasoning_raw)[:100] + "..."

    # 1. Comparisons
    comp_lines = ""
    for item in comparisons:
        me = item.get("me_val", "?")
        them = item.get("them_val", "?")
        icon = item.get("icon", "")
        cat = item.get("category", "🔹")
        comp_lines += f"{cat} <b>{me} → {them}</b> {icon}\n"

    # 2. Tech Stack
    stack_verdict = tech.get("verdict", "N/A")
    matches = ", ".join(tech.get("matches", [])[:5])
    gaps = ", ".join(tech.get("gaps", [])[:3])

    stack_section = f"🛠 <b>Stack:</b> {stack_verdict}\n"
    if matches:
        stack_section += f"   ➕ <i>{matches}</i>\n"
    if gaps:
        stack_section += f"   ➖ <i>{gaps}</i>\n"

    # 3. Score Breakdown (Expanded)
    # Allows you to see exactly where you lost points (e.g. Culture or Level)
    s_tech = scores.get("skills_match", "-")
    s_lvl = scores.get("career_level_alignment", "-")
    s_exp = scores.get("experience_relevance", "-")
    s_cult = scores.get("culture_fit", "-")

    score_line = f"Tech: {s_tech} | Lvl: {s_lvl} | Exp: {s_exp} | Cult: {s_cult}"

    message = f"""
📊 <b>JOB {index}/{total} | {avg_score:.1f}/10</b>

🏢 <b>{title}</b>
🏭 {company}

<b>⚔️ THE MATCH UP</b>
{comp_lines}{stack_section}
🎯 <b>SCORES:</b> {score_line}

💡 <b>TAKEAWAY:</b>
<i>{summary}</i>

🔗 <a href="{job_url}">Apply Now</a>
{"=" * 30}
""".strip()

    return message


def format_summary_message_telegram(jobs, run_name: str, profile: str = "") -> str:
    avg_of_avgs = sum(job[3] for job in jobs) / len(jobs) if jobs else 0
    profile_line = f"👤 <b>{profile}</b>\n" if profile else ""
    message = f"""
🎯 <b>JOB SEARCH RESULTS</b>
{profile_line}Run: <code>{run_name}</code>
Found: <b>{len(jobs)}</b> Matches | Avg: <b>{avg_of_avgs:.1f}/10</b>

<b>Top Matches:</b>
"""
    for i, job in enumerate(jobs[:3], 1):
        message += f"\n{i}. {job[0]} @ {job[1]} ({job[3]:.1f}/10)"

    message += f"\n\n📬 Details below..."
    return message
