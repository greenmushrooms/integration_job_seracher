import json


def format_job_message_telegram(job, index: int, total: int) -> str:
    """Format a single job posting for Telegram"""
    title, company, location, avg_score, match_scores, reasoning, job_url = job

    # Parse match_scores if it's a JSON string
    if isinstance(match_scores, str):
        match_scores = json.loads(match_scores)

    message = f"""
📊 <b>JOB {index}/{total} | Score: {avg_score}/10</b>

🏢 <b>{title}</b>
🏭 {company}
📍 {location}

⭐ <b>SCORES:</b>
- Skills: {match_scores.get("skills_match", "N/A")}
- Experience: {match_scores.get("experience_relevance", "N/A")}
- Keywords: {match_scores.get("keywords_ats", "N/A")}
- Level: {match_scores.get("career_level_alignment", "N/A")}
- Culture: {match_scores.get("soft_skills_cultural_fit", "N/A")}

💡 <b>WHY IT MATCHES:</b>
{reasoning}

🔗 <a href="{job_url}">Apply Now</a>

{"=" * 40}
""".strip()

    return message


def format_summary_message_telegram(jobs, run_name: str) -> str:
    """Create a summary message for Telegram"""
    avg_of_avgs = sum(job[3] for job in jobs) / len(jobs) if jobs else 0

    message = f"""
🎯 <b>JOB SEARCH RESULTS</b>
Run: <code>{run_name}</code>

Found <b>{len(jobs)}</b> high-quality matches!
Average Score: <b>{avg_of_avgs:.1f}/10</b>

<b>Top matches:</b>
"""
    for i, job in enumerate(jobs[:5], 1):
        message += f"\n{i}. {job[0]} at {job[1]} ({job[3]}/10)"

    message += f"\n\n📬 Details for all {len(jobs)} jobs coming below..."
    return message
