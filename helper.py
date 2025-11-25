import json


def format_job_message_telegram(job, index: int, total: int) -> str:
    title, company, location, avg_score, match_scores, reasoning_json, job_url = job

    # Parse scores
    scores = json.loads(match_scores) if isinstance(match_scores, str) else match_scores

    # Parse comparisons
    try:
        comparisons = json.loads(reasoning_json)
        # Build the polite list
        comp_text = ""
        for item in comparisons:
            # Output: 💰 Salary: Significantly Higher (✅)
            cat_map = {"Salary": "💰", "Level": "🧗", "Stack": "🛠", "Remote": "🏠"}
            emoji = cat_map.get(item["category"], "🔹")

            comp_text += (
                f"{emoji} <b>{item['category']}:</b> {item['verdict']} {item['icon']}\n"
            )

    except:
        comp_text = "No comparison data available."

    message = f"""
📊 <b>JOB {index}/{total} | Score: {avg_score:.1f}/10</b>

🏢 <b>{title}</b>
🏭 {company}

<b>⚔️ THE MATCH UP:</b>
{comp_text}
⭐ <b>DETAILS:</b>
Stack Match: {scores.get("skills_match")}/10
Career Move: {scores.get("career_level_alignment")}/10

🔗 <a href="{job_url}">Apply Now</a>
{"=" * 30}
""".strip()

    return message


def format_summary_message_telegram(jobs, run_name: str) -> str:
    """Create a summary message for Telegram"""
    avg_of_avgs = sum(job[3] for job in jobs) / len(jobs) if jobs else 0

    message = f"""
🎯 <b>JOB SEARCH RESULTS</b>
Run: <code>{run_name}</code>

Found <b>{len(jobs)}</b> matches > 7.5
Avg Score: <b>{avg_of_avgs:.1f}/10</b>

<b>Top 3:</b>
"""
    for i, job in enumerate(jobs[:3], 1):
        message += f"\n{i}. {job[0]} ({job[3]}/10)"

    message += f"\n\n📬 Details coming below..."
    return message
