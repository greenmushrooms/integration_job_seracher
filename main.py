import json
import os
from datetime import datetime, timedelta

from rapidfuzz import fuzz

import pandas as pd
import requests
from jobspy import scrape_jobs
from pandas import DataFrame
from prefect import flow, runtime, task
from prefect_dbt import PrefectDbtRunner, PrefectDbtSettings
from sqlalchemy import create_engine, text

from agent_eval import ClaudeJobEvaluator
from helper import format_job_message_telegram, format_summary_message_telegram

USE_QUEUE = os.getenv("USE_QUEUE", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Profile-specific fewshot eval prompt builder (v3_fewshot variant)
# ---------------------------------------------------------------------------

_FEWSHOT_EXAMPLES = {
    "Slava": """
Example 1 — Pivot:
Job: {"title": "Pharmacy Technician", "summary": "Dispense medications, manage inventory, assist pharmacists in a retail pharmacy setting. Requires pharmacy technician certification."}
{"verdict": "Pivot", "match_scores": {"skills_match": 1, "career_level_alignment": 4, "experience_relevance": 1, "culture_fit": 3}, "job_in_one_line": "Retail pharmacy technician role dispensing medications", "why_you_fit": "No relevant overlap with data engineering background", "key_gap": "Completely different domain and function — requires pharmacy certification"}

Example 2 — Lateral:
Job: {"title": "Senior Data Engineer", "summary": "Build and maintain cloud data pipelines using Spark, Python, and dbt. Design data models, mentor junior engineers, collaborate with analytics teams."}
{"verdict": "Lateral", "match_scores": {"skills_match": 9, "career_level_alignment": 7, "experience_relevance": 9, "culture_fit": 7}, "job_in_one_line": "Senior DE role building data pipelines with dbt and Spark", "why_you_fit": "Direct stack match — Python, dbt, cloud pipelines, mentoring all align with current experience", "key_gap": "Senior title is one level below current Lead DE role"}

Example 3 — Step Up:
Job: {"title": "Staff Data Engineer / Tech Lead", "summary": "Technical lead for data platform engineering team. Own architecture decisions, lead 5-8 engineers, set engineering standards across the org, interface with VP-level stakeholders."}
{"verdict": "Step Up", "match_scores": {"skills_match": 8, "career_level_alignment": 9, "experience_relevance": 8, "culture_fit": 7}, "job_in_one_line": "Staff-level DE tech lead owning platform architecture and team leadership", "why_you_fit": "Current Lead DE experience with mentoring and architecture directly prepares for Staff-level ownership", "key_gap": "Larger team scope and VP-level stakeholder management is a stretch"}""",

    "Kezia": """
Example 1 — Pivot:
Job: {"title": "Electrician Apprentice", "summary": "Install and maintain electrical systems in residential and commercial buildings. Requires electrical apprenticeship certification and physical site work."}
{"verdict": "Pivot", "match_scores": {"skills_match": 1, "career_level_alignment": 3, "experience_relevance": 1, "culture_fit": 2}, "job_in_one_line": "Trades apprenticeship in electrical installation and maintenance", "why_you_fit": "No relevant overlap with business analysis or Salesforce background", "key_gap": "Completely different domain — requires trades certification and physical site work"}

Example 2 — Lateral:
Job: {"title": "Business Systems Analyst", "summary": "Gather requirements from stakeholders, document processes, manage Salesforce CRM configuration, support Agile delivery teams with user stories and UAT testing."}
{"verdict": "Lateral", "match_scores": {"skills_match": 9, "career_level_alignment": 7, "experience_relevance": 9, "culture_fit": 7}, "job_in_one_line": "BSA role with Salesforce, Agile delivery and requirements gathering", "why_you_fit": "Direct match — Salesforce, Agile, user stories, UAT are core to current role at MLSE", "key_gap": "Same seniority level with no meaningful career step up"}

Example 3 — Step Up:
Job: {"title": "Lead Business Analyst / Product Owner", "summary": "Lead a team of BAs, own the product backlog, drive roadmap decisions with C-suite stakeholders, accountable for delivery across multiple workstreams."}
{"verdict": "Step Up", "match_scores": {"skills_match": 8, "career_level_alignment": 9, "experience_relevance": 8, "culture_fit": 7}, "job_in_one_line": "Lead BA/PO role with team ownership and executive stakeholder management", "why_you_fit": "Strong BA foundation with Agile and SDLC experience positions well for team lead ownership", "key_gap": "Managing a team of BAs and C-suite accountability is a meaningful stretch beyond current scope"}""",
}

def build_eval_prompt(profile: str) -> str:
    """Build profile-specific v3_fewshot eval prompt.

    Uses {{placeholder}} syntax for Go router variable substitution.
    Examples are embedded inline — no Python .format() so no brace escaping needed.
    """
    examples = _FEWSHOT_EXAMPLES.get(profile, "")
    return (
        "You are an honest career advisor.\n\n"
        + examples
        + "\n\n--- NOW EVALUATE ---\n"
        "Candidate:\n{{candidate_json}}\n\n"
        "Job:\nTitle: {{title}}\nSummary: {{summary}}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "verdict": "Step Up" | "Lateral" | "Title Regression" | "Pivot",\n'
        '  "match_scores": {\n'
        '    "skills_match": <1-10>,\n'
        '    "career_level_alignment": <1-10>,\n'
        '    "experience_relevance": <1-10>,\n'
        '    "culture_fit": <1-10>\n'
        '  },\n'
        '  "job_in_one_line": "what this role does and its domain/industry",\n'
        '  "why_you_fit": "strongest overlap between candidate and this role in one sentence",\n'
        '  "key_gap": "biggest mismatch, risk, or missing requirement in one sentence"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_engine():
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return create_engine(connection_string)


def get_queue_engine():
    dsn = os.getenv("LLM_QUEUE_DSN")
    return create_engine(dsn)


def write_to_db(df: DataFrame, schema: str, table_name: str) -> None:
    df.to_sql(
        name=table_name,
        con=get_db_engine(),
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
    )


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram_message(message_text: str, chat_id: str) -> bool:
    """Send a message via Telegram using simple requests"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, data=data)
        result = response.json()
        if not result["ok"]:
            print(f"Telegram rejected message: {result.get('description')} — preview: {message_text[:120]}")
        return result["ok"]
    except Exception as e:
        print(f"Error sending message: {e}")
        return False


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

CA_PROVINCES = {
    "ON", "BC", "AB", "QC", "MB", "SK", "NS", "NB", "PE", "NL",
    "Ontario", "British Columbia", "Alberta", "Quebec", "Manitoba",
    "Saskatchewan", "Nova Scotia", "New Brunswick", "Newfoundland",
}


UK_TERMS = {"United Kingdom", "UK", "England", "Scotland", "Wales", "London"}


def _country_for_location(location: str) -> str:
    for term in UK_TERMS:
        if term in location:
            return "uk"
    for province in CA_PROVINCES:
        if province in location:
            return "canada"
    return "usa"


# ---------------------------------------------------------------------------
# Profile / config helpers
# ---------------------------------------------------------------------------

def load_search_configs() -> list[dict]:
    """Return search config for all active profiles from adm.job_search_config."""
    query = text("""
        SELECT c.profile, c.titles, c.locations, c.searches
        FROM adm.job_search_config c
        JOIN adm.resume r ON r.profile = c.profile AND r.is_active = TRUE
    """)
    with get_db_engine().connect() as conn:
        rows = conn.execute(query).fetchall()
    return [
        {"profile": r[0], "titles": r[1], "locations": r[2], "searches": r[3]}
        for r in rows
    ]


def load_resume(profile: str) -> tuple[str, str]:
    """Load (resume_body, telegram_chat_id) for a profile.
    Returns resume_json as string if available, otherwise raw resume_body."""
    query = text("""
        SELECT resume_body, telegram_chat_id, resume_json
        FROM adm.resume
        WHERE profile = :profile AND is_active = TRUE
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    with get_db_engine().connect() as conn:
        result = conn.execute(query, {"profile": profile}).fetchone()
    if result:
        resume_body, telegram_chat_id, resume_json = result
        resume = json.dumps(resume_json) if resume_json else resume_body
        return resume, telegram_chat_id
    raise ValueError(f"No active resume found for profile: {profile}")


def load_telegram_chat_id(profile: str) -> str:
    query = text("""
        SELECT telegram_chat_id
        FROM adm.resume
        WHERE profile = :profile AND is_active = TRUE
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    with get_db_engine().connect() as conn:
        result = conn.execute(query, {"profile": profile}).fetchone()
    if result and result[0]:
        return result[0]
    raise ValueError(f"No telegram_chat_id found for profile: {profile}")


def _is_blocked(title: str, blocklist: list[str], threshold: int = 85) -> bool:
    title_lower = title.lower()
    return any(
        fuzz.partial_ratio(title_lower, term.lower()) >= threshold
        for term in blocklist
    )


def load_jobs(profile: str, limit: int = 3, hours: int = 72) -> pd.DataFrame:
    query = text("""
        SELECT j.id, j.description, j.title, j.company,
               COALESCE(c.blocklist, '{}'::text[]) AS blocklist
        FROM public.jobspy_jobs j
        LEFT JOIN adm.job_search_config c ON c.profile = j.sys_profile
        WHERE j.sys_profile = :profile
          AND j.date_posted >= CURRENT_DATE - MAKE_INTERVAL(hours => :hours)
          AND NOT EXISTS (
              SELECT 1 FROM public.evaluated_jobs e
              WHERE e.job_id = j.id
          )
        ORDER BY j.id DESC
        LIMIT :limit
    """)
    with get_db_engine().connect() as conn:
        df = pd.read_sql_query(query, conn, params={"profile": profile, "limit": limit, "hours": hours})

    if df.empty:
        return df

    blocklist = df["blocklist"].iloc[0] or []
    if blocklist:
        before = len(df)
        df = df[~df["title"].apply(lambda t: _is_blocked(str(t), blocklist))]
        blocked = before - len(df)
        if blocked:
            print(f"Blocked {blocked} jobs for {profile} via fuzzy blocklist")

    return df.drop(columns=["blocklist"])


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------

@task()
def find_and_process(
    title: str, location: str, profile: str, searches: int = 70
) -> DataFrame:
    country = _country_for_location(location)
    try:
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "google"],
            search_term=title,
            google_search_term=f"{title} jobs near {location} since yesterday",
            location=location,
            results_wanted=searches,
            hours_old=72,
            country_indeed=country,
            linkedin_fetch_description=True,
        )
    except Exception as e:
        print(f"WARNING: scrape failed for '{title}' in {location}: {e}")
        return DataFrame()
    jobs["sys_run_name"] = runtime.flow_run.name
    jobs["sys_profile"] = profile

    print(f"Found {len(jobs)} jobs for '{title}' in {location}")
    write_to_db(jobs, "jobspy", "import_jobs")
    return jobs.head()


@task()
def run_dbt():
    vars_json = f'{{"run_name": "{runtime.flow_run.name}"}}'
    cli_args = ["build", "--vars", vars_json]
    PrefectDbtRunner(
        settings=PrefectDbtSettings(
            project_dir="data__job_searcher", profiles_dir="data__job_searcher"
        )
    ).invoke(cli_args)


@task()
def send_telegram_notifications(region_jobs, run_name: str, chat_id: str, profile: str = "", total_evaluated: int = 0):
    if not region_jobs:
        print("No jobs to send")
        return
    # region_jobs is list of (region, job_row)
    jobs = [job for _, job in region_jobs]
    summary = format_summary_message_telegram(region_jobs, run_name, profile, total_evaluated)
    send_telegram_message(summary, chat_id)

    current_region = None
    idx = 0
    for region, job in region_jobs:
        if region != current_region:
            current_region = region
            idx = 0
            region_count = sum(1 for r, _ in region_jobs if r == region)
            header = f"{REGION_EMOJI.get(region, '🌍')} <b>{REGION_LABEL.get(region, region.title())}</b> — {region_count} match{'es' if region_count != 1 else ''}"
            send_telegram_message(header, chat_id)
        idx += 1
        job_message = format_job_message_telegram(job, idx, sum(1 for r, _ in region_jobs if r == region))
        send_telegram_message(job_message, chat_id)
    print(f"Sent Telegram messages for {profile}: {len(jobs)} jobs across regions")


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def _push_profile_to_queue(
    profile: str, resume: str, jobs_df: pd.DataFrame, run_name: str
) -> int:
    """Push unevaluated jobs for a profile onto the LLM queue as two-stage DAG.

    Pushes to job_extract (7B) which auto-creates job_eval (14B) with depends_on.
    The scheduler sees the DAG and batches by model to minimize swaps.
    """
    from llm_queue import LLMQueueClient

    dsn = os.getenv("LLM_QUEUE_DSN")
    worker_url = os.getenv("LLM_QUEUE_WORKER_URL")

    payloads = [
        {
            "job_id": str(job.get("id")),
            "company": job.get("company", ""),
            "title": job.get("title", ""),
            "sys_profile": profile,
            "sys_run_name": run_name,
            "pack_id": run_name,
            "inputs": {
                "description": job.get("description", ""),
            },
            "next_step": {
                "topic": "job_eval",
                "step": "eval",
                "prompt": build_eval_prompt(profile),
                "inputs": {"candidate_json": resume},
            },
        }
        for _, job in jobs_df.iterrows()
    ]

    client = LLMQueueClient(dsn=dsn, worker_url=worker_url)
    task_ids = client.push_batch("job_extract", payloads)
    print(f"Pushed {len(task_ids)} jobs for {profile} to job_extract queue")
    return len(task_ids)


def _drain_queue_results(profile: str, run_name: str) -> list[str]:
    """
    Read all 'done' queue tasks for a profile not yet in evaluated_jobs.
    Writes results to evaluated_jobs. Returns list of written job_ids.
    """
    topic = "job_eval"

    # Existing evaluated job_ids for this profile
    existing_query = text(
        "SELECT job_id FROM public.evaluated_jobs WHERE sys_profile = :profile"
    )
    with get_db_engine().connect() as conn:
        existing = {row[0] for row in conn.execute(existing_query, {"profile": profile})}

    # Done tasks from queue DB
    done_query = text("""
        SELECT payload, result
        FROM llm_queue.tasks
        WHERE topic = :topic AND status = 'done'
          AND payload->>'sys_profile' = :profile
    """)
    with get_queue_engine().connect() as conn:
        rows = conn.execute(done_query, {"topic": topic, "profile": profile}).fetchall()

    new_rows = [
        (payload, result) for payload, result in rows
        if payload.get("job_id") not in existing
    ]

    if not new_rows:
        print(f"No new queue results for {profile} ({len(rows)} total done in queue)")
        return []

    CANONICAL_KEYS = {"skills_match", "career_level_alignment", "experience_relevance", "culture_fit"}

    result_rows = []
    written_ids = []
    for payload, result in new_rows:
        scores = result.get("match_scores", {})
        # Only average the 4 canonical keys — ignore any extra keys the model invented
        canonical = {k: v for k, v in scores.items() if k in CANONICAL_KEYS}
        avg_score = sum(canonical.values()) / len(canonical) if canonical else 0
        result_rows.append({
            "job_id": payload["job_id"],
            "avg_score": avg_score,
            "match_scores": json.dumps(canonical),
            "reasoning": json.dumps({
                "verdict": result.get("verdict"),
                "summary": result.get("job_in_one_line") or result.get("one_line_summary") or result.get("summary"),
                "why_you_fit": result.get("why_you_fit"),
                "key_gap": result.get("key_gap"),
            }),
            "sys_run_name": run_name,
            "sys_profile": profile,
        })
        written_ids.append(payload["job_id"])

    write_to_db(pd.DataFrame(result_rows), "public", "evaluated_jobs")
    print(f"Wrote {len(result_rows)} results for {profile} to evaluated_jobs")
    return written_ids


REGION_EMOJI = {"canada": "🇨🇦", "uk": "🇬🇧", "usa": "🇺🇸"}
REGION_LABEL = {"canada": "Canada", "uk": "United Kingdom", "usa": "United States"}


def _get_top_jobs_for_profile(
    profile: str, min_score: float = 6.9, hours: int = 48, per_region: int = 10
):
    """Top scored jobs per region for a profile, unsent, within the last N hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    query = text("""
        SELECT
            j.title, j.company, j.location,
            e.avg_score, e.match_scores, e.reasoning,
            COALESCE(j.job_url_direct, j.job_url) as job_url,
            e.job_id
        FROM public.evaluated_jobs e
        INNER JOIN public.jobspy_jobs j ON e.job_id = j.id
        WHERE e.sys_profile = :profile
          AND e.avg_score >= :min_score
          AND e.created_at >= :since
          AND e.notified_at IS NULL
          AND j.date_posted >= CURRENT_DATE - INTERVAL '3 days'
        ORDER BY e.avg_score DESC
        LIMIT 200
    """)
    with get_db_engine().connect() as conn:
        all_jobs = conn.execute(
            query, {"profile": profile, "min_score": min_score, "since": since}
        ).fetchall()

    # Group top N per region
    buckets: dict[str, list] = {"canada": [], "uk": [], "usa": []}
    for job in all_jobs:
        region = _country_for_location(job[2] or "")
        if len(buckets[region]) < per_region:
            buckets[region].append(job)

    # Return ordered: canada → uk → usa, preserving region info
    result = []
    for region in ("canada", "uk", "usa"):
        result.extend((region, job) for job in buckets[region])
    return result


def _mark_jobs_notified(job_ids: list[str]) -> None:
    if not job_ids:
        return
    query = text("""
        UPDATE public.evaluated_jobs
        SET notified_at = NOW()
        WHERE job_id = ANY(:job_ids)
    """)
    with get_db_engine().connect() as conn:
        conn.execute(query, {"job_ids": job_ids})
        conn.commit()


# ---------------------------------------------------------------------------
# Multi-profile queue pipeline (primary / overnight pattern)
# ---------------------------------------------------------------------------

@flow()
def load_jobs_flow():
    """
    Scrape jobs for ALL active profiles and push to LLM queue.
    Schedule: 6 PM Toronto — queue drains overnight with local LLM.
    """
    configs = load_search_configs()
    print(f"Running for {len(configs)} profiles: {[c['profile'] for c in configs]}")

    for config in configs:
        for title in config["titles"]:
            for location in config["locations"]:
                find_and_process(
                    title=title,
                    location=location,
                    profile=config["profile"],
                    searches=config["searches"],
                )

    run_dbt()

    run_name = runtime.flow_run.name
    for config in configs:
        profile = config["profile"]
        resume, _ = load_resume(profile)
        cap = len(config["titles"]) * len(config["locations"]) * config["searches"] * 2
        jobs_df = load_jobs(profile, limit=cap)

        if jobs_df.empty:
            print(f"No unevaluated jobs for {profile}")
            continue

        _push_profile_to_queue(profile, resume, jobs_df, run_name)


@flow()
def notify_matches_flow(min_score: float = 6.9):
    """
    Drain LLM queue results for ALL active profiles and send Telegram alerts.
    Schedule: 8 AM Toronto — after overnight queue processing.
    """
    configs = load_search_configs()
    run_name = runtime.flow_run.name

    for config in configs:
        profile = config["profile"]
        print(f"Processing {profile}...")

        written = _drain_queue_results(profile, run_name)
        print(f"Drained {len(written)} new results for {profile}")

        jobs = _get_top_jobs_for_profile(profile=profile, min_score=min_score)
        chat_id = load_telegram_chat_id(profile)

        if jobs:
            job_ids = [job[-1] for _, job in jobs]
            send_telegram_notifications(jobs, run_name, chat_id, profile, total_evaluated=len(written))
            _mark_jobs_notified(job_ids)
        else:
            print(f"No qualifying jobs (>= {min_score}) for {profile} ({len(written)} evaluated)")


# ---------------------------------------------------------------------------
# Legacy single-shot flow (Claude direct, no queue) — kept for ad-hoc runs
# ---------------------------------------------------------------------------

@task()
def get_top_jobs(run_name: str, min_score: float = 7.5):
    query = text("""
        SELECT
            j.title, j.company, j.location,
            e.avg_score, e.match_scores, e.reasoning,
            COALESCE(j.job_url_direct, j.job_url) as job_url
        FROM public.evaluated_jobs e
        INNER JOIN public.jobspy_jobs j ON e.job_id = j.id
        WHERE e.sys_run_name = :run_name AND e.avg_score >= :min_score
        ORDER BY e.avg_score DESC
    """)
    with get_db_engine().connect() as conn:
        result = conn.execute(query, {"run_name": run_name, "min_score": min_score})
        jobs = result.fetchall()
    print(f"Found {len(jobs)} jobs with score >= {min_score}")
    return jobs


@flow()
def process_jobs(profile: str, run_name: str = None, searches: int = 20):
    sys_run_name = run_name or runtime.flow_run.name
    resume, _ = load_resume(profile)
    jobs_df = load_jobs(profile, limit=searches)

    print(f"Loaded resume: {len(resume)} characters, {len(jobs_df)} jobs")

    if USE_QUEUE:
        _push_profile_to_queue(profile, resume, jobs_df, sys_run_name)
    else:
        evaluator = ClaudeJobEvaluator()
        results = evaluator.evaluate(resume, jobs_df)
        results["sys_run_name"] = sys_run_name
        results["sys_profile"] = profile
        write_to_db(results, "public", "evaluated_jobs")


@flow()
def notify_top_jobs(profile: str, min_score: float = 6.9, run_name: str = None):
    sys_run_name = run_name or runtime.flow_run.name
    chat_id = load_telegram_chat_id(profile)
    jobs = get_top_jobs(sys_run_name, min_score)
    if jobs:
        send_telegram_notifications(jobs, sys_run_name, chat_id, profile)
    return len(jobs)


@flow()
def get_jobs(
    title: str = "Director of Finance",
    locations: list[str] = ["Mississauga, On"],
    profile: str = "Ray",
    searches: int = 30,
    min_score: float = 7.5,
):
    parent_run_name = runtime.flow_run.name
    for location in locations:
        find_and_process(title=title, location=location, profile=profile, searches=searches)
    run_dbt()
    process_jobs(profile, parent_run_name, searches * len(locations) * 2)
    notify_top_jobs(profile=profile, min_score=min_score, run_name=parent_run_name)
    return parent_run_name


if __name__ == "__main__":
    get_jobs()
