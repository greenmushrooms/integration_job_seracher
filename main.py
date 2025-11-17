import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from jobspy import scrape_jobs
from pandas import DataFrame
from prefect import flow, runtime, task
from prefect_dbt import PrefectDbtRunner, PrefectDbtSettings
from sqlalchemy import create_engine, text

from agent_eval import ClaudeJobEvaluator
from helper import format_job_message_telegram, format_summary_message_telegram

# Load environment variables
load_dotenv()

# Get database credentials
DB_HOST = os.getenv("DATABASE_HOST")
DB_PORT = os.getenv("DATABASE_PORT")
DB_NAME = os.getenv("DATABASE_NAME")
DB_USER = os.getenv("DATABSE_USER")
DB_PASSWORD = os.getenv("DATABSE_PASSWORD")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CONNECTION_STRING = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
ENGINE = create_engine(CONNECTION_STRING)


def send_telegram_message(message_text: str) -> bool:
    """Send a message via Telegram using simple requests"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, data=data)
        result = response.json()
        return result["ok"]
    except Exception as e:
        print(f"Error sending message: {e}")
        return False


@task()
def find_and_process(title: str, location: str, profile: str) -> DataFrame:
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "google"],
        search_term=title,
        google_search_term="Data engineer jobs near Toronto, Ontario since yesterday",
        location=location,
        results_wanted=20,
        hours_old=72,
        country_indeed="canada",
        linkedin_fetch_description=True,
    )
    jobs["sys_run_name"] = runtime.flow_run.name
    jobs["sys_profile"] = profile

    print(f"we are running flow {runtime.flow_run.name}")
    print(f"Found {len(jobs)} jobs")
    print(jobs.head())

    write_to_db(jobs, "jobspy", "import_jobs")
    return jobs.head()


def load_jobs(profile: str, limit: int = 3) -> pd.DataFrame:
    query = text("""
        SELECT id, description, title, company
        FROM public.jobspy_jobs j
        WHERE sys_profile = :profile
          AND NOT EXISTS (
              SELECT *
              FROM public.evaluated_jobs e
              WHERE e.job_id = j.id
          )
        ORDER BY id DESC
        LIMIT :limit
    """)

    with ENGINE.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"profile": profile, "limit": limit})

    return df


def load_resume(profile: str) -> str:
    query = text("""
        SELECT resume_body
        FROM adm.resume
        WHERE profile = :profile
          AND is_active = TRUE
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    with ENGINE.connect() as conn:
        result = conn.execute(query, {"profile": profile}).fetchone()

    if result:
        return result[0]
    else:
        raise ValueError(f"No active resume found for profile: {profile}")


def write_to_db(df: DataFrame, schema: str, table_name: str) -> None:
    df.to_sql(
        name=table_name,
        con=ENGINE,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
    )


@task()
def run_dbt():
    vars_json = f'{{"run_name": "{runtime.flow_run.name}"}}'
    cli_args = ["build", "--vars", vars_json]

    PrefectDbtRunner(
        settings=PrefectDbtSettings(
            project_dir="data__job_searcher", profiles_dir="data__job_searcher"
        )
    ).invoke(cli_args)


@flow()
def process_jobs(profile: str):
    resume = load_resume(profile)
    jobs_df = load_jobs(profile, limit=30)

    print(f"Loaded resume: {len(resume)} characters")
    print(f"Loaded {len(jobs_df)} jobs\n")

    evaluator = ClaudeJobEvaluator()
    results = evaluator.evaluate(resume, jobs_df)
    results["sys_run_name"] = runtime.flow_run.name
    results["sys_profile"] = profile

    write_to_db(results, "public", "evaluated_jobs")


@task()
def get_top_jobs(run_name: str, min_score: float = 7.5):
    """Retrieve top-scoring jobs from the database"""
    query = text("""
        SELECT
            j.title,
            j.company,
            j.location,
            e.avg_score,
            e.match_scores,
            e.reasoning,
            COALESCE(j.job_url_direct, j.job_url) as job_url
        FROM public.evaluated_jobs as e
        INNER JOIN public.jobspy_jobs as j
            ON e.job_id = j.id
        WHERE e.sys_run_name = :run_name
         -- AND e.avg_score > :min_score
        ORDER BY e.avg_score DESC
        limit 1
    """)

    with ENGINE.connect() as conn:
        result = conn.execute(query, {"run_name": run_name, "min_score": min_score})
        jobs = result.fetchall()

    print(f"Found {len(jobs)} jobs with score > {min_score}")
    return jobs


@task()
def send_telegram_notifications(jobs, run_name: str):
    """Send Telegram notifications for top jobs"""
    if not jobs:
        print("No jobs to send")
        return

    # Send summary first
    summary = format_summary_message_telegram(jobs, run_name)
    send_telegram_message(summary)
    print(f"Sent summary to Telegram")

    # Send individual job details
    for i, job in enumerate(jobs, 1):
        job_message = format_job_message_telegram(job, i, len(jobs))
        send_telegram_message(job_message)
        print(f"Sent job {i}/{len(jobs)} to Telegram")

    print(f"Successfully sent {len(jobs) + 1} Telegram messages")


@flow()
def notify_top_jobs(min_score: float = 7.5):
    """Flow to retrieve and send top job matches via SMS"""
    run_name = runtime.flow_run.name
    print(f"Notifying top jobs for run: {run_name}")

    jobs = get_top_jobs("graceful-chowchow", min_score)

    if jobs:
        send_telegram_notifications(jobs, run_name)
    else:
        print(f"No jobs found with score > {min_score}")

    return len(jobs)


@flow()
def get_jobs(
    title: str = "data engineer", location: str = "Toronto, On", profile: str = "Slava"
):
    # print(f"searching for {title} jobs in {location}")
    # thing = find_and_process(title=title, location=location, profile=profile)
    # run_dbt()
    # process_jobs(profile)
    notify_top_jobs(min_score=7.5)
    # return thing


if __name__ == "__main__":
    get_jobs()
