import os
from jobspy import scrape_jobs
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pandas import DataFrame
from prefect import flow, task
from prefect_dbt import PrefectDbtRunner, PrefectDbtSettings


# Load environment variables
load_dotenv()

# Get database credentials
DB_HOST = os.getenv("DATABASE_HOST")
DB_PORT = os.getenv("DATABASE_PORT")
DB_NAME = os.getenv("DATABASE_NAME")
DB_USER = os.getenv("DATABSE_USER")
DB_PASSWORD = os.getenv("DATABSE_PASSWORD")
SCHEMA_NAME = "jobspy"


@task()
def find_and_process(title: str, location: str) -> DataFrame:
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "google"],  # "zip_recruiter",
        search_term=title,
        google_search_term="Data engineer jobs near Toronto, Ontario since yesterday",
        location=location,
        results_wanted=20,
        hours_old=72,
        country_indeed="canada",
        linkedin_fetch_description=True,
    )
    print(f"Found {len(jobs)} jobs")
    print(jobs.head())
    write_jobs(jobs)
    return jobs.head()


def write_jobs(jobs: DataFrame) -> None:
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)

    table_name = "jobs"

    jobs.to_sql(
        name=table_name,
        con=engine,
        schema=SCHEMA_NAME,
        if_exists="append",
        index=False,
        method="multi",
    )
    return


@task()
def run_dbt():
    PrefectDbtRunner(
        settings=PrefectDbtSettings(project_dir="test", profiles_dir="examples/run_dbt")
    ).invoke(["build"])


@flow()
def get_jobs(title: str = "data engineer", location: str = "Toronto, On"):
    print(f"seraching for {title} jobs into {location}")
    thing = find_and_process(title=title, location=location)
    return thing


if __name__ == "__main__":
    get_jobs()
