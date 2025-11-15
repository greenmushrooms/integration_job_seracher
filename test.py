import json
import os

import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame
from sqlalchemy import create_engine, text

from agent_eval import ClaudeJobEvaluator

load_dotenv()
DB_HOST = os.getenv("DATABASE_HOST")
DB_PORT = os.getenv("DATABASE_PORT")
DB_NAME = os.getenv("DATABASE_NAME")
DB_USER = os.getenv("DATABSE_USER")
DB_PASSWORD = os.getenv("DATABSE_PASSWORD")


def load_resume(profile: str) -> str:
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)

    query = text("""
        SELECT resume_body
        FROM adm.resume
        WHERE profile = :profile
          AND is_active = TRUE
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"profile": profile}).fetchone()

    if result:
        return result[0]  # fetchone() returns a tuple, get first element
    else:
        raise ValueError(f"No active resume found for profile: {profile}")


def load_jobs(profile: str, limit: int = 3) -> pd.DataFrame:
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)

    query = text("""
        SELECT id, description, title, company
        FROM public.jobspy_jobs
        WHERE sys_profile = :profile
        ORDER BY id desc
        LIMIT :limit

    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"profile": profile, "limit": limit})

    return df


def write_to_db(jobs: DataFrame, schema: str, table_name: str) -> None:
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)

    jobs.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
    )
    return


resume = load_resume("Slava")
jobs_df = load_jobs("Slava", limit=3)

print(f"Loaded resume: {len(resume)} characters")
print(f"Loaded {len(jobs_df)} jobs\n")

# Evaluate jobs
evaluator = ClaudeJobEvaluator()
results = evaluator.evaluate(resume, jobs_df)

write_to_db(results, "public", "evaluated_jobs")
