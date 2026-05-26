"""One-off: scrape + queue jobs for Cait only (subset of load_jobs_flow)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, runtime

from main import (
    find_and_process,
    load_jobs,
    load_resume,
    load_search_configs,
    run_dbt,
    _push_profile_to_queue,
)


@flow(name="load_jobs_flow_cait")
def load_jobs_flow_cait():
    configs = [c for c in load_search_configs() if c["profile"] == "Cait"]
    if not configs:
        print("No active Cait config found")
        return

    config = configs[0]
    profile = config["profile"]
    print(f"Running for {profile}: {config['titles']} x {config['locations']} ({config['searches']})")

    for title in config["titles"]:
        for location in config["locations"]:
            find_and_process(
                title=title,
                location=location,
                profile=profile,
                searches=config["searches"],
            )

    run_dbt()

    run_name = runtime.flow_run.name
    resume, _ = load_resume(profile)
    cap = len(config["titles"]) * len(config["locations"]) * config["searches"] * 2
    jobs_df = load_jobs(profile, limit=cap)

    if jobs_df.empty:
        print(f"No unevaluated jobs for {profile}")
        return

    _push_profile_to_queue(profile, resume, jobs_df, run_name)


if __name__ == "__main__":
    load_jobs_flow_cait()
