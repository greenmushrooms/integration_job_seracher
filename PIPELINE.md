# Job Searcher Pipeline

## Architecture

```
00:00 Toronto  →  load_jobs_flow
                  ├── for each profile in adm.job_search_config
                  │   └── for each title × location
                  │       └── scrape jobs → import_jobs
                  ├── dbt build (dedup → jobspy_jobs)
                  └── for each profile
                      └── push unevaluated jobs → llm_queue.tasks

overnight       →  llm-queue-worker (Ollama, local LLM)
                   processes job_eval tasks slowly

08:00 Toronto  →  notify_matches_flow
                  └── for each profile
                      ├── drain done queue tasks → evaluated_jobs
                      └── top matches (score ≥ 6.9, last 48h) → Telegram
```

## Adding a New Profile

No redeploy needed. Just two DB inserts:

```sql
-- 1. Add resume
INSERT INTO adm.resume (profile, resume_title, resume_body, telegram_chat_id, is_active)
VALUES ('Name', 'Title', '...resume text...', <telegram_chat_id>, TRUE);

-- 2. Add search config
INSERT INTO adm.job_search_config (profile, titles, locations, searches) VALUES (
    'Name',
    ARRAY['job title one', 'job title two'],
    ARRAY['Toronto, ON', 'New York, NY'],
    20  -- results per title×location combo
);
```

Both flows pick up all active profiles automatically on next run.

## Deployments

| Name | Schedule | Entrypoint | Purpose |
|---|---|---|---|
| `load-jobs` | `0 0 * * *` Toronto | `main.py:load_jobs_flow` | Scrape + push to queue |
| `notify-matches` | `0 8 * * *` Toronto | `main.py:notify_matches_flow` | Drain + Telegram |

> **Note:** `job-search-deployment` (old Claude-direct monolith) should be deleted from
> Prefect UI — it conflicts with the queue pipeline by consuming unevaluated jobs at 7 AM
> before notify-matches can drain them.

## Running the Migration

```bash
docker exec hub_db psql -U user_job_searcher -d job_searcher \
  -f /path/to/migrations/001_multi_profile_pipeline.sql
```

## Rebuilding & Redeploying

```bash
# 1. Rebuild image
docker build -t job-searcher:latest .

# 2. Redeploy both flows
docker run --rm --network project-hub-network \
  -e PREFECT_API_URL=http://prefect-server-dev:4200/api \
  job-searcher:latest \
  prefect deploy --all
```

## Search Config (`adm.job_search_config`)

| Profile | Titles | Locations | Searches/combo |
|---|---|---|---|
| Slava | `data engineer` | Toronto ON | 100 |
| Kezia | business analyst, senior BA, business systems analyst, technical BA, technology consultant, business consultant | Toronto ON, New York NY, Chicago IL, Dallas TX, Los Angeles CA | 20 |

Kezia: 6 titles × 5 locations × 20 searches = up to **600 raw jobs/night** (after dbt dedup, ~200–300 unique).
At ~5 min/job on CPU Ollama, allow 20–25 hrs. **Fix GPU first** to bring this to ~2–3 hrs.
