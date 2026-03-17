-- Multi-profile queue pipeline migration
-- Creates job_search_config table and adds created_at to evaluated_jobs
--
-- NOTE: adm schema is owned by hub_user, not user_job_searcher.
-- Run CREATE TABLE and GRANT as hub_user, remaining statements as user_job_searcher.
--
-- As hub_user:
--   docker exec hub_db psql -U hub_user -d job_searcher -f this_file.sql
-- Then re-run inserts + ALTER as user_job_searcher if needed.

-- Search config per profile (arrays of titles and locations)
CREATE TABLE IF NOT EXISTS adm.job_search_config (
    profile   text PRIMARY KEY,
    titles    text[]  NOT NULL,
    locations text[]  NOT NULL,
    searches  int     NOT NULL DEFAULT 20
);

INSERT INTO adm.job_search_config (profile, titles, locations, searches) VALUES
(
    'Slava',
    ARRAY['data engineer'],
    ARRAY['Toronto, ON'],
    100
),
(
    'Kezia',
    ARRAY[
        'business analyst',
        'senior business analyst',
        'business systems analyst',
        'technical business analyst',
        'technology consultant',
        'business consultant'
    ],
    ARRAY[
        'Toronto, ON',
        'New York, NY',
        'Chicago, IL',
        'Dallas, TX',
        'Los Angeles, CA'
    ],
    20
)
ON CONFLICT (profile) DO NOTHING;

GRANT SELECT, INSERT, UPDATE ON adm.job_search_config TO user_job_searcher;

-- Add created_at so notify flow can filter to jobs evaluated today
ALTER TABLE public.evaluated_jobs
    ADD COLUMN IF NOT EXISTS created_at timestamp DEFAULT NOW();
