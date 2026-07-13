-- Per-profile Telegram alert threshold. NULL = fall back to the
-- notify_matches_flow min_score parameter (deployment default 6.9).
-- Lives on adm.job_search_config (pipeline-owned, never web-edited),
-- same as blocklist/exact_blocklist.
ALTER TABLE adm.job_search_config ADD COLUMN IF NOT EXISTS notify_min_score real;
