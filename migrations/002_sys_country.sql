-- Track scrape country alongside each job so bucketing doesn't depend on
-- parsing free-text location strings (which misclassifies UK rows like
-- 'Cheltenham, ENG, GB' as USA).

ALTER TABLE jobspy.import_jobs ADD COLUMN IF NOT EXISTS sys_country text;
ALTER TABLE public.jobspy_jobs ADD COLUMN IF NOT EXISTS sys_country text;
