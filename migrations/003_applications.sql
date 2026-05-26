-- Track which good jobs have been applied to / skipped / progressed to interview.
-- Written via the /api/v1/jobs/<id>/status endpoint so an external LLM can
-- update state as it helps apply.

CREATE TABLE IF NOT EXISTS public.applications (
    job_id      text        NOT NULL,
    sys_profile text        NOT NULL,
    status      text        NOT NULL CHECK (status IN ('applied', 'skipped', 'interview')),
    notes       text,
    updated_at  timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, sys_profile)
);

CREATE INDEX IF NOT EXISTS applications_profile_status_idx
    ON public.applications (sys_profile, status);
