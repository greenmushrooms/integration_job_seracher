-- Per-entry search rows: each row is one (title, location, results) scrape,
-- replacing the titles[] x locations[] cross-product read of
-- adm.job_search_config. load_jobs_flow iterates these rows directly; the
-- box-db-sync morning flow keeps them in sync with the web app's
-- web.job_searches (web edits win, row-set diff per profile).
--
-- adm.job_search_config STAYS: it still owns blocklist/exact_blocklist
-- (pipeline-owned, never web-edited). Its titles/locations/searches columns
-- are legacy after this migration.
--
-- Idempotent: the seed only fires for profiles with no rows here yet,
-- expanding the old cross product so the scrape set is preserved exactly.
CREATE TABLE IF NOT EXISTS adm.job_searches (
    profile    text NOT NULL,
    sort_order int  NOT NULL,
    title      text NOT NULL,
    location   text NOT NULL,
    searches   int  NOT NULL DEFAULT 20 CHECK (searches >= 1),
    PRIMARY KEY (profile, sort_order)
);

INSERT INTO adm.job_searches (profile, sort_order, title, location, searches)
SELECT c.profile,
       row_number() OVER (PARTITION BY c.profile ORDER BY t.ord, l.ord) - 1,
       t.title, l.location, c.searches
FROM adm.job_search_config c
CROSS JOIN LATERAL unnest(c.titles)    WITH ORDINALITY AS t(title, ord)
CROSS JOIN LATERAL unnest(c.locations) WITH ORDINALITY AS l(location, ord)
WHERE NOT EXISTS (SELECT 1 FROM adm.job_searches s WHERE s.profile = c.profile);
