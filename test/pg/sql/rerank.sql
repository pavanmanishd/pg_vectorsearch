-- Rerank correctness: index scan results must match sequential scan

-- Use higher dimensionality to exercise the full rerank path
-- (low-dim vectors can sometimes bypass rerank entirely)
CREATE TABLE rerank_test (id serial, v vec32(32));

INSERT INTO rerank_test (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vec32(32)
    FROM generate_series(1, 500) i;

-- The query vector, hoisted into an immutable function so the planner folds
-- it to a literal. Read with a subquery it becomes an InitPlan whose own
-- sequential scan lands in every plan and cost below.
SELECT v::text AS rqv FROM rerank_test WHERE id = 42 \gset
SELECT format($f$
    CREATE FUNCTION rqv() RETURNS vec32(32)
        LANGUAGE sql IMMUTABLE PARALLEL SAFE
        AS $b$ SELECT %L::vec32(32) $b$
$f$, :'rqv') \gexec

-- L2 distance
CREATE INDEX idx_rerank_l2 ON rerank_test USING prism (v)
    WITH (centroid_compression = true);

-- Compute ground truth via sequential scan
SET enable_indexscan = off;
CREATE TEMP TABLE truth_l2 AS
    SELECT id, v <-> rqv() AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

-- Compute index scan results (exercises rerank)
-- Small tables, so a sequential scan is the plan the cost model correctly
-- prefers and the scan below is forced through the index. The rows come
-- back either way, so this pins the plan.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM rerank_test
    ORDER BY v <-> rqv() LIMIT 10;
CREATE TEMP TABLE result_l2 AS
    SELECT id, v <-> rqv() AS dist
    FROM rerank_test
    ORDER BY v <-> rqv() LIMIT 10;
RESET enable_seqscan;

-- All ground truth top-10 must appear in index results
SELECT count(*) AS matching_ids
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Distances must match exactly (rerank uses full-precision vectors)
SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Cosine distance
CREATE INDEX idx_rerank_cos ON rerank_test
    USING prism (v vec32_cosine_ops)
    WITH (centroid_compression = true);

SET enable_indexscan = off;
CREATE TEMP TABLE truth_cos AS
    SELECT id, v <=> rqv() AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

SET enable_seqscan = off;
CREATE TEMP TABLE result_cos AS
    SELECT id, v <=> rqv() AS dist
    FROM rerank_test
    ORDER BY v <=> rqv() LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS matching_ids
    FROM truth_cos t JOIN result_cos r USING (id);

SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_cos t JOIN result_cos r USING (id);

-- A NULL query vector must not crash the scan. A subquery that matches no
-- row yields NULL, every distance is NULL, so all rows tie and any rows
-- satisfy the ORDER BY and the LIMIT.
SET enable_seqscan = off;

-- The generic plan keeps the argument a parameter, so no InitPlan lands in
-- the plan and the index scan is what the test pins. Executing it is what
-- covers the NULL a client binds to a parameter.
SET plan_cache_mode = force_generic_plan;
PREPARE null_q(vec32) AS
    SELECT count(*) FROM (
        SELECT id FROM rerank_test ORDER BY v <-> $1 LIMIT 5) t;
EXPLAIN (COSTS OFF) EXECUTE null_q(NULL);
EXECUTE null_q(NULL);

-- The issue's reproducer is cosine: the substitute must be a valid vector
-- for the cosine distance too.
PREPARE null_q_cos(vec32) AS
    SELECT count(*) FROM (
        SELECT id FROM rerank_test ORDER BY v <=> $1 LIMIT 5) t;
EXPLAIN (COSTS OFF) EXECUTE null_q_cos(NULL);
EXECUTE null_q_cos(NULL);
DEALLOCATE null_q_cos;
DEALLOCATE null_q;
RESET plan_cache_mode;

-- A rescan whose argument switches between NULL and a real vector: the
-- cached substitute must not leak into the non-NULL row, and two NULL rows
-- exercise the cache itself. The LATERAL top-k reaches the index as a
-- nestloop parameter, so this is a real rescan.
EXPLAIN (COSTS OFF)
SELECT q.k, s.id
FROM (VALUES (1, NULL::vec32(32)), (2, NULL::vec32(32)), (3, rqv())) q(k, v),
     LATERAL (SELECT id FROM rerank_test ORDER BY v <-> q.v LIMIT 5) s;
SELECT q.k, count(s.id) AS rows
FROM (VALUES (1, NULL::vec32(32)), (2, NULL::vec32(32)), (3, rqv())) q(k, v),
     LATERAL (SELECT id FROM rerank_test ORDER BY v <-> q.v LIMIT 5) s
GROUP BY q.k
ORDER BY q.k;

-- The real vector's top 5 must be the index's real top 5.
SELECT count(*) AS real_ids_match
FROM (VALUES (3, rqv())) q(k, v),
     LATERAL (SELECT id FROM rerank_test ORDER BY v <-> q.v LIMIT 5) s
JOIN result_l2 r USING (id);

-- The same query through a vec16 index: the substitute is built as float32
-- directly, so this pins that the vec16 path does not crash on it.
CREATE TABLE rerank_half (id serial, v vec16(8));
INSERT INTO rerank_half (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 7) j
    )::vec16(8)
    FROM generate_series(1, 500) i;
CREATE INDEX idx_rerank_half ON rerank_half USING prism (v);
SET plan_cache_mode = force_generic_plan;
PREPARE half_q(vec16) AS
    SELECT count(*) FROM (
        SELECT id FROM rerank_half ORDER BY v <-> $1 LIMIT 5) t;
EXPLAIN (COSTS OFF) EXECUTE half_q(NULL);
EXECUTE half_q(NULL);
DEALLOCATE half_q;
RESET plan_cache_mode;
RESET enable_seqscan;
DROP TABLE rerank_half;

-- Cleanup
DROP TABLE rerank_test;

DROP FUNCTION rqv();
