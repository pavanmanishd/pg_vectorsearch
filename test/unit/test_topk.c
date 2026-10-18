/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_topk.c - Unit tests for bounded top-K collection
 *
 * Tests cover:
 * - Basic create/init/destroy
 * - Insert and threshold behavior
 * - Extract sorted order
 * - Brute-force validation against linear scan
 * - Error-aware threshold and candidate buffering
 * - Edge cases (k=1, all equal distances, empty)
 */

#include <math.h>
#include <string.h>

#include "algo/topk.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(TopK);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Basic API tests
 * ---------------------------------------------------------------- */

TEST(topk_create_destroy)
{
	VsTopK *topk = vs_topk_create(10);
	ASSERT_NOT_NULL(topk, "create should succeed");
	ASSERT_EQ(0, topk->cand_count, "cand_count should be 0");
	ASSERT_EQ(10, topk->k, "k should be 10");
	vs_topk_destroy(topk);
}

TEST(topk_init_cleanup)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0");
	ASSERT_EQ(5, topk.k, "k should be 5");
	vs_topk_cleanup(&topk);
}

TEST(topk_init_defines_src_stamp)
{
	/* Every insert stamps cur_src onto the entry, and only the query scan
	 * ever sets it, so init has to leave it defined. The poison makes a
	 * missing initialization show up as the poison rather than as a
	 * chance zero. */
	VsTopK topk;
	memset(&topk, 0xAB, sizeof(topk));
	vs_topk_init(&topk, 3);
	vs_topk_insert(&topk, 1.0f, 0.0f, 7);

	ASSERT_EQ(1, topk.cand_count, "insert buffered the candidate");
	ASSERT_EQ(0, topk.candidates[0].src, "init defines the src stamp");
	vs_topk_cleanup(&topk);
}

TEST(topk_create_reports_heap_capacity)
{
	/* k_capacity is what a reset compares a new k against; read wrong, the
	 * reset skips its reallocation and leaves ub_heap sized for the
	 * original k while k says otherwise -- inserts then run past it. */
	VsTopK *topk = vs_topk_create(2);
	ASSERT_EQ(2, topk->k_capacity, "create reports its heap capacity");
	ASSERT_EQ(0, topk->cur_src, "create defines the src stamp");

	vs_topk_reset_to_k(topk, 64);
	for (uint32_t i = 0; i < 64; i++)
		vs_topk_insert(topk, (float)(64 - i), 0.0f, i + 1);
	ASSERT_EQ(64, topk->ub_count, "the threshold heap grew with k");

	vs_topk_destroy(topk);
}

TEST(topk_threshold_empty)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold should be INFINITY when not full");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Insert and threshold (zero error)
 * ---------------------------------------------------------------- */

TEST(topk_insert_under_capacity)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 5.0f, 0.0f, 1);
	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold still INFINITY (< k upper bounds)");

	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	vs_topk_insert(&topk, 7.0f, 0.0f, 3);

	Distance thresh = vs_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(
			7.0f, thresh, 1e-6f, "threshold should be max upper_bound (7.0)");
	vs_topk_cleanup(&topk);
}

TEST(topk_insert_tightens_threshold)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 5.0f, 0.0f, 1);
	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	vs_topk_insert(&topk, 7.0f, 0.0f, 3);

	/* Insert better upper bound → tightens threshold */
	vs_topk_insert(&topk, 4.0f, 0.0f, 4);

	Distance thresh = vs_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(5.0f, thresh, 1e-6f, "threshold should tighten to 5.0");
	vs_topk_cleanup(&topk);
}

TEST(topk_insert_prunes_bad_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);
	vs_topk_insert(&topk, 3.0f, 0.0f, 3);

	/* lower_bound=10 >= threshold=3 → pruned */
	vs_topk_insert(&topk, 10.0f, 0.0f, 99);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* Should only have the first 3 entries */
	ASSERT_EQ(3, count, "pruned entry should not appear");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

TEST(topk_extract_sorted_ascending)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	vs_topk_insert(&topk, 9.0f, 0.0f, 9);
	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	vs_topk_insert(&topk, 3.0f, 0.0f, 3);
	vs_topk_insert(&topk, 7.0f, 0.0f, 7);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	ASSERT_EQ(5, count, "should extract 5 entries");

	for (uint32_t i = 1; i < count; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result[%u] (%.1f) should be <= result[%u] (%.1f)",
				i - 1,
				results[i - 1].distance,
				i,
				results[i].distance);
		ASSERT_TRUE(results[i - 1].distance <= results[i].distance, msg);
	}

	/* Verify IDs match expected order */
	ASSERT_EQ(1, results[0].id, "first should be id=1 (dist 1.0)");
	ASSERT_EQ(3, results[1].id, "second should be id=3 (dist 3.0)");
	ASSERT_EQ(5, results[2].id, "third should be id=5 (dist 5.0)");
	ASSERT_EQ(7, results[3].id, "fourth should be id=7 (dist 7.0)");
	ASSERT_EQ(9, results[4].id, "fifth should be id=9 (dist 9.0)");
	vs_topk_cleanup(&topk);
}

TEST(topk_extract_resets)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(2, count, "should extract 2");

	vs_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0 after reset");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Brute-force validation
 * ---------------------------------------------------------------- */

TEST(topk_brute_force_validation)
{
	const uint32_t k = 10;
	const uint32_t n = 100;

	VsTopK topk;
	vs_topk_init(&topk, k);

	/* Insert n candidates with pseudo-random distances, zero error */
	float *all_dists = vs_alloc(n * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
	{
		all_dists[i] = (float)((i * 97 + 13) % 1000) / 10.0f;
		vs_topk_insert(&topk, all_dists[i], 0.0f, i);
	}

	VsTopKEntry *results = vs_alloc(topk.cand_count * sizeof(*results));
	uint32_t	 count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* With zero error, threshold pruning eliminates candidates
	 * whose distance >= Kth-smallest, so count should be <= k+1
	 * (boundary candidates may survive). Check the first k. */
	ASSERT_TRUE(count >= k, "should have at least k results");

	/* Brute force: sort all distances and take first k */
	for (uint32_t i = 0; i < n; i++)
	{
		for (uint32_t j = i + 1; j < n; j++)
		{
			if (all_dists[j] < all_dists[i])
			{
				float tmp	 = all_dists[i];
				all_dists[i] = all_dists[j];
				all_dists[j] = tmp;
			}
		}
	}

	/* Verify first k results match brute force */
	for (uint32_t i = 0; i < k; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result[%u] dist %.1f should match brute-force %.1f",
				i,
				results[i].distance,
				all_dists[i]);
		ASSERT_FLOAT_EQ(all_dists[i], results[i].distance, 1e-6f, msg);
	}

	vs_free(results);
	vs_free(all_dists);
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Error-aware behavior
 * ---------------------------------------------------------------- */

TEST(topk_error_affects_threshold)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* dist=3, error=1 → upper_bound=4 */
	vs_topk_insert(&topk, 3.0f, 1.0f, 1);
	/* dist=2, error=0.5 → upper_bound=2.5 */
	vs_topk_insert(&topk, 2.0f, 0.5f, 2);

	/* Threshold = max of {4.0, 2.5} = 4.0 (Kth-smallest ub) */
	ASSERT_FLOAT_EQ(
			4.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold should be worst upper_bound");
	vs_topk_cleanup(&topk);
}

TEST(topk_keeps_high_error_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* dist=2, error=0 → ub=2 */
	vs_topk_insert(&topk, 2.0f, 0.0f, 1);
	/* dist=3, error=0 → ub=3 */
	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	/* Threshold = 3.0 */

	/* dist=10, error=9 → lb=1, ub=19
	 * lower_bound=1 < threshold=3 → NOT pruned.
	 * This candidate might be the closest! */
	vs_topk_insert(&topk, 10.0f, 9.0f, 3);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* All three should be kept (high-error entry not evicted) */
	ASSERT_EQ(3, count, "high-error candidate should be kept");

	bool has_3 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 3)
			has_3 = true;
	}
	ASSERT_TRUE(has_3, "high-error entry (id=3) must be in results");
	vs_topk_cleanup(&topk);
}

TEST(topk_buffers_more_than_k)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* All with overlapping error bounds — none can be pruned */
	vs_topk_insert(&topk, 5.0f, 3.0f, 1); /* lb=2, ub=8 */
	vs_topk_insert(&topk, 6.0f, 3.0f, 2); /* lb=3, ub=9 */
	vs_topk_insert(&topk, 7.0f, 3.0f, 3); /* lb=4, ub=10 */

	/* threshold = 9.0 (Kth-smallest ub from {8,9})
	 * Entry 3: lb=4 < 9 → kept */
	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	ASSERT_TRUE(
			count > 2, "should buffer more than K with overlapping bounds");
	ASSERT_EQ(3, count, "all 3 entries should be in buffer");
	vs_topk_cleanup(&topk);
}

TEST(topk_error_preserved_in_extract)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	vs_topk_insert(&topk, 2.0f, 0.5f, 1);
	vs_topk_insert(&topk, 4.0f, 1.0f, 2);
	vs_topk_insert(&topk, 6.0f, 0.1f, 3);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(3, count, "should extract 3");

	/* Results sorted by distance ascending, error preserved */
	ASSERT_FLOAT_EQ(2.0f, results[0].distance, 1e-6f, "first dist=2.0");
	ASSERT_FLOAT_EQ(0.5f, results[0].error, 1e-6f, "first error=0.5");
	ASSERT_FLOAT_EQ(4.0f, results[1].distance, 1e-6f, "second dist=4.0");
	ASSERT_FLOAT_EQ(1.0f, results[1].error, 1e-6f, "second error=1.0");
	ASSERT_FLOAT_EQ(6.0f, results[2].distance, 1e-6f, "third dist=6.0");
	ASSERT_FLOAT_EQ(0.1f, results[2].error, 1e-6f, "third error=0.1");
	vs_topk_cleanup(&topk);
}

TEST(topk_filters_stale_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* First two: threshold starts at INFINITY, so accepted */
	vs_topk_insert(&topk, 1.0f, 0.0f, 1); /* lb=1, ub=1 */
	vs_topk_insert(&topk, 2.0f, 0.0f, 2); /* lb=2, ub=2 */
	/* threshold = 2.0 */

	/* This tightens the threshold heap */
	vs_topk_insert(&topk, 0.5f, 0.0f, 3); /* lb=0.5, ub=0.5 */
	/* threshold ub_heap now {1.0, 0.5} → root=1.0 */

	/* Entry id=2 (lb=2) >= final threshold=1.0 → stale, filtered */
	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	ASSERT_EQ(2, count, "stale entry should be filtered");

	bool has_2 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 2)
			has_2 = true;
	}
	ASSERT_FALSE(has_2, "id=2 (lb=2 >= threshold=1) should be filtered");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(topk_k_equals_one)
{
	VsTopK topk;
	vs_topk_init(&topk, 1);

	vs_topk_insert(&topk, 10.0f, 0.0f, 10);
	ASSERT_FLOAT_EQ(
			10.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold with one entry");

	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_FLOAT_EQ(
			5.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold after better insert");

	vs_topk_insert(&topk, 20.0f, 0.0f, 20);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* id=10 (lb=10 >= threshold=5) should be filtered.
	 * id=20 (lb=20 >= threshold=5) should be pruned.
	 * Only id=5 remains. */
	ASSERT_EQ(1, count, "should extract 1");
	ASSERT_EQ(5, results[0].id, "should be the best");
	vs_topk_cleanup(&topk);
}

TEST(topk_all_equal_distances)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	for (uint32_t i = 0; i < 10; i++)
		vs_topk_insert(&topk, 42.0f, 0.0f, i);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* With zero error: ub=42 for all. threshold=42.
	 * lb=42 >= threshold=42 → later entries pruned.
	 * Only the first 5 (before threshold fills) survive. */
	ASSERT_EQ(5, count, "should extract 5");

	for (uint32_t i = 0; i < count; i++)
	{
		ASSERT_FLOAT_EQ(
				42.0f,
				results[i].distance,
				1e-6f,
				"all distances should be equal");
	}
	vs_topk_cleanup(&topk);
}

TEST(topk_extract_empty)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	VsTopKEntry results[1];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(0, count, "empty extraction should return 0");
	vs_topk_cleanup(&topk);
}

TEST(topk_reset)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);
	ASSERT_TRUE(topk.cand_count > 0, "has candidates before reset");

	vs_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count after reset");
	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold after reset should be INFINITY");

	/* Should work again after reset */
	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_EQ(1, topk.cand_count, "cand_count after re-insert");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Capped extraction (quickselect + dedup) vs brute force
 * ---------------------------------------------------------------- */

/* Adversarial distance distributions for the quickselect partition:
 * all-equal (pivot-equal band), sorted, reverse-sorted, two-valued,
 * scattered, and negative (inner product), each with duplicate ids
 * (SOAR-style replicas). The capped extract must return exactly the cap
 * best unique ids, sorted; bounds bugs in the partition surface under
 * the sanitizer jobs. */
TEST(topk_extract_capped_adversarial)
{
	const uint32_t n	  = 500;
	const uint32_t k	  = 10;
	const uint32_t caps[] = {1, 10, 33, 160, 500, 1000};

	for (int dist_kind = 0; dist_kind < 6; dist_kind++)
	{
		for (size_t c = 0; c < sizeof(caps) / sizeof(caps[0]); c++)
		{
			VsTopK topk;
			vs_topk_init(&topk, k);
			/* Every id twice (replica), error large enough that no
			 * candidate is threshold-pruned: the buffer holds all. */
			for (uint32_t i = 0; i < n; i++)
			{
				float d;
				switch (dist_kind)
				{
				case 0:
					d = 1.0f;
					break;
				case 1:
					d = (float)i;
					break;
				case 2:
					d = (float)(n - i);
					break;
				case 3:
					d = (float)(i % 2);
					break;
				case 4:
					d = (float)((i * 7919u) % 257u);
					break;
				default:
					d = -(float)((i * 7919u) % 257u);
					break;
				}
				vs_topk_insert(&topk, d, 1000.0f, (uint64_t)(i % 250) + 1);
			}

			VsTopKEntry *res = vs_alloc(topk.cand_count * sizeof(VsTopKEntry));
			uint32_t	 out;
			vs_topk_extract_sorted_capped(&topk, res, &out, caps[c]);

			uint32_t expect = caps[c] < 250 ? caps[c] : 250;
			ASSERT_EQ(expect, out, "capped extract returns cap uniques");
			for (uint32_t i = 1; i < out; i++)
				ASSERT_TRUE(
						res[i - 1].distance <= res[i].distance,
						"capped extract sorted ascending");
			for (uint32_t i = 0; i < out; i++)
				for (uint32_t j = i + 1; j < out; j++)
					ASSERT_TRUE(res[i].id != res[j].id, "no duplicate ids");

			vs_free(res);
			vs_topk_cleanup(&topk);
		}
	}
}

/*
 * The unique extract has no dedup pass, so it is the direct check on the
 * distance sort: negative distances (inner product) come first, the two
 * zeros tie, and the order is ascending end to end.
 */
TEST(topk_extract_unique_negative_distances)
{
	const float dists[] =
			{-3.5f, 0.0f, 2.5f, -0.0f, -1.25f, 4.0f, -2.0f, 0.5f, -4.5f, 1.5f};
	const uint32_t n = sizeof(dists) / sizeof(dists[0]);

	VsTopK topk;

	vs_topk_init(&topk, 8);
	for (uint32_t i = 0; i < n; i++)
		vs_topk_insert_unique(&topk, dists[i], 0.0f, (uint64_t)(i + 1));

	VsTopKEntry *res = vs_alloc(topk.cand_count * sizeof(VsTopKEntry));
	uint32_t	 out;

	vs_topk_extract_sorted_unique(&topk, res, &out);

	ASSERT_EQ(8, out, "unique extract returns k entries");
	ASSERT_EQ(-4.5f, res[0].distance, "most negative distance first");
	ASSERT_EQ(1.5f, res[out - 1].distance, "smallest admitted distance last");
	for (uint32_t i = 1; i < out; i++)
		ASSERT_TRUE(
				res[i - 1].distance <= res[i].distance,
				"unique extract sorted ascending");

	/* -0.0 and +0.0 are the same distance, so the sort must keep the order
	 * they arrived in instead of ranking one below the other. */
	ASSERT_EQ(2, (int)res[4].id, "the +0.0 row sorts first at equal keys");
	ASSERT_EQ(4, (int)res[5].id, "the -0.0 row keeps its arrival order");

	vs_free(res);
	vs_topk_cleanup(&topk);
}

/*
 * The pool-style path: append scored candidates without a heap, then
 * select the best keep in one pass. Mirrors the rerank, which computes
 * exact distances for a whole pool before choosing the top k.
 */
TEST(topk_append_extract_best)
{
	const uint32_t n	= 1000;
	const uint32_t keep = 50;

	VsTopK topk;

	vs_topk_init(&topk, keep);

	float *all_dists = vs_alloc(n * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
	{
		all_dists[i] = (float)((i * 97 + 13) % 1000) / 10.0f;
		vs_topk_append(&topk, all_dists[i], (uint64_t)(i + 1));
	}

	uint32_t out = vs_topk_extract_best(&topk, keep);

	ASSERT_EQ(keep, out, "extract_best returns keep entries");
	for (uint32_t i = 1; i < out; i++)
		ASSERT_TRUE(
				topk.candidates[i - 1].distance <= topk.candidates[i].distance,
				"extract_best sorted ascending");

	/* Brute force: sort all distances and compare the first keep. */
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t j = i + 1; j < n; j++)
			if (all_dists[j] < all_dists[i])
			{
				float tmp	 = all_dists[i];
				all_dists[i] = all_dists[j];
				all_dists[j] = tmp;
			}

	for (uint32_t i = 0; i < keep; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"best[%u] dist %.1f should match brute-force %.1f",
				i,
				topk.candidates[i].distance,
				all_dists[i]);
		ASSERT_FLOAT_EQ(all_dists[i], topk.candidates[i].distance, 1e-6f, msg);
	}

	vs_free(all_dists);
	vs_topk_cleanup(&topk);
}

/* Equal distances keep the order they were appended in: the selection is
 * stable, which is what lets the rerank produce TID order among ties by
 * appending in TID order. */
TEST(topk_append_extract_best_ties)
{
	const float	   dists[] = {5.0f, 1.0f, 5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 3.0f};
	const uint32_t n	   = sizeof(dists) / sizeof(dists[0]);
	const uint32_t keep	   = 5;

	VsTopK topk;

	vs_topk_init(&topk, keep);
	for (uint32_t i = 0; i < n; i++)
		vs_topk_append(&topk, dists[i], (uint64_t)(i + 1));

	uint32_t out = vs_topk_extract_best(&topk, keep);

	ASSERT_EQ(keep, out, "extract_best returns keep entries");

	/* The 1.0s arrive first (ids 2, 4, 6), then the 3.0s (ids 5, 8). */
	const uint64_t expect[] = {2, 4, 6, 5, 8};

	for (uint32_t i = 0; i < out; i++)
	{
		char msg[96];

		snprintf(msg, sizeof(msg), "tie %u keeps its append order", i);
		ASSERT_EQ((int)expect[i], (int)topk.candidates[i].id, msg);
	}

	vs_topk_cleanup(&topk);
}

/* An empty pool, a keep above the pool, and negative distances. */
TEST(topk_append_extract_best_edges)
{
	VsTopK topk;

	vs_topk_init(&topk, 8);

	ASSERT_EQ(0, vs_topk_extract_best(&topk, 8), "empty pool selects nothing");

	vs_topk_append(&topk, -2.5f, 1);
	vs_topk_append(&topk, 0.0f, 2);
	vs_topk_append(&topk, -7.5f, 3);

	uint32_t out = vs_topk_extract_best(&topk, 8);

	ASSERT_EQ(3, out, "keep above the pool returns the whole pool");
	ASSERT_FLOAT_EQ(
			-7.5f, topk.candidates[0].distance, 1e-6f, "most negative first");
	ASSERT_FLOAT_EQ(0.0f, topk.candidates[2].distance, 1e-6f, "largest last");

	vs_topk_cleanup(&topk);
}

/* A reserve covers the appends that follow, with no reallocation. */
TEST(topk_reserve_presizes)
{
	VsTopK topk;

	vs_topk_init(&topk, 0);
	vs_topk_reserve(&topk, 1000);

	ASSERT_TRUE(topk.cand_capacity >= 1000, "reserve grows the buffer");

	uint32_t cap = topk.cand_capacity;

	for (uint32_t i = 0; i < 1000; i++)
		vs_topk_append(&topk, (float)i, (uint64_t)(i + 1));

	ASSERT_EQ(1000, topk.cand_count, "all appends kept");
	ASSERT_EQ((int)cap, (int)topk.cand_capacity, "no reallocation needed");

	vs_topk_cleanup(&topk);
}

/* The TID sort ping-pongs between two buffers and skips bytes that are
 * uniform across every id, so the number of passes varies with the table:
 * the sorted order must land back in `order` whichever parity it ends on.
 * One block runs two passes, 512 blocks with byte-sized offsets run three. */
TEST(topk_sort_by_tid_pass_counts)
{
	enum
	{
		N = 512
	};

	VsTopKEntry cand[N];
	uint32_t	order[N];
	uint32_t	scratch[N];

	/* One block: bytes 0 and 1 vary, so two passes. */
	for (uint32_t i = 0; i < N; i++)
		cand[i] = (VsTopKEntry){.distance = 0.0f,
								.id		  = ((uint64_t)7 << 16) | (N - i)};

	for (uint32_t i = 0; i < N; i++)
		order[i] = i;

	vs_topk_sort_by_tid(order, scratch, cand, N);

	for (uint32_t i = 1; i < N; i++)
		ASSERT_TRUE(
				cand[order[i - 1]].id < cand[order[i]].id,
				"single-pass result is sorted in order");

	/* 512 blocks with offsets below 256: bytes 0, 2 and 3 vary, so three
	 * passes run. */
	for (uint32_t i = 0; i < N; i++)
		cand[i] = (VsTopKEntry){.distance = 0.0f,
								.id = ((uint64_t)(N - i) << 16) | (i & 0xFF)};

	for (uint32_t i = 0; i < N; i++)
		order[i] = i;

	vs_topk_sort_by_tid(order, scratch, cand, N);

	for (uint32_t i = 1; i < N; i++)
		ASSERT_TRUE(
				cand[order[i - 1]].id < cand[order[i]].id,
				"three-pass result is sorted in order");
}
