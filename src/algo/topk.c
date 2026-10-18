/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * topk.c - Top-K collection
 *
 * Threshold heap: max-heap of K Distance values (upper bounds).
 * Candidate buffer: growable array of all entries that passed
 * the threshold check.
 *
 * Used by both standalone and PG builds.
 */

#include <stdlib.h>
#include <string.h>

#include "algo/topk.h"
#include "core/idset.h"
#include "core/memory.h"
#include "core/platform.h"

#define VS_TOPK_INITIAL_CAP_MIN 32

/* ----------------------------------------------------------------
 * Threshold heap (max-heap of Distance values)
 * ---------------------------------------------------------------- */

static void
ub_sift_up(Distance *heap, uint64_t *ids, uint32_t i)
{
	while (i > 0)
	{
		uint32_t parent = (i - 1) / 2;
		if (heap[i] <= heap[parent])
			break;
		Distance tmp = heap[i];
		heap[i]		 = heap[parent];
		heap[parent] = tmp;
		uint64_t tid = ids[i];
		ids[i]		 = ids[parent];
		ids[parent]	 = tid;
		i			 = parent;
	}
}

static void
ub_sift_down(Distance *heap, uint64_t *ids, uint32_t count)
{
	uint32_t i = 0;
	for (;;)
	{
		uint32_t left	 = 2 * i + 1;
		uint32_t right	 = 2 * i + 2;
		uint32_t largest = i;

		if (left < count && heap[left] > heap[largest])
			largest = left;
		if (right < count && heap[right] > heap[largest])
			largest = right;

		if (largest == i)
			break;

		Distance tmp  = heap[i];
		heap[i]		  = heap[largest];
		heap[largest] = tmp;
		uint64_t tid  = ids[i];
		ids[i]		  = ids[largest];
		ids[largest]  = tid;
		i			  = largest;
	}
}

static void
ub_sift_down_from(Distance *heap, uint64_t *ids, uint32_t count, uint32_t i)
{
	for (;;)
	{
		uint32_t left	 = 2 * i + 1;
		uint32_t right	 = 2 * i + 2;
		uint32_t largest = i;

		if (left < count && heap[left] > heap[largest])
			largest = left;
		if (right < count && heap[right] > heap[largest])
			largest = right;

		if (largest == i)
			break;

		Distance tmp  = heap[i];
		heap[i]		  = heap[largest];
		heap[largest] = tmp;
		uint64_t tid  = ids[i];
		ids[i]		  = ids[largest];
		ids[largest]  = tid;
		i			  = largest;
	}
}

/* ----------------------------------------------------------------
 * Distance sort
 * ---------------------------------------------------------------- */

/*
 * Monotonic 32-bit key for a float distance: comparing keys compares
 * distances. Flips the sign bit of non-negative values and inverts
 * negative ones, the usual IEEE-754 total-order mapping, and maps every
 * NaN to the largest key so NaNs sort after every number.
 */
static inline uint32_t
distance_key(Distance d)
{
	union
	{
		float	 f;
		uint32_t u;
	} v = {.f = d};

	/* -0.0 and +0.0 are the same distance, so key them alike. */
	if (v.u == 0x80000000u)
		v.u = 0;

	/* NaN: exponent all ones with a non-zero mantissa, either sign. */
	if ((v.u & 0x7fffffffu) > 0x7f800000u)
		return UINT32_MAX;

	return (v.u & 0x80000000u) ? ~v.u : (v.u | 0x80000000u);
}

/* Scratch of at least n entries, grown geometrically in the top-K's
 * context. */
static VsTopKEntry *
sort_buf(VsTopK *topk, uint32_t n)
{
	if (topk->sort_buf_cap < n)
	{
		uint32_t want = n;
		uint32_t grow = topk->sort_buf_cap * 2;

		if (grow > want)
			want = grow;

		topk->sort_buf =
				vs_memctx_alloc(topk->memctx, want * sizeof(VsTopKEntry));
		topk->sort_buf_cap = want;
	}

	return topk->sort_buf;
}

/*
 * Sort entries by distance ascending with a four-pass LSD radix sort over
 * the distance keys. Stable, so equal distances keep their arrival order.
 * The two buffers ping-pong and the even pass count lands the sorted
 * order back in `a`.
 */
static void
radix_sort_by_distance(VsTopKEntry *a, VsTopKEntry *scratch, uint32_t n)
{
	VsTopKEntry *src = a;
	VsTopKEntry *dst = scratch;

	for (uint32_t shift = 0; shift < 32; shift += 8)
	{
		uint32_t count[256] = {0};

		for (uint32_t i = 0; i < n; i++)
			count[(distance_key(src[i].distance) >> shift) & 0xFF]++;

		uint32_t sum = 0;
		for (uint32_t d = 0; d < 256; d++)
		{
			uint32_t c = count[d];

			count[d] = sum;
			sum += c;
		}

		for (uint32_t i = 0; i < n; i++)
			dst[count[(distance_key(src[i].distance) >> shift) & 0xFF]++] =
					src[i];

		VsTopKEntry *tmp = src;

		src = dst;
		dst = tmp;
	}
}

/* ----------------------------------------------------------------
 * Init / cleanup / create / destroy
 * ---------------------------------------------------------------- */

void
vs_topk_init(VsTopK *topk, uint32_t k)
{
	topk->memctx	 = vs_memctx_create(NULL, "topk");
	topk->k			 = k;
	topk->k_capacity = k;
	topk->ub_heap	 = vs_memctx_alloc(topk->memctx, k * sizeof(Distance));
	topk->ub_ids	 = vs_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
	topk->ub_count	 = 0;

	uint32_t cap = k * 2;
	if (cap < VS_TOPK_INITIAL_CAP_MIN)
		cap = VS_TOPK_INITIAL_CAP_MIN;
	topk->candidates =
			vs_memctx_alloc(topk->memctx, cap * sizeof(VsTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
	topk->sort_buf		= NULL;
	topk->sort_buf_cap	= 0;
	/* Stamped onto every insert; only the query scan ever sets it. */
	topk->cur_src = 0;
}

void
vs_topk_cleanup(VsTopK *topk)
{
	if (topk == NULL)
		return;
	if (topk->memctx != NULL)
	{
		vs_memctx_delete(topk->memctx);
		topk->memctx = NULL;
	}
	topk->ub_heap	   = NULL;
	topk->ub_ids	   = NULL;
	topk->candidates   = NULL;
	topk->sort_buf	   = NULL;
	topk->sort_buf_cap = 0;
}

VsTopK *
vs_topk_create(uint32_t k)
{
	/* The handle comes from the caller's context rather than from the arena
	 * it owns. vs_topk_reset_to_k resets that arena when k outgrows it,
	 * which would reclaim the object being reset through. */
	VsTopK *topk = vs_alloc(sizeof(VsTopK));
	vs_topk_init(topk, k);
	return topk;
}

void
vs_topk_destroy(VsTopK *topk)
{
	if (topk == NULL)
		return;
	vs_topk_cleanup(topk);
	vs_free(topk);
}

void
vs_topk_reset(VsTopK *topk)
{
	vs_topk_reset_to_k(topk, topk->k);
}

void
vs_topk_reset_to_k(VsTopK *topk, uint32_t k)
{
	/* O(1) reset: buffers are retained across resets (grow-only), so a
	 * reused top-K reaches a steady state with zero allocator traffic.
	 * Only a k above the allocated capacity re-allocates — reset the
	 * arena first so the outgrown blocks are reclaimed rather than
	 * leaked into the context. The candidate buffer keeps its
	 * high-water capacity, avoiding the doubling-regrowth copies that
	 * a fresh buffer paid on every use. */
	if (vs_unlikely(k > topk->k_capacity))
	{
		uint32_t cand_cap = topk->cand_capacity;
		uint32_t min_cap  = k * 2;

		if (cand_cap < min_cap)
			cand_cap = min_cap;
		if (cand_cap < VS_TOPK_INITIAL_CAP_MIN)
			cand_cap = VS_TOPK_INITIAL_CAP_MIN;

		vs_memctx_reset(topk->memctx);
		topk->ub_heap = vs_memctx_alloc(topk->memctx, k * sizeof(Distance));
		topk->ub_ids  = vs_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
		topk->candidates =
				vs_memctx_alloc(topk->memctx, cand_cap * sizeof(VsTopKEntry));
		topk->cand_capacity = cand_cap;
		topk->sort_buf		= NULL;
		topk->sort_buf_cap	= 0;
		topk->k_capacity	= k;
	}

	topk->k			 = k;
	topk->ub_count	 = 0;
	topk->cand_count = 0;
}

void
vs_topk_insert_unique(
		VsTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold. Written NaN-safe: a NaN
	 * estimate (e.g. a zero vector normalized for cosine) fails every
	 * comparison, so the naive lb >= threshold guard would admit it --
	 * and one NaN upper bound in the heap corrupts the threshold for
	 * the rest of the scan. */
	if (!(lb < vs_topk_threshold(topk)))
		return;

	/* No dedup scan: caller guarantees ids are unique. */
	if (topk->ub_count < topk->k)
	{
		topk->ub_heap[topk->ub_count] = ub;
		topk->ub_ids[topk->ub_count]  = id;
		topk->ub_count++;
		ub_sift_up(topk->ub_heap, topk->ub_ids, topk->ub_count - 1);
	}
	else if (ub < topk->ub_heap[0])
	{
		topk->ub_heap[0] = ub;
		topk->ub_ids[0]	 = id;
		ub_sift_down(topk->ub_heap, topk->ub_ids, topk->ub_count);
	}

	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
		vs_topk_reserve(topk, topk->cand_capacity + 1);

	topk->candidates[topk->cand_count++] = (VsTopKEntry){
			.distance = distance,
			.error	  = error,
			.id		  = id,
			.src	  = topk->cur_src,
	};
}

/* ----------------------------------------------------------------
 * Insert
 * ---------------------------------------------------------------- */

void
vs_topk_insert(VsTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold. Written NaN-safe: a NaN
	 * estimate (e.g. a zero vector normalized for cosine) fails every
	 * comparison, so the naive lb >= threshold guard would admit it --
	 * and one NaN upper bound in the heap corrupts the threshold for
	 * the rest of the scan. */
	if (!(lb < vs_topk_threshold(topk)))
		return;

	/* Dedup: check if this ID is already in the heap. If so,
	 * only keep the entry with the better (smaller) upper bound.
	 * This prevents duplicates (e.g. SOAR replicas) from occupying
	 * multiple heap slots and inflating the threshold. */
	for (uint32_t i = 0; i < topk->ub_count; i++)
	{
		if (topk->ub_ids[i] == id)
		{
			if (ub < topk->ub_heap[i])
			{
				topk->ub_heap[i] = ub;
				ub_sift_down_from(
						topk->ub_heap, topk->ub_ids, topk->ub_count, i);
			}
			goto append;
		}
	}

	/* Update threshold heap */
	if (topk->ub_count < topk->k)
	{
		topk->ub_heap[topk->ub_count] = ub;
		topk->ub_ids[topk->ub_count]  = id;
		topk->ub_count++;
		ub_sift_up(topk->ub_heap, topk->ub_ids, topk->ub_count - 1);
	}
	else if (ub < topk->ub_heap[0])
	{
		topk->ub_heap[0] = ub;
		topk->ub_ids[0]	 = id;
		ub_sift_down(topk->ub_heap, topk->ub_ids, topk->ub_count);
	}

append:
	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
		vs_topk_reserve(topk, topk->cand_capacity + 1);

	/* Append to candidate buffer */
	topk->candidates[topk->cand_count++] = (VsTopKEntry){
			.distance = distance,
			.error	  = error,
			.id		  = id,
			.src	  = topk->cur_src,
	};
}

/*
 * Deduplicate a distance-sorted candidate prefix in place, keeping the
 * first (best-distance) entry per ID, tracked in an id set. A rescan
 * of prior survivors per entry would go quadratic in the survivor
 * count; the set keeps this pass linear. Returns the deduplicated
 * count, additionally truncated to `cap` unique entries when cap > 0.
 */
static uint32_t
dedup_sorted_prefix(VsTopKEntry *results, uint32_t n, uint32_t cap)
{
	if (n <= 1)
		return n;

	VsIdSet seen;
	vs_idset_init(&seen, n);

	uint32_t w = 0;
	for (uint32_t r = 0; r < n; r++)
	{
		if (!vs_idset_test_add(&seen, results[r].id))
			continue;
		results[w++] = results[r];
		if (cap > 0 && w == cap)
			break;
	}
	vs_idset_cleanup(&seen);
	return w;
}

/*
 * Partially order `results` so the `want` smallest-distance entries
 * occupy the front (in arbitrary order within each side). Iterative
 * Hoare quickselect with median-of-three pivots: expected O(n), no
 * recursion, no external randomness.
 */
static void
quickselect_by_distance(VsTopKEntry *results, uint32_t n, uint32_t want)
{
	uint32_t lo = 0, hi = n;

	while (hi - lo > 1 && want > lo && want < hi)
	{
		uint32_t mid = lo + (hi - lo) / 2;
		Distance a	 = results[lo].distance;
		Distance b	 = results[mid].distance;
		Distance c	 = results[hi - 1].distance;
		Distance pivot;
		if ((a <= b && b <= c) || (c <= b && b <= a))
			pivot = b;
		else if ((b <= a && a <= c) || (c <= a && a <= b))
			pivot = a;
		else
			pivot = c;

		/* The median-of-three pivot exists in the range, so each scan
		 * has a stopper; the explicit bounds checks make that local
		 * rather than an invariant to trust (j is unsigned, so an
		 * unchecked descent past lo would wrap). */
		uint32_t i = lo, j = hi - 1;
		while (i <= j)
		{
			while (i < hi - 1 && results[i].distance < pivot)
				i++;
			while (j > lo && results[j].distance > pivot)
				j--;
			if (i <= j)
			{
				VsTopKEntry tmp = results[i];
				results[i]		= results[j];
				results[j]		= tmp;
				i++;
				if (j == 0)
					break;
				j--;
			}
		}
		/* Entries [lo, j] <= pivot <= entries [i, hi). Recurse into the
		 * side containing the selection boundary. */
		if (want <= j + 1)
			hi = j + 1;
		else if (want >= i)
			lo = i;
		else
			return; /* boundary falls in the pivot-equal middle band */
	}
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

/* The uncapped form of vs_topk_extract_sorted_capped. */
void
vs_topk_extract_sorted(VsTopK *topk, VsTopKEntry *results, uint32_t *count_out)
{
	vs_topk_extract_sorted_capped(topk, results, count_out, 0);
}

/*
 * Extract the threshold survivors sorted by distance ascending and
 * deduplicated (first, i.e. best-distance, entry per id), keeping at
 * most the best `cap` unique candidates; cap == 0 keeps them all. The
 * survivor count is unbounded when distance estimates are noisy -- a
 * loose threshold admits most scanned entries, every SOAR replica
 * included -- so sorting all survivors is wasted work when only a
 * small prefix is kept. A quickselect partition narrows to the
 * smallest 3*cap entries first: every id occurs at most twice (primary
 * posting plus at most one SOAR replica), so the smallest 2*cap
 * entries already contain the best cap unique ids and 3*cap leaves
 * margin.
 */
void
vs_topk_extract_sorted_capped(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out, uint32_t cap)
{
	Distance threshold = vs_topk_threshold(topk);

	uint32_t out = 0;
	for (uint32_t i = 0; i < topk->cand_count; i++)
	{
		Distance lb = topk->candidates[i].distance - topk->candidates[i].error;
		if (lb <= threshold)
			results[out++] = topk->candidates[i];
	}

	uint32_t sel = out;
	if (cap > 0 && out > cap * 3)
	{
		sel = cap * 3;
		quickselect_by_distance(results, out, sel);
	}

	if (sel > 1)
		radix_sort_by_distance(results, sort_buf(topk, sel), sel);

	*count_out = dedup_sorted_prefix(results, sel, cap);
}

/*
 * Same as vs_topk_extract_sorted but skips the duplicate-id pass.
 * For callers whose ids are unique by construction (array indices,
 * beam-search buffer positions), the dedup scan can never remove
 * anything and is quadratic in the survivor count.
 */
void
vs_topk_extract_sorted_unique(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out)
{
	Distance threshold = vs_topk_threshold(topk);

	/* Filter stale candidates and copy survivors to results */
	uint32_t out = 0;
	for (uint32_t i = 0; i < topk->cand_count; i++)
	{
		Distance lb = topk->candidates[i].distance - topk->candidates[i].error;
		if (lb <= threshold)
			results[out++] = topk->candidates[i];
	}

	/* Sort by distance ascending */
	if (out > 1)
		radix_sort_by_distance(results, sort_buf(topk, out), out);

	*count_out = out;
}

/*
 * Grow the candidate buffer to hold at least `capacity` entries, never
 * shrinking it. The append path knows how many entries it will score, so it
 * reserves once instead of doubling through a series of reallocations.
 */
void
vs_topk_reserve(VsTopK *topk, uint32_t capacity)
{
	if (topk->cand_capacity >= capacity)
		return;

	uint32_t	 want = topk->cand_capacity * 2;
	VsTopKEntry *new_buf;

	if (want < capacity)
		want = capacity;

	new_buf = vs_memctx_alloc(topk->memctx, want * sizeof(VsTopKEntry));
	memcpy(new_buf, topk->candidates, topk->cand_count * sizeof(VsTopKEntry));
	topk->candidates	= new_buf;
	topk->cand_capacity = want;
}

/*
 * Append a scored candidate without a threshold check or heap update. The
 * buffer grows geometrically like the insert path's.
 */
void
vs_topk_append(VsTopK *topk, Distance distance, uint64_t id)
{
	if (topk->cand_count == topk->cand_capacity)
		vs_topk_reserve(topk, topk->cand_capacity + 1);

	topk->candidates[topk->cand_count++] = (VsTopKEntry){
			.distance = distance,
			.error	  = 0.0f,
			.id		  = id,
			.src	  = topk->cur_src,
	};
}

/*
 * Select the best `keep` appended candidates by sorting the buffer once,
 * stable, so candidates with equal distances keep the order the caller
 * appended them in. The heap is not consulted and no candidate is dropped
 * before the selection.
 */
uint32_t
vs_topk_extract_best(VsTopK *topk, uint32_t keep)
{
	uint32_t n = topk->cand_count;

	if (n == 0)
		return 0;
	if (n > 1)
		radix_sort_by_distance(topk->candidates, sort_buf(topk, n), n);

	return n < keep ? n : keep;
}

/*
 * Sort candidate indices by encoded TID, block then offset, with a six-pass
 * LSD radix sort over the 48-bit id, skipping a byte that is uniform across
 * every id. The rerank's read stream coalesces adjacent block numbers into
 * larger reads, so this order matters for cold data. Passes ping-pong
 * between `order` and `scratch`, so the sorted order is copied back when the
 * last pass landed in `scratch`; `scratch` must hold `count` entries.
 */
void
vs_topk_sort_by_tid(
		uint32_t		  *order,
		uint32_t		  *scratch,
		const VsTopKEntry *candidates,
		uint32_t		   count)
{
	uint32_t *src = order;
	uint32_t *dst = scratch;

	if (count == 0)
		return;

	for (uint32_t shift = 0; shift < 48; shift += 8)
	{
		uint32_t counts[256] = {0};

		for (uint32_t i = 0; i < count; i++)
			counts[(candidates[src[i]].id >> shift) & 0xFF]++;

		/* A byte that is the same for every TID orders nothing: the top
		 * bytes are zero below 2^24 heap blocks, so most tables skip them. */
		if (counts[(candidates[src[0]].id >> shift) & 0xFF] == count)
			continue;

		uint32_t sum = 0;

		for (uint32_t d = 0; d < 256; d++)
		{
			uint32_t c = counts[d];

			counts[d] = sum;
			sum += c;
		}

		for (uint32_t i = 0; i < count; i++)
			dst[counts[(candidates[src[i]].id >> shift) & 0xFF]++] = src[i];

		uint32_t *tmp = src;

		src = dst;
		dst = tmp;
	}

	/* The pass count varies with the skips, so the result is not always in
	 * `order` when the loop ends. */
	if (src != order)
		memcpy(order, src, count * sizeof(uint32_t));
}
