/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_refine.c - Unit tests for the page-backed leaf-refinement kernel
 *
 * prism_refine_write_means divides a leaf's accumulated sums by its counts
 * and hands each mean to the head writer. For cosine the written reference
 * must be unit norm: the mean of normalized rows drifts below unit norm,
 * every other centroid write path normalizes for the same reason, and a
 * reference that is not unit norm biases the RaBitQ residuals it centers.
 */

#include <math.h>
#include <string.h>

#include "index/posting_build.h"
#include "vs_test.h"

TEST_GROUP(Refine);
TEST_MEMCTX_FIXTURE();

/* Capture the vector the kernel hands to the head writer. */
typedef struct
{
	uint32_t  calls;
	uint32_t  leaf;
	Dimension dim;
	float	 *out;
} CaptureCtx;

static void
capture_write_head(void *arg, uint32_t leaf, const float *vec)
{
	CaptureCtx *c = (CaptureCtx *)arg;

	c->calls++;
	c->leaf = leaf;
	memcpy(c->out, vec, (size_t)c->dim * sizeof(float));
}

static double
norm_of(const float *v, Dimension dim)
{
	double sum = 0.0;

	for (Dimension i = 0; i < dim; i++)
		sum += (double)v[i] * v[i];
	return sqrt(sum);
}

/* Two orthogonal unit rows in one leaf: the mean has norm 1/sqrt(2), so a
 * cosine reference must be re-normalized to unit norm. */
TEST(refine_write_means_normalizes_cosine)
{
	enum
	{
		DIM	   = 8,
		LEAVES = 2
	};

	double	   sums[LEAVES * DIM] = {0};
	uint64_t   counts[LEAVES]	  = {0};
	float	   scratch[DIM];
	float	   out[DIM];
	CaptureCtx cap = {.dim = DIM, .out = out};

	sums[1 * DIM + 0] = 1.0;
	sums[1 * DIM + 1] = 1.0;
	counts[1]		  = 2;

	prism_refine_write_means(
			sums,
			counts,
			0,
			LEAVES,
			DIM,
			true,
			scratch,
			capture_write_head,
			&cap);

	ASSERT_EQ(1, cap.calls, "only the non-empty leaf is written");
	ASSERT_EQ(1, cap.leaf, "the non-empty leaf is the one written");
	ASSERT_FLOAT_EQ(
			1.0f,
			(float)norm_of(out, DIM),
			1e-6f,
			"cosine reference must be unit norm");
	ASSERT_FLOAT_EQ(
			(float)(1.0 / sqrt(2.0)),
			out[0],
			1e-6f,
			"mean direction is preserved");
	ASSERT_FLOAT_EQ(
			(float)(1.0 / sqrt(2.0)),
			out[1],
			1e-6f,
			"mean direction is preserved");
}

/* L2 references are plain means: the kernel must not normalize them. */
TEST(refine_write_means_keeps_l2_mean)
{
	enum
	{
		DIM	   = 8,
		LEAVES = 1
	};

	double	   sums[LEAVES * DIM] = {0};
	uint64_t   counts[LEAVES]	  = {0};
	float	   scratch[DIM];
	float	   out[DIM];
	CaptureCtx cap = {.dim = DIM, .out = out};

	sums[0]	  = 1.0;
	sums[1]	  = 1.0;
	counts[0] = 2;

	prism_refine_write_means(
			sums,
			counts,
			0,
			LEAVES,
			DIM,
			false,
			scratch,
			capture_write_head,
			&cap);

	ASSERT_EQ(1, cap.calls, "the leaf is written");
	ASSERT_FLOAT_EQ(0.5f, out[0], 1e-6f, "L2 mean is not normalized");
	ASSERT_FLOAT_EQ(0.5f, out[1], 1e-6f, "L2 mean is not normalized");
	ASSERT_FLOAT_EQ(0.0f, out[2], 1e-6f, "untouched components stay zero");
}

/* A leaf whose routed rows cancel to the origin keeps a zero reference: the
 * normalization must leave it finite rather than divide by zero. */
TEST(refine_write_means_zero_mean_stays_finite)
{
	enum
	{
		DIM	   = 4,
		LEAVES = 1
	};

	double	   sums[LEAVES * DIM] = {0};
	uint64_t   counts[LEAVES]	  = {0};
	float	   scratch[DIM];
	float	   out[DIM];
	CaptureCtx cap = {.dim = DIM, .out = out};

	counts[0] = 2;

	prism_refine_write_means(
			sums,
			counts,
			0,
			LEAVES,
			DIM,
			true,
			scratch,
			capture_write_head,
			&cap);

	ASSERT_EQ(1, cap.calls, "the leaf is written");
	for (Dimension i = 0; i < DIM; i++)
		ASSERT_FLOAT_EQ(0.0f, out[i], 1e-6f, "zero mean stays zero");
}

/* Empty leaves keep their sample-trained head: nothing is written for them,
 * and a fully empty tile writes nothing at all. */
TEST(refine_write_means_skips_empty_leaves)
{
	enum
	{
		DIM	   = 4,
		LEAVES = 3
	};

	double	   sums[LEAVES * DIM] = {0};
	uint64_t   counts[LEAVES]	  = {0};
	float	   scratch[DIM];
	float	   out[DIM];
	CaptureCtx cap = {.dim = DIM, .out = out};

	counts[2]		  = 1;
	sums[2 * DIM + 0] = 1.0;

	prism_refine_write_means(
			sums,
			counts,
			0,
			LEAVES,
			DIM,
			true,
			scratch,
			capture_write_head,
			&cap);

	ASSERT_EQ(1, cap.calls, "only the populated leaf is written");
	ASSERT_EQ(2, cap.leaf, "the populated leaf is the one written");
	ASSERT_FLOAT_EQ(1.0f, out[0], 1e-6f, "single-row leaf keeps its row");
}
