# TokenGate Calibration Report

**Calibration Run ID:** `calib_sha256:c_6bea5ea6_a800a8de`

## Executive Summary

**Selected Policy:** `policy_0400`

**Selection Method:** pareto3_utopia

**Selected Parameters:**
- `min_words`: 60
- `max_words`: 100
- `silence_timer_ms`: 1000
- `max_wait_timeout_ms`: 4000

## Literature-Informed Grid Justification

**min_words_range:** Larger chunks for 'bucket not pipe' behavior (30-70 words)

**max_words_range:** Typical sentence ranges and practical chunk sizes (80-160 words)

**silence_timer_range:** Streaming pause thresholds from prior studies (1-3s)

**max_wait_timeout_range:** Absolute upper bound to flush accumulated chunks (4-8s)

## Policy Validity Filter Results

- **Valid policies:** 625
- **Invalid policies:** 0

## Constraint Filter Results

- **Total policies evaluated:** 625
- **Survivor policies:** 250
- **Rejected policies:** 375

## Selection Rule Explanation

**Method Used:** pareto3_utopia

3-objective Pareto frontier analysis with utopia-distance selection was used. Objectives: minimize TTFF (Time To First Flush), minimize flush count (BufferAgent calls), maximize chunk size. First, non-dominated points (Pareto frontier) were identified in the 3D objective space. Then, the policy with minimum dimension-normalized Euclidean distance to the utopia point (1, 1, 1) in goodness space was selected. Normalization bounds were computed from survivor policies using percentile-based methods (P05/P95, or min/max for small-N cases).

## Normalization Bounds

Normalization bounds were computed from survivor policies:

- **ttff_content_p50_ms**: [1242ms, 1502ms] (method: p05_p95) ✅ Active
- **flush_count_mean**: [46.4, 53.75] (method: p05_p95) ✅ Active
- **chunk_size_p50**: [57, 66] (method: p05_p95) ✅ Active

## Top 5 Policies

| Policy ID | min_words | max_words | silence_timer_ms | max_wait_timeout_ms | TTFF (p50) | Chunk Size (p50) | Weighted Score |
|-----------|-----------|-----------|-------------------|---------------------|------------|------------------|----------------|
| policy_0400 | 60 | 100 | 1000 | 4000 | 1502.0 | 66.0 | 0.6175 |
| policy_0401 | 60 | 100 | 1000 | 5000 | 1502.0 | 66.0 | 0.6175 |
| policy_0402 | 60 | 100 | 1000 | 6000 | 1502.0 | 66.0 | 0.6175 |
| policy_0403 | 60 | 100 | 1000 | 7000 | 1502.0 | 66.0 | 0.6175 |
| policy_0404 | 60 | 100 | 1000 | 8000 | 1502.0 | 66.0 | 0.6175 |

## Spam Sensitivity Analysis

Spam metrics recomputed for different α values. Includes:
- Selected policy
- Top 5 policies by utopia distance
- Top 5 policies by weighted score

| Policy ID | Rank (Utopia) | Utopia Dist | Rank (Weighted) | α=0.5 | α=0.6 | α=0.7 | α=0.8 |
|-----------|---------------|-------------|-----------------|-------|-------|-------|-------|
| **policy_0400** (selected) | 1 | 0.5774 | 1 | 5.30% | 7.06% | 8.83% | 10.76% |
| policy_0401 | 2 | 0.5774 | 2 | 5.30% | 7.06% | 8.83% | 10.76% |
| policy_0402 | 3 | 0.5774 | 3 | 5.30% | 7.06% | 8.83% | 10.76% |
| policy_0403 | 4 | 0.5774 | 4 | 5.30% | 7.06% | 8.83% | 10.76% |
| policy_0404 | 5 | 0.5774 | 5 | 5.30% | 7.06% | 8.83% | 10.76% |

## Reproducibility

- **Trace Dataset Hash:** `sha256:c173b1f838a346f2aab8c44e10e3807140c4f28cf105f315ba55fbbf681b12b8`
- **MAS Run ID:** `mas_1d5b227a_gpt4omini_0ad5f2b4_d22060bf`
- **EXAID Commit:** `a800a8de806ad556616561fa12aeb8bb4bf8a805`
- **Config Hash:** `6bea5ea6048899d61eefb8da3dc7d7260f4044df4ce92e7f669575a0db646525`

