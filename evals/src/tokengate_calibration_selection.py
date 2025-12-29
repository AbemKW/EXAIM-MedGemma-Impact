"""Selection logic for TokenGate calibration."""

from typing import Callable, List, Optional, Tuple

from .tokengate_calibration_models import PolicyMetrics


def normalize_value(value: float, lower: float, upper: float, invert: bool = False) -> float:
    """Normalize value to [0, 1] range."""
    if value < lower:
        normalized = 0.0
    elif value > upper:
        normalized = 1.0
    else:
        normalized = (value - lower) / (upper - lower) if upper > lower else 0.0

    if invert:
        return 1.0 - normalized
    return normalized


def normalize_to_goodness(
    value: float,
    lo: float,
    hi: float,
    lower_is_better: bool,
) -> float:
    """
    Convert raw metric value to [0, 1] goodness space.

    Args:
        value: Raw metric value
        lo: Lower bound for normalization
        hi: Upper bound for normalization
        lower_is_better: True if lower values are better (e.g., TTFF, flush_count)

    Returns:
        Goodness value in [0, 1] where 1.0 is best
    """
    if hi == lo:
        # Degenerate bounds - return neutral value (shouldn't happen if bounds computed correctly)
        return 0.5

    # Normalize to [0, 1]
    norm = max(0.0, min(1.0, (value - lo) / (hi - lo)))

    # Convert to goodness: if lower is better, invert
    if lower_is_better:
        return 1.0 - norm
    return norm


def compute_weighted_score(
    policy_metrics: PolicyMetrics,
    config: dict,
    computed_bounds: Optional[dict] = None,
    dropped_metrics: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Compute weighted objective function score.

    If computed_bounds and dropped_metrics are provided, uses data-driven bounds
    and renormalizes weights when metrics are dropped.

    Args:
        policy_metrics: PolicyMetrics to score
        config: Configuration dict
        computed_bounds: Optional computed bounds dict (if None, uses config bounds)
        dropped_metrics: Optional list of dropped metric names (if None, uses all metrics)

    Returns:
        Weighted score, or None if all weights were dropped
    """
    selection_config = config.get("selection", {}).get("weighted_score", {})
    weights = selection_config.get("weights", {})

    if dropped_metrics is None:
        dropped_metrics = []

    # Determine active metrics (not dropped)
    all_metrics = [
        "ttff_content_p50_ms",
        "flush_count_mean",
        "chunk_size_p50",
        "spam_pct_mean",
        "worst_wait_p95_ms",
    ]
    active_metrics = [m for m in all_metrics if m not in dropped_metrics]

    if not active_metrics:
        # All metrics dropped - return None to signal lexicographic fallback
        return None

    # Compute active weights sum for renormalization
    active_weights_sum = sum(weights.get(m, 0.0) for m in active_metrics)

    if active_weights_sum == 0:
        # All active weights are zero - return None
        return None

    # Use computed bounds if provided, otherwise fallback to config bounds
    if computed_bounds is not None:
        bounds = computed_bounds
    else:
        bounds = selection_config.get("normalization_bounds", {})

    score = 0.0

    # TTFF (lower is better)
    if "ttff_content_p50_ms" in active_metrics and policy_metrics.ttff_content_p50_ms is not None:
        w = weights.get("ttff_content_p50_ms", 0.0)
        if w > 0:
            # Renormalize weight
            normalized_w = w / active_weights_sum
            if computed_bounds and "ttff_content_p50_ms" in computed_bounds:
                b = computed_bounds["ttff_content_p50_ms"]
                lo = b.get("lo_ms", b.get("lo", 5000))
                hi = b.get("hi_ms", b.get("hi", 30000))
            else:
                b = bounds.get("ttff_content_p50_ms", {"lower": 5000, "upper": 30000})
                lo = b.get("lower", 5000)
                hi = b.get("upper", 30000)
            score += normalized_w * normalize_value(policy_metrics.ttff_content_p50_ms, lo, hi, invert=True)

    # Flush count (lower is better)
    if "flush_count_mean" in active_metrics and policy_metrics.flush_count_mean is not None:
        w = weights.get("flush_count_mean", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "flush_count_mean" in computed_bounds:
                b = computed_bounds["flush_count_mean"]
                lo = b.get("lo", 10)
                hi = b.get("hi", 150)
            else:
                b = bounds.get("flush_count_mean", {"lower": 10, "upper": 150})
                lo = b.get("lower", 10)
                hi = b.get("upper", 150)
            score += normalized_w * normalize_value(policy_metrics.flush_count_mean, lo, hi, invert=True)

    # Chunk size (higher is better)
    if "chunk_size_p50" in active_metrics and policy_metrics.chunk_size_p50 is not None:
        w = weights.get("chunk_size_p50", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "chunk_size_p50" in computed_bounds:
                b = computed_bounds["chunk_size_p50"]
                lo = b.get("lo", 30)
                hi = b.get("hi", 160)
            else:
                b = bounds.get("chunk_size_p50", {"lower": 30, "upper": 160})
                lo = b.get("lower", 30)
                hi = b.get("upper", 160)
            score += normalized_w * normalize_value(policy_metrics.chunk_size_p50, lo, hi, invert=False)

    # Spam (lower is better)
    if "spam_pct_mean" in active_metrics and policy_metrics.spam_pct_mean is not None:
        w = weights.get("spam_pct_mean", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "spam_pct_mean" in computed_bounds:
                b = computed_bounds["spam_pct_mean"]
                lo = b.get("lo", 0)
                hi = b.get("hi", 10)
            else:
                b = bounds.get("spam_pct_mean", {"lower": 0, "upper": 10})
                lo = b.get("lower", 0)
                hi = b.get("upper", 10)
            score += normalized_w * normalize_value(policy_metrics.spam_pct_mean, lo, hi, invert=True)

    # Worst wait (lower is better)
    if "worst_wait_p95_ms" in active_metrics and policy_metrics.worst_wait_p95_ms is not None:
        w = weights.get("worst_wait_p95_ms", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "worst_wait_p95_ms" in computed_bounds:
                b = computed_bounds["worst_wait_p95_ms"]
                lo = b.get("lo", 10000)
                hi = b.get("hi", 60000)
            else:
                b = bounds.get("worst_wait_p95_ms", {"lower": 10000, "upper": 60000})
                lo = b.get("lower", 10000)
                hi = b.get("upper", 60000)
            score += normalized_w * normalize_value(policy_metrics.worst_wait_p95_ms, lo, hi, invert=True)

    return score


def build_pareto_frontier_3d(
    points: List[Tuple[PolicyMetrics, List[float]]],
    active_dimensions: List[int],
) -> List[Tuple[PolicyMetrics, List[float]]]:
    """
    Build k-dimensional Pareto frontier (non-dominated points).

    Excludes dropped metrics from dominance test by using only active dimensions.

    A point a dominates point b if:
    - For all active dimensions i: a[i] >= b[i] (all objectives better or equal)
    - AND exists dimension j where a[j] > b[j] (strictly better in at least one)

    Args:
        points: List of (policy_metrics, goodness_vector) tuples
        active_dimensions: List of dimension indices to use (e.g., [0, 1, 2] if all active, [0, 2] if dimension 1 dropped)

    Returns:
        List of non-dominated points (Pareto frontier)
    """
    if not points:
        return []

    if not active_dimensions:
        # No active dimensions - return all points (degenerate case)
        return points

    frontier = []
    for i, (pm_i, vec_i) in enumerate(points):
        is_dominated = False

        # Extract active dimensions for point i
        active_i = [vec_i[d] for d in active_dimensions]

        for j, (pm_j, vec_j) in enumerate(points):
            if i == j:
                continue

            # Extract active dimensions for point j
            active_j = [vec_j[d] for d in active_dimensions]

            # Check if point j dominates point i
            # j dominates i if: all active_j[k] >= active_i[k] AND exists k where active_j[k] > active_i[k]
            all_better_or_equal = all(
                active_j[k] >= active_i[k] for k in range(len(active_dimensions))
            )
            strictly_better = any(
                active_j[k] > active_i[k] for k in range(len(active_dimensions))
            )

            if all_better_or_equal and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            frontier.append((pm_i, vec_i))

    return frontier


def select_pareto_utopia(
    survivor_metrics: List[PolicyMetrics],
    computed_bounds: dict,
    dropped_metrics: List[str],
    config: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[PolicyMetrics, str, dict]:
    """
    Select best policy using k-dimensional Pareto frontier + utopia-distance selection.

    Process:
    1. Extract 3D objective values (TTFF, flush_count, chunk_size) from survivors
    2. Normalize to goodness space using computed bounds
    3. Exclude dropped metrics from both Pareto dominance AND distance computation
    4. Build k-dimensional Pareto frontier (k = number of active dimensions)
    5. Compute dimension-normalized Euclidean distance to utopia point (1, 1, ..., 1)
    6. Select point with minimum distance
    7. Use deterministic tie-breaking if distances are equal

    Args:
        survivor_metrics: List of PolicyMetrics that passed constraints
        computed_bounds: Dict mapping metric names to bounds metadata
        dropped_metrics: List of metric names that were dropped (degenerate bounds)
        config: Configuration dict (for weighted score fallback)
        logger: Optional logger for debug messages

    Returns:
        Tuple of (selected_policy, selection_method, metadata_dict)
    """
    import math

    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    if not survivor_metrics:
        raise ValueError("No survivor policies to select from")

    # Define metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    metric_lower_is_better = [
        True,
        True,
        False,
    ]  # TTFF and flush_count: lower is better; chunk_size: higher is better

    # Determine active dimensions (exclude dropped metrics)
    active_dimensions = []
    for i, metric_name in enumerate(metric_order):
        if metric_name not in dropped_metrics:
            active_dimensions.append(i)

    if not active_dimensions:
        # All metrics dropped - use lexicographic fallback
        log("DEBUG: All metrics dropped, using lexicographic fallback")
        # Deterministic tie-breaking order: lower flush_count, higher chunk_size, lower TTFF, smallest policy_id
        sorted_policies = sorted(
            survivor_metrics,
            key=lambda pm: (
                pm.flush_count_mean if pm.flush_count_mean is not None else float("inf"),
                -(pm.chunk_size_p50 if pm.chunk_size_p50 is not None else -float("inf")),
                pm.ttff_content_p50_ms if pm.ttff_content_p50_ms is not None else float("inf"),
                pm.policy_id,
            ),
        )
        return (
            sorted_policies[0],
            "lexicographic_fallback",
            {"dropped_metrics": dropped_metrics},
        )

    # Extract and normalize to goodness space
    points = []
    for pm in survivor_metrics:
        goodness_vector = []
        for i, metric_name in enumerate(metric_order):
            if i in active_dimensions:
                value = getattr(pm, metric_name, None)
                if value is None:
                    # Skip policies with missing values
                    break

                bounds = computed_bounds[metric_name]
                if metric_name == "ttff_content_p50_ms":
                    lo = bounds["lo_ms"]
                    hi = bounds["hi_ms"]
                else:
                    lo = bounds["lo"]
                    hi = bounds["hi"]

                goodness = normalize_to_goodness(value, lo, hi, metric_lower_is_better[i])
                goodness_vector.append(goodness)

        if len(goodness_vector) == len(active_dimensions):
            points.append((pm, goodness_vector))

    if not points:
        # No valid points - fallback to weighted score
        log(
            "DEBUG: No valid points with all required metrics, falling back to weighted_score"
        )
        for pm in survivor_metrics:
            pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)
        best_policy = max(
            survivor_metrics,
            key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1,
        )
        return best_policy, "weighted_fallback", {"dropped_metrics": dropped_metrics}

    # After compressing vectors to active dimensions, remap active_dimensions to [0, 1, ..., k-1]
    # This fixes the index mismatch bug when metrics are dropped
    k = len(points[0][1]) if points else 0
    compressed_active_dimensions = list(range(k))

    # Build k-dimensional Pareto frontier
    frontier = build_pareto_frontier_3d(points, compressed_active_dimensions)

    log(
        f"DEBUG: Pareto frontier: {len(frontier)} non-dominated points (filtered from {len(points)} total)"
    )
    log(f"DEBUG: Active dimensions: {len(active_dimensions)} (excluded dropped metrics: {dropped_metrics})")

    if not frontier:
        # Empty frontier - fallback to weighted score
        log("DEBUG: Empty Pareto frontier, falling back to weighted_score")
        for pm in survivor_metrics:
            pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)
        best_policy = max(
            survivor_metrics,
            key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1,
        )
        return best_policy, "weighted_fallback", {"dropped_metrics": dropped_metrics}

    # Compute dimension-normalized Euclidean distance to utopia
    k = len(active_dimensions)
    utopia = [1.0] * k  # Utopia point: all goodness values = 1.0

    distances = []
    for pm, goodness_vec in frontier:
        # Dimension-normalized distance: sqrt(mean((1 - goodness_i)^2))
        squared_diffs = [(1.0 - goodness_vec[i]) ** 2 for i in range(k)]
        distance = math.sqrt(sum(squared_diffs) / k)
        distances.append((pm, goodness_vec, distance))

    # Find minimum distance
    min_distance = min(dist for _, _, dist in distances)

    # Collect candidates with minimum distance (within tolerance)
    tolerance = 1e-10
    candidates = [
        (pm, vec, dist) for pm, vec, dist in distances if abs(dist - min_distance) < tolerance
    ]

    if len(candidates) == 1:
        selected_policy = candidates[0][0]
        log(f"DEBUG: Selected policy {selected_policy.policy_id} with utopia distance {min_distance:.6f}")
        return (
            selected_policy,
            "pareto3_utopia",
            {"dropped_metrics": dropped_metrics, "utopia_distance": min_distance},
        )

    # Tie-breaking: deterministic order
    log(
        f"DEBUG: {len(candidates)} candidates tied at distance {min_distance:.6f}, applying tie-breaking"
    )
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            x[0].flush_count_mean if x[0].flush_count_mean is not None else float("inf"),
            -(x[0].chunk_size_p50 if x[0].chunk_size_p50 is not None else -float("inf")),
            x[0].ttff_content_p50_ms if x[0].ttff_content_p50_ms is not None else float("inf"),
            x[0].policy_id,
        ),
    )

    selected_policy = sorted_candidates[0][0]
    log(f"DEBUG: Selected policy {selected_policy.policy_id} after tie-breaking")
    return (
        selected_policy,
        "pareto3_utopia",
        {"dropped_metrics": dropped_metrics, "utopia_distance": min_distance},
    )


def compute_utopia_distances_for_all(
    survivor_metrics: List[PolicyMetrics],
    computed_bounds: dict,
    dropped_metrics: List[str],
) -> List[Tuple[PolicyMetrics, float]]:
    """
    Compute utopia distances for all survivor policies (not just frontier).

    This is used for ranking and reporting top policies by utopia distance.

    Args:
        survivor_metrics: List of all survivor policies
        computed_bounds: Computed normalization bounds
        dropped_metrics: List of dropped metric names

    Returns:
        List of (policy_metrics, utopia_distance) tuples, sorted by distance (ascending)
    """
    import math

    # Define metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    metric_lower_is_better = [True, True, False]

    # Determine active dimensions
    active_dimensions = []
    for i, metric_name in enumerate(metric_order):
        if metric_name not in dropped_metrics:
            active_dimensions.append(i)

    if not active_dimensions:
        # All metrics dropped - return empty list
        return []

    k = len(active_dimensions)
    utopia = [1.0] * k

    distances = []
    for pm in survivor_metrics:
        goodness_vector = []
        for i, metric_name in enumerate(metric_order):
            if i in active_dimensions:
                value = getattr(pm, metric_name, None)
                if value is None:
                    break

                bounds = computed_bounds[metric_name]
                if metric_name == "ttff_content_p50_ms":
                    lo = bounds["lo_ms"]
                    hi = bounds["hi_ms"]
                else:
                    lo = bounds["lo"]
                    hi = bounds["hi"]

                goodness = normalize_to_goodness(value, lo, hi, metric_lower_is_better[i])
                goodness_vector.append(goodness)

        if len(goodness_vector) == k:
            # Compute dimension-normalized distance
            squared_diffs = [(1.0 - goodness_vector[i]) ** 2 for i in range(k)]
            distance = math.sqrt(sum(squared_diffs) / k)
            distances.append((pm, distance))

    # Sort by distance (ascending - lower is better)
    distances.sort(key=lambda x: x[1])
    return distances
