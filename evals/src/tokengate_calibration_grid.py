"""Policy grid generation and validation for TokenGate calibration."""

from typing import List, Tuple

from .tokengate_calibration_models import Policy


def generate_policy_grid(config: dict) -> List[Policy]:
    """Generate all parameter combinations from grid."""
    grid = config["parameter_grid"]
    policies = []

    policy_idx = 0
    for min_words in grid["min_words"]:
        for max_words in grid["max_words"]:
            for silence_timer_ms in grid["silence_timer_ms"]:
                for max_wait_timeout_ms in grid["max_wait_timeout_ms"]:
                    policy_id = f"policy_{policy_idx:04d}"
                    policies.append(
                        Policy(
                            policy_id=policy_id,
                            min_words=min_words,
                            max_words=max_words,
                            silence_timer_ms=silence_timer_ms,
                            max_wait_timeout_ms=max_wait_timeout_ms,
                            boundary_cues=grid.get("boundary_cues", ".?!\n"),
                        )
                    )
                    policy_idx += 1

    return policies


def filter_valid_policies(
    policies: List[Policy],
    config: dict,
) -> Tuple[List[Policy], List[Tuple[Policy, str]]]:
    """Filter out invalid parameter combinations."""
    constraints = config.get("validity_constraints", {})
    valid_policies = []
    invalid_policies = []

    for policy in policies:
        reasons = []

        # Constraint 1: min_words < max_words (strictly)
        if constraints.get("min_words_lt_max_words", True):
            if policy.min_words >= policy.max_words:
                reasons.append(
                    f"min_words ({policy.min_words}) >= max_words ({policy.max_words})"
                )

        # Constraint 2: max_wait_timeout_ms >= silence_timer_ms
        if constraints.get("max_wait_gte_silence", True):
            if policy.max_wait_timeout_ms < policy.silence_timer_ms:
                reasons.append(
                    f"max_wait_timeout_ms ({policy.max_wait_timeout_ms}) < "
                    f"silence_timer_ms ({policy.silence_timer_ms})"
                )

        # Constraint 3: Optional gap between min and max
        min_gap = constraints.get("min_gap_between_min_max")
        if min_gap is not None:
            if policy.max_words < policy.min_words + min_gap:
                reasons.append(
                    f"max_words ({policy.max_words}) < min_words ({policy.min_words}) + "
                    f"{min_gap}"
                )

        if reasons:
            invalid_policies.append((policy, "; ".join(reasons)))
        else:
            valid_policies.append(policy)

    return valid_policies, invalid_policies
