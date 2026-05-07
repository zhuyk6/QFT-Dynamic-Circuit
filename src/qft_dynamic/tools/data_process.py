def calc_tvd(ideal_prob: dict[int, float], noisy_counts: dict[int, int]) -> float:
    """
    Calculate the Total Variation Distance (TVD) between the ideal probability
    and the noisy frequency distribution.

    Args:
        ideal_prob (dict[int, float]): Ideal probability distribution.
        noisy_counts (dict[int, int]): Noisy counts from the experiment.

    Returns:
        float: The Total Variation Distance (TVD).
    """
    all_keys: set[int] = set(ideal_prob.keys()) | set(noisy_counts.keys())

    total_noisy: int = sum(noisy_counts.values())
    tvd_sum: float = 0.0
    for k in all_keys:
        p_ideal: float = ideal_prob.get(k, 0.0)
        p_noisy: float = noisy_counts.get(k, 0) / total_noisy
        tvd_sum += abs(p_ideal - p_noisy)

    return 0.5 * tvd_sum


def calc_fidelity(p: dict[int, float], q: dict[int, float]) -> float:
    """
    Calculate the fidelity between two probability distributions.

    fidelity(p, q) = sum(sqrt(p_i * q_i))^2
    """
    union_keys: set[int] = set(p.keys()) | set(q.keys())
    fidelity_sum = sum((p.get(k, 0.0) * q.get(k, 0.0)) ** 0.5 for k in union_keys)
    return fidelity_sum**2
