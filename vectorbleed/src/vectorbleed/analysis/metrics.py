"""Metrics computation for VectorBleed experiment analysis."""

import numpy as np


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score between reference and hypothesis text."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words or not hyp_words:
        return 0.0

    lcs_len = _lcs_length(ref_words, hyp_words)

    precision = lcs_len / len(hyp_words)
    recall = lcs_len / len(ref_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def compute_topic_reconstruction_accuracy(
    actual_topics: list[str],
    inferred_topics: list[str],
) -> float:
    """Compute what percentage of actual topics were correctly inferred.

    Uses fuzzy matching — a topic is "inferred" if any inferred topic
    has significant word overlap with it.
    """
    if not actual_topics:
        return 0.0

    matches = 0
    for actual in actual_topics:
        actual_words = set(actual.lower().split())
        for inferred in inferred_topics:
            inferred_words = set(inferred.lower().split())
            overlap = len(actual_words & inferred_words) / max(len(actual_words), 1)
            if overlap >= 0.3:  # 30% word overlap threshold
                matches += 1
                break

    return matches / len(actual_topics)


def compute_information_leakage_bits(
    score_distribution_same: list[float],
    score_distribution_cross: list[float],
) -> float:
    """Estimate information leakage in bits from score distributions.

    Uses KL divergence between same-tenant and cross-tenant score distributions
    as a proxy for information leakage.
    """
    if not score_distribution_same or not score_distribution_cross:
        return 0.0

    # Bin the distributions
    bins = np.linspace(0, 1, 50)
    hist_same, _ = np.histogram(score_distribution_same, bins=bins, density=True)
    hist_cross, _ = np.histogram(score_distribution_cross, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_same = hist_same + eps
    hist_cross = hist_cross + eps

    # Normalize
    hist_same = hist_same / hist_same.sum()
    hist_cross = hist_cross / hist_cross.sum()

    # KL divergence (bits)
    kl_div = np.sum(hist_cross * np.log2(hist_cross / hist_same))

    return max(0.0, float(kl_div))


def compute_isolation_strength(
    total_queries: int,
    cross_tenant_leaks: int,
    max_leak_score: float,
) -> dict:
    """Compute overall isolation strength metrics.

    Returns:
        Dictionary with isolation metrics
    """
    leak_rate = cross_tenant_leaks / max(total_queries, 1)

    # Isolation grade
    if leak_rate == 0 and max_leak_score < 0.3:
        grade = "A"
        assessment = "Strong isolation — no detectable leakage"
    elif leak_rate < 0.05:
        grade = "B"
        assessment = "Good isolation — minimal leakage signals"
    elif leak_rate < 0.2:
        grade = "C"
        assessment = "Moderate isolation — some leakage detected"
    elif leak_rate < 0.5:
        grade = "D"
        assessment = "Weak isolation — significant leakage"
    else:
        grade = "F"
        assessment = "Failed isolation — widespread cross-tenant access"

    return {
        "grade": grade,
        "assessment": assessment,
        "leak_rate": leak_rate,
        "max_leak_score": max_leak_score,
        "total_queries": total_queries,
        "cross_tenant_leaks": cross_tenant_leaks,
    }
