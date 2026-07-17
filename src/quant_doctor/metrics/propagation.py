"""Cross-layer error-propagation metrics — Phase 2.

Whether damage stays localized or grows with depth is a key discriminator
between failure modes (cf. QEP, NeurIPS 2025: quantization error grows roughly
exponentially across layers). These operate on the per-layer cosine profile.
"""

from __future__ import annotations

from statistics import median


def depth_trend(cosines: list[float]) -> float:
    """Pearson correlation between layer index and cosine.

    Strongly negative → damage worsens monotonically with depth (the signature
    of accumulating signal degradation). Near zero → no depth trend (a localized
    collapse or uniform format bug).
    """
    n = len(cosines)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(cosines) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, cosines))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in cosines)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x**0.5 * var_y**0.5)


def cliff_gap(cosines: list[float]) -> float:
    """How far the worst layer sits below the typical layer.

    median(cosine) - min(cosine). Large → a sharp cliff (one layer detonates
    while the rest are fine). Small → smooth/uniform damage.
    """
    return median(cosines) - min(cosines)


def clean_prefix_len(cosines: list[float], threshold: float) -> int:
    """Number of leading layers that are healthy before the first damaged one.

    A long clean prefix followed by a cliff is the computation-collapse pattern
    (information intact, then destroyed). Zero prefix (damage from layer 0) leans
    toward a format/decode bug.
    """
    n = 0
    for c in cosines:
        if c >= threshold:
            n += 1
        else:
            break
    return n
