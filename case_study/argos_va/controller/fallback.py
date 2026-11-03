"""Guaranteed fallback via grouped QoS reduction (Algorithm 2).

When a decision is rejected, generate candidates by lowering every non-zero stream's QoS one
level at a time (uniform grouped reduction), keeping the relative allocation. This yields at
most L-1 candidates instead of the L^N of exhaustive search. Among the validated candidates
pick the one with the largest total QoS. The all-zero decision d_safe is always the last
candidate and is always accepted (its safety follows from the system model, not prediction),
so a validated decision always exists.
"""
from __future__ import annotations


def grouped_candidates(d_rej: tuple[int, ...]) -> list[tuple[int, ...]]:
    cands: list[tuple[int, ...]] = []
    d = list(d_rej)
    while any(q > 0 for q in d):
        d = [q - 1 if q > 0 else 0 for q in d]
        cands.append(tuple(d))
    return cands  # last element is all-zero == d_safe


def grouped_fallback(d_rej: tuple[int, ...], is_accepted) -> tuple[int, ...]:
    """`is_accepted(decision) -> bool` must accept d_safe unconditionally (Eq. 8)."""
    validated = [c for c in grouped_candidates(d_rej) if is_accepted(c)]
    if not validated:  # d_safe is always accepted, so this is defensive only
        return tuple(0 for _ in d_rej)
    return max(validated, key=sum)
