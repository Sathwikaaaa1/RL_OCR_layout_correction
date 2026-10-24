"""Multiplicity-aware statistics for saved PPO page predictions."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from rlocr.gated_decoder import sign_flip_pvalue
from rlocr.real_stratified import paired_summary


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise correction, returned in original order."""
    values = np.asarray(p_values,dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(values,kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank,index in enumerate(order):
        running = max(running,min(1.0,(len(values)-rank)*values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def compare_pages(method: Sequence[float], reference: Sequence[float]) -> dict:
    result = paired_summary(method,reference,samples=10_000,seed=314159)
    delta = np.asarray(method,dtype=float)-np.asarray(reference,dtype=float)
    delta[np.abs(delta)<1e-12] = 0  # Remove arithmetic noise from seed averaging.
    result["sign_flip_p"] = sign_flip_pvalue(delta,10_000,314159)
    return result
