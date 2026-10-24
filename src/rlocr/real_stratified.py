"""Paired page statistics for frozen real-data predictions (no fitting)."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

METHODS = ("Native", "XY-cut", "Raw GAT", "Gated GAT", "Sequential PPO v2", "Page-level PPO")
SEEDS = (1337, 4242, 90210)
SEED = 314159


def paired_summary(method: Sequence[float], native: Sequence[float], *,
                   samples: int = 10_000, seed: int = SEED) -> dict[str, Any]:
    """Paired bootstrap of per-page differences; bounded temporary memory."""
    values, baseline = np.asarray(method,dtype=float), np.asarray(native,dtype=float)
    if values.shape != baseline.shape or values.ndim != 1 or samples < 1:
        raise ValueError("Need equally sized 1D arrays and positive sample count")
    if not np.isfinite(values).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite scores are not valid observations")
    delta = values-baseline
    if not len(delta):
        return {"pages":0,"mean_tau":None,"mean_difference":None,"ci95":None,
                "better":0,"worse":0,"same":0,"worst_single_page_loss":None,"worst_signed_difference":None}
    rng = np.random.default_rng(seed)
    means = np.concatenate([delta[rng.integers(0,len(delta),size=(min(512,samples-start),len(delta)))].mean(axis=1)
                            for start in range(0,samples,512)])
    return {"pages":len(delta),"mean_tau":float(values.mean()),"mean_difference":float(delta.mean()),
            "ci95":[float(value) for value in np.quantile(means,[.025,.975])],
            "better":int((delta>1e-9).sum()),"worse":int((delta < -1e-9).sum()),"same":int((np.abs(delta)<=1e-9).sum()),
            "worst_single_page_loss":float(max(0,-delta.min())),"worst_signed_difference":float(delta.min())}


def strata_for(row: dict) -> dict[str,str]:
    """No learned-model scores define strata. Native outcome is GT-conditioned."""
    return {"overall":"all", "layout":row["layout_type"],
            "line_count":"<=96" if row["number_of_lines"] <= 96 else ">96",
            "native_outcome":("excluded" if row["native_tau"] is None else "native-correct" if row["native_tau"] == 1.0 else "native-wrong"),
            "native_xy_agreement":">=0.98" if row["native_xy_tau"] >= .98 else "<0.98"}


def stratify(rows: list[dict]) -> dict:
    result = {}
    for dimension in strata_for(rows[0]):
        result[dimension] = {}
        for label in sorted({strata_for(row)[dimension] for row in rows}):
            selected = [row for row in rows if strata_for(row)[dimension] == label]
            scored = [row for row in selected if row["status"] == "scored"]
            result[dimension][label] = {"selected_pages":len(selected),"excluded_pages":len(selected)-len(scored),
                "methods":{method:paired_summary([row["scores"][method] for row in scored],
                                                 [row["native_tau"] for row in scored]) for method in METHODS}}
    return result


def switched_summary(native: Sequence[float], method: Sequence[float], switched: Sequence[bool]) -> dict:
    baseline, values, mask = np.asarray(native), np.asarray(method), np.asarray(switched,dtype=bool)
    differences = values-baseline
    improved = int(((differences>1e-9)&mask).sum())
    worse = int(((differences < -1e-9)&mask).sum())
    same = int(((np.abs(differences)<=1e-9)&mask).sum())
    count = int(mask.sum())
    fixed = int(((baseline<1)&(values==1)&mask).sum())
    broken = int(((baseline==1)&(values<1-1e-9)&mask).sum())
    return {"pages_switched":count,"improved":improved,"worse":worse,"same":same,
            "fraction_improved":improved/count if count else None,
            "fraction_worse":worse/count if count else None,"fraction_same":same/count if count else None,
            "fully_fixed":fixed,"native_correct_broken":broken,
            "net_pages_fixed_minus_broken":fixed-broken,"net_improved_minus_worsened":improved-worse}
