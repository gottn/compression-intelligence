"""Drift of a checkpoint against its base from per-token bits on identical tokens (exp06). Pure.

drift = 100 (Σ bits_post − Σ bits_base) / Σ bits_base, in %, per corpus and per token class; the fresh value is the
unweighted mean over the fresh corpora (exp03-exp05 convention). Classes are exp05's (cilab.tokclass): the POS scheme
(knowledge / middle / style) and the frequency scheme (rare / frequent). Contrasts are differences of fresh class
drifts, in percentage points: contrast_ks = drift_knowledge − drift_style, contrast_rf = drift_rare − drift_frequent.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from cilab.tokclass import FREQ_CLASSES, POS_CLASSES, class_sums

SCHEMES: Mapping[str, tuple[str, ...]] = {"pos": POS_CLASSES, "freq": FREQ_CLASSES}


def pct_change(base_bits: float, post_bits: float) -> float:
    return 100.0 * (post_bits - base_bits) / base_bits


def class_drift(base: np.ndarray, post: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Drift (%) per class code 0..n_classes−1."""
    if base.shape != post.shape:
        raise ValueError(f"base {base.shape} vs post {post.shape}")
    b, p = class_sums(base, codes, n_classes), class_sums(post, codes, n_classes)
    return 100.0 * (p - b) / b


def drift_row(base: Mapping[str, np.ndarray], post: Mapping[str, np.ndarray],
              classes: Mapping[str, Mapping[str, np.ndarray]], fresh: Sequence[str]) -> dict[str, float]:
    """Drift of `post` against `base` (per-token bits per corpus): dbpb_<corpus> for every corpus, dbpb_fresh, and the
    fresh-mean drift_<class> of both schemes with the two contrasts. classes[corpus][scheme] are per-token codes."""
    row = {f"dbpb_{c}": pct_change(float(base[c].sum()), float(post[c].sum())) for c in sorted(base)}
    row["dbpb_fresh"] = float(np.mean([row[f"dbpb_{c}"] for c in fresh]))
    for scheme, names in SCHEMES.items():
        per = np.array([class_drift(base[c], post[c], classes[c][scheme], len(names)) for c in fresh])
        row.update({f"drift_{n}": float(v) for n, v in zip(names, per.mean(axis=0), strict=True)})
    row["contrast_ks"] = row["drift_knowledge"] - row["drift_style"]
    row["contrast_rf"] = row["drift_rare"] - row["drift_frequent"]
    return row
