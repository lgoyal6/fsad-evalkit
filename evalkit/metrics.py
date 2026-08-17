"""The three numbers a defect-detection claim needs, plus the one it usually hides.

  image_auroc  - "is this image defective at all?"      threshold-free, per image
  pixel_auroc  - "which pixels?"                        threshold-free, per pixel
  aupro        - "which pixels?" but every defect region counts equally, so one
                 8000-pixel scratch cannot drown out twenty 40-pixel pinholes.
                 Integrated to FPR 0.3 and normalised, as defined in Bergmann et al.,
                 "Uninformed Students" (CVPR 2020), and used by every MVTec AD paper.

AUPRO is the number that moves when a model is genuinely good at localisation rather
than good at guessing that a picture contains something. It is almost always the
lowest of the three, which is why a single unlabelled percentage is unfalsifiable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. labels in {0, 1}."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels).ravel()
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0  # average rank for ties
        i = j + 1
    return float((ranks[labels[order] == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def image_auroc(image_scores: np.ndarray, labels: np.ndarray) -> float:
    return auroc(image_scores, labels)


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    return auroc(maps.ravel(), masks.ravel())


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3, n_thresh: int = 300) -> float:
    """Per-region overlap vs false-positive rate, integrated to `fpr_limit`."""
    regions: list[np.ndarray] = []  # score values inside each connected GT region
    for m, gt in zip(maps, masks):
        if gt.max() == 0:
            continue
        lab, n = label(gt)
        for r in range(1, n + 1):
            regions.append(np.sort(m[lab == r]))
    if not regions:
        return float("nan")

    normal_scores = np.sort(maps[masks == 0].ravel())
    lo = min(float(maps.min()), float(normal_scores[0]))
    hi = float(maps.max())
    thresholds = np.linspace(hi, lo, n_thresh)  # ascending FPR

    # overlap(region, t) = fraction of that region's pixels with score >= t
    pro = np.zeros(n_thresh)
    for r in regions:
        pro += (len(r) - np.searchsorted(r, thresholds, side="left")) / len(r)
    pro /= len(regions)
    fpr = (len(normal_scores) - np.searchsorted(normal_scores, thresholds, side="left")) / len(
        normal_scores
    )

    keep = fpr <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p = fpr[keep], pro[keep]
    if f[-1] < fpr_limit:  # extend flat to the limit so every run integrates the same span
        f, p = np.append(f, fpr_limit), np.append(p, p[-1])
    return float(np.trapezoid(p, f) / fpr_limit)
