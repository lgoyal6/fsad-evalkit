"""PatchCore-style kNN anomaly scoring over frozen patch features.

Memory bank = every patch feature from the k normal support images. Anomaly score for
a test patch = Euclidean distance to its nearest neighbour in the bank. Pixel map =
that grid upsampled to image size and Gaussian-smoothed (sigma=4, as in the paper).
Image score = max of the pixel map.

No coreset subsampling: at k <= 10 support images the bank holds k*256 <= 2560 vectors,
so exact search is already fast and coreset selection would only lose recall. Add it
before you scale to hundreds of support images.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


class PatchCore:
    def __init__(self, smoothing_sigma: float = 4.0):
        self.bank: np.ndarray | None = None
        self.sigma = smoothing_sigma

    def fit(self, support_feats: np.ndarray) -> "PatchCore":
        """support_feats: (k, grid, grid, dim) from k NORMAL images."""
        self.bank = support_feats.reshape(-1, support_feats.shape[-1]).astype(np.float32)
        self._bank_sq = (self.bank**2).sum(1)
        return self

    def _nn_dist(self, q: np.ndarray) -> np.ndarray:
        # ||q - b||^2 = |q|^2 - 2 q.b + |b|^2 ; only the min over b is needed
        d2 = self._bank_sq[None, :] - 2.0 * (q @ self.bank.T)
        return np.sqrt(np.maximum((q**2).sum(1) + d2.min(1), 0.0))

    def score(self, feats: np.ndarray, image_size: int) -> np.ndarray:
        """feats: (n, grid, grid, dim). Returns pixel_maps (n, image_size, image_size)."""
        n, g, _, d = feats.shape
        maps = np.empty((n, image_size, image_size), np.float32)
        for i in range(n):
            grid_scores = self._nn_dist(feats[i].reshape(-1, d)).reshape(g, g)
            m = zoom(grid_scores, image_size / g, order=1)
            maps[i] = gaussian_filter(m, self.sigma)
        return maps


def image_score(maps: np.ndarray, mode: str = "max", top_frac: float = 0.01) -> np.ndarray:
    """Collapse a pixel map to one number per image.

    This choice is almost never stated in a vendor benchmark, and it moves the headline
    "detection accuracy" by a lot:

      max      - PatchCore's own choice. One pixel decides the image. On a clean, well
                 covered surface it is excellent; when normal images carry sensor noise
                 the max is an extreme-value statistic of ~50k noisy pixels and the
                 defect signal is buried in it.
      top1pct  - mean of the highest-scoring 1% of pixels. Same predictions, same model,
                 far more robust, and it usually reports a higher number.

    Neither is wrong. Reporting one of them as "detection accuracy" without saying which
    is what makes a single percentage unfalsifiable.
    """
    flat = maps.reshape(maps.shape[0], -1)
    if mode == "max":
        return flat.max(1)
    if mode == "top1pct":
        n = max(1, int(round(top_frac * flat.shape[1])))
        return np.partition(flat, -n, axis=1)[:, -n:].mean(1)
    raise ValueError(f"unknown image score mode {mode!r}")
