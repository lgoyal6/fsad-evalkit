"""Datasets for few-shot anomaly detection.

Two sources:
  * `toy`   - procedurally generated, ships with the repo, no download, deterministic.
  * `mvtec` - real MVTec AD, loaded from a directory you supply yourself.

The toy set exists so `python run.py` works offline. Every number produced on it is
SYNTHETIC and is labelled that way in the output. It is a plumbing test for the
harness, not evidence about any model's real-world accuracy.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

IMG = 224  # multiple of 14 so DINOv2 (patch 14) tiles it exactly


@dataclass
class Split:
    """One category. Images are float32 HxWx3 in [0, 1]."""

    name: str
    train_normal: np.ndarray  # (N, H, W, 3) - the pool few-shot k is drawn from
    test_images: np.ndarray  # (M, H, W, 3)
    test_labels: np.ndarray  # (M,) 1 = anomalous
    test_masks: np.ndarray  # (M, H, W) uint8, all-zero for normal images


# --------------------------------------------------------------------------- toy


def _noise(rng: np.random.Generator, sigma: float) -> np.ndarray:
    z = gaussian_filter(rng.standard_normal((IMG, IMG)), sigma, mode="wrap")
    return z / (z.std() + 1e-8)


@lru_cache(maxsize=None)
def _template(kind: str) -> np.ndarray:
    """The part, as the camera always sees it: fixed per category, no per-image jitter.

    Industrial inspection is a fixed camera looking at the same part in the same jig.
    Randomising global appearance per image would make every *normal* test image as
    far from the support set as an anomalous one, and the whole benchmark collapses to
    chance. That failure is easy to build by accident; it is one reason a benchmark
    needs to be published rather than described.
    """
    rng = np.random.default_rng(abs(zlib.crc32(kind.encode())))
    y, x = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    if kind == "weave":
        base = (
            0.50
            + 0.11 * np.sin(2 * np.pi * x / 8.0)
            + 0.11 * np.sin(2 * np.pi * y / 8.0)
            + 0.05 * _noise(rng, 3.0)
        )
    else:  # "grain"
        base = 0.50 + 0.09 * _noise(rng, 1.2) + 0.06 * _noise(rng, 6.0)
    return base.astype(np.float32)


def _texture(rng: np.random.Generator, kind: str) -> np.ndarray:
    """One acquisition of the part: template + sensor grain + slight lighting drift."""
    base = _template(kind) + 0.035 * _noise(rng, 0.8) + rng.normal(0.0, 0.010)
    tint = rng.normal(0.0, 0.008, size=3).astype(np.float32)
    return np.clip(base[..., None] + tint, 0.0, 1.0).astype(np.float32)


def _defect(rng: np.random.Generator) -> np.ndarray:
    """Return a signed additive field; nonzero only inside the defect.

    Amplitude and area are chosen to land in the same *regime* as the MVTec AD texture
    categories: defects are high-contrast enough that "does this image contain one" is
    nearly saturated, but small enough that "which pixels" is not. That regime is the
    whole reason image-level AUROC and pixel-level AUPRO have to be reported separately.
    The target was fixed before any model was run; see README "Why these constants".
    """
    field = np.zeros((IMG, IMG), np.float32)
    for _ in range(rng.integers(1, 3)):
        amp = float(rng.choice([-1.0, 1.0])) * rng.uniform(0.18, 0.30)
        if rng.random() < 0.5:  # scratch
            cx, cy = rng.integers(30, IMG - 30, size=2)
            ang, ln = rng.uniform(0, np.pi), rng.integers(30, 75)
            t = np.linspace(-ln / 2, ln / 2, int(ln) * 3)
            xs = np.clip((cx + t * np.cos(ang)).astype(int), 0, IMG - 1)
            ys = np.clip((cy + t * np.sin(ang)).astype(int), 0, IMG - 1)
            stamp = np.zeros((IMG, IMG), np.float32)
            stamp[ys, xs] = 1.0
            stamp = gaussian_filter(stamp, rng.uniform(0.8, 1.6))
        else:  # blob
            cx, cy = rng.integers(25, IMG - 25, size=2)
            rx, ry = rng.integers(6, 15, size=2)
            yy, xx = np.mgrid[0:IMG, 0:IMG]
            stamp = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
            stamp = gaussian_filter(stamp, 1.5)
        field += amp * (stamp / (stamp.max() + 1e-8))
    return field


def make_toy(category: str, n_train: int, n_test_each: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    train = np.stack([_texture(rng, category) for _ in range(n_train)])

    imgs, labels, masks = [], [], []
    for _ in range(n_test_each):  # normal
        imgs.append(_texture(rng, category))
        labels.append(0)
        masks.append(np.zeros((IMG, IMG), np.uint8))
    for _ in range(n_test_each):  # anomalous
        base, field = _texture(rng, category), _defect(rng)
        imgs.append(np.clip(base + field[..., None], 0.0, 1.0).astype(np.float32))
        labels.append(1)
        # ground truth = where the injected field actually changed the image
        masks.append((np.abs(field) > 0.25 * np.abs(field).max()).astype(np.uint8))

    order = np.random.default_rng(seed + 1).permutation(len(imgs))
    return Split(
        name=category,
        train_normal=train,
        test_images=np.stack(imgs)[order],
        test_labels=np.array(labels)[order],
        test_masks=np.stack(masks)[order],
    )


# ------------------------------------------------------------------------- mvtec


def load_mvtec(root: str, category: str, n_test_each: int | None = None) -> Split:
    """Load one MVTec AD category from an extracted `mvtec_anomaly_detection` dir.

    Layout expected (this is MVTec's own layout, unmodified):
        <root>/<category>/train/good/*.png
        <root>/<category>/test/<defect_type>/*.png     ("good" = normal)
        <root>/<category>/ground_truth/<defect_type>/*_mask.png

    MVTec AD is free for non-commercial use but requires accepting their licence and
    downloading it yourself: https://www.mvtec.com/company/research/datasets/mvtec-ad
    Nothing in this repo downloads it.
    """
    from PIL import Image

    def _load(p, gray=False):
        im = Image.open(p).convert("L" if gray else "RGB").resize((IMG, IMG), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        return a

    cdir = os.path.join(root, category)
    if not os.path.isdir(cdir):
        raise FileNotFoundError(f"no category {category!r} under {root!r}")

    gdir = os.path.join(cdir, "train", "good")
    train = np.stack([_load(os.path.join(gdir, f)) for f in sorted(os.listdir(gdir))])

    imgs, labels, masks = [], [], []
    tdir = os.path.join(cdir, "test")
    for defect in sorted(os.listdir(tdir)):
        ddir = os.path.join(tdir, defect)
        if not os.path.isdir(ddir):
            continue
        files = sorted(os.listdir(ddir))
        if n_test_each is not None:
            files = files[:n_test_each]
        for f in files:
            imgs.append(_load(os.path.join(ddir, f)))
            if defect == "good":
                labels.append(0)
                masks.append(np.zeros((IMG, IMG), np.uint8))
            else:
                labels.append(1)
                mp = os.path.join(cdir, "ground_truth", defect, f.replace(".png", "_mask.png"))
                m = _load(mp, gray=True) if os.path.exists(mp) else np.zeros((IMG, IMG), np.float32)
                masks.append((m > 0.5).astype(np.uint8))

    return Split(category, train, np.stack(imgs), np.array(labels), np.stack(masks))
