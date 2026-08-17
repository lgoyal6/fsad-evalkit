"""Patch feature extractors.

Both return (n_images, grid, grid, dim) float32 patch features on a 16x16 grid for a
224x224 input.

  * `toy`    - fixed random projection of raw patches. No weights, no download, runs
               offline in milliseconds. It is a stand-in so the harness is runnable
               without pulling 90 MB of weights; it is NOT DINOv2 and results obtained
               with it say nothing about DINOv2.
  * `dinov2` - the real thing (ViT-S/14, self-supervised, frozen). Downloads weights
               on first use, then caches. This is the backbone the eval card assumes.

Both apply PatchCore's "locally aware" 3x3 neighbourhood average over the patch grid,
which is the step that makes a frozen backbone competitive at localisation.
"""

from __future__ import annotations

import numpy as np

GRID = 16  # 224 / 14
PATCH = 14

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _neighbourhood_pool(feats: np.ndarray) -> np.ndarray:
    """3x3 average pool over the patch grid, stride 1, edge-replicated."""
    p = np.pad(feats, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(feats)
    for dy in range(3):
        for dx in range(3):
            out += p[:, dy : dy + GRID, dx : dx + GRID, :]
    return out / 9.0


class ToyBackbone:
    name = "toy"
    note = "fixed random projection of raw pixels (NOT DINOv2)"

    def __init__(self, dim: int = 96, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((PATCH * PATCH * 3, dim)).astype(np.float32)
        self.proj /= np.sqrt(PATCH * PATCH * 3)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        n = images.shape[0]
        x = (images - _IMAGENET_MEAN) / _IMAGENET_STD
        # (n, GRID, PATCH, GRID, PATCH, 3) -> flatten each patch
        x = x.reshape(n, GRID, PATCH, GRID, PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(n, GRID, GRID, PATCH * PATCH * 3)
        return _neighbourhood_pool(x @ self.proj)


class DINOv2Backbone:
    name = "dinov2"
    note = "DINOv2 ViT-S/14 (frozen, self-supervised), timm weights"

    def __init__(self, model: str = "vit_small_patch14_dinov2.lvd142m", threads: int = 0):
        import timm
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model = timm.create_model(model, pretrained=True, num_classes=0, img_size=224)
        self.model.eval()

    def __call__(self, images: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = (images - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
        outs = []
        with torch.no_grad():
            for i in range(0, t.shape[0], 8):
                f = self.model.forward_features(t[i : i + 8])
                f = f[:, self.model.num_prefix_tokens :, :]  # drop CLS / registers
                outs.append(f.numpy())
        f = np.concatenate(outs).reshape(-1, GRID, GRID, outs[0].shape[-1])
        return _neighbourhood_pool(f.astype(np.float32))


def build(name: str, threads: int = 0):
    if name == "toy":
        return ToyBackbone()
    if name == "dinov2":
        return DINOv2Backbone(threads=threads)
    raise ValueError(f"unknown backbone {name!r}")
