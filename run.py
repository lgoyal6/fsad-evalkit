#!/usr/bin/env python3
"""Fill in the eval card.

    python run.py                                  # offline, toy data, ~30 s
    python run.py --backbone dinov2                # real DINOv2 ViT-S/14 features
    python run.py --data mvtec --mvtec-root PATH   # real MVTec AD (you supply it)

Prints one row per (category, k) with image AUROC, pixel AUROC, pixel AUPRO, and CPU
p50/p99 latency, plus a mean row. Writes the same thing to results.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import zlib

import numpy as np

from evalkit import backbones, data, metrics
from evalkit.patchcore import PatchCore, image_score

TOY_CATEGORIES = ["weave", "grain"]


def evaluate(split: data.Split, backbone, shots: list[int], seed: int, latency_reps: int):
    test_feats = backbone(split.test_images)
    rows = []
    for k in shots:
        if k > len(split.train_normal):
            continue
        idx = np.random.default_rng(seed).choice(len(split.train_normal), k, replace=False)
        core = PatchCore().fit(backbone(split.train_normal[idx]))
        maps = core.score(test_feats, data.IMG)

        # end-to-end single-image CPU latency: feature extraction + kNN + map.
        # A p99 estimated from N samples is a single order statistic; at N=100 it is the
        # slowest-but-one observation and is noisy. Raise --latency-reps for a stable one.
        lat = []
        for i in range(latency_reps):
            one = split.test_images[i % len(split.test_images) : i % len(split.test_images) + 1]
            t0 = time.perf_counter()
            core.score(backbone(one), data.IMG)
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        pct = lambda q: lat[int(round(q * (len(lat) - 1)))]  # noqa: E731

        rows.append(
            {
                "category": split.name,
                "k": k,
                "image_auroc_max": metrics.image_auroc(
                    image_score(maps, "max"), split.test_labels
                ),
                "image_auroc_top1pct": metrics.image_auroc(
                    image_score(maps, "top1pct"), split.test_labels
                ),
                "pixel_auroc": metrics.pixel_auroc(maps, split.test_masks),
                "pixel_aupro": metrics.aupro(maps, split.test_masks),
                "latency_p50_ms": pct(0.50),
                "latency_p95_ms": pct(0.95),
                "latency_p99_ms": pct(0.99),
                "n_latency_samples": len(lat),
                "n_test": int(len(split.test_labels)),
                "n_anomalous": int(split.test_labels.sum()),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["toy", "mvtec"], default="toy")
    ap.add_argument("--mvtec-root", default=None, help="dir holding the extracted MVTec AD")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--backbone", choices=["toy", "dinov2"], default="toy")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (0 = default)")
    ap.add_argument("--latency-reps", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=30)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.data == "mvtec" and not args.mvtec_root:
        ap.error("--data mvtec needs --mvtec-root (this repo never downloads MVTec AD)")

    synthetic = args.data == "toy"
    backbone = backbones.build(args.backbone, threads=args.threads)

    cats = args.categories or (TOY_CATEGORIES if synthetic else ["bottle", "grid", "screw"])
    rows = []
    for c in cats:
        split = (
            # zlib.crc32, not hash(): str hashing is salted per process, so hash() would
            # silently give a different dataset on every run.
            data.make_toy(
                c,
                n_train=16,
                n_test_each=args.test_per_class,
                seed=args.seed + zlib.crc32(c.encode()) % 100000,
            )
            if synthetic
            else data.load_mvtec(args.mvtec_root, c, args.test_per_class)
        )
        rows += evaluate(split, backbone, args.shots, args.seed, args.latency_reps)

    banner = "  [SYNTHETIC DATA - toy set, not evidence about real defects]" if synthetic else ""
    print(f"\ndataset={args.data}  backbone={backbone.name} ({backbone.note}){banner}")
    print(f"host={platform.machine()} {platform.system()}  threads={args.threads}\n")
    cols = ["imgAUROC max", "imgAUROC top1%", "pix AUROC", "pix AUPRO", "p50 ms", "p99 ms"]
    keys = [
        "image_auroc_max",
        "image_auroc_top1pct",
        "pixel_auroc",
        "pixel_aupro",
        "latency_p50_ms",
        "latency_p99_ms",
    ]
    hdr = f"| {'category':<10} | {'k':>2} | " + " | ".join(f"{c:>14}" for c in cols) + " |"
    print(hdr)
    print("|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|")

    def line(label, k, get):
        vals = " | ".join(f"{get(key):>14.3f}" for key in keys)
        print(f"| {label:<10} | {k:>2} | {vals} |")

    for r in rows:
        line(r["category"], r["k"], lambda key, r=r: r[key])
    for k in sorted({r["k"] for r in rows}):
        sub = [r for r in rows if r["k"] == k]
        line("MEAN", k, lambda key, sub=sub: statistics.fmean(r[key] for r in sub))

    meta = {
        "dataset": args.data,
        "synthetic": synthetic,
        "backbone": backbone.name,
        "backbone_note": backbone.note,
        "host": f"{platform.machine()} {platform.system()}",
        "threads": args.threads,
        "python": sys.version.split()[0],
        "seed": args.seed,
        "rows": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {args.out}")
    if synthetic:
        print("NOTE: every number above is SYNTHETIC. Re-run with --data mvtec for real ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
