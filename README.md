# fsad-evalkit

A small, runnable few-shot anomaly-detection baseline that fills in an eval card:
**DINOv2 frozen features + PatchCore-style kNN memory bank**, reporting image-level
AUROC, pixel-level AUROC, pixel-level AUPRO, a few-shot sweep over k = 1/2/5/10, and CPU
p50/p95/p99 inference latency.

The companion document is `../EVAL_CARD.md`. The harness exists to show that every field
on that card is cheap to produce: this is 369 lines of code and runs on a laptop CPU.

---

## The short version

**What I noticed.** Allus publishes a table claiming 99.9 on Defect Detection against
YOLO11's 92.5. Their other table names the benchmark and the parameter count per row. This
one names no dataset, no metric, no protocol and no units, which makes the number
unfalsifiable rather than wrong.

**Why that matters more than it sounds.** In few-shot anomaly detection there are at least
three legitimate numbers you could call "detection accuracy", and they disagree wildly on
the same predictions. From one DINOv2 ViT-S/14 run in this repo:

| category | k | image AUROC | pixel AUROC | pixel AUPRO |
|---|---:|---:|---:|---:|
| weave | 5 | **1.000** | 0.956 | 0.853 |
| grain | 5 | 0.927 | 0.873 | **0.614** |

**Same model, same prediction set, same run. Image AUROC reaches a perfect 1.000 on one
category while pixel AUPRO on another sits at 0.614.** A reader handed "99.9" cannot tell
which of these it is, so they cannot check it and neither can a competitor. Image-level AUROC
is also the easiest of the three, and localisation AUPRO is the one that tells you whether
the model found the defect or just noticed the image was odd.

**What I built.** `EVAL_CARD.md`, a fill-in-the-blank card specifying dataset, which metric
and why, few-shot k at 1/2/5/10, the train and test protocol, and CPU latency percentiles.
Plus a runnable DINOv2 plus PatchCore-style kNN baseline that fills the card in, so it is a
worked example rather than a lecture.

**Latency, since "99.2% compliance in 30 minutes" is a throughput claim.** p50 55.1 ms,
p95 92.2 ms, p99 123.4 ms on one CPU thread, 100 samples.

**What it is not.** Every number here is SYNTHETIC, measured on a toy dataset that ships with
the repo so it runs offline in a couple of minutes. The DINOv2 features are real, the images
are not. This demonstrates the protocol; it is not a competitive result and it says nothing
about what Allus measured.


## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pillow torch timm
.venv/bin/python run.py
```

`python run.py` uses the built-in toy dataset and the toy backbone. It is fully offline,
downloads nothing, and takes about 40 seconds.

```bash
.venv/bin/python run.py --backbone dinov2     # real DINOv2 ViT-S/14 (~85 MB, first run only)
```

## What is real and what is not

| | Default (`--backbone toy`) | `--backbone dinov2` |
|---|---|---|
| Features | fixed random projection of raw pixels | real DINOv2 ViT-S/14, frozen |
| Weights downloaded | none | ~85 MB, cached after first run |
| Purpose | plumbing test, runs anywhere offline | the actual baseline |

**The dataset is synthetic in both cases.** The toy set is procedurally generated
textures with injected scratches and blobs. Every number produced on it is labelled
SYNTHETIC in the output, and it is a demonstration of the protocol, not a competitive
result or evidence about any real defect. To get real numbers, point it at MVTec AD.

## Pointing it at real MVTec AD

This repo never downloads MVTec AD. Get it yourself (free for non-commercial use,
requires accepting their licence) from
https://www.mvtec.com/company/research/datasets/mvtec-ad, extract it, then:

```bash
.venv/bin/python run.py --data mvtec --mvtec-root /path/to/mvtec_anomaly_detection \
  --backbone dinov2 --categories bottle cable capsule carpet grid hazelnut leather \
  metal_nut pill screw tile toothbrush transistor wood zipper
```

That is the only change required. `evalkit/data.py::load_mvtec` reads MVTec's own
directory layout unmodified (`<cat>/train/good`, `<cat>/test/<defect>`,
`<cat>/ground_truth/<defect>/*_mask.png`), so nothing needs renaming.

## Layout

| File | What it does |
|---|---|
| `run.py` | CLI, sweeps k, prints the table, writes `results.json` |
| `evalkit/data.py` | toy dataset generator + MVTec AD loader |
| `evalkit/backbones.py` | toy random-projection and DINOv2 patch feature extractors |
| `evalkit/patchcore.py` | kNN memory bank, anomaly maps, image-score aggregation |
| `evalkit/metrics.py` | AUROC, pixel AUROC, AUPRO |

## Method

Patch features are taken on a 16x16 grid (224px input, patch size 14) and averaged over
a 3x3 neighbourhood, which is PatchCore's "locally aware" step. The memory bank is every
patch feature from the k normal support images. A test patch scores as the Euclidean
distance to its nearest bank neighbour; the grid is upsampled and Gaussian-smoothed
(sigma 4) into a pixel map; the image score is the max of that map, or the mean of the
top 1% of pixels.

No coreset subsampling: at k <= 10 the bank holds at most 2,560 vectors, so exact search
is already fast and coreset selection would only lose recall. Add it before scaling to
hundreds of support images.

## Two deliberate choices worth knowing about

**Image-score aggregation is reported both ways.** Collapsing a pixel map to one number
per image is an unstated protocol choice in most vendor tables, and it moves the headline
number. On weak features the two aggregations differ by 10.5 AUROC points on identical
predictions. Reporting both is the point.

**Why these constants.** The toy defects use amplitude 0.18 to 0.30 over sensor grain of
0.035, with small defect areas. That target was set before any model was run, and the
goal was stated in advance: land in the same *regime* as the MVTec AD texture categories,
where image-level detection is near-saturated but localisation is not. That regime is
what makes reporting image AUROC and pixel AUPRO separately necessary. An earlier version
of the toy set randomised global texture phase per image, which made every normal test
image as far from the support set as an anomalous one and collapsed the whole benchmark
to chance (image AUROC 0.49). That is an easy mistake to make and an invisible one if the
benchmark is never published; it is one concrete reason to publish the harness.

## Reproducibility notes

- Fully seeded. Dataset seeds use `zlib.crc32`, not `hash()`, because Python salts string
  hashing per process and `hash()` would silently give a different dataset every run.
- `--threads 1` by default so latency numbers are not confounded by thread count.
- A p99 estimated from N samples is one order statistic. The default `--latency-reps 100`
  makes the reported p99 the second-slowest observation, which is noisy. Use
  `--latency-reps 2000` for a p99 you would defend in public.
- Single seed per configuration by default. For a real report, run several seeds and
  publish the spread, especially at k=1 where variance is largest.
