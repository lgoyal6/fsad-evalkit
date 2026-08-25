# Eval card: industrial defect detection

A fill-in-the-blank card for reporting a defect-detection accuracy number so that
someone outside the company can tell what was measured, and in principle disagree with
it. Everything below is standard practice in the anomaly-detection literature; none of
it is exotic, and none of it requires disclosing training data, model architecture, or
customer images.

The test it has to pass: **a reader who has never seen your system should be able to
rebuild your evaluation from this card alone and get the same number.** If a field is
blank, a reader cannot check the claim, and a claim that cannot be checked is not
evidence, however good the underlying system is.

Copy this file, fill it in, publish it next to the number. Fields marked **required**
are the ones that make the difference between a number and an assertion.

---

## 0. The headline

> AllusONE achieves **____%** on **____** .

Fill the second blank with a metric name, not a task name. "Defect Detection" is a task.
"Image-level AUROC" is a metric. A percentage attached to a task, with no metric, cannot
be right or wrong, so it cannot be checked. Section 2 is about exactly this.

---

## 1. Dataset (**required**)

| Field | Value |
|---|---|
| Dataset name and version | |
| Public or proprietary | |
| Where to obtain it | |
| Number of categories | |
| Normal / anomalous test images | |
| Image resolution evaluated at | |
| Pixel-level ground-truth masks available | yes / no |

**The field standard is MVTec AD and VisA.** Report on at least one of them, even if
your product runs on customer data.

- **MVTec AD** (Bergmann et al., CVPR 2019). 15 categories, 5 texture and 10 object,
  3629 normal training images, 1725 test images, per-pixel ground-truth masks. Free for
  non-commercial use, requires accepting a licence:
  https://www.mvtec.com/company/research/datasets/mvtec-ad
- **VisA** (Zou et al., ECCV 2022). 12 categories, 10,821 images, 9,621 normal and 1,200
  anomalous, per-pixel masks, CC BY 4.0: https://github.com/amazon-science/spot-diff

The reason to report a public set even when it is not your target domain is not that it
resembles your customers' parts. It is that it is the only way a reader can separate
"this model is good" from "this test set was easy." A proprietary benchmark can be both
completely honest and completely uninformative to an outsider, because the outsider
cannot see the denominator.

If the headline number comes from proprietary customer data, say so, and report the
public-set number alongside it. Two numbers, clearly labelled, is far stronger than one
unlabelled number.

**Per-category results are required, not just the mean.** MVTec means hide enormous
spread; a model at 0.99 mean can sit at 0.85 on screw and 1.00 on leather. The category
where you are weakest is the one a prospective customer needs, and publishing it is the
single most credible thing on the card.

---

## 2. Metric (**required**): why one number is not enough

Report all three. They measure different things and they routinely differ by 30+ points
on the same run, on the same predictions.

| Metric | Question it answers | Typical use |
|---|---|---|
| **Image-level AUROC** | Does this image contain a defect at all? | Sorting good parts from bad on a line |
| **Pixel-level AUROC** | Which pixels are defective? | Localisation, heavily inflated by large defects |
| **Pixel-level AUPRO** | Which pixels, with every defect region weighted equally? | The honest localisation number |

**Why AUPRO is the one that matters for localisation.** Pixel AUROC averages over
pixels, so one 8,000-pixel scratch dominates twenty 40-pixel pinholes. In manufacturing
the pinholes are usually the expensive ones. AUPRO (per-region overlap, integrated to
a false-positive rate of 0.3 and normalised) weights every connected defect region
equally, so a model cannot score well by finding only the big obvious flaws. Defined in
Bergmann et al., "Uninformed Students" (CVPR 2020); used by essentially every MVTec AD
paper since.

**This is the crux.** In our own baseline run (DINOv2 ViT-S/14 + PatchCore kNN, toy set,
k=10, `grain` category, see `results_dinov2_toy.json`), the *same model*, on the *same images*, in the
*same run*, produced:

| image-level AUROC | pixel-level AUROC | pixel-level AUPRO |
|---|---|---|
| 0.913 | 0.875 | 0.631 |

All three are legitimate, correctly computed numbers describing one set of predictions.
They span 28 points. On the easier `weave` category the same run reports image-level
AUROC **1.000** alongside AUPRO 0.853, so the model that looks perfect on one metric is
visibly imperfect on another.

If a table says "99.9" with no metric named, a reader cannot tell which of these it is,
and therefore cannot tell whether the number is remarkable or ordinary. That is what
"unfalsifiable" means here: not that the number is wrong, but that no observation could
show it to be wrong.

Also state:

- **Threshold-free or thresholded?** AUROC and AUPRO are threshold-free. If you instead
  report accuracy, precision, recall, or F1, you must state the operating threshold and
  how it was chosen, because those numbers are meaningless without it.
- **If you report accuracy on a class-imbalanced test set, state the class balance.**
  A set that is 70% normal gives 70% accuracy to a model that says "good" every time.
- **Image-score aggregation** (the one almost nobody states). A pixel-level model has to
  be collapsed to one number per image, and how you do that moves the headline. Max over
  the anomaly map is PatchCore's own choice; mean of the top 1% of pixels is more robust
  to sensor noise. Same model, same predictions, different headline: in our runs the two
  aggregations agree closely on strong features (0.913 vs 0.920 with DINOv2) but differ
  by 10.5 AUROC points on weaker ones (0.734 vs 0.839), because a max over ~50,000 noisy
  pixels is an extreme-value statistic. State which you use.

---

## 3. Comparison protocol (**required if you name a competitor**)

| Field | Value |
|---|---|
| Baseline name and exact version / checkpoint | |
| Baseline numbers: reproduced by you, or quoted from a paper | |
| If reproduced: the command you ran | |
| Was the baseline tuned on this data? How much? | |
| Same test split for both systems | yes / no |
| Same input resolution for both | yes / no |

Two specific hazards worth pre-empting:

- **Supervised vs unsupervised is not a like-for-like comparison.** YOLO11 is a
  supervised detector that needs labelled defect boxes; a zero-shot or few-shot anomaly
  model needs only normal images. Whichever direction the comparison runs, say how many
  labels each side got. A supervised detector given 5 labelled defects and one given
  5,000 are different systems wearing the same name.
- **A dash in a comparison table needs a reason.** If a baseline is blank on a row
  because the task is outside what it does, write that. An unexplained blank reads as a
  loss the baseline never contested.

---

## 4. Few-shot protocol (**required if you claim fast onboarding**)

"Fast few-shot fine-tuning" and "30 minutes to a working solution" are claims about how
little data you need. That is a measurable quantity, so measure it.

Report the full curve, not one point:

| k (normal support images) | 1 | 2 | 5 | 10 | full |
|---|---|---|---|---|---|
| Image-level AUROC | | | | | |
| Pixel-level AUPRO | | | | | |

| Field | Value |
|---|---|
| What k counts (normal images only? labelled defects too?) | |
| How the k support images were selected (random / curated) | |
| Number of random seeds, and the spread across them | |
| Any per-category hyperparameter tuning | |

**State the seed spread.** At k=1 the variance across seeds is large, and a single
favourable draw is not a result. Mean and standard deviation over at least 5 seeds is
the norm.

**Say plainly whether any labelled defect images were used.** This is the most
consequential fact in a few-shot claim and the easiest to leave ambiguous.

---

## 5. Latency and throughput (**required if you claim a time-to-value number**)

"Industrial-grade vision solutions in 30 minutes with >99.95% detection and 99.2%
compliance" is a throughput claim, and throughput claims need a denominator. Thirty
minutes of what, on what hardware, at what frame rate?

| Field | Value |
|---|---|
| Hardware (exact CPU / GPU / edge SKU) | |
| Precision (fp32 / fp16 / int8) | |
| Batch size | |
| Threads / concurrency | |
| Input resolution | |
| Latency **p50** (ms/image) | |
| Latency **p95** (ms/image) | |
| Latency **p99** (ms/image) | |
| Number of latency samples | |
| Warm or cold (are weights already resident?) | |
| What the timer covers (preprocess? postprocess? I/O?) | |

**Report p99, not just mean.** On a production line the tail is the number that decides
whether you keep up with the belt; a good mean with a bad p99 drops parts. And state the
sample count: a "p99" computed from 30 samples is the second-slowest observation and is
mostly noise. Use at least 1,000 samples for a p99 you would defend.

For an edge claim, the CPU number is the one that matters, because that is what runs on
an industrial PC without a GPU.

---

## 6. Reproducibility

| Field | Value |
|---|---|
| Is there a runnable artifact? | |
| Model card / weights available | |
| Eval code available | |
| Exact command to regenerate the table | |
| Random seeds | |
| Date of measurement, and library versions | |

You do not have to open-source the model to be credible here. Publishing the *evaluation
harness* while keeping the weights private is enough: it lets an outsider run your
protocol against a public baseline and confirm the protocol is fair, which is most of
what they actually doubt.

---

## 7. Known limits (**required**, and the most persuasive section on the card)

A benchmark table with no failure modes reads as marketing regardless of whether it is
true. Name at least three real ones:

- Categories or defect types where the model is weakest, with numbers.
- Conditions that degrade it: lighting change, new part geometry, camera swap, occlusion,
  motion blur, reflective or transparent surfaces.
- What happens with a domain shift the support set never covered.
- The label-noise story. Every real industrial dataset has disagreeing inspectors, and a
  ceiling on measured accuracy that comes from the labels rather than the model. If your
  test labels were single-annotated, the headline number carries that uncertainty and
  saying so costs you nothing.

Publishing a negative result is the cheapest credibility available. A vendor who says
"we are at 0.87 on reflective metal and here is why" is more believable on their 0.999
than one who reports only the 0.999.

---

## Filled-in worked example

The harness in this repo is a runnable DINOv2 + PatchCore-style kNN baseline that
fills in sections 1, 2, 4 and 5 automatically, and `results.json` is the output. It runs
offline on a toy set in about a minute, and points at real MVTec AD with one flag. It is
not a competitive system; it exists to show that every field on this card is cheap to
produce.
