# Dataset Analysis — Construction-PPE (accuracy-improvement study)

**Dataset:** Ultralytics Construction-PPE, downloaded 2026-08-07 (178 MB).
**Counts verified:** train 1,132 images / 1,142 label files (10 labels have no matching image — harmless, YOLO skips them), val 143, test 141. No empty label files.
**Method:** all statistics computed directly from the YOLO label files; domain findings from manual review of sampled images.

## 1. Per-class statistics (train split, 9,191 boxes)

| Class | Instances | Images | % of boxes | Median box area | Boxes < 1% of image |
|---|---|---|---|---|---|
| Person | 1,790 | 1,077 | 19.5% | 22.8% | 2% |
| helmet | 1,357 | 840 | 14.8% | 1.2% | 44% |
| vest | 1,283 | 837 | 14.0% | 6.4% | 10% |
| boots | 1,251 | 538 | 13.6% | 1.3% | 40% |
| gloves | 1,162 | 571 | 12.6% | 0.7% | 72% |
| none | 654 | 349 | 7.1% | 5.1% | 19% |
| goggles | 427 | 393 | 4.6% | 0.7% | 78% |
| no_gloves | 442 | 190 | 4.8% | 1.0% | 51% |
| no_helmet | 400 | 232 | 4.4% | 1.8% | 30% |
| no_goggle | 337 | 216 | 3.7% | 0.5% | 77% |
| **no_boots** | **88** | **28** | **1.0%** | 0.6% | 78% |

Class imbalance: **20.3 : 1** (Person 1,790 vs no_boots 88).

## 2. Findings

### Finding 1 — The weak classes are rare AND tiny

Every class with poor V1 mAP@50 sits at the intersection of few instances and small boxes. `no_goggle` boxes have a **median area of 0.5% of the image** — at 640×640 training resolution that is roughly a 45×45-pixel object, at the edge of what a detector can resolve. `goggles` (78% tiny) survives on volume (427 instances); `no_goggle` (337, 77% tiny) and `no_boots` (88, 78% tiny) do not.

### Finding 2 — The violation-class images are off-domain (root cause)

240 train images contain a `no_*` class. In a random sample of 12 of them, **12 / 12 were not construction imagery at all**: bodybuilders in gyms, politicians at podiums, a chef cooking, soccer players, office workers at desks, decades-old family photos. The `no_*` labels appear to be sourced from generic "person without PPE" stock photos.

Consequences:

- The model learns *"no_helmet ≈ person in an office/gym photo"* — a context cue, not a violation cue — so it fails to fire on real job-site violations (we observed exactly this: workers without hard hats at a rail station passing as "COMPLIANT" in the V1 demo).
- The V1 per-class numbers for `no_*` classes measure performance on stock photos, not on the deployment domain.
- Label quality in this subset is also loose (e.g., a swimmer wearing swim goggles on her head is labeled `no_goggle`).

### Finding 3 — `no_boots` validation metrics are statistical noise

Val contains **4 `no_boots` instances in 2 images**. Any mAP reported for this class (V1: 0.164) is meaningless at this sample size. Test has 23 instances in 6 images — still thin. Conclusions about `no_boots` should not be drawn from either split.

### Finding 4 — Evaluation has headroom on the test split

V1 was only ever evaluated on val (143 images). The untouched test split (141 images, 1,251 boxes) gives a second, independent measurement and should be reported in the final results.

## 3. What this means for accuracy — the V2 plan

Each change is matched to a finding (implemented in `notebooks/04_training_v2.ipynb`):

| Change | Addresses | Mechanism |
|---|---|---|
| Train at **imgsz 960** (was 640) | Finding 1 | A 0.5%-area box goes from ~45 px to ~68 px — materially easier to detect. Biggest single lever for gloves/goggles/no_goggle. |
| **Oversample rare-class images** (no_boots ×5, other no_* ×2 extra copies) | Finding 1 | Rebalances the 20:1 skew without touching val/test. |
| **60 epochs, patience 15** (was 40) | — | Higher resolution + oversampling changes the loss landscape; V1's plateau at ~epoch 35 doesn't transfer. |
| Evaluate on **val and test** | Finding 4 | Two independent measurements; report both. |
| **Compliance rule redesigned to positive evidence** (in `src/predict_compliance.py`) | Finding 2 | The old rule trusted the weakest, off-domain classes to fire on violations. The new rule requires *positive* detection of required gear (helmet, vest — the 0.80–0.85 mAP classes) and treats `no_*` hits as supplementary evidence. This fixes the observed false-"COMPLIANT" failure mode regardless of retraining. |

**Expectation (honest):** overall mAP@50 should land meaningfully above the 0.61 baseline — the tiny-object and imbalance fixes are well-established levers — but the `no_*` ceiling is set by off-domain data that retraining cannot fully overcome. Getting violation classes truly right requires in-domain violation imagery (SH17, 8,099 images, or targeted collection). Numbers in the final report come only from the executed V2 run.

*(Sampled label-quality evidence: `docs/label_qa_samples.jpg`.)*
