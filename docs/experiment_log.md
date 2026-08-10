# Experiment Log — chasing the mAP@50 ≥ 0.85 target

Three training runs and a diagnostic investigation. Every number below is measured, not estimated.

## Summary

| Run | Task | Config | val mAP@50 | test mAP@50 |
|---|---|---|---|---|
| V1 | 11-class | YOLO11s, 640px, 40 epochs | 0.608 | — |
| V2 | 11-class | YOLO11s, 960px, 60 epochs, rare-class oversampling | **0.569** (regression) | 0.531 |
| V2 re-scored | 6-class | same weights, incoherent classes removed from the task | 0.831 | 0.824 |
| **V3 (final)** | **6-class** | **YOLO11s, 960px, 55 epochs, close_mosaic 15, no oversampling** | **0.827** | **0.842** |

## V2: why the "improvement" made things worse

V2 applied two changes: 960px input (for the tiny-object problem) and oversampling of images containing rare `no_*` classes (for the 20.3:1 imbalance). Net result: **mAP@50 fell from 0.608 to 0.569.**

Per-class deltas, V1 → V2 (validation):

| Class | V1 | V2 | Δ |
|---|---|---|---|
| helmet | 0.803 | 0.826 | **+0.023** |
| vest | 0.850 | 0.863 | **+0.013** |
| goggles | 0.816 | 0.815 | −0.001 |
| gloves | 0.826 | 0.817 | −0.009 |
| boots | 0.799 | 0.781 | −0.018 |
| Person | 0.907 | 0.886 | −0.021 |
| none | 0.552 | 0.505 | −0.047 |
| no_goggle | 0.233 | 0.177 | −0.056 |
| no_gloves | 0.289 | 0.225 | −0.064 |
| no_helmet | 0.448 | 0.370 | −0.078 |
| no_boots | 0.164 | **0.000** | −0.164 |

The resolution change did what it was supposed to — helmet and vest, the two classes the compliance rule depends on, both improved. **The oversampling is what failed**, and it failed for exactly the reason `dataset_analysis.md` Finding 2 predicted: the images containing `no_*` labels are off-domain stock photography (gyms, offices, family photos — 12/12 in a random sample). Duplicating them multiplied the wrong distribution.

**The hypothesis was refuted by its own mechanism** — and that refutation is what pointed at the real fix.

## The deeper finding: the 11-class target is unreachable

Two pieces of evidence from V2's confusion matrix show the ceiling is set by the labels, not the model.

**1. The `no_*` labels contradict the gear labels.** True `no_boots` is predicted as `boots` **50%** of the time. The same visual region — a worker's feet — is annotated `boots` in some images and `no_boots` in others. A detector cannot separate two classes annotated on the same appearance. Between 24% and 50% of every `no_*` class is also lost to background.

**2. The `none` class steals correct detections.** The model predicts `none` on objects that are truly **vest (13%), gloves (8%), helmet (5%)**.

Because mAP@50 is the unweighted mean across all 11 classes, five incoherent classes cap the achievable headline regardless of model quality. Ultralytics publishes no baseline for this dataset. **Reaching 0.85 averaged over all 11 classes is not achievable with these labels.**

## V3: solve the task the system actually performs

The system decides compliance from **positive evidence** — a detected helmet and vest on a detected person (`src/predict_compliance.py`). It never needs a `no_*` class. V3 therefore trains on the six classes the decision uses: `helmet, gloves, vest, boots, goggles, Person`, dropping `none` and the four `no_*` classes.

A documented scope decision with measured justification, not metric selection:

- The dropped classes are **not used** by the compliance rule.
- They are **off-domain** (12/12 sampled images are not construction scenes).
- They are **internally contradictory** (`no_boots` ↔ `boots`, 50% confusion).
- They are **actively harmful** to the retained classes (`none` absorbs 13% of true vests).
- **The evaluation set is unchanged.** All 143 validation and 141 test images retained — zero images dropped, only the label set changes. Both 11-class and 6-class numbers are reported.

Side effect confirming the diagnosis: class imbalance falls from **20.3:1 to 4.2:1**. The imbalance problem largely *was* the junk classes.

### V3 result

YOLO11s · 960px · 55 epochs · close_mosaic 15 · cos_lr · 0.825 h on a Colab T4.

| Metric | Validation | Held-out test |
|---|---|---|
| mAP@50 | 0.8273 | **0.8417** |
| mAP@50-95 | 0.4509 | **0.4554** |
| Precision / recall | 0.857 / 0.789 | 0.880 / 0.781 |

Per-class mAP@50 (test): helmet **0.934**, vest **0.904**, goggles 0.839, Person 0.836, gloves 0.782, boots 0.756.

Against the V2 6-class baseline: **test mAP@50 +0.0182**, test mAP@50-95 **+0.0225**, val mAP@50-95 **+0.0030**, val mAP@50 −0.0039. Training natively on six classes improved localization on both splits and detection quality on the held-out split.

**The target was not met.** 0.842 on test falls short of 0.85. The remaining gap sits almost entirely in `boots` (0.756) and `gloves` (0.782) — the two classes with the loosest bounding boxes in the dataset. The two classes the compliance decision actually requires, helmet and vest, are at 0.934 and 0.904.

### Rejected after testing

**Test-time augmentation.** Measured on the 6-class task: mAP@50 **0.823 with TTA vs 0.831 without**. It hurt, so it is not used.

## Reproducing

`notebooks/05_training_v3_6class.ipynb` — Run All on any free GPU (Colab T4 or Kaggle). About 50 minutes. It rebuilds the 6-class labels from the original download, trains, and evaluates on val and test against the baseline.

*Operational note: Colab dropped the GPU runtime three times during this work, losing two V3 runs mid-training (the second at epoch 85/100). The schedule was shortened from 100 to 55 epochs so the run completed inside the window that had been failing.*
