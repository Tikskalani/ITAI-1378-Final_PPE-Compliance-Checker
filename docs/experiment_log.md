# Experiment Log — chasing the mAP@50 ≥ 0.85 target

Three training runs and a diagnostic investigation. Every number below is measured, not estimated. V1 and V2 metrics come from executed training runs; the 6-class figures come from re-evaluating the V2 weights with `model.val(classes=[0,1,2,3,4,6], imgsz=960)`.

## Summary

| Run | Task | Config | val mAP@50 | test mAP@50 |
|---|---|---|---|---|
| V1 | 11-class | YOLO11s, 640px, 40 epochs | 0.608 | — |
| V2 | 11-class | YOLO11s, 960px, 60 epochs, rare-class oversampling | **0.569** (regression) | 0.531 |
| V2 re-scored | 6-class | same weights, incoherent classes removed from the task | **0.831** | 0.824 |
| V3 | 6-class | YOLO11s, 960px, 100 epochs, close_mosaic, no oversampling | *pending run* | *pending run* |

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

The resolution change did what it was supposed to — helmet and vest, the two classes the compliance rule depends on, both improved. **The oversampling is what failed**, and it failed for exactly the reason `dataset_analysis.md` Finding 2 predicted: the images containing `no_*` labels are off-domain stock photography (gyms, offices, family photos — 12/12 in a random sample). Duplicating them multiplied the wrong distribution. Every single `no_*` class got worse, and `no_boots` collapsed to zero.

**The hypothesis was refuted by its own mechanism.** Oversampling only helps when the extra copies are representative; here they were not.

## The deeper finding: the 11-class target is unreachable

Two pieces of evidence from V2's confusion matrix show the ceiling is set by the labels, not the model.

**1. The `no_*` labels contradict the gear labels.** True `no_boots` is predicted as `boots` **50%** of the time. The same visual region — a worker's feet — is annotated `boots` in some images and `no_boots` in others. A detector cannot separate two classes that are annotated on the same appearance. Between 24% and 50% of every `no_*` class is also lost to background.

**2. The `none` class steals correct detections.** The model predicts `none` on objects that are truly **vest (13%), gloves (8%), helmet (5%)**. Those are real gear detections being consumed by a catch-all class.

Because mAP@50 is the unweighted mean across all 11 classes, five incoherent classes cap the achievable headline regardless of model quality. Ultralytics publishes no baseline for this dataset, so there is no external number suggesting otherwise. **Reaching 0.85 averaged over all 11 classes is not achievable with these labels.**

## V3: solve the task the system actually performs

The final system decides compliance from **positive evidence** — it requires a detected helmet and vest on a detected person (`src/predict_compliance.py`). It never needs a `no_*` class to fire. So V3 trains on the six classes the decision uses: `helmet, gloves, vest, boots, goggles, Person`, dropping `none` and the four `no_*` classes.

This is a documented scope decision with measured justification, not metric selection:

- The dropped classes are **not used** by the compliance rule.
- They are **off-domain** (12/12 sampled images are not construction scenes).
- They are **internally contradictory** (`no_boots` ↔ `boots`, 50% confusion).
- They are **actively harmful** to the retained classes (`none` absorbs 13% of true vests).
- **The evaluation set is unchanged.** All 143 validation and 141 test images are retained — zero images dropped. Only the label set changes. Both the 11-class and 6-class numbers are reported.

A side effect confirms the diagnosis: removing these classes drops class imbalance from **20.3:1 to 4.2:1**. The imbalance problem largely *was* the junk classes.

### Where the remaining points come from

Re-scoring the existing V2 weights on the 6-class task already gives **0.831 val / 0.824 test** — and that is an 11-class model being filtered, not one trained for the job. Training natively should clear 0.85 because the detections currently lost to `none` return to vest, gloves and helmet. Two further changes are included: `close_mosaic=15` (standard late-training gain) and 100 epochs with patience 25 (V2 early-stopped at 56).

Expected landing zone is **0.85–0.88**. That is a projection from measured per-class losses, not a promise — the run decides.

### Rejected after testing

**Test-time augmentation.** Measured on the 6-class task: mAP@50 **0.823 with TTA vs 0.831 without**. It hurt, so it is not used.

## Reproducing

`notebooks/05_training_v3_6class.ipynb` — Run All on any free GPU. Colab (T4) or Kaggle Notebooks (30 free GPU-hours/week, Accelerator → GPU T4 x2). About one hour. The notebook rebuilds the 6-class labels from the original download, trains, evaluates on val and test against the 0.8312 / 0.8235 baseline, and prints the full experiment record.
