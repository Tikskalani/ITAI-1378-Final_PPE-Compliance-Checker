# Results

All figures measured from executed runs. Nothing estimated.

## Summary

| Run | Task | Config | val mAP@50 | test mAP@50 |
|---|---|---|---|---|
| V1 | 11-class | YOLO11s, 640px, 40 epochs | 0.608 | — |
| V2 | 11-class | YOLO11s, 960px, 60 epochs, oversampling | 0.569 | 0.531 |
| V2 re-scored | 6-class | same weights, incoherent classes removed | 0.831 | 0.824 |
| **V3 (final)** | **6-class** | **YOLO11s, 960px, 55 epochs, close_mosaic** | **0.827** | **0.842** |

## V3 — the final model

YOLO11s · 960px · 55 epochs · batch 8 · close_mosaic 15 · cos_lr · **0.825 h on a Colab T4**

| Metric | Validation (143 img / 945 inst) | Held-out test (141 img / 1,032 inst) |
|---|---|---|
| mAP@50 | 0.8273 | **0.8417** |
| mAP@50-95 | 0.4509 | **0.4554** |
| Precision | 0.8572 | 0.8796 |
| Recall | 0.7887 | 0.7805 |

Speed (val, 960px, T4): 14.4 ms preprocess + 24.5 ms inference + 2.3 ms postprocess = **~41 ms/image**.

Test is the more rigorous figure — `best.pt` is chosen on validation fitness, so test is the only split never used for model selection.

### Per-class mAP@50

| Class | val | test |
|---|---|---|
| helmet | 0.804 | **0.934** |
| vest | 0.841 | **0.904** |
| goggles | 0.809 | 0.839 |
| Person | 0.896 | 0.836 |
| gloves | 0.817 | 0.782 |
| boots | 0.797 | 0.756 |
| **mean** | **0.827** | **0.842** |

The two classes the compliance rule requires — helmet and vest — are the strongest on the held-out split.

## V3 vs the V2 6-class baseline

| Metric | V2 (6-cls) | V3 | Δ |
|---|---|---|---|
| val mAP@50 | 0.8312 | 0.8273 | −0.0039 |
| val mAP@50-95 | 0.4479 | 0.4509 | **+0.0030** |
| test mAP@50 | 0.8235 | 0.8417 | **+0.0182** |
| test mAP@50-95 | 0.4329 | 0.4554 | **+0.0225** |

Training natively on six classes improved localization quality (mAP@50-95) on both splits and detection quality on the held-out split. Validation is essentially unchanged.

## V1 → V2 per-class deltas (11-class, validation)

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

Higher resolution helped the gear classes the rule depends on. Oversampling degraded every violation class, because the images it duplicated are off-domain stock photography (see `docs/dataset_analysis.md`).

## Rejected after measurement

**Test-time augmentation** — 0.823 with vs 0.831 without on the 6-class task. Not used.

## Files

- `v3_run/` — **final model artifacts**: training curves, confusion matrices, `results.csv`, 12 demo outputs
- `v2_run/` — V2 curves, confusion matrices, 8 demo outputs
- `results_curves.png`, `confusion_matrix.png`, `val_batch0_pred.jpg`, `compliance_samples/` — V1 artifacts

Trained weights are not stored in the repo; `notebooks/05_training_v3_6class.ipynb` regenerates `best_v3.pt` with one Run-All on a GPU (~50 min).
