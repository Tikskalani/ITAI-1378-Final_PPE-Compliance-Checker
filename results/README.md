# Results

All figures measured from executed runs. Nothing estimated.

## Summary

| Run | Task | Config | val mAP@50 | test mAP@50 |
|---|---|---|---|---|
| V1 | 11-class | YOLO11s, 640px, 40 epochs, batch 16 | 0.608 | — |
| V2 | 11-class | YOLO11s, 960px, 60 epochs, rare-class oversampling | 0.569 | 0.531 |
| **V2 re-scored** | **6-class** | same weights, incoherent classes removed | **0.831** | **0.824** |

**Headline (6-class task, validation):** mAP@50 **0.831** · mAP@50-95 **0.448** · precision 0.833 · recall 0.784
**Held-out test:** mAP@50 **0.824** · mAP@50-95 0.433 · precision 0.857 · recall 0.773
**Speed:** ~20 ms/image end-to-end on a Tesla T4.

The 6-class figures come from `model.val(classes=[0,1,2,3,4,6], imgsz=960)` on the V2 weights. All 143 validation and 141 test images are retained — only the label set changes.

## Per-class mAP@50 (validation)

| Kept — used by the compliance rule | | Dropped — incoherent labels | |
|---|---|---|---|
| Person | 0.886 | none | 0.505 |
| vest | 0.863 | no_helmet | 0.370 |
| helmet | 0.826 | no_gloves | 0.225 |
| gloves | 0.817 | no_goggle | 0.177 |
| goggles | 0.815 | no_boots | 0.000 |
| boots | 0.781 | | |
| **mean** | **0.831** | **mean** | **0.255** |

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

## Reading the results

- The detector is strong on people and real gear (0.78–0.89) and weak only on classes whose labels are internally contradictory.
- V2's confusion matrix shows true `no_boots` predicted as `boots` **50%** of the time, and the `none` class absorbing true vest 13% / gloves 8% / helmet 5%.
- Rejected after measurement: **test-time augmentation** (0.823 with vs 0.831 without).

## Files

- `results_curves.png`, `confusion_matrix.png`, `val_batch0_pred.jpg` — V1 artifacts
- `compliance_samples/` — 6 annotated V1 outputs
- `v2_run/` — V2 curves, confusion matrices, `results.csv`, and 8 demo outputs including a **NON-COMPLIANT** case

Trained weights are not stored in the repo; `notebooks/02_training.ipynb` (V1) or `notebooks/05_training_v3_6class.ipynb` (V3) regenerate them with one Run-All on a GPU.
