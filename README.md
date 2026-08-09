# PPE Compliance Checker — Final Project

**Course:** ITAI 1378 — Computer Vision · Final Project
**Team:** Trilok Kalani · Matthew Jenkins

> A computer-vision system that looks at a job-site image, detects each worker, and flags whether they are wearing the required safety gear — hard hat, hi-vis vest, and safety goggles — labeling each person "compliant" or "non-compliant."

**Demo video:** [ADD LINK — YouTube (Unlisted) or Google Drive with "Anyone with link"]
**Presentation:** [`docs/presentation.pdf`](docs/presentation.pdf)
**Midterm proposal repo:** https://github.com/Tikskalani/ITAI-1378-Midterm_PPE-Compliance-Checker

---

## 1. What the system does

The system takes an image of a work area and runs a fine-tuned YOLO11 object detector that locates each person and each PPE item. Per-worker logic then associates detected gear to the person it belongs to (gear box center inside the person box) and outputs a bounding box plus a **COMPLIANT** / **NON-COMPLIANT** label for each individual. A worker is compliant only when the required gear (helmet + vest by default) is **positively detected** on them; any overlapping `no_helmet` / `no_goggle` detection adds a violation. (This positive-evidence rule replaced the original rule after the dataset analysis — see `docs/dataset_analysis.md`, Finding 2; the V1 rule is kept as `--rule legacy`.)

*Scope note vs. the proposal:* the midterm specified hard hat + vest + goggles as the required set. The final defaults to hard hat + vest because goggles are the smallest object in the dataset (median box 0.7% of image area, recall 0.75) — hard-requiring them would falsely flag roughly a quarter of compliant workers. Goggles enforcement remains available via `--require helmet,vest,goggles`, and any `no_goggle` detection still counts as a violation under the default rule.

```
[ Job-site image ]
        |
        v
[ YOLO11 detector ]  -- detects person + hard hat + vest + goggles (+ no_* violations)
        |
        v
[ Per-worker compliance logic ]  -- associate gear -> person
        |                           required set = {hard hat, vest} (+ goggles via --require)
        v
[ Output: box per worker + "Compliant" / "Non-compliant" ]
```

Annotated example outputs are in [`results/compliance_samples/`](results/compliance_samples/).

## 2. Technical approach

| Element | Choice | Why |
|---|---|---|
| CV technique | Multi-class object detection | Locates and identifies people plus several gear types in a single pass. |
| Model | YOLO11s (Ultralytics), fine-tuned | Strong accuracy/speed trade-off; trains on a free Colab T4 in ~17 minutes. |
| Framework | PyTorch + Ultralytics | Standard, open-source; built-in tracking (ByteTrack) for the video extension. |
| Compliance logic | Rule-based, per worker | Deterministic and explainable — associate detected gear to each person box; default required set = hard hat + vest (goggles configurable via `--require`). |

## 3. Dataset

**Ultralytics Construction-PPE** — 1,416 images (1,132 train / 143 val / 141 test), 11 classes:
`helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`.
The explicit `no_*` labels let the compliance rule flag violations directly from the detector. The dataset downloads automatically through Ultralytics (~178 MB); details and licenses in [`data/README.md`](data/README.md).

Identified scale-up path: **SH17** (8,099 images, ~75,994 instances) for improving the rare violation classes.

## 4. Results

**Run:** YOLO11s · 40 epochs · imgsz 640 · batch 16 · Google Colab Tesla T4 · ~17 min · Ultralytics 8.4.92
**Evaluation:** held-out validation set — 143 images, 1,172 instances.

| Metric | Target | Achieved |
|---|---|---|
| mAP@50 | ≥ 0.85 | **0.61** |
| mAP@50-95 | — | **0.30** |
| Mean precision / recall | — | 0.66 / 0.59 |
| Inference latency | < 1 s / image | **~20 ms / image end-to-end on T4** (3.8 ms pre + 12.6 ms inference + 3.3 ms post) |

### Per-class results (validation)

| Class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Person | 0.86 | 0.91 | **0.91** | 0.53 |
| vest | 0.85 | 0.83 | **0.85** | 0.53 |
| gloves | 0.86 | 0.78 | **0.83** | 0.39 |
| goggles | 0.79 | 0.75 | **0.82** | 0.38 |
| helmet | 0.83 | 0.81 | **0.80** | 0.43 |
| boots | 0.76 | 0.73 | **0.80** | 0.45 |
| none | 0.57 | 0.58 | 0.55 | 0.22 |
| no_helmet | 0.55 | 0.42 | 0.45 | 0.16 |
| no_gloves | 0.49 | 0.23 | 0.29 | 0.08 |
| no_goggle | 0.38 | 0.15 | 0.23 | 0.08 |
| no_boots | 0.36 | 0.25 | 0.16 | 0.08 |

*Correction vs. the midterm materials: the midterm slides/README listed per-class `metrics.box.maps` values (which are per-class mAP@50-95) under the heading "per-class mAP@50". The table above uses the correct columns from the validation printout in [`notebooks/03_training_results.ipynb`](notebooks/03_training_results.ipynb).*

### Reading the results

- Training converged cleanly — losses fall smoothly and mAP plateaus around epoch 35–40 with no overfitting ([`results/results_curves.png`](results/results_curves.png)).
- The detector is **strong on people and common gear** (mAP@50 0.80–0.91 for Person, vest, gloves, goggles, helmet, boots).
- It is **weak on the rare violation classes** (`no_goggle` 0.23, `no_boots` 0.16), which have very few training examples (as few as 4 instances in val). Since the compliance rule relies partly on `no_*` detections, this is the main limitation.
- The overall mAP@50 of 0.61 is pulled down by those rare classes; the clearest improvement path is more violation-class data (SH17 scale-up or oversampling), not a different architecture.

## 5. Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_exploration.ipynb          data exploration + pretrained baseline
│   ├── 02_training.ipynb             V1 training + evaluation + compliance demo (Colab, Run-All)
│   ├── 03_training_results.ipynb     the executed V1 run with all outputs (mAP@50 0.61)
│   └── 04_training_v2.ipynb          V2 accuracy-improvement training (960px, oversampling)
├── src/
│   └── predict_compliance.py         run the trained model on any image or folder
├── data/
│   └── README.md                     dataset sources, classes, licenses
├── docs/
│   ├── AI_usage_log.md               how AI tools were used in this project
│   ├── dataset_analysis.md           measured dataset statistics + accuracy-improvement plan
│   ├── label_qa_samples.jpg          evidence: off-domain violation-class images
│   └── presentation.pdf              final presentation slides
└── results/
    ├── README.md                     metrics summary
    ├── results_curves.png            training/val loss + P/R/mAP curves
    ├── confusion_matrix.png          per-class confusion matrix
    ├── val_batch0_pred.jpg           validation-batch predictions
    └── compliance_samples/           6 annotated Compliant / Non-compliant outputs
```

## 6. Reproducing the results

Trained weights (`best.pt`, ~19 MB) are not stored in the repo; one Run-All regenerates them.

1. Open `notebooks/02_training.ipynb` in Google Colab, set the runtime to **T4 GPU**, and **Runtime → Run all**. The Construction-PPE dataset downloads automatically; training takes ~17 minutes and the notebook saves `best.pt`, all metric plots, and the annotated compliance demo images, then zips them for download.
2. Run the trained model on any image locally:

```bash
pip install -r requirements.txt
python src/predict_compliance.py --weights best.pt --source your_image.jpg --out out/
```

## 7. Challenges & what we learned

| Challenge | What happened / mitigation |
|---|---|
| Rare violation classes score low | Confirmed in results (`no_*` mAP@50 0.16–0.45). Identified fix: scale to SH17 or oversample violations. |
| Small dataset caps overall accuracy | 1,416 images is an honest baseline (0.61); more data is the path to the 0.85 target. |
| Colab torch/Ultralytics install conflict | Install cell auto-restarts the runtime once, then skips on later runs. |
| **Lost laptop late in the project** | All code, notebooks, and results survived because they were committed to GitHub — including the executed notebook, from which every figure in this repo was recovered. Version control saved the project. |
| Video (V2) more complex than time allowed | Scoped as stretch/capstone; the image pipeline is the graded deliverable. |

## 8. Accuracy improvement (V2)

A full statistical analysis of the dataset ([`docs/dataset_analysis.md`](docs/dataset_analysis.md)) found that the weak classes are simultaneously rare (20.3:1 imbalance), tiny (median `no_goggle` box = 0.5% of image area), and — critically — trained on **off-domain stock photos** (12/12 sampled violation-class images were gyms, offices, and family photos, not construction sites).

[`notebooks/04_training_v2.ipynb`](notebooks/04_training_v2.ipynb) implements the fixes: training at 960px (tiny objects), oversampling rare-class images (imbalance), 60 epochs with early stopping, and evaluation on both val and test. The compliance rule in `src/predict_compliance.py` was redesigned to require positive gear detections instead of trusting the off-domain `no_*` classes. V2 metrics are added here from the executed run only.

## References

- OSHA 29 CFR 1926.95–102 — PPE requirements for construction (head, eye, high-visibility).
- Ultralytics YOLO11 documentation and Construction-PPE dataset — https://docs.ultralytics.com/datasets/detect/construction-ppe/
- Mughees et al., *SH17: A Dataset for Human Safety and PPE Detection in Manufacturing Industry*, arXiv:2407.04590 (2024).
- Roboflow Universe — PPE datasets — https://universe.roboflow.com/browse/construction/ppe
