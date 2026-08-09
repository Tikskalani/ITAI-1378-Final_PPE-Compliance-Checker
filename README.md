# PPE Compliance Checker — Final Project

**Course:** ITAI 1378 — Computer Vision · Final Project
**Team:** Trilok Kalani · Matthew Jenkins

> A computer-vision system that looks at a job-site image, detects each worker, and flags whether they are wearing the required safety gear — hard hat, hi-vis vest, and safety goggles — labeling each person "compliant" or "non-compliant."

**Demo video:** [ADD LINK — YouTube (Unlisted) or Google Drive with "Anyone with link"]
**Presentation:** [`docs/presentation.pdf`](docs/presentation.pdf)
**Experiment log:** [`docs/experiment_log.md`](docs/experiment_log.md) · **Dataset analysis:** [`docs/dataset_analysis.md`](docs/dataset_analysis.md)
**Midterm proposal repo:** https://github.com/Tikskalani/ITAI-1378-Midterm_PPE-Compliance-Checker

---

## 1. What the system does

The system takes an image of a work area and runs a fine-tuned YOLO11 object detector that locates each person and each PPE item. Per-worker logic then associates detected gear to the person it belongs to (gear box center inside the person box) and outputs a bounding box plus a **COMPLIANT** / **NON-COMPLIANT** label for each individual.

A worker is compliant only when the required gear (helmet + vest by default) is **positively detected** on them; any overlapping `no_helmet` / `no_goggle` detection adds a violation. This positive-evidence rule replaced the original rule after the dataset analysis (see `docs/dataset_analysis.md`, Finding 2); the original is kept as `--rule legacy`.

*Scope note vs. the proposal:* the midterm specified hard hat + vest + goggles as the required set. The final defaults to hard hat + vest because goggles are the smallest object in the dataset (median box 0.7% of image area, recall 0.75) — hard-requiring them would falsely flag roughly a quarter of compliant workers. Goggles enforcement remains available via `--require helmet,vest,goggles`.

```
[ Job-site image ]
        |
        v
[ YOLO11 detector ]  -- detects person + hard hat + vest + goggles
        |
        v
[ Per-worker compliance logic ]  -- associate gear -> person
        |                           required set = {hard hat, vest} (+ goggles via --require)
        v
[ Output: box per worker + "Compliant" / "Non-compliant" ]
```

Annotated outputs: [`results/v2_run/compliance_out_v2/`](results/v2_run/compliance_out_v2/) (includes a NON-COMPLIANT case) and [`results/compliance_samples/`](results/compliance_samples/).

## 2. Technical approach

| Element | Choice | Why |
|---|---|---|
| CV technique | Multi-class object detection | Locates and identifies people plus several gear types in a single pass. |
| Model | YOLO11s (Ultralytics), fine-tuned | Strong accuracy/speed trade-off; trains on a free Colab T4. |
| Framework | PyTorch + Ultralytics | Standard, open-source; built-in tracking (ByteTrack) for the video extension. |
| Compliance logic | Rule-based, per worker | Deterministic and explainable — default required set = hard hat + vest (goggles configurable). |

## 3. Dataset

**Ultralytics Construction-PPE** — 1,416 images (1,132 train / 143 val / 141 test), 11 classes:
`helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`. Downloads automatically (~178 MB); details in [`data/README.md`](data/README.md).

A full statistical analysis of the labels is in [`docs/dataset_analysis.md`](docs/dataset_analysis.md). It drives every decision below.

## 4. Results

Three training runs. All figures measured; nothing estimated.

| Run | Task | Config | val mAP@50 | test mAP@50 |
|---|---|---|---|---|
| V1 | 11-class | YOLO11s, 640px, 40 epochs | 0.608 | — |
| V2 | 11-class | YOLO11s, 960px, 60 epochs, rare-class oversampling | 0.569 | 0.531 |
| **V2 re-scored** | **6-class** | same weights, incoherent classes removed from the task | **0.831** | **0.824** |

**Headline: mAP@50 = 0.831** on the six classes the compliance decision uses, against a 0.85 target — with precision 0.83, recall 0.78, mAP@50-95 0.448, and ~20 ms/image end-to-end on a T4 (target was < 1 s/image).

### Per-class mAP@50 — kept vs dropped

| Kept (used by the rule) | mAP@50 | | Dropped | mAP@50 |
|---|---|---|---|---|
| Person | 0.886 | | none | 0.505 |
| vest | 0.863 | | no_helmet | 0.370 |
| helmet | 0.826 | | no_gloves | 0.225 |
| gloves | 0.817 | | no_goggle | 0.177 |
| goggles | 0.815 | | no_boots | **0.000** |
| boots | 0.781 | | | |
| **mean** | **0.831** | | **mean** | **0.255** |

### Why five classes were dropped

The six kept classes average 0.831. The five dropped classes average 0.255 — and the reason is the labels, not the model:

- **They are off-domain.** In a random sample of 12 training images containing `no_*` labels, **12 of 12** were not construction scenes: gyms, offices, banquets, family photos. The `no_*` subset appears to be generic "person without PPE" stock photography.
- **They are self-contradictory.** V2's confusion matrix shows true `no_boots` predicted as `boots` **50%** of the time. The same visual region is annotated `boots` in some images and `no_boots` in others; no amount of training separates two classes drawn on the same appearance.
- **They actively damage the classes we keep.** The model predicts `none` on objects that are truly **vest (13%), gloves (8%), helmet (5%)** — real gear detections consumed by a catch-all class.
- **They are not used.** The positive-evidence compliance rule never needs a `no_*` detection to fire.

**The evaluation set is unchanged.** All 143 validation and 141 test images are retained — zero images dropped, only the label set changes. Both the 11-class and 6-class figures are reported above. Removing these classes also drops class imbalance from **20.3:1 to 4.2:1**; the imbalance problem largely *was* the junk classes.

### The V2 experiment

V2 changed two things: 960px input (for tiny objects) and oversampling of rare `no_*` images (for imbalance). Net result was a **regression**, 0.608 → 0.569.

The resolution change worked — helmet **+0.023** and vest **+0.013**, the two classes the compliance rule depends on. The oversampling failed, and failed for exactly the reason the dataset analysis predicted: it duplicated off-domain images, multiplying the wrong distribution. Every `no_*` class got worse and `no_boots` collapsed to zero. The hypothesis was refuted by its own mechanism, which is what pointed at the real fix.

**Also tested and rejected:** test-time augmentation, measured at 0.823 vs 0.831 without. It hurt, so it is not used.

*Correction vs. the midterm materials: the midterm listed per-class `metrics.box.maps` values (which are mAP@50-95) under the heading "per-class mAP@50". All tables here use the correct columns.*

## 5. Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_exploration.ipynb          data exploration + pretrained baseline
│   ├── 02_training.ipynb             V1 training (Colab, Run-All)
│   ├── 03_training_results.ipynb     the executed V1 run with all outputs
│   ├── 04_training_v2.ipynb          V2 experiment (960px + oversampling)
│   └── 05_training_v3_6class.ipynb   V3 — 6-class training targeting mAP@50 >= 0.85
├── src/
│   └── predict_compliance.py         run the model on any image or folder
├── data/
│   └── README.md                     dataset sources, classes, licenses
├── docs/
│   ├── AI_usage_log.md               how AI tools were used
│   ├── dataset_analysis.md           measured dataset statistics + root cause
│   ├── experiment_log.md             V1 -> V2 -> V3 experiment record
│   ├── label_qa_samples.jpg          evidence: off-domain violation-class images
│   └── presentation.pdf              final presentation slides
└── results/
    ├── README.md                     metrics summary
    ├── results_curves.png            V1 training curves
    ├── confusion_matrix.png          V1 confusion matrix
    ├── val_batch0_pred.jpg           V1 validation predictions
    ├── compliance_samples/           6 annotated V1 outputs
    └── v2_run/                       V2 curves, confusion matrices, 8 demo outputs
```

## 6. Reproducing the results

Trained weights are not stored in the repo; one Run-All regenerates them.

1. **V1 baseline (~17 min):** open `notebooks/02_training.ipynb` in Colab, set **T4 GPU**, **Runtime → Run all**.
2. **V3, the 6-class run (~1 hr):** open `notebooks/05_training_v3_6class.ipynb` on Colab or Kaggle with a GPU and Run All. It rebuilds the 6-class labels from the original download, trains, and evaluates on both val and test against the 0.8312 / 0.8235 baseline.
3. **Run the model on any image:**

```bash
pip install -r requirements.txt
python src/predict_compliance.py --weights best.pt --source your_image.jpg --imgsz 960 --out out/
```

## 7. Challenges & what we learned

| Challenge | What happened |
|---|---|
| V2 "improvement" made things worse | 0.608 → 0.569. Oversampling multiplied off-domain images. Diagnosed from per-class deltas and the confusion matrix. |
| The 11-class target was unreachable | Contradictory labels (`no_boots` ↔ `boots`, 50% confusion) cap the achievable mean regardless of model quality. |
| A junk class was eating real detections | `none` absorbed 13% of true vests. Found in the confusion matrix, not in the headline number. |
| **Lost laptop late in the project** | All code, notebooks and results survived because they were committed to GitHub — including the executed notebook, from which every figure was recovered. Version control saved the project. |
| Video (V2 capstone) beyond scope | Kept as future work; the image pipeline is the graded deliverable. |

The broader lesson: the headline metric was measuring the dataset's defects more than the system's quality. Diagnosing that took a confusion matrix and a random sample of training images — not a bigger model.

## References

- OSHA 29 CFR 1926.95–102 — PPE requirements for construction (head, eye, high-visibility).
- Ultralytics YOLO11 documentation and Construction-PPE dataset — https://docs.ultralytics.com/datasets/detect/construction-ppe/
- Mughees et al., *SH17: A Dataset for Human Safety and PPE Detection in Manufacturing Industry*, arXiv:2407.04590 (2024).
- Roboflow Universe — PPE datasets — https://universe.roboflow.com/browse/construction/ppe
