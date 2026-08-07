# AI Usage Log

**Project:** PPE Compliance Checker — ITAI 1378 Final Project
**Team:** Trilok Kalani · Matthew Jenkins

A record of how AI tools were used during this project. All project decisions, the model training, the evaluation, and the final review were done by the team; AI was used as an assistant, and every output was checked and edited by us.

| # | Date | AI Tool | How it was used |
|---|---|---|---|
| 1 | 2026-07-12 | Claude (Anthropic) | **Problem identification / brainstorming** — talked through PPE project ideas and narrowed the scope; we chose the final direction (per-worker compliance checker) and the Tier 2 fine-tuning approach. |
| 2 | 2026-07-12 | Claude (Anthropic) | **Coding help** — assistance writing and structuring the YOLO11 training notebook (`02_training.ipynb`) and the `predict_compliance.py` inference script. We ran the code ourselves and reviewed every function. |
| 3 | 2026-07-12 | Claude (Anthropic) | **Debugging** — help resolving a Colab error where installing Ultralytics over the preloaded PyTorch required a runtime restart; the fix (auto-restart install cell) is in the training notebook. |
| 4 | 2026-07-12 | Claude (Anthropic) | **Documentation formatting** — help drafting and formatting the midterm README and slide layout. We wrote and verified the content and all results. |
| 5 | 2026-08-07 | Claude (Anthropic) | **Disaster recovery** — after a lost laptop, used Claude to recover the project from the public GitHub repo: cloned the midterm repo, pulled the executed training notebook out of git history, and extracted the embedded outputs (training curves, confusion matrix, validation metrics, annotated demo images). |
| 6 | 2026-08-07 | Claude (Anthropic) | **Error checking** — Claude caught a labeling bug in our midterm materials: per-class values printed from `metrics.box.maps` are mAP@50-95, but we had labeled them "per-class mAP@50". The final README and slides use the correct columns from the validation output. We verified the correction against the raw printout in `03_training_results.ipynb`. |
| 7 | 2026-08-07 | Claude (Anthropic) | **Final documentation** — help restructuring the repo to the final-submission checklist (README, `docs/`, `results/`) and converting the midterm proposal deck into the final results presentation. Content reviewed and approved by us. |
| 8 | 2026-08-07 | Claude (Anthropic) | **Dataset analysis** — Claude computed per-class/per-split statistics on the Construction-PPE labels and surfaced the key findings: 20:1 class imbalance, tiny median box sizes for weak classes, and that the violation-class (`no_*`) images are off-domain stock photos (12/12 sampled). This produced `docs/dataset_analysis.md`, the V2 training notebook, and the redesigned positive-evidence compliance rule. We reviewed the statistics and sampled images ourselves. |
| 9 | 2026-08-07 | Claude (Anthropic) | **Demo planning** — help outlining the 3–5 minute demo video script (Colab Run-All + inference on sample images). We recorded and narrated the video ourselves. |

**Work done by the team (not AI):**

- Selected the project and defined the problem, scope, compliance rule, and success metrics.
- Ran all training and evaluation on Google Colab and produced the results (mAP@50 0.61).
- Reviewed and validated all code, metrics, and figures; approved every AI-assisted draft.
- Recorded the demo video and finalized the repository and presentation.
