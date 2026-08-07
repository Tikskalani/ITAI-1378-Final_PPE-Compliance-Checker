"""
PPE Compliance Checker — standalone inference.

After training with notebooks/02_training.ipynb (V1) or notebooks/04_training_v2.ipynb (V2)
you get a `best.pt`. This script runs that model on an image (or a folder) and labels each
worker Compliant / Non-compliant, then saves annotated copies.

Usage:
    python src/predict_compliance.py --weights best.pt --source path/to/image_or_folder
    python src/predict_compliance.py --weights best.pt --source site.jpg --out out/ --conf 0.35
    python src/predict_compliance.py --weights best.pt --source site.jpg --rule legacy

Compliance rules (per detected Person):

  positive (default) — require POSITIVE detection of each required gear item
      (default: helmet + vest, the model's strongest classes) on the person;
      any overlapping `no_helmet` / `no_goggle` detection adds a violation.
      Rationale: the `no_*` training data is off-domain stock photography
      (see docs/dataset_analysis.md, Finding 2), so absence-of-gear must not
      depend on the weakest classes firing. This fixes the V1 failure mode
      where workers without hard hats passed as COMPLIANT.

  legacy — the V1 rule: NON-COMPLIANT only if a `no_helmet` / `no_goggle` box
      overlaps the person or no `vest` overlaps them. Kept for comparison.

Dataset classes (Ultralytics Construction-PPE):
    helmet, gloves, vest, boots, goggles, none, Person,
    no_helmet, no_goggle, no_gloves, no_boots
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO

VIOLATION_LABELS = {"no_helmet", "no_goggle"}
DEFAULT_REQUIRED = ("helmet", "vest")


def center_in(box, person):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def judge(near, rule, required):
    """Return the list of problems for one person given nearby detection labels."""
    if rule == "legacy":
        problems = [n.replace("no_", "no ") for n in near if n in VIOLATION_LABELS]
        if "vest" not in near:
            problems.append("no vest")
        return problems
    # positive-evidence rule
    problems = [f"missing {g}" for g in required if g not in near]
    problems += [n.replace("no_", "no ") for n in near if n in VIOLATION_LABELS]
    return problems


def annotate(model, img_path, conf=0.35, imgsz=640, rule="positive", required=DEFAULT_REQUIRED):
    names = model.names
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    r = model(img_path, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = [(names[int(c)], list(map(float, b))) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
    persons = [b for n, b in dets if n.lower() == "person"]
    others = [(n, b) for n, b in dets if n.lower() != "person"]
    if not persons:  # fall back to an image-level judgement
        persons = [[0, 0, img.shape[1], img.shape[0]]]

    n_ok = n_bad = 0
    for p in persons:
        near = [n for n, b in others if center_in(b, p)]
        problems = judge(near, rule, required)
        ok = not problems
        n_ok += ok
        n_bad += (not ok)
        color = (0, 170, 0) if ok else (0, 0, 220)  # BGR
        label = "COMPLIANT" if ok else "NON-COMPLIANT: " + ", ".join(problems)
        x1, y1, x2, y2 = map(int, p)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img, n_ok, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to trained best.pt")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="compliance_out", help="output folder")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; use 960 for a V2 model trained at 960")
    ap.add_argument("--rule", choices=["positive", "legacy"], default="positive",
                    help="positive = require detected gear (default); legacy = V1 rule")
    ap.add_argument("--require", default=",".join(DEFAULT_REQUIRED),
                    help="comma-separated gear required under the positive rule "
                         "(default: helmet,vest; add goggles to enforce eye protection "
                         "at the cost of more false flags — goggles are small objects)")
    args = ap.parse_args()

    required = tuple(g.strip() for g in args.require.split(",") if g.strip())
    model = YOLO(args.weights)
    if os.path.isdir(args.source):
        paths = [p for p in sorted(glob.glob(os.path.join(args.source, "*")))
                 if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    else:
        paths = [args.source]

    os.makedirs(args.out, exist_ok=True)
    total_ok = total_bad = 0
    for p in paths:
        img, n_ok, n_bad = annotate(model, p, conf=args.conf, imgsz=args.imgsz,
                                    rule=args.rule, required=required)
        outp = os.path.join(args.out, os.path.basename(p))
        cv2.imwrite(outp, img)
        total_ok += n_ok
        total_bad += n_bad
        print(f"{os.path.basename(p)}: {n_ok} compliant, {n_bad} non-compliant -> {outp}")
    print(f"\nDone. {len(paths)} image(s). Totals: {total_ok} compliant, {total_bad} non-compliant.")


if __name__ == "__main__":
    main()
