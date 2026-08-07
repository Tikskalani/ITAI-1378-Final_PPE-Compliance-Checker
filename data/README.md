# Data — PPE Compliance Checker

This folder documents the datasets. Raw images and model weights are not committed (see `.gitignore`); they are downloaded from the links below. All data is public and already labeled, so no manual collection is required.

## Sources

| Dataset | Size | Classes / labels | Link |
|---|---|---|---|
| Ultralytics Construction-PPE (V1) | 1,416 images (1,132 train / 143 val / 141 test) | 11 classes (helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots) | https://docs.ultralytics.com/datasets/detect/construction-ppe/ |
| SH17 (V2 scale-up) | 8,099 images · ~75,994 instances | 17 PPE + body-part classes | https://github.com/ahmadmughees/SH17dataset |
| Roboflow Universe — PPE | Thousands of images | presence + NO- variants | https://universe.roboflow.com/browse/construction/ppe |

Each dataset's license is listed on its linked page.

## Classes used (V1)
`helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots`

## Preparation
1. Construction-PPE downloads automatically through Ultralytics on the first training run.
2. Images are standardized to YOLO format at 640×640; the provided train/val/test split is kept.
3. Light augmentation (brightness, blur, partial occlusion) is applied for site-lighting robustness.
