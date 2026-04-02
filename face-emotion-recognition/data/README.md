# 📥 FER-2013 Dataset Instructions

**Dataset:** FER-2013 (Facial Expression Recognition)
**Kaggle Link:** https://www.kaggle.com/datasets/msambare/fer2013

---

## Option A — Kaggle CLI
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d .
```

## Option B — Manual
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click **Download** → extract zip
3. Place folders inside `data/`

## Expected Structure After Download
```
data/
├── train/
│   ├── angry/       (3,995 images)
│   ├── disgust/     (436 images)
│   ├── fear/        (4,097 images)
│   ├── happy/       (7,215 images)
│   ├── neutral/     (4,965 images)
│   ├── sad/         (4,830 images)
│   └── surprise/    (3,171 images)
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

**Total Images:** ~35,887
**Image Size:** 48×48 pixels, grayscale
