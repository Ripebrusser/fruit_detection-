# Dataset Documentation

Everything below was measured from the dataset in this repository on 2026-11-09 using
`SRC/data/inspect_dataset.py`, `SRC/data/find_cross_sections.py` and `SRC/data/prepare_dataset.py`.
Raw evidence lives in `Dataset/interim/`.

## 1. Source

- **Name**: Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality
- **Publisher**: Mendeley Data — <https://data.mendeley.com/datasets/bdd69gyhv8/1>
- **Local location**: `Dataset/Original Image/Original Image/`
- **No archive file** is present in the repository; only the extracted folder tree was supplied.
  The pre-augmented archive was **not** used, and no additional data was downloaded.
- **Licence**: **CC BY 4.0** (as published on the Mendeley Data record). Attribution and the full
  licence note are in `LICENSES.md`.
- **Collection context** (from the companion Data in Brief article): images were gathered between
  16 and 31 March 2022 from fruit shops and fields in Savar, Dhaka, Bangladesh — consistent with the
  EXIF capture dates found in this repository.

## 2. Actual structure

```
Dataset/
└── Original Image/
    └── Original Image/
        ├── FreshApple/          (200 .jpg)
        ├── FreshBanana/         (200 .jpg)
        ├── FreshGrape/          (200 .jpg)
        ├── FreshGuava/          (200 .jpg)
        ├── FreshJujube/         (200 .jpg)
        ├── FreshOrange/         (200 .jpg)
        ├── FreshPomegranate/    (200 .jpg)
        ├── FreshStrawberry/     (200 .jpg)
        ├── RottenApple/         (200 .jpg)
        ├── RottenBanana/        (200 .jpg)
        ├── RottenGrape/         (200 .jpg)
        ├── RottenGuava/         (200 .jpg)
        ├── RottenJujube/        (200 .jpg)
        ├── RottenOrange/        (200 .jpg)
        ├── RottenPomegranate/   (200 .jpg)
        └── RottenStrawberry/    (200 .jpg)
```

File naming is `<Class> (<n>).jpg` with `n` = 1…200 in every folder.

## 3. Verified counts

| Property | Measured value |
|----------|----------------|
| Total files | **3,200** |
| File formats | 3,200 `.jpg` (100 %) |
| Readable / decodable by Pillow | **3,200 (100 %)** |
| Corrupted or truncated files | **0** |
| Classes | **16** (8 fruits × 2 conditions) |
| Images per class | **200** (perfectly balanced) |
| Fresh images | 1,600 |
| Rotten images | 1,600 |
| Images per fruit | 400 |
| Smallest image | 251 × 301 px |
| Largest image | 4160 × 4160 px |
| Median size | 2492 × 2330 px |
| Images below 224 px on any side | **0** |
| Images with EXIF capture date | 3,000 |
| EXIF capture period | March–April **2022** only (2,974 distinct timestamps) |
| Camera makes recorded | Symphony (1,750), "--" (355), Xiaomi (105) |
| Mean brightness | 156.3 / 255 (0 images below 40 → no near-black photos) |

The 3,200 originals are all at least 224 px on both sides, so the required 224 × 224 resize
always downscales and never invents pixels.

## 4. Duplicates and suspicious content

Two levels of duplicate detection were run:

| Method | Groups | Images involved |
|--------|--------|-----------------|
| SHA-256 identical file | **7 groups** | 14 files (7 redundant copies) |
| Perceptual hash (pHash), Hamming ≤ 4 | 38 groups | 90 images |
| pHash Hamming ≤ 6 | 98 groups | 241 images |
| pHash Hamming ≤ 8 | 188 groups | 586 images (rejected as too loose) |

- All **7 exact duplicate groups are inside `FreshOrange`** (e.g. `FreshOrange (27).jpg` =
  `FreshOrange (77).jpg`). They are genuine re-uploads of the same file.
- The pHash ≤ 4 level is used for split grouping — it is the strictest level that still catches
  visibly re-encoded copies without merging unrelated fruit. pHash ≤ 8 was measured and rejected
  because it merged 586 images including obviously different fruits, i.e. it produced
  false positives.
- **1 cross-class pHash ≤ 4 group** exists (two files that are near-identical but sit in different
  class folders); 13 such groups exist at the looser ≤ 6 level. This is a small amount of label
  noise inherent to the dataset. Both copies are placed in the **same** split so the model never
  sees an identical image in training and testing, and the issue is recorded in the risk register.
- No images with extreme exposure were found; 11 images are very bright (mean > 220) but readable.
- Visual review also found a handful of non-fruit photographs (apple orchard, strawberry field and
  market-wide shots, e.g. `FreshStrawberry (166)–(170)`, `FreshApple (35)–(39)`). They carry the
  correct fruit label but no single fruit, which adds label/interior noise. They are kept — they are
  part of the published dataset — and their effect is discussed in the evaluation limitations.

## 5. Whole fruit vs cross-section (critical finding)

**Method.** Every class folder was rendered as a labelled thumbnail grid
(`Dataset/interim/review_grids/classes/<Class>.jpg`, 200 images per sheet, numeric file order) and
inspected visually. A background-masked colour analysis (`Dataset/interim/fruit_colour_scores.csv`)
was computed as a second signal.

**Result.**

- The overwhelming majority of images (>95 %) are **whole-fruit photographs** of the intact fruit,
  usually on a plain white, grey or pale-blue surface with one to a few specimens.
- A **small number of cross-section photographs exist**, concentrated in:
  - `FreshApple` and `RottenApple` — apples cut in half, showing pale flesh, core and seeds
    (e.g. `RottenApple (189).jpg`, `RottenApple (190).jpg`, `RottenApple (194).jpg`,
    `RottenApple (195).jpg`, `RottenApple (132).jpg`).
  - `FreshPomegranate` / `RottenPomegranate` — opened fruit with visible arils.
  - Isolated cases in `RottenOrange` (torn fruit) and `RottenJujube` (one split fruit).
  - `FreshGrape` contains a few "bunch close-up" shots, but they are still whole grapes.
- Cross-sections are **not labelled**, **not evenly distributed**, and represent at most a low
  single-digit percentage overall. There is no reliable way to know the exact count without
  annotating all 3,200 images by hand, which was not done; the naming heuristic in
  `SRC/data/find_cross_sections.py` is a *screening aid*, not a ground truth (its pale-fraction score
  also fires on light-skinned whole fruit, so the candidate montages were the deciding evidence).

**Decision.** The project therefore does **not** claim validated cross-section support:

- The model is trained and evaluated on the dataset as published; whatever cross-section images
  exist are naturally present in training and test, so the network is not blind to cut fruit.
- No separate cross-section accuracy is reported as a validated metric, because the subset cannot be
  identified with confidence. This is stated in the UI and README.
- The pipeline keeps an optional `is_cross_section` column in the manifest (all values `unknown`),
  and `SRC/models/predictor.py` exposes a `image_type` field, so a future curated subset can enable
  cross-section validation without redesigning the application.

**Data that would be needed to add it properly:** a balanced, explicitly labelled set of cut-fruit
photographs per class (both halves and slices), several hundred per class, shot in the same variety
of lighting/backgrounds as real user photos.

## 6. Shelf life / time-to-spoilage labels — none available

Checked exhaustively:

- No CSV, JSON, XML, text or spreadsheet metadata accompanies the images (3,200 files, all `.jpg`).
- Folder and file names encode only fruit and condition.
- EXIF carries camera make, orientation and a capture timestamp (March–April 2022). All images were
  captured in a single short period, so EXIF dates are **acquisition dates, not storage age**.
- No storage temperature, harvest date, or days-to-spoilage field exists anywhere.

**Conclusion:** a shelf-life regression model is not feasible with this dataset. Any "days until
rotten" figure would be invented. The system reports:

> Remaining shelf life: Not available — insufficient time-based training data.

## 7. Class definition used by the model

Index order is fixed in `Models/class_names.json`:

| Idx | Class | Idx | Class |
|-----|-------|-----|-------|
| 0 | FreshApple | 8 | RottenApple |
| 1 | FreshBanana | 9 | RottenBanana |
| 2 | FreshOrange | 10 | RottenOrange |
| 3 | FreshGrape | 11 | RottenGrape |
| 4 | FreshGuava | 12 | RottenGuava |
| 5 | FreshJujube | 13 | RottenJujube |
| 6 | FreshPomegranate | 14 | RottenPomegranate |
| 7 | FreshStrawberry | 15 | RottenStrawberry |

Fruit identity = index // 2 mapped onto `FRUITS`
(`Apple, Banana, Orange, Grape, Guava, Jujube, Pomegranate, Strawberry`).
Condition = fresh when index % 2 == 0, otherwise rotten.

## 8. Split policy

- Stratified per class, 70 / 15 / 15, seeded (`seed = 42`).
- Duplicate groups are atomic: every member of a group goes to the same split.
- Augmentation is generated on the fly from the training split only; validation and test arrays are
  stored unaugmented.
- Exact counts are written to `Models/class_names.json → split_counts` and to
  `Dataset/dataset_manifest.csv`.
