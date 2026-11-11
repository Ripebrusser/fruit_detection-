# Final feature status

Status legend: **Working** — implemented, exercised end to end, evidence recorded.
**Partial** — works with a measured, documented limit. **Not supported** — deliberately absent
because the data does not exist to support it.

| # | Feature | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Correct fruit identification (8 fruits) | Working | 97.50 % fruit-level accuracy on 480 test images |
| 2 | Fresh / rotten classification | Working | 98.75 % condition accuracy; every fruit ≥ 0.967 |
| 3 | Fresh % and rotten % | Working | summed softmax probabilities, always sum to 100 % (asserted) |
| 4 | Fruit identification confidence | Working | fruit-level probability; matches `fruit_probabilities[fruit]` exactly |
| 5 | Condition confidence, shown separately | Working (new) | `condition_confidence` in the API and a second meter in the UI |
| 6 | One consistent verdict per upload | Working (fixed) | single `verdict` object drives every card; 0 contradictions in 480 test payloads |
| 7 | Visual spoilage assessment | Working | "Shows visible signs of spoilage" / "Appears visually fresh" / "Uncertain" / "Not assessed" |
| 8 | Food-safety disclaimer | Working | exact required text on every result and in the footer |
| 9 | Unknown / invalid image handling | **Partial** | declines to name a fruit for 4.58 % of test images and 71.7 % of synthetic non-fruit probes; unseen fruit *species* still receive a closest-match answer with explicit confidence |
| 10 | Blurry / dark / tiny / multiple-fruit handling | Working | measured per case in `gate_probe.json`; quality is advice, confidence decides |
| 11 | Drag-and-drop, preview, analyse, loading, clear | Working | driven in the browser (file injection → analyse → result → clear → re-upload) |
| 12 | Result dashboard (cards, bars, chips, banner) | Working | rendered from real API responses in Chromium |
| 13 | Responsive layout | **Partial** | breakpoints at 980/640 px reviewed; not exercised on a physical device |
| 14 | Error handling (wrong type, corrupt, empty, oversized, too small) | Working | HTTP 400/413 with JSON messages; covered by tests |
| 15 | Upload security | Working | `secure_filename`, allow-list, 8 MB / 40 MP caps, traversal blocked |
| 16 | Model card with measured metrics | Working | pulled live from `metrics.json`, including the gate's own operating point |
| 17 | Automated tests | Working | 67 passed in 58.92 s |
| 18 | Per-fruit / per-class evaluation | Working (new) | `Evaluation/metrics/per_fruit_report.{json,md}`, `misclassified.csv` |
| 19 | Architecture comparison (16-class vs multi-task) | Working (new) | trained and measured; 16-class retained |
| 20 | **Remaining shelf life (days)** | **Not supported** | no storage age, temperature or time-to-spoilage labels exist; EXIF dates are Mar–Apr 2022 with no storage meaning. UI: "Remaining shelf life cannot currently be estimated from the available dataset." |
| 21 | **Cross-section (cut fruit) analysis** | **Not supported** | no cut/whole label; cut fruit appears only as a small unlabelled minority in apple, pomegranate and orange sets. Stated as not validated in the hero and About sections. |
| 22 | Fruit-ripeness grading beyond fresh/rotten | Not supported | the dataset has two condition classes per fruit, not a ripeness scale |
| 23 | Grain / non-fruit object detection | Not supported | no labelled non-fruit class; synthetic probes are the only negatives |

## What "Working" means here

For features 1–20 the claim rests on one of: a run over the 480-image test split, an automated
test, an HTTP request against the running server, or a scripted browser interaction. The
browser checks recorded in this round were:

1. Load `/` → hero, upload card and performance tiles render; no console errors; all network
   requests 200.
2. Inject a `RottenPomegranate` test photograph → preview appears, Analyse enables.
3. Analyse → banner "Rotten pomegranate", fruit chip "Fruit identified", condition chip
   "Condition confident", fresh 0.1 %, rotten 99.9 %, condition confidence 99.9 %, spoilage
   "Shows visible signs of spoilage", one blur advisory, shelf-life message as specified.
4. Clear → results hidden, dropzone restored, Analyse disabled again.
5. Inject a generated noise PNG → Analyse → banner "Fruit not identified", fruit "Not
   identified", condition "Not assessed", percentages "—", edibility "Not assessed", muted
   condition card. Refresh and re-upload behave identically.

## Known gaps that are code-level, not data-level

- **Unseen fruit species.** The softmax always returns a fruit. A photograph of a mango is
  answered with the closest of the eight. The gate flags it when confidence is low, which is
  the honest limit of a closed-set classifier without real negative examples. Fixing this
  properly needs a labelled "not one of these" dataset or an embedding-distance gate fitted on
  real negatives — neither is available inside this project.
- **Apple is the weakest fruit** (0.900 fruit-level recall), from fresh-apple ↔ fresh-pomegranate
  confusion. Not a code defect; visible in `misclassified.csv`.
- **Temperature scaling does not help** this model (ECE 1.45 % → 1.65 %). It is retained because
  the gate thresholds were fitted on calibrated scores; removing it would require refitting the
  gate, with no measured benefit either way.

## What is *not* claimed

- No shelf-life estimate, in days or otherwise, anywhere in the UI or API.
- No cross-section validation, for any fruit.
- No guarantee of food safety.
- No claim that the model recognises fruits outside the eight trained types.
