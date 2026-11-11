# Final fix register

Every defect below was observed on this codebase, not hypothesised. Each entry records how it
was reproduced, what actually caused it, what changed, and how the fix was verified.

Severity: **S1** wrong or contradictory result shown to a user · **S2** invalid measurement or
protocol violation · **S3** robustness/quality.

---

## F-01 — Rotten pomegranate reported as "unable to identify" *and* "looks rotten"

- **Severity:** S1
- **Reported:** a photograph of a rotten pomegranate produced an uncertain identification and,
  further down the same page, a confident rotten prediction.
- **Reproduced:** 30 held-out `RottenPomegranate` test images run through the real predictor.
  **14 of 30 were rejected** as unidentifiable. Thirteen of those fourteen had fruit-level
  confidence ≥ 0.85 and named the right fruit (one was genuinely misclassified as
  `RottenApple` at 0.497). Example: `top1=0.999`, `fruit confidence=1.000`,
  `rotten=0.999` — rejected anyway.
- **Root cause:** three separate faults compounded.
  1. The rejection gate fired on *image quality*. The blur threshold was the 1st percentile of
     Laplacian variance on the test split, which smooth-skinned fruit (banana, pomegranate)
     naturally sit at the bottom of. **47 of 55 rejections across the whole test split were
     "the image looks blurry"**, including images classified at 100.0% confidence.
  2. Identity was gated on the **16-class top-1 probability**, so a fruit whose fresh and
     rotten classes each took ~45% of the mass looked "unidentified" even though the fruit
     itself was certain.
  3. The front end rewrote only the fruit name when the gate fired. The fresh/rotten bars,
     the condition badge and the edibility label kept rendering the confident prediction
     underneath the "uncertain" message — the exact contradiction the user saw.
- **Fix:**
  - `SRC/models/predictor.py`: identity is now gated on **fruit-level confidence** (the summed
    probability of that fruit's two classes) plus a fruit-margin floor. Condition is a separate
    decision with its own threshold and confidence. Quality measurements became *warnings*.
  - The payload gained a single authoritative `verdict` object
    (`identified` / `identified_condition_uncertain` / `unidentified`) which the UI uses to
    drive every card.
  - `Website/static/js/app.js`: an unidentified image now shows "Not assessed" for condition,
    em-dashes for the percentages, a muted card and "Not assessed" for edibility. No card can
    disagree with the banner.
- **Verified:** **2 of 30** rejected (both genuine model/confidence failures), 0 contradictory
  payloads across all 480 test images, and the browser flow re-tested end to end.
- **Files:** `SRC/models/predictor.py`, `Website/static/js/app.js`,
  `Website/templates/index.html`, `Website/static/css/style.css`

## F-02 — Acceptance thresholds were fitted on the test split

- **Severity:** S2
- **Observed:** `Models/ood_config.json` declared
  `"calibrated_on": "validation split for temperature; test split for thresholds"`, and
  `SRC/models/evaluate.py` derived the blur, brightness and top-1 thresholds from the split it
  was also evaluating on.
- **Impact:** every rejection number was measured on data used to set the threshold — an
  optimistic and non-reproducible protocol, and exactly what the project rules forbid.
- **Fix:** new `SRC/models/calibrate_gate.py` fits temperature, fruit confidence, fruit margin
  and condition thresholds **on the validation split only**, writing `Models/gate_config.json`.
  `evaluate.py` no longer writes thresholds; it loads them and *reports* test behaviour.
- **Verified:** `Evaluation/test_results/gate_calibration.json` records the validation target
  (≤2% false rejection) and the achieved 1.87%. The test split is evaluated once.

## F-03 — An unidentified image still carried a condition verdict

- **Severity:** S1
- **Found by:** the new consistency test, immediately after F-01 was implemented.
- **Detail:** `verdict.condition` returned `"rotten"` for an image the same payload called
  unidentified, because the condition check ran independently of whose fruit had been named.
- **Fix:** `verdict.condition` is now only published when a fruit has been identified and the
  condition is confident.
- **Verified:** `Tests/test_verdict_consistency.py::test_no_payload_contradicts_itself` over a
  sample of the test split.

## F-04 — Edibility verdict produced for images with no identified fruit

- **Severity:** S1 (same family as F-01)
- **Detail:** `_edibility()` computed "Appears visually fresh" from the fresh/rotten split even
  when the fruit was rejected, implying a safety-adjacent observation about an unknown object.
- **Fix:** `_edibility()` returns `not_assessed` when the fruit is unidentified, `uncertain`
  when the condition split is not decisive, and only then the fresh/spoilage labels.

## F-05 — `SRC/models/gate_report.py` read removed configuration keys

- **Severity:** S3
- **Detail:** the script accessed `predictor.config["min_top1_probability"]`,
  `["max_entropy_ratio"]`, `["min_blur_variance"]` and `["min_brightness"]`, all of which the
  two-stage gate replaced. It would crash with `KeyError`.
- **Fix:** rewritten against `load_gate_config()`, and it now measures identity and condition
  outcomes separately across test images, synthetic probes, degraded images and flat colour.

## F-06 — Dead and duplicated measurement code in `evaluate.py`

- **Severity:** S3
- **Detail:** `min_blur` and `min_brightness` were computed from the test split and then never
  used once threshold writing was removed; a second `classification_report` import shadowed the
  module-level one and raised `UnboundLocalError` when the per-class block was added.
- **Fix:** unused quality computation deleted (reference values now live in
  `Models/gate_config.json`, fitted on validation); the duplicate import removed and the block
  reuses the report that was already computed.

## F-07 — Bug in the first version of the per-fruit report

- **Severity:** S3
- **Detail:** the manifest merge keyed on a mapped Series and produced `KeyError:
  ['relative_path'] not in index`.
- **Fix:** `file_name` is now materialised as a column before merging.

## F-08 — Shelf-life wording did not match the required text

- **Severity:** S2 (honesty of the reported feature)
- **Detail:** the message read "Remaining shelf life: Not available — insufficient time-based
  training data."
- **Fix:** now exactly "Remaining shelf life cannot currently be estimated from the available
  dataset." Reason text unchanged and still honest about the missing labels.

## F-09 — Model card described a gate that did not behave as claimed

- **Severity:** S2
- **Detail:** the site advertised "calibrated confidence with an explicit uncertainty gate"
  while the gate rejected 11.5% of genuine test photographs, 85% of them for image quality
  alone. The claim was not false, but the operated point was indefensible.
- **Fix:** the gate was re-fitted (F-02) and the site now prints its own operating point and the
  measured rejection rate (`#gate-summary`), so the claim is checkable rather than asserted.

## F-10 — Cosmetic placeholder left in the performance tiles

- **Severity:** S3 (introduced during this work, fixed before delivery)
- **Detail:** a "§" character was used as the value of the cross-section tile.
- **Fix:** replaced with "No", which is also the truthful answer.

---

## Issues found but deliberately *not* "fixed"

| Item | Why it was left alone |
| --- | --- |
| Cross-section support | No cut/whole label exists in the dataset and only a small unlabelled minority shows cut fruit. A classifier cannot be validated on it, and inventing cut images would fabricate data. The site states it is not validated. |
| Shelf-life estimation | No capture dates, storage temperatures or days-to-spoilage labels. EXIF timestamps all fall in Mar–Apr 2022 and carry no storage meaning. Any day figure would be invented. |
| Unseen fruit species rejection | The 8-fruit softmax always has an answer. Without real photographs of other fruits (not downloadable under the project rules) the only defensible negatives are synthetic probes. Measured honestly and reported; the UI shows confidence rather than pretending certainty. |
| 15 misclassified test images | Model behaviour, not a code defect. Listed in `Evaluation/test_results/misclassified.csv` for inspection. |

## Defects confirmed *absent*

Checked explicitly rather than assumed:

- **Class mapping** — `Models/class_names.json` pairs each index with its fruit
  (`fruits[i // 2]`), 16 names / 8 fruits, and the mapping reproduces 96.88% test accuracy.
  A permutation would collapse accuracy to ~6%.
- **Preprocessing parity** — the serving path (PIL → `preprocess_for_model` → 224×224 BILINEAR
  RGB → model with `preprocess_input` baked in) and the evaluation path (cached `uint8` `.npy`)
  produced **480/480 identical predictions**.
- **Single inference path** — only `FruitPredictor.predict_array` touches the network; the
  Flask layer adds file handling, nothing else. No hardcoded or random predictions exist in
  the codebase (grep for `random`, `np.random` inside `Website/` returns only the test
  fixtures' noise generators).
- **Upload handling** — `secure_filename`, extension allow-list, 8 MB cap, 40 MP cap,
  64 px minimum side, `Image.verify()` before decode, upload deletion only inside the upload
  directory, HTTP errors preserved instead of becoming 500s.
