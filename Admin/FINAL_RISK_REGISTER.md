# Final risk register

Post-audit position, 9 November 2026. Each risk states what it is, what was measured, what
reduces it, and what remains. "Residual" is the risk left after the mitigations in this
repository — not after the mitigations we would like to have.

Scoring: likelihood × impact, High / Medium / Low.

| ID | Risk | Likelihood | Impact | Residual | Mitigation in place |
| --- | --- | --- | --- | --- | --- |
| R1 | **A user treats a fresh verdict as proof of food safety** | Medium | High | Medium | Mandatory exact disclaimer on every result; edibility worded as a visual observation; internal contamination explicitly called invisible in a photograph |
| R2 | **Unseen fruit species answered confidently** | Medium | Medium | **Medium–High** | Gate + separate confidence + explicit "closest supported fruit" language. Measured: 71.7 % of 120 synthetic non-fruit probes rejected. Real mango/persimmon photographs are *not* covered by that number and may still pass |
| R3 | **Ambiguous fresh/rotten reported as certain** | Low | High | Low | Condition now has its own threshold (0.673) and its own confidence; below it the UI says "condition unclear" and the spoilage label says "Uncertain" |
| R4 | **Genuine photograph refused** ("unable to identify") | Low | Medium | Low | Previously 11.5 % of test images, now 4.58 %; blur/darkness no longer reject anything. Rotten pomegranates: 14/30 → 2/30 |
| R5 | **Metrics overstate real-world performance** | Medium | High | Medium | Training used studio-style images on plain backgrounds; 480 test images from the same pool give ±2–4 pp band; per-class metrics rest on 30 images. Stated in the README and evaluation report |
| R6 | **Test-set leakage through threshold tuning** | — | High | **Closed** | All thresholds now fitted on validation (`calibrate_gate.py`); `evaluate.py` only reports. Verified by reading `gate_config.json`'s `calibrated_on` |
| R7 | **Duplicate images spanning splits inflating accuracy** | Low | High | Low | SHA-256 + pHash (Hamming ≤ 4) union-find grouping; 104 groups, **0 spanning splits**, asserted by a test |
| R8 | **Contradictory result shown to the user** | Low | High | Low | Single `verdict` object; UI derives every card from it; regression test over the test split asserts 0 contradictions |
| R9 | **Shelf-life or cross-section claims creep back in** | Low | High | Low | `shelf_life.available` is hard `False` with a fixed message; `cross_section_validated` is hard `False`; both asserted by tests, and the wording is on the page itself |
| R10 | **Upload abuse** (huge file, decompression bomb, traversal, stored payloads) | Low | Medium | Low | 8 MB cap, 40 MP cap, 64 px floor, `Image.verify()` before decode, `secure_filename`, delete restricted to the upload directory, 1-hour TTL sweep |
| R11 | **CPU-only inference too slow for demo use** | Low | Low | Low | 83.5 ms mean per image on CPU; the measured worst case in a live request was 1.3 s including upload and JSON round trip |
| R12 | **Environment drift** (Python 3.14 has no TensorFlow wheels; native Windows TF is CPU-only) | Medium | Medium | Medium | Pinned `.venv` on Python 3.12.16 + TF 2.20.0, recorded in `requirements.txt` and the README; WSL2 GPU path documented but not used |
| R13 | **Single-machine, single-user demonstration** | Medium | Medium | Accepted | Flask development server, no authentication, no request queue. Adequate for a college review; not a deployment configuration, and the startup log says so |
| R14 | **Dataset licensing / attribution missed** | Low | Medium | Low | CC BY 4.0 attribution in `LICENSES.md`, the README and the page footer, with DOI |

## Risks that remain genuinely open

**R2 is the one to keep saying out loud.** A closed-set classifier asked about a fruit it has
never seen will answer. The gate reduces the chance of a confident wrong answer but cannot
eliminate it, and the only negatives available in this project are synthetic patterns, which are
easier to reject than a real photograph of a different species. The honest statement for the
review is: *"it recognises eight fruits; for anything else it reports its closest match and how
confident it is, and it will sometimes be wrong."* Obtaining a small labelled set of
non-supported fruit photographs (say 20 per species) would let an embedding-distance gate be
fitted and measured properly — that is the one dataset addition that would materially change
this risk.

**R5 is the other.** The 96.88 % figure describes studio-style dataset images, not phone photos
of a fruit bowl. Any demonstration should show both: a dataset-style image where it looks
excellent, and a cluttered real photo where it may not.

## Closed since the first release

| Was | Now |
| --- | --- |
| Gate rejected 11.5 % of genuine test images | 4.58 % |
| 47 of 55 rejections were "blurry" on correctly classified fruit | 0 |
| 14/30 rotten pomegranates reported as unidentified | 2/30, both genuine uncertainty |
| Rejection and confident prediction shown together on one page | impossible by construction |
| Thresholds fitted on the test split | fitted on validation, reported on test |
| Unidentified images carried a condition label | no condition, no edibility claim |
