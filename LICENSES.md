# Licences and Attribution

## 1. Dataset

**Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality**

- Authors: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin
- Publisher: Mendeley Data, Version 1, published 8 April 2022
- DOI: [10.17632/bdd69gyhv8.1](https://doi.org/10.17632/bdd69gyhv8.1)
- Dataset page: <https://data.mendeley.com/datasets/bdd69gyhv8/1>
- Companion article: Sultana, Jahan & Uddin, *"An extensive dataset for successful recognition of
  fresh and rotten fruits"*, Data in Brief 44 (2022) 108552,
  DOI [10.1016/j.dib.2022.108552](https://doi.org/10.1016/j.dib.2022.108552) (open access, CC BY).
- **Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)** — as stated on the
  Mendeley Data record. Attribution is given above; the dataset is reused unmodified.

The raw images are **not** redistributed by this repository's code and are excluded from version
control via `.gitignore`. Only derived artefacts (split manifest, trained weights, metrics) are
produced by the project.

### Note on the augmented archive

The Mendeley record also publishes an augmented set (12,335 images derived from the 3,200 originals
with rotation, flipping, zooming and shearing). **This project deliberately does not use it.** All
augmentation is performed here, after the split, using the training split only, as required to keep
validation and test data clean.

## 2. Project code

Written for this B.Tech coursework submission. No licence is granted for redistribution; the code is
provided "as is", without warranty of any kind. It is not a food-safety instrument and must not be
used to make consumption decisions.

## 3. Third-party software

| Component | Version used | Licence |
|-----------|--------------|---------|
| TensorFlow / Keras | 2.20.0 | Apache-2.0 |
| MobileNetV2 ImageNet weights (Keras Applications) | bundled with Keras | Apache-2.0 |
| NumPy | 2.2.6 | BSD-3-Clause |
| pandas | 2.x | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| SciPy | 1.18.1 | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | 5.0.0 | Apache-2.0 |
| Pillow | 11.x | MIT-CMU / HPND |
| Flask | 3.1.3 | BSD-3-Clause |
| Werkzeug | 3.1.9 | BSD-3-Clause |
| imagehash | 4.3+ | BSD-2-Clause |
| matplotlib | 3.9+ | PSF-based (matplotlib licence) |
| seaborn | 0.13.2 | BSD-3-Clause |
| pytest | 9.2.0 | MIT |
| uv (environment tool) | 0.13.7 | Apache-2.0 / MIT |

Refer to each project's own licence text for the authoritative terms.
