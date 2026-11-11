"""Inference pipeline: fruit identity, condition scores, uncertainty and warnings.

The predictor answers three separate questions instead of one fused verdict:

* **Fruit identity** — "is this a supported fruit, and which one?" Scored with the summed
  probability of that fruit's fresh and rotten classes, gated by a threshold calibrated on
  the validation split.
* **Condition** — "does it look fresh or rotten, and how decisive is that split?" Has its
  own confidence. A weak condition marks the freshness verdict as uncertain; it never
  discards the fruit identification.
* **Capture quality** — blur, exposure and resolution. These are reported as warnings
  only. Quality is evidence about the photograph, not about which species it depicts.

Keeping those apart is what stops a single result page from saying "Unable to identify
this fruit confidently" while also showing a confident rotten score. The symmetrical
failure — a clear photograph of a supported fruit rejected because its skin is smooth —
is avoided the same way.

The predictor is deliberately explicit about what it does *not* know:

* Condition percentages are summed model probabilities, not physical measurements.
* Shelf life is always unavailable: the training data has no time labels.
* Edibility is described as a visual observation, never a food-safety verdict.

Usage (programmatic):
    from SRC.models.predictor import FruitPredictor
    predictor = FruitPredictor()
    result = predictor.analyze(pil_image)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from SRC.utils.imaging import preprocess_for_model, quality_checks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "Models"
DEFAULT_MODEL_PATH = MODELS_DIR / "final_model" / "fruit_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
GATE_CONFIG_PATH = MODELS_DIR / "gate_config.json"
LEGACY_OOD_CONFIG_PATH = MODELS_DIR / "ood_config.json"

SAFETY_DISCLAIMER = (
    "This result is an AI-based visual assessment only. It cannot guarantee food safety. "
    "Inspect the fruit carefully and do not consume it if you suspect spoilage."
)
SHELF_LIFE_MESSAGE = (
    "Remaining shelf life cannot currently be estimated from the available dataset."
)
UNKNOWN_MESSAGE = (
    "Unable to identify this fruit confidently. Please upload a clearer image of one fruit."
)
RETRY_HINT = "Use a well-lit photograph of a single fruit filling most of the frame."

DEFAULT_GATE_CONFIG: dict = {
    "version": 2,
    "temperature": 1.0,
    "fruit": {"min_confidence": 0.0, "min_margin": 0.0},
    "condition": {"min_confidence": 0.0},
    "quality": {
        "min_blur_variance": 0.0,
        "min_brightness": 0.0,
        "max_brightness": 255.0,
    },
    "calibrated_on": {"split": "defaults (no gate_config.json found)"},
}


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    total = float(probabilities.sum())
    if total <= 0:
        return np.full_like(probabilities, 1.0 / len(probabilities))
    return probabilities / total


def load_gate_config(path: Path | str = GATE_CONFIG_PATH) -> dict:
    """Read the serving gate, upgrading the flat v1 schema if an old file is present."""
    candidate = Path(path)
    if not candidate.exists() and candidate == GATE_CONFIG_PATH and LEGACY_OOD_CONFIG_PATH.exists():
        candidate = LEGACY_OOD_CONFIG_PATH
    if not candidate.exists():
        return json.loads(json.dumps(DEFAULT_GATE_CONFIG))

    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if raw.get("version") == 2:
        config = json.loads(json.dumps(DEFAULT_GATE_CONFIG))
        config.update(raw)
        return config

    # v1 layout: flat thresholds, and quality values that used to reject outright.
    return {
        "version": 2,
        "temperature": float(raw.get("temperature", 1.0)),
        "fruit": {
            "min_confidence": float(raw.get("min_top1_probability", 0.0)),
            "min_margin": float(raw.get("min_fruit_margin", 0.0)),
        },
        "condition": {"min_confidence": 0.0},
        "quality": {
            "min_blur_variance": float(raw.get("min_blur_variance", 0.0)),
            "min_brightness": float(raw.get("min_brightness", 0.0)),
            "max_brightness": float(raw.get("max_brightness", 255.0)),
        },
        "calibrated_on": {"split": raw.get("calibrated_on", "legacy v1 configuration")},
    }


class PredictorUnavailable(RuntimeError):
    """Raised when the trained model has not been produced yet."""


class FruitPredictor:
    """Wraps the trained Keras model with project-specific aggregation logic."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        class_names_path: Path | str = CLASS_NAMES_PATH,
        gate_config_path: Path | str = GATE_CONFIG_PATH,
        temperature: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        if not self.model_path.exists():
            raise PredictorUnavailable(
                f"trained model not found at {self.model_path}; run `python -m SRC.models.train`"
            )
        if not self.class_names_path.exists():
            raise PredictorUnavailable(
                f"class mapping not found at {self.class_names_path}; "
                "run `python -m SRC.data.prepare_dataset`"
            )

        meta = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        self.class_names: list[str] = meta["class_names"]
        self.fruits: list[str] = meta["fruits"]
        self.input_size: int = int(meta.get("input_size", 224))

        self.config = load_gate_config(gate_config_path)
        if temperature is not None:
            self.config["temperature"] = float(temperature)

        import tensorflow as tf

        self._tf = tf
        self.model = tf.keras.models.load_model(self.model_path)

    # ------------------------------------------------------------------ helpers
    @property
    def config_path(self) -> Path:
        return GATE_CONFIG_PATH

    def _temperature(self) -> float:
        return max(float(self.config.get("temperature", 1.0)), 1e-3)

    def _calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Temperature scaling applied in probability space: p ∝ p^(1/T)."""
        temp = self._temperature()
        if abs(temp - 1.0) < 1e-6:
            return probabilities
        with np.errstate(divide="ignore"):
            logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temp
        logits -= logits.max()
        exp = np.exp(logits)
        return _normalise(exp)

    def _fruit_scores(self, probabilities: np.ndarray) -> dict[str, float]:
        scores = {fruit: 0.0 for fruit in self.fruits}
        for idx, name in enumerate(self.class_names):
            fruit = name.replace("Fresh", "").replace("Rotten", "")
            scores[fruit] += float(probabilities[idx])
        return scores

    # ---------------------------------------------------------------- inference
    def predict_array(self, rgb: np.ndarray) -> dict:
        """Run the network on an RGB uint8 array and return calibrated outputs."""
        prepared = preprocess_for_model(rgb, self.input_size)[None, ...]
        started = time.perf_counter()
        raw = self.model.predict(prepared, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        raw = _normalise(np.asarray(raw, dtype=np.float64))
        calibrated = self._calibrate(raw)
        order = np.argsort(calibrated)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        fruit_scores = self._fruit_scores(calibrated)
        fruit_rank = sorted(fruit_scores.items(), key=lambda kv: kv[1], reverse=True)
        fresh = float(sum(calibrated[i] for i, n in enumerate(self.class_names) if n.startswith("Fresh")))
        rotten = float(sum(calibrated[i] for i, n in enumerate(self.class_names) if n.startswith("Rotten")))
        entropy = float(-(calibrated * np.log(np.clip(calibrated, 1e-12, 1.0))).sum())
        max_entropy = float(np.log(len(self.class_names)))

        return {
            "class_probabilities": {
                name: float(calibrated[i]) for i, name in enumerate(self.class_names)
            },
            "top_class": self.class_names[top1],
            "top_class_probability": float(calibrated[top1]),
            "runner_up_probability": float(calibrated[top2]),
            "class_margin": float(calibrated[top1] - calibrated[top2]),
            "fruit_probabilities": fruit_scores,
            "predicted_fruit": fruit_rank[0][0],
            "fruit_confidence": float(fruit_rank[0][1]),
            "fruit_margin": float(fruit_rank[0][1] - fruit_rank[1][1]),
            "fresh_probability": fresh,
            "rotten_probability": rotten,
            "entropy": entropy,
            "entropy_ratio": entropy / max_entropy,
            "inference_ms": elapsed_ms,
            "temperature": self._temperature(),
        }

    # ------------------------------------------------------------------ analysis
    def _fruit_assessment(self, pred: dict) -> dict:
        """Decide whether the fruit can be named, independent of its condition."""
        cfg = self.config.get("fruit", {})
        min_confidence = float(cfg.get("min_confidence", 0.0))
        min_margin = float(cfg.get("min_margin", 0.0))
        reasons: list[str] = []
        if pred["fruit_confidence"] < min_confidence:
            reasons.append(
                f"the model's confidence in any single fruit ({pred['fruit_confidence']:.0%}) is "
                f"below the calibrated acceptance level ({min_confidence:.0%})"
            )
        if pred["fruit_margin"] < min_margin:
            reasons.append(
                f"the two most likely fruits are too close to separate "
                f"(margin {pred['fruit_margin']:.0%})"
            )
        status = "uncertain" if reasons else "identified"
        return {
            "status": status,
            "fruit": pred["predicted_fruit"],
            "confidence": pred["fruit_confidence"],
            "margin": pred["fruit_margin"],
            "threshold": min_confidence,
            "reasons": reasons,
            "message": UNKNOWN_MESSAGE if reasons else None,
            "retry_hint": RETRY_HINT if reasons else None,
        }

    def _condition_assessment(self, pred: dict) -> dict:
        """Decide whether the fresh/rotten split is decisive. Never rejects the image."""
        cfg = self.config.get("condition", {})
        min_confidence = float(cfg.get("min_confidence", 0.0))
        fresh, rotten = pred["fresh_probability"], pred["rotten_probability"]
        condition = "fresh" if fresh >= rotten else "rotten"
        confidence = max(fresh, rotten)
        reasons: list[str] = []
        if confidence < min_confidence:
            reasons.append(
                f"fresh and rotten are too close ({confidence:.0%} at the calibrated "
                f"acceptance level of {min_confidence:.0%})"
            )
        return {
            "status": "uncertain" if reasons else "confident",
            "condition": condition,
            "confidence": confidence,
            "margin": abs(fresh - rotten),
            "threshold": min_confidence,
            "reasons": reasons,
        }

    def _quality_warnings(self, quality: dict, pred: dict) -> list[str]:
        """Blur and exposure become advice, never a reason to discard a prediction."""
        cfg = self.config.get("quality", {})
        notes: list[str] = []
        if quality["blur_variance"] < float(cfg.get("min_blur_variance", 0.0)):
            notes.append(
                "This photograph looks soft or out of focus. A sharper image would make the "
                "condition assessment more reliable."
            )
        if quality["brightness"] < float(cfg.get("min_brightness", 0.0)):
            notes.append("The photograph is unusually dark; brighten it if the result looks wrong.")
        if quality["brightness"] > float(cfg.get("max_brightness", 255.0)):
            notes.append("The photograph is over-exposed; some surface detail may be lost.")
        if min(quality["width"], quality["height"]) < self.input_size:
            notes.append(
                f"The uploaded image is smaller than {self.input_size}×{self.input_size} pixels, "
                "so detail is lost when it is resized for the model."
            )
        if not 0.02 < quality["mask_fraction"] < 0.995:
            notes.append(
                "The fruit could not be separated from the background, which usually means it "
                "does not fill enough of the frame."
            )
        if pred["class_margin"] < 0.10:
            notes.append(
                "The model is not decisively confident about the exact class, though the fruit "
                "identity above is based on the combined fresh and rotten evidence."
            )
        return notes

    def _edibility(self, condition_status: str, fruit_status: str, pred: dict) -> dict:
        if fruit_status != "identified":
            return {
                "status": "not_assessed",
                "label": "Not assessed",
                "reason": "The fruit was not identified confidently, so no condition "
                          "assessment is offered.",
                "disclaimer": SAFETY_DISCLAIMER,
                "note": "Visual freshness is not proof of safety: contamination and internal "
                        "spoilage can be invisible in a photograph.",
            }
        if condition_status != "confident":
            return {
                "status": "uncertain",
                "label": "Visual assessment is uncertain",
                "reason": "Fresh and rotten probabilities are too close to call from this "
                          "photograph.",
                "disclaimer": SAFETY_DISCLAIMER,
                "note": "Visual freshness is not proof of safety: contamination and internal "
                        "spoilage can be invisible in a photograph.",
            }
        if pred["rotten_probability"] >= 0.50:
            status, label = "shows_spoilage", "Shows visible signs of spoilage"
        else:
            status, label = "appears_fresh", "Appears visually fresh"
        return {
            "status": status,
            "label": label,
            "reason": None,
            "disclaimer": SAFETY_DISCLAIMER,
            "note": "Visual freshness is not proof of safety: contamination and internal "
                    "spoilage can be invisible in a photograph.",
        }

    def analyze(self, image, source_name: str | None = None) -> dict:
        """Full analysis payload consumed by the website and the tests.

        ``image`` may be a PIL image or an RGB uint8 numpy array.
        """
        if hasattr(image, "convert"):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        quality = quality_checks(rgb)
        pred = self.predict_array(rgb)
        fruit = self._fruit_assessment(pred)
        condition = self._condition_assessment(pred)
        edibility = self._edibility(condition["status"], fruit["status"], pred)

        warnings = self._quality_warnings(quality, pred)
        if fruit["status"] != "identified":
            warnings.insert(
                0,
                "This photograph does not match any of the eight supported fruits well enough "
                "to name one. The closest match is shown only for reference.",
            )
        elif condition["status"] != "confident":
            warnings.insert(
                0,
                "The fruit is identified, but the fresh/rotten split is too close to call. "
                "Treat the condition as unknown.",
            )

        if fruit["status"] != "identified":
            state = "unidentified"
            headline = "Fruit not identified"
        elif condition["status"] != "confident":
            state = "identified_condition_uncertain"
            headline = f"{fruit['fruit']} — condition unclear"
        else:
            state = "identified"
            headline = (
                f"Rotten {fruit['fruit'].lower()}"
                if condition["condition"] == "rotten"
                else f"Fresh {fruit['fruit'].lower()}"
            )

        return {
            "ok": True,
            # Closest model output. Only meaningful together with ``verdict.state``.
            "fruit": pred["predicted_fruit"],
            "condition": condition["condition"],
            "fresh_probability": pred["fresh_probability"],
            "rotten_probability": pred["rotten_probability"],
            "identification_confidence": pred["fruit_confidence"],
            "condition_confidence": condition["confidence"],
            "predicted_class": pred["top_class"],
            "class_probabilities": pred["class_probabilities"],
            "fruit_probabilities": pred["fruit_probabilities"],
            "verdict": {
                "state": state,
                "headline": headline,
                "fruit": fruit["fruit"] if fruit["status"] == "identified" else None,
                # A condition is only published when a fruit has been named: a fresh/rotten
                # split on an unrecognised image says nothing about any particular fruit.
                "condition": condition["condition"]
                if (fruit["status"] == "identified" and condition["status"] == "confident")
                else None,
                "fruit_confidence": fruit["confidence"],
                "condition_confidence": condition["confidence"],
            },
            "fruit_identification": fruit,
            "condition_assessment": condition,
            "edibility": edibility,
            "shelf_life": {
                "available": False,
                "days_estimate": None,
                "message": SHELF_LIFE_MESSAGE,
                "reason": "The training dataset contains only fresh/rotten labels with no storage "
                          "age or capture timeline, so days-until-spoilage cannot be learned.",
            },
            # Kept for callers written against v1: this is the fruit-identity gate.
            "uncertainty": {
                "rejected": fruit["status"] != "identified",
                "reasons": fruit["reasons"],
                "message": fruit["message"],
                "source": source_name or "upload",
            },
            "warnings": warnings,
            "image_info": quality,
            "model": {
                "name": "MobileNetV2 (ImageNet) transfer learning, 16-class softmax",
                "input_size": self.input_size,
                "temperature": pred["temperature"],
                "calibrated": pred["temperature"] != 1.0,
                "gate": {
                    "fruit_min_confidence": float((self.config.get("fruit") or {}).get("min_confidence", 0.0)),
                    "condition_min_confidence": float(
                        (self.config.get("condition") or {}).get("min_confidence", 0.0)
                    ),
                    "calibrated_on": self.config.get("calibrated_on"),
                },
            },
            "inference_ms": round(pred["inference_ms"], 2),
            "notes": {
                "probabilities": "Freshness/rotten percentages are summed model probabilities, "
                                 "not the physical proportion of the fruit that is fresh or rotten.",
                "scope": "Validated for whole-fruit photographs of the eight supported fruits only.",
            },
        }
