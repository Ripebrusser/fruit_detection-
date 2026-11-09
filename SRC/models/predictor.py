"""Inference pipeline: fruit identity, condition scores, uncertainty and warnings.

The predictor is deliberately explicit about what it does *not* know:

* Condition percentages are summed model probabilities, not physical measurements.
* Shelf life is always reported as unavailable: the training data has no time labels.
* A calibrated confidence gate decides when an image is too uncertain to name a fruit.
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
OOD_CONFIG_PATH = MODELS_DIR / "ood_config.json"

SAFETY_DISCLAIMER = (
    "This result is an AI-based visual assessment only. It cannot guarantee food safety. "
    "Inspect the fruit carefully and do not consume it if you suspect spoilage."
)
SHELF_LIFE_MESSAGE = "Remaining shelf life: Not available — insufficient time-based training data."
UNKNOWN_MESSAGE = "Unable to identify this fruit confidently. Please upload a clearer image of one fruit."

DEFAULT_OOD_CONFIG = {
    "min_top1_probability": 0.0,   # replaced by the calibrated value when available
    "min_fruit_margin": 0.0,
    "max_entropy_ratio": 1.0,
    "min_blur_variance": 0.0,
    "min_brightness": 0.0,
    "max_brightness": 255.0,
    "temperature": 1.0,
    "calibrated_on": "defaults (no ood_config.json found)",
}


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    total = float(probabilities.sum())
    if total <= 0:
        return np.full_like(probabilities, 1.0 / len(probabilities))
    return probabilities / total


class PredictorUnavailable(RuntimeError):
    """Raised when the trained model has not been produced yet."""


class FruitPredictor:
    """Wraps the trained Keras model with project-specific aggregation logic."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        class_names_path: Path | str = CLASS_NAMES_PATH,
        ood_config_path: Path | str = OOD_CONFIG_PATH,
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

        if Path(ood_config_path).exists():
            self.config = {**DEFAULT_OOD_CONFIG, **json.loads(Path(ood_config_path).read_text(encoding="utf-8"))}
        else:
            self.config = dict(DEFAULT_OOD_CONFIG)
        if temperature is not None:
            self.config["temperature"] = float(temperature)

        import tensorflow as tf

        self._tf = tf
        self.model = tf.keras.models.load_model(self.model_path)

    # ------------------------------------------------------------------ helpers
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
    def _uncertainty(self, raw: np.ndarray | None, pred: dict, quality: dict, input_kind: str) -> dict:
        """Apply the calibrated rejection gate and collect human-readable reasons."""
        reasons: list[str] = []
        cfg = self.config
        if pred["top_class_probability"] < cfg["min_top1_probability"]:
            reasons.append(
                f"the top class probability ({pred['top_class_probability']:.0%}) is below the "
                f"calibrated acceptance level ({cfg['min_top1_probability']:.0%})"
            )
        if pred["fruit_margin"] < cfg["min_fruit_margin"]:
            reasons.append(
                f"the two most likely fruits are too close (margin {pred['fruit_margin']:.0%})"
            )
        if pred["entropy_ratio"] > cfg["max_entropy_ratio"]:
            reasons.append("the class distribution is nearly uniform")
        if quality["blur_variance"] < cfg["min_blur_variance"]:
            reasons.append("the image looks blurry")
        if quality["brightness"] < cfg["min_brightness"]:
            reasons.append("the image is very dark")
        if quality["brightness"] > cfg["max_brightness"]:
            reasons.append("the image is over-exposed")
        return {
            "rejected": bool(reasons),
            "reasons": reasons,
            "message": UNKNOWN_MESSAGE if reasons else None,
            "source": input_kind,
        }

    def _edibility(self, pred: dict) -> dict:
        fresh, rotten = pred["fresh_probability"], pred["rotten_probability"]
        if rotten >= 0.80:
            status, label = "shows_spoilage", "Shows visible signs of spoilage"
        elif fresh >= 0.80:
            status, label = "appears_fresh", "Appears visually fresh"
        else:
            status, label = "uncertain", "Visual assessment is uncertain"
        return {
            "status": status,
            "label": label,
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
        uncertainty = self._uncertainty(None, pred, quality, source_name or "upload")
        edibility = self._edibility(pred)

        warnings: list[str] = []
        if uncertainty["rejected"]:
            warnings.append(
                "This image falls outside what the model can identify reliably. "
                "The prediction below is the closest supported fruit but should not be trusted."
            )
        if quality["width"] < 224 or quality["height"] < 224:
            warnings.append("The uploaded image is small; details may be lost when resized to 224×224.")
        if not 0.02 < quality["mask_fraction"] < 0.995:
            warnings.append(
                "The fruit is hard to separate from the background; results may be less reliable."
            )
        if pred["class_margin"] < 0.10 and not uncertainty["rejected"]:
            warnings.append("The model is not decisively confident about the exact class.")

        return {
            "ok": True,
            "fruit": pred["predicted_fruit"],
            "condition": "fresh" if pred["fresh_probability"] >= pred["rotten_probability"] else "rotten",
            "fresh_probability": pred["fresh_probability"],
            "rotten_probability": pred["rotten_probability"],
            "identification_confidence": pred["fruit_confidence"],
            "predicted_class": pred["top_class"],
            "class_probabilities": pred["class_probabilities"],
            "fruit_probabilities": pred["fruit_probabilities"],
            "edibility": edibility,
            "shelf_life": {
                "available": False,
                "days_estimate": None,
                "message": SHELF_LIFE_MESSAGE,
                "reason": "The training dataset contains only fresh/rotten labels with no storage "
                          "age or capture timeline, so days-until-spoilage cannot be learned.",
            },
            "uncertainty": uncertainty,
            "warnings": warnings,
            "image_info": quality,
            "model": {
                "name": "MobileNetV2 (ImageNet) transfer learning, 16-class softmax",
                "input_size": self.input_size,
                "temperature": pred["temperature"],
                "calibrated": pred["temperature"] != 1.0,
            },
            "inference_ms": round(pred["inference_ms"], 2),
            "notes": {
                "probabilities": "Freshness/rotten percentages are summed model probabilities, "
                                 "not the physical proportion of the fruit that is fresh or rotten.",
                "scope": "Validated for whole-fruit photographs of the eight supported fruits only.",
            },
        }
