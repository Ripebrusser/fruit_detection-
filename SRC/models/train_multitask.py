"""Multi-task comparison: 8-class fruit identity + 2-class condition.

The shipped model is a single 16-class softmax, and the website derives fruit identity by
summing the two classes of each fruit. That is coherent by construction — the same weights
produce both numbers — so the audit question is empirical rather than architectural: does an
explicit two-head model measure better on the identical split?

This script trains one, with the same backbone, the same augmentation, the same frozen/fine
tune schedule (8 + 8 epochs) and the same train/validation/test split as the 16-class model,
then writes a side-by-side comparison. Nothing here replaces the shipped model: it exists to
answer the question with measurements instead of opinion.

Usage:
    python -m SRC.models.train_multitask
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from SRC.models.train import (
    AUTOTUNE,
    IMAGE_SIZE,
    MODELS_DIR,
    SEED,
    build_augmentation,
    load_split,
    set_seeds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTITASK_DIR = MODELS_DIR / "multitask"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
METRICS_PATH = PROJECT_ROOT / "Evaluation" / "metrics" / "metrics.json"
COMPARISON_PATH = PROJECT_ROOT / "Evaluation" / "metrics" / "architecture_comparison.json"
COMPARISON_MD = PROJECT_ROOT / "Evaluation" / "metrics" / "architecture_comparison.md"


def make_multitask_dataset(x, y, *, training: bool, batch_size: int, augment=None):
    """Yield ((image,), {"fruit": ..., "condition": ...}) with training-identical preprocessing."""
    import tensorflow as tf

    fruit = y // 2
    condition = y % 2
    ds = tf.data.Dataset.from_tensor_slices((x, {"fruit": fruit, "condition": condition}))
    if training:
        ds = ds.shuffle(min(len(x), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda i, l: (tf.cast(i, tf.float32) / 255.0,
                      {"fruit": l["fruit"], "condition": l["condition"]}),
        num_parallel_calls=AUTOTUNE,
    )
    if training and augment is not None:
        ds = ds.map(lambda i, l: (augment(i, training=True), l), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, l: (i * 255.0, l), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_multitask(dropout: float = 0.3):
    """Same backbone and head width as the 16-class model, split into two outputs."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    fruit = layers.Dense(8, activation="softmax", name="fruit_probs")(x)
    condition = layers.Dense(2, activation="softmax", name="condition_probs")(x)
    return keras.Model(inputs, {"fruit": fruit, "condition": condition}, name="mobilenetv2_multitask")


def main() -> int:
    import tensorflow as tf
    from tensorflow import keras

    set_seeds(SEED)
    MULTITASK_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    train_ds = make_multitask_dataset(
        x_train, y_train, training=True, batch_size=32, augment=build_augmentation()
    )
    val_ds = make_multitask_dataset(x_val, y_val, training=False, batch_size=32)

    model = build_multitask()
    losses = {"fruit": "sparse_categorical_crossentropy", "condition": "sparse_categorical_crossentropy"}
    metrics = {"fruit": ["accuracy"], "condition": ["accuracy"]}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=losses, metrics=metrics
    )

    started = time.time()
    checkpoint = CHECKPOINT_DIR / "multitask.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(str(CHECKPOINT_DIR / "multitask_log.csv")),
    ]

    print("\n=== Stage A: frozen backbone ===")
    hist_a = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks, verbose=2)

    print("\n=== Stage B: fine-tuning the top 40 layers ===")
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=losses, metrics=metrics
    )
    hist_b = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks, verbose=2)

    history: dict[str, list] = {}
    for hist in (hist_a, hist_b):
        for key, values in hist.history.items():
            history.setdefault(key, []).extend(float(v) for v in values)
    (MULTITASK_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best = tf.keras.models.load_model(checkpoint)
    probs = best.predict(x_test, batch_size=32, verbose=0)
    fruit_pred = probs["fruit"].argmax(axis=1)
    condition_pred = probs["condition"].argmax(axis=1)
    fruit_true = y_test // 2
    condition_true = y_test % 2

    combined_pred = fruit_pred * 2 + condition_pred
    accuracy_16 = float((combined_pred == y_test).mean())
    fruit_accuracy = float((fruit_pred == fruit_true).mean())
    condition_accuracy = float((condition_pred == condition_true).mean())

    summary = {
        "model": "mobilenet_v2 multi-task (8 fruit + 2 condition heads)",
        "epochs_run": len(history.get("loss", [])),
        "train_seconds": round(time.time() - started, 1),
        "test_split_n": int(len(y_test)),
        "accuracy_16_equivalent": accuracy_16,
        "fruit_accuracy": fruit_accuracy,
        "condition_accuracy": condition_accuracy,
        "best_val_loss": float(min(history.get("val_loss", [0.0]))),
    }
    (MULTITASK_DIR / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best.save(MULTITASK_DIR / "multitask_model.keras")

    write_comparison(summary)
    print(json.dumps(summary, indent=2))
    return 0


def write_comparison(multitask: dict) -> None:
    single = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    rows = [
        ("16-class accuracy", single.get("accuracy"), multitask["accuracy_16_equivalent"]),
        ("fresh vs rotten accuracy", single.get("condition_level_accuracy"), multitask["condition_accuracy"]),
        ("fruit identity accuracy", single.get("fruit_level_accuracy"), multitask["fruit_accuracy"]),
        ("macro F1 (16-class)", single.get("f1_macro"), None),
    ]
    payload = {
        "single_head_16_class": {
            "accuracy": single.get("accuracy"),
            "condition_level_accuracy": single.get("condition_level_accuracy"),
            "fruit_level_accuracy": single.get("fruit_level_accuracy"),
            "f1_macro": single.get("f1_macro"),
            "parameters_note": "shared trunk, one 16-unit softmax",
        },
        "multi_task": multitask,
        "comparison": [
            {
                "metric": name,
                "single_head": a,
                "multi_task": b,
                "difference_multi_minus_single": (None if (a is None or b is None) else b - a),
            }
            for name, a, b in rows
        ],
        "protocol": "Identical train/validation/test splits, identical backbone, identical "
                    "augmentation, identical 8 + 8 epoch schedule. The multi-task model was "
                    "selected by validation loss; the single-head model by validation accuracy. "
                    "The test split was evaluated once per model.",
    }
    COMPARISON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Architecture comparison: 16-class softmax vs two heads",
        "",
        "Same data, same backbone, same schedule (8 frozen epochs + 8 fine-tune epochs), one",
        "evaluation pass each on the untouched test split.",
        "",
        "| Metric | 16-class single head | Multi-task (8 + 2) | Difference |",
        "| --- | --- | --- | --- |",
    ]
    for entry in payload["comparison"]:
        single_value = "—" if entry["single_head"] is None else f"{entry['single_head']:.4f}"
        multi_value = "—" if entry["multi_task"] is None else f"{entry['multi_task']:.4f}"
        diff = entry["difference_multi_minus_single"]
        diff_value = "—" if diff is None else f"{diff:+.4f}"
        lines.append(f"| {entry['metric']} | {single_value} | {multi_value} | {diff_value} |")
    lines += [
        "",
        f"Multi-task training time: {multitask['train_seconds']:.0f} s on CPU.",
        "",
        "The shipped model keeps the 16-class head unless the multi-task model is measurably",
        "better here; see `Admin/FINAL_MODEL_EVALUATION_REPORT.md` for the decision.",
        "",
    ]
    COMPARISON_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
