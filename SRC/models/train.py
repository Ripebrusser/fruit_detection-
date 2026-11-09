"""Train the 16-class fruit condition classifier.

Two models are supported:

* ``mobilenet`` (default, the project model): MobileNetV2 ImageNet weights, a frozen
  backbone followed by a fine-tuned top block, trained on the cached 224x224 arrays.
* ``baseline``: a small from-scratch CNN, kept only as a comparison point.

Augmentation is applied **inside the training pipeline only**; validation and test
arrays are fed unaugmented. The MobileNetV2 ``preprocess_input`` scaling lives inside
the saved model so that serving cannot drift from training.

Usage:
    python -m SRC.models.train --model mobilenet
    python -m SRC.models.train --model both
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "Dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "Models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_DIR = MODELS_DIR / "final_model"

IMAGE_SIZE = 224
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = PROCESSED_DIR / f"{split}_images.npy"
    y_path = PROCESSED_DIR / f"{split}_labels.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"missing {x_path.name}; run `python -m SRC.data.prepare_dataset --stage cache` first"
        )
    return np.load(x_path), np.load(y_path)


def build_augmentation() -> keras.Sequential:
    """Geometric + photometric augmentation for the training split only."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.06, fill_mode="reflect", seed=SEED),
            layers.RandomZoom(0.12, fill_mode="reflect", seed=SEED),
            layers.RandomTranslation(0.06, 0.06, fill_mode="reflect", seed=SEED),
            layers.RandomBrightness(0.10, value_range=(0.0, 1.0), seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="train_augmentation",
    )


def make_dataset(
    x: np.ndarray, y: np.ndarray, *, training: bool, batch_size: int, augment: keras.Sequential | None = None
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 2048), seed=SEED, reshuffle_each_iteration=True)
    # Augment in the [0, 1] range (that is what the Keras augmentation layers assume),
    # then hand the model 0-255 pixels because preprocess_input is part of the model.
    ds = ds.map(lambda i, l: (tf.cast(i, tf.float32) / 255.0, l), num_parallel_calls=AUTOTUNE)
    if training and augment is not None:
        ds = ds.map(lambda i, l: (augment(i, training=True), l), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, l: (i * 255.0, l), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_mobilenet(n_classes: int, dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    # preprocess_input is baked in: serving only has to supply 0-255 RGB pixels.
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
    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_fruit_classifier")


def build_baseline_cnn(n_classes: int) -> keras.Model:
    """Small from-scratch CNN used only to compare against transfer learning."""
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def callbacks_for(tag: str, patience: int) -> list[keras.callbacks.Callback]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / f"{tag}.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(str(CHECKPOINT_DIR / f"{tag}_log.csv")),
    ]


def merge_histories(*histories: keras.callbacks.History) -> dict:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def train_mobilenet(args) -> dict:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    n_classes = int(max(y_train.max(), y_val.max())) + 1
    print(f"train {x_train.shape}  val {x_val.shape}  classes {n_classes}")

    train_ds = make_dataset(
        x_train, y_train, training=True, batch_size=args.batch_size, augment=build_augmentation()
    )
    val_ds = make_dataset(x_val, y_val, training=False, batch_size=args.batch_size)

    model = build_mobilenet(n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda line: print(line))

    started = time.time()
    print("\n=== Stage A: frozen backbone, training the classifier head ===")
    hist_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks_for("mobilenet_head", patience=3),
        verbose=2,
    )

    print("\n=== Stage B: fine-tuning the top of the backbone ===")
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[: -args.finetune_layers]:
        layer.trainable = False
    # BatchNorm statistics stay frozen: with ~2.2k images, updating them hurts.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"trainable parameters in stage B: {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=callbacks_for("mobilenet_finetune", patience=4),
        verbose=2,
    )

    history = merge_histories(hist_head, hist_finetune)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Stage B can end below stage A's best validation accuracy, so the shipped model is
    # chosen by measured validation accuracy rather than by which stage ran last.
    selection = finalize_best_model()
    model_path = PROJECT_ROOT / selection["model_path"]

    elapsed = time.time() - started
    result = {
        "model": "mobilenet_v2",
        "n_classes": n_classes,
        "epochs_run": len(history.get("loss", [])),
        "best_val_accuracy": selection["chosen_val_accuracy"],
        "checkpoint_selection": selection["candidates"],
        "chosen_stage": selection["chosen"],
        "train_seconds": round(elapsed, 1),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "history_path": str((FINAL_DIR / "history.json").relative_to(PROJECT_ROOT)),
        "batch_size": args.batch_size,
        "trainable_params_stage_b": int(trainable),
    }
    (FINAL_DIR / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def finalize_best_model() -> dict:
    """Copy the stage checkpoint with the best validation accuracy to final_model/."""
    x_val, y_val = load_split("val")
    candidates = {
        "stage_a_frozen_backbone": CHECKPOINT_DIR / "mobilenet_head.keras",
        "stage_b_finetuned": CHECKPOINT_DIR / "mobilenet_finetune.keras",
    }
    scores: dict[str, float] = {}
    for name, path in candidates.items():
        if not path.exists():
            continue
        candidate_model = keras.models.load_model(path)
        _, accuracy = candidate_model.evaluate(
            x_val.astype("float32"), y_val, batch_size=32, verbose=0
        )
        scores[name] = float(accuracy)
    if not scores:
        raise FileNotFoundError(f"no stage checkpoints found in {CHECKPOINT_DIR}")

    chosen = max(scores, key=lambda key: scores[key])
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    target = FINAL_DIR / "fruit_model.keras"
    shutil.copyfile(candidates[chosen], target)
    summary_path = FINAL_DIR / "checkpoint_selection.json"
    summary_path.write_text(
        json.dumps(
            {"candidates": scores, "chosen": chosen, "chosen_val_accuracy": scores[chosen]}, indent=2
        ),
        encoding="utf-8",
    )
    return {
        "candidates": scores,
        "chosen": chosen,
        "chosen_val_accuracy": scores[chosen],
        "model_path": str(target.relative_to(PROJECT_ROOT)),
    }


def train_baseline(args) -> dict:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    n_classes = int(max(y_train.max(), y_val.max())) + 1
    train_ds = make_dataset(
        x_train, y_train, training=True, batch_size=args.batch_size, augment=build_augmentation()
    )
    val_ds = make_dataset(x_val, y_val, training=False, batch_size=args.batch_size)

    model = build_baseline_cnn(n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.baseline_epochs,
        callbacks=callbacks_for("baseline_cnn", patience=4),
        verbose=2,
    )
    out_dir = MODELS_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "baseline_model.keras")
    (out_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    result = {
        "model": "baseline_cnn",
        "epochs_run": len(history.history.get("loss", [])),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "train_seconds": round(time.time() - started, 1),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("mobilenet", "baseline", "both", "finalize"), default="mobilenet"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--finetune-layers", type=int, default=40)
    parser.add_argument("--baseline-epochs", type=int, default=12)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    args = parser.parse_args(argv)

    set_seeds()
    print(f"TensorFlow {tf.__version__}; GPUs visible: {tf.config.list_physical_devices('GPU')}")

    if args.model == "finalize":
        print(json.dumps(finalize_best_model(), indent=2))
        return 0
    if args.model in ("mobilenet", "both"):
        train_mobilenet(args)
    if args.model in ("baseline", "both"):
        train_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
