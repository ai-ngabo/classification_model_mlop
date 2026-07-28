"""
Corn Leaf Disease Classifier
----------------------------
Defines model architecture, training (two-stage fine-tuning),
evaluation, saving/loading, and lightweight retraining.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import IMG_SIZE, NUM_CLASSES, CLASS_NAMES

# Paths 

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "corn_disease_model.keras")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Model building
def build_model(img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES,
                backbone: str = "mobilenetv2", dropout: float = 0.3,
                trainable_base: bool = False):
    """Build classifier with pretrained backbone + custom head."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Augmentation (training only)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)

    # Rescale to [-1,1]
    x = layers.Rescaling(2.0, offset=-1.0)(x)

    # Backbone
    if backbone.lower() == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                    input_shape=(img_size, img_size, 3))
    else:
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                 input_shape=(img_size, img_size, 3))
    base.trainable = trainable_base
    x = base(x, training=False)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="corn_disease_classifier")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile with accuracy + top-2 accuracy metrics."""
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    return model


def set_backbone_trainable(model, trainable: bool, unfreeze_from: int = -40):
    """Unfreeze top backbone layers for fine-tuning."""
    import tensorflow as tf
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if not base:
        return model

    base.trainable = trainable
    if trainable:
        for layer in base.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model

# Class weights

def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[int, float]:
    """Inverse-frequency class weights, normalized to mean 1.0."""
    labels = np.asarray(labels).ravel()
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights /= weights.mean()
    return {i: float(w) for i, w in enumerate(weights)}


def labels_from_dataset(ds) -> np.ndarray:
    """Extract integer labels from a tf.data.Dataset."""
    out = [np.asarray(y) for _, y in ds]
    return np.concatenate(out).ravel() if out else np.empty((0,), dtype=np.int64)


# Callbacks

def default_callbacks(checkpoint_path: Optional[str] = None, patience: int = 5, monitor: str = "val_loss"):
    import tensorflow as tf
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]
    if checkpoint_path:
        cbs.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, save_best_only=True, verbose=1))
    return cbs


# Training

def train_model(train_ds, val_ds, backbone: str = "mobilenetv2",
                stage1_epochs: int = 8, stage2_epochs: int = 12,
                stage1_lr: float = 1e-3, stage2_lr: float = 1e-5,
                class_weight: Optional[Dict[int, float]] = None,
                checkpoint_path: Optional[str] = None, verbose: int = 1):
    """Two-stage training: head first, then fine-tune backbone."""
    model = build_model(backbone=backbone, trainable_base=False)
    model = compile_model(model, learning_rate=stage1_lr)

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=stage1_epochs,
                   class_weight=class_weight, callbacks=default_callbacks(checkpoint_path, patience=4),
                   verbose=verbose)

    model = set_backbone_trainable(model, True, unfreeze_from=-40)
    model = compile_model(model, learning_rate=stage2_lr)

    h2 = model.fit(train_ds, validation_data=val_ds, epochs=stage2_epochs,
                   class_weight=class_weight, callbacks=default_callbacks(checkpoint_path, patience=5),
                   verbose=verbose)

    history = {
        "stage1": {k: [float(v) for v in vals] for k, vals in h1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in h2.history.items()},
    }
    return model, history


# Evaluation

def evaluate_model(model, test_ds, class_names: List[str] = None) -> Dict:
    """Evaluate model: accuracy, per-class metrics, confusion matrix, ROC-AUC."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report

    class_names = class_names or CLASS_NAMES
    y_true, y_prob = [], []

    for xb, yb in test_ds:
        y_prob.append(model.predict(xb, verbose=0))
        y_true.append(np.asarray(yb))

    y_true_arr = np.concatenate(y_true).ravel()
    y_prob_arr = np.concatenate(y_prob)
    y_pred_arr = y_prob_arr.argmax(axis=1)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec, rec, f1, support = precision_recall_fscore_support(y_true_arr, y_pred_arr,
                                                             labels=list(range(len(class_names))), zero_division=0)
    macro = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))

    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "roc_auc_ovr_macro": auc,
        "per_class": {
            class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "n_test_samples": int(y_true_arr.shape[0]),
        "classification_report": classification_report(y_true_arr, y_pred_arr, target_names=class_names, zero_division=0),
    }


def print_evaluation(results: Dict) -> None:
    """Readable evaluation summary."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Test samples        : {results['n_test_samples']}")
    print(f"Accuracy            : {results['accuracy']:.4f}")
    print(f"Macro Precision     : {results['macro_precision']:.4f}")
    print(f"Macro Recall        : {results['macro_recall']:.4f}")
    print(f"Macro F1            : {results['macro_f1']:.4f}")
    print(f"Weighted F1         : {results['weighted_f1']:.4f}")
    print(f"ROC-AUC (OvR macro) : {results['roc_auc_ovr_macro']:.4f}")
    print("-" * 70)
    print(results["classification_report"])


# Persistence + versioning

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_model(model, path: str = PROD_MODEL_PATH,
               metrics: Optional[Dict] = None,
               version: Optional[str] = None,
               notes: str = "") -> Dict:
    """Save model and update metadata sidecar."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

    meta = load_metadata()
    if version is None:
        prev = meta.get("version", "v0")
        try:
            n = int(str(prev).lstrip("v")) + 1
        except Exception:
            n = 1
        version = f"v{n}"

    meta.update({
        "version": version,
        "model_path": path,
        "saved_at": _now_iso(),
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "notes": notes,
    })
    if metrics:
        meta["metrics"] = {k: v for k, v in metrics.items() if k != "classification_report"}
    save_metadata(meta)
    return meta


def load_model(path: str = PROD_MODEL_PATH):
    """Load a saved Keras model."""
    import tensorflow as tf
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    return tf.keras.models.load_model(path)


def load_metadata(path: str = METADATA_PATH) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(meta: Dict, path: str = METADATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# Light retraining

def light_retrain(X_new: np.ndarray, y_new: np.ndarray,
                  base_model_path: str = PROD_MODEL_PATH,
                  epochs: int = 5, batch_size: int = 16,
                  learning_rate: float = 1e-4,
                  validation_split: float = 0.2) -> Tuple[object, Dict]:
    """Warm-start from production model and retrain briefly on new data."""
    import tensorflow as tf

    model = load_model(base_model_path)
    model = compile_model(model, learning_rate=learning_rate)
    cw = compute_class_weights(y_new)

    t0 = time.time()
    hist = model.fit(
        X_new, y_new,
        epochs=epochs, batch_size=batch_size,
        validation_split=validation_split if len(y_new) >= 10 else 0.0,
        class_weight=cw,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3,
                                                    restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    elapsed = time.time() - t0

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["elapsed_seconds"] = elapsed
    history["n_samples"] = int(len(y_new))
    history["class_weights"] = cw
    return model, history


def should_promote(new_metrics: Dict, current_metrics: Optional[Dict], threshold: float = 0.0) -> Tuple[bool, str]:
    """Decide if retrained model should replace production (based on macro F1)."""
    if not current_metrics:
        return True, "No incumbent metrics; promoting by default."

    new_f1 = new_metrics.get("macro_f1", 0.0)
    old_f1 = current_metrics.get("macro_f1", 0.0)
    delta = new_f1 - old_f1

    if delta > threshold:
        return True, f"Promoted: macro F1 improved {old_f1:.4f} -> {new_f1:.4f} (+{delta:.4f})."
    return False, f"Rejected: macro F1 {new_f1:.4f} did not beat incumbent {old_f1:.4f} (delta {delta:+.4f})."


# model checkpoint

if __name__ == "__main__":
    print("Model module self-check")
    print("-" * 40)
    w = compute_class_weights(np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3]))
    print("Class weights on synthetic imbalance:", w)
    ok, msg = should_promote({"macro_f1": 0.97}, {"macro_f1": 0.95})
    print("Promotion check:", ok, "|", msg)
    ok, msg = should_promote({"macro_f1": 0.93}, {"macro_f1": 0.95})
    print("Promotion check:", ok, "|", msg)