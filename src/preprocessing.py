"""
preprocessing.py
================
Data loading, cleaning, transformation, and augmentation
for the Corn Leaf Disease classifier.

Usable by both:
  * training/retraining jobs (batch mode)
  * FastAPI service (single-image inference)

Dataset: PlantVillage (TFDS 'plant_village'), subset to 4 corn classes.
"""

import os
import io
from typing import Tuple, List, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE: int = 224          # native input size for MobileNetV2/EfficientNet
CHANNELS: int = 3
NUM_CLASSES: int = 4

# Global PlantVillage indices for corn classes
CORN_GLOBAL_INDICES: List[int] = [7, 8, 9, 10]

# Local class names (index-aligned)
CLASS_NAMES: List[str] = [
    "Gray_Leaf_Spot",
    "Common_Rust",
    "Healthy",
    "Northern_Leaf_Blight",
]

# Friendly labels for UI
CLASS_DISPLAY: Dict[int, str] = {
    0: "Gray Leaf Spot (Cercospora)",
    1: "Common Rust",
    2: "Healthy",
    3: "Northern Leaf Blight",
}

# Map global dataset index -> local index
_GLOBAL_TO_LOCAL: Dict[int, int] = {g: i for i, g in enumerate(CORN_GLOBAL_INDICES)}


# ---------------------------------------------------------------------------
# Dataset loading (training path)
# ---------------------------------------------------------------------------

def load_corn_dataset(split: str = "train", img_size: int = IMG_SIZE,
                      batch_size: int = 32, shuffle: bool = True,
                      augment: bool = False):
    """Load PlantVillage, keep corn classes, resize/normalize, return tf.data.Dataset."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    ds = tfds.load("plant_village", split=split, as_supervised=True)
    corn_set = tf.constant(CORN_GLOBAL_INDICES, dtype=tf.int64)

    def _is_corn(image, label):
        return tf.reduce_any(tf.equal(tf.cast(label, tf.int64), corn_set))

    keys = tf.constant(CORN_GLOBAL_INDICES, dtype=tf.int64)
    vals = tf.constant(list(range(NUM_CLASSES)), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)

    def _remap(image, label):
        return image, table.lookup(tf.cast(label, tf.int64))

    def _prep(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.filter(_is_corn).map(_remap).map(_prep, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _augment_pair(image, label):
    """Training-time augmentation."""
    import tensorflow as tf
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0), label


def make_train_test(img_size: int = IMG_SIZE, batch_size: int = 32, train_frac: int = 80):
    """Return (train_ds, test_ds) split from PlantVillage."""
    train_split = f"train[:{train_frac}%]"
    test_split = f"train[{train_frac}%:]"
    train_ds = load_corn_dataset(train_split, img_size, batch_size, shuffle=True, augment=True)
    test_ds = load_corn_dataset(test_split, img_size, batch_size, shuffle=False, augment=False)
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Single-image preprocessing (inference path)
# ---------------------------------------------------------------------------

def preprocess_image_bytes(data: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Convert raw image bytes -> batch (1, img_size, img_size, 3)."""
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_path(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Same as preprocess_image_bytes but from file path."""
    with open(path, "rb") as f:
        return preprocess_image_bytes(f.read(), img_size)


# ---------------------------------------------------------------------------
# Retraining-data ingestion
# ---------------------------------------------------------------------------

def load_images_from_folder(folder: str, img_size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load retraining images from class-named subfolders."""
    from PIL import Image
    name_to_local = {name: i for i, name in enumerate(CLASS_NAMES)}
    X, y, used = [], [], []

    if not os.path.isdir(folder):
        return np.empty((0, img_size, img_size, 3), np.float32), np.empty((0,), np.int64), []

    for class_dir in sorted(os.listdir(folder)):
        full_dir = os.path.join(folder, class_dir)
        if not os.path.isdir(full_dir) or class_dir not in name_to_local:
            continue
        local = name_to_local[class_dir]
        for fname in sorted(os.listdir(full_dir)):
            fpath = os.path.join(full_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize((img_size, img_size))
                X.append(np.asarray(img, dtype=np.float32) / 255.0)
                y.append(local)
                used.append(fpath)
            except Exception:
                continue

    if not X:
        return np.empty((0, img_size, img_size, 3), np.float32), np.empty((0,), np.int64), []

    return np.stack(X), np.asarray(y, dtype=np.int64), used


def count_uploaded_images(folder: str) -> int:
    """Count valid retraining images in a folder."""
    _, y, _ = load_images_from_folder(folder)
    return int(y.shape[0])


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Corn classes (local -> display):")
    for i in range(NUM_CLASSES):
        print(f"  {i}: {CLASS_DISPLAY[i]}  (global idx {CORN_GLOBAL_INDICES[i]})")
    print("IMG_SIZE:", IMG_SIZE, "NUM_CLASSES:", NUM_CLASSES)
