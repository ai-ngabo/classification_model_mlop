"""
All data loading, cleaning, transformation, and augmentation logic for the
Corn Leaf Disease classifier.

This module is intentionally importable by BOTH:
  * the training / retraining job (batch mode), and
  * the FastAPI service (single-image inference mode)
"""

from __future__ import annotations

import os
import io
from typing import Tuple, List, Dict

import numpy as np

# Constants
IMG_SIZE: int = 224          # native input size for MobileNetV2/EfficientNet
CHANNELS: int = 3
NUM_CLASSES: int = 4

# Global plant_village indices that correspond to corn, in a fixed order.
CORN_GLOBAL_INDICES: List[int] = [7, 8, 9, 10]

# Human-readable class names, index-aligned with CORN_GLOBAL_INDICES.
CLASS_NAMES: List[str] = [
    "Gray_Leaf_Spot",       # local 0  (global 7)
    "Common_Rust",          # local 1  (global 8)
    "Healthy",              # local 2  (global 9)
    "Northern_Leaf_Blight",  # local 3  (global 10)
]

# Friendly labels for display in the UI.
CLASS_DISPLAY: Dict[int, str] = {
    0: "Gray Leaf Spot (Cercospora)",
    1: "Common Rust",
    2: "Healthy",
    3: "Northern Leaf Blight",
}

# Map from the dataset's global index -> our local 0..3 index.
_GLOBAL_TO_LOCAL: Dict[int, int] = {
    g: i for i, g in enumerate(CORN_GLOBAL_INDICES)
}


# Data acquisition + subset (batch / training path)

def load_corn_dataset(
    split: str = "train",
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
):
    """
    Load the PlantVillage dataset, keep ONLY the 4 corn classes,
    remap their labels to local indices 0..3, resize + normalize, and return
    a batched tf.data.Dataset ready for training or evaluation.

    Returns
    tf.data.Dataset yielding (image_batch, label_batch)
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    ds = tfds.load("plant_village", split=split, as_supervised=True)

    corn_set = tf.constant(CORN_GLOBAL_INDICES, dtype=tf.int64)

    def _is_corn(image, label):
        return tf.reduce_any(tf.equal(tf.cast(label, tf.int64), corn_set))

    # Build a lookup tensor for global->local remapping.
    # keys sorted for tf.lookup; values are the local indices.
    keys = tf.constant(CORN_GLOBAL_INDICES, dtype=tf.int64)
    vals = tf.constant(list(range(NUM_CLASSES)), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1,
    )

    def _remap(image, label):
        local = table.lookup(tf.cast(label, tf.int64))
        return image, local

    def _prep(image, label):
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.filter(_is_corn).map(_remap).map(_prep,
                                             num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _augment_pair(image, label):
    """On-the-fly augmentation applied during training only."""
    import tensorflow as tf
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_train_test(
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    train_frac: int = 80,
):
    """
    Convenience helper returning (train_ds, test_ds) with an 80/20 style split
    carved from plant_village's single 'train' split.
    """
    train_split = f"train[:{train_frac}%]"
    test_split = f"train[{train_frac}%:]"
    train_ds = load_corn_dataset(train_split, img_size, batch_size,
                                 shuffle=True, augment=True)
    test_ds = load_corn_dataset(test_split, img_size, batch_size,
                                shuffle=False, augment=False)
    return train_ds, test_ds


# Single-image preprocessing (inference / API path)

def preprocess_image_bytes(data: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Turn raw uploaded image bytes into a model-ready batch of shape
    (1, img_size, img_size, 3), float32 in [0, 1].

    Uses Pillow so the API container does not need full TensorFlow just to
    decode an image (keeps the inference image lighter).
    """
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_path(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Same as preprocess_image_bytes but reads from a file path on disk."""
    with open(path, "rb") as f:
        return preprocess_image_bytes(f.read(), img_size)


# Retraining-data ingestion (batch upload path)

def load_images_from_folder(
    folder: str,
    img_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a folder of uploaded retraining images arranged as:

        folder/
          Gray_Leaf_Spot/*.jpg
          Common_Rust/*.jpg
          Healthy/*.jpg
          Northern_Leaf_Blight/*.jpg

    Class subfolder names must match CLASS_NAMES. Returns (X, y, filepaths)
    where X is (N, img_size, img_size, 3) float32 and y is (N,) int labels.

    Files whose subfolder name is not a recognized class are skipped and
    reported back to the caller via the returned filepaths list only including
    successfully loaded files.
    """
    from PIL import Image

    name_to_local = {name: i for i, name in enumerate(CLASS_NAMES)}
    X: List[np.ndarray] = []
    y: List[int] = []
    used: List[str] = []

    if not os.path.isdir(folder):
        return np.empty((0, img_size, img_size, 3), np.float32), \
            np.empty((0,), np.int64), []

    for class_dir in sorted(os.listdir(folder)):
        full_dir = os.path.join(folder, class_dir)
        if not os.path.isdir(full_dir):
            continue
        if class_dir not in name_to_local:
            # Unknown class folder -> skip silently; caller can validate names.
            continue
        local = name_to_local[class_dir]
        for fname in sorted(os.listdir(full_dir)):
            fpath = os.path.join(full_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(
                    (img_size, img_size))
                X.append(np.asarray(img, dtype=np.float32) / 255.0)
                y.append(local)
                used.append(fpath)
            except Exception:
                # Corrupt / non-image file -> skip.
                continue

    if not X:
        return np.empty((0, img_size, img_size, 3), np.float32), \
            np.empty((0,), np.int64), []

    return np.stack(X), np.asarray(y, dtype=np.int64), used


def count_uploaded_images(folder: str) -> int:
    """Count valid class-folder images sitting in a retrain upload directory."""
    _, y, _ = load_images_from_folder(folder)
    return int(y.shape[0])


if __name__ == "__main__":
    # Small self-check that does not require the dataset download.
    print("Corn classes (local -> display):")
    for i in range(NUM_CLASSES):
        print(f"  {i}: {CLASS_DISPLAY[i]}  (global idx {CORN_GLOBAL_INDICES[i]})")
    print("IMG_SIZE:", IMG_SIZE, "NUM_CLASSES:", NUM_CLASSES)


# FOLDER-BASED ACQUISITION (Kaggle "corn-or-maize-leaf-disease-dataset")
# Map any source folder name (case-insensitive) -> our canonical class name.
FOLDER_ALIASES: Dict[str, str] = {
    "gray_leaf_spot": "Gray_Leaf_Spot",
    "grey_leaf_spot": "Gray_Leaf_Spot",
    "cercospora_leaf_spot gray_leaf_spot": "Gray_Leaf_Spot",
    "common_rust": "Common_Rust",
    "rust": "Common_Rust",
    "healthy": "Healthy",
    "blight": "Northern_Leaf_Blight",
    "northern_leaf_blight": "Northern_Leaf_Blight",
}


def _canonical_class(folder_name: str) -> Optional[str]:
    """Return our canonical class name for a source folder, or None if unknown."""
    key = folder_name.strip().lower()
    return FOLDER_ALIASES.get(key)


def find_class_root(root: str) -> Optional[str]:
    """
    Given a downloaded/extracted dataset directory (which may be nested),
    locate the folder that directly contains the class subfolders.

    Returns the path whose immediate subdirectories include at least two
    recognized corn classes, or None if not found.
    """
    root = os.path.abspath(root)
    for cur, dirs, _files in os.walk(root):
        recognized = [d for d in dirs if _canonical_class(d) is not None]
        if len(recognized) >= 2:
            return cur
    return None


def prepare_disk_split(
    source_root: str,
    train_dir: str,
    test_dir: str,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """
    Copy a folder-per-class dataset into the repo's data/train and data/test
    directories, using our CANONICAL class names (so 'Blight' becomes
    'Northern_Leaf_Blight'). This satisfies the required data/train + data/test
    layout and guarantees folder names match CLASS_NAMES exactly.

    Returns a summary dict: {'train': {class: n}, 'test': {class: n}}.
    """
    import random as _random
    import shutil as _shutil

    class_root = find_class_root(source_root)
    if class_root is None:
        raise FileNotFoundError(
            f"Could not find class subfolders under {source_root}. "
            f"Expected folders like Common_Rust/, Gray_Leaf_Spot/, Blight/, Healthy/."
        )

    # Clean destination class folders
    for d in (train_dir, test_dir):
        for cname in CLASS_NAMES:
            os.makedirs(os.path.join(d, cname), exist_ok=True)

    rng = _random.Random(seed)
    summary = {"train": {}, "test": {}}
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    for folder in sorted(os.listdir(class_root)):
        src_dir = os.path.join(class_root, folder)
        if not os.path.isdir(src_dir):
            continue
        canonical = _canonical_class(folder)
        if canonical is None:
            continue

        files = [f for f in sorted(os.listdir(src_dir))
                 if f.lower().endswith(valid_ext)]
        rng.shuffle(files)
        n_train = int(len(files) * train_frac)
        train_files = files[:n_train]
        test_files = files[n_train:]

        for subset, flist, ddir in (("train", train_files, train_dir),
                                    ("test", test_files, test_dir)):
            dst_dir = os.path.join(ddir, canonical)
            for i, fname in enumerate(flist):
                ext = os.path.splitext(fname)[1].lower()
                dst = os.path.join(dst_dir, f"{canonical}_{i:05d}{ext}")
                _shutil.copyfile(os.path.join(src_dir, fname), dst)
            summary[subset][canonical] = summary[subset].get(canonical, 0) + len(flist)

    return summary


def build_datasets_from_directory(
    directory: str,
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
):
    """
    Build a batched, normalized tf.data.Dataset from a class-subfolder
    directory (data/train or data/test). Label indices are pinned to the
    order of CLASS_NAMES so they match the rest of the pipeline exactly.
    """
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,          # pins label order to our scheme
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )

    def _norm(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    ds = ds.map(_norm, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)