"""
Load test for Corn Leaf Disease API.
Sends POST /predict with corn leaf images and occasional GET /health checks.
Used to measure latency and throughput when scaling API replicas.
"""

import os
import io
import glob
import random
from locust import HttpUser, task, between, events

# Directory for sample images (default: data/test)
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "data/test")
_SAMPLE_IMAGES: list[bytes] = []


def _load_sample_images(max_images: int = 40) -> None:
    """Load up to max_images from SAMPLE_DIR into memory."""
    global _SAMPLE_IMAGES
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths: list[str] = []
    for cls_dir in glob.glob(os.path.join(SAMPLE_DIR, "*")):
        if os.path.isdir(cls_dir):
            for pat in patterns:
                paths.extend(glob.glob(os.path.join(cls_dir, pat)))
    random.shuffle(paths)
    for p in paths[:max_images]:
        try:
            with open(p, "rb") as f:
                _SAMPLE_IMAGES.append(f.read())
        except Exception:
            continue


def _generate_fallback_image() -> bytes:
    """Generate a random JPEG if no sample images exist."""
    try:
        import numpy as np
        from PIL import Image
        arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        # Minimal valid 1x1 JPEG as last resort / safety net
        return bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300"
            "080606070605080707070909080a0c140d0c0b0b0c1912130f"
            "141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c3032"
            "3534341f27393d38323c2e333432ffc0000b080001000101011"
            "100ffc4001f0000010501010101010100000000000000000102"
            "030405060708090a0bffda0008010100003f00d2cf20ffd9"
        )


@events.test_start.add_listener
def _on_start(environment, **kwargs):
    """Load images at test start, or use fallback if none found."""
    _load_sample_images()
    if _SAMPLE_IMAGES:
        print(f"[locust] Loaded {len(_SAMPLE_IMAGES)} images from {SAMPLE_DIR}")
    else:
        _SAMPLE_IMAGES.append(_generate_fallback_image())
        print(f"[locust] No images found; using fallback.")


class CornPredictUser(HttpUser):
    """Simulates a client sending requests to the API."""
    wait_time = between(0.1, 0.5)

    @task(9)
    def predict(self):
        """Send POST /predict with a random image."""
        img = random.choice(_SAMPLE_IMAGES)
        files = {"file": ("leaf.jpg", io.BytesIO(img), "image/jpeg")}
        with self.client.post("/predict", files=files,
                              catch_response=True, name="POST /predict") as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("503: model not loaded")
            else:
                resp.failure(f"{resp.status_code}: {resp.text[:100]}")

    @task(1)
    def health(self):
        """Send GET /health check."""
        with self.client.get("/health", catch_response=True,
                             name="GET /health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"{resp.status_code}")
