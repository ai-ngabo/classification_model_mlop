"""
Corn Leaf Disease Classifier API
--------------------------------
FastAPI service exposing endpoints for prediction, health, metrics,
and retraining. Model is loaded once per container for low latency.

Endpoints:
- GET  /                : Service banner + endpoint index
- GET  /health          : Model status, uptime, version
- GET  /metrics         : Current model metrics + counters
- POST /predict         : Single image -> class + confidence
- POST /upload-retrain  : Upload ZIP of class-foldered images
- POST /retrain         : Trigger background retraining job
- GET  /retrain/status  : Status of most recent retraining job
"""

from __future__ import annotations
import os, io, time, json, shutil, zipfile, tempfile, threading
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Project imports
from src.preprocessing import CLASS_NAMES, CLASS_DISPLAY, NUM_CLASSES, IMG_SIZE, load_images_from_folder, count_uploaded_images
from src.prediction import predict_from_bytes, model_status, reload_model, get_model
from src import model as model_mod

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETRAIN_DIR = os.environ.get("RETRAIN_DIR", os.path.join(PROJECT_ROOT, "data", "retrain"))
TEST_DIR = os.environ.get("TEST_DIR", os.path.join(PROJECT_ROOT, "data", "test"))
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "50"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
os.makedirs(RETRAIN_DIR, exist_ok=True)

# FastAPI app setup 
app = FastAPI(
    title="Corn Leaf Disease Classifier API",
    description="Serves predictions and retraining for a MobileNetV2 corn leaf disease model (4 classes).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # UI is separate; relax CORS for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()

# Retraining job state
_retrain_lock = threading.Lock()
_retrain_state: Dict = {
    "status": "idle",      # idle | running | completed | failed | rejected
    "started_at": None,
    "finished_at": None,
    "message": "",
    "result": None,
}

def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def _uptime_seconds() -> float:
    """Return API uptime in seconds."""
    return round(time.time() - _START_TIME, 1)

# Startup: warm model 
@app.on_event("startup")
def _warm_model() -> None:
    try:
        get_model()
        print("[startup] Model loaded and ready.")
    except Exception as e:
        print(f"[startup] Model not loaded: {e}")

# Basic info endpoints
@app.get("/")
def root() -> Dict:
    return {
        "service": "Corn Leaf Disease Classifier API",
        "version": app.version,
        "classes": CLASS_NAMES,
        "endpoints": {
            "GET /health": "model status + uptime",
            "GET /metrics": "model metrics + counters",
            "POST /predict": "single-image prediction",
            "POST /upload-retrain": "upload ZIP of images",
            "POST /retrain": "trigger retraining",
            "GET /retrain/status": "latest retraining status",
        },
    }

@app.get("/health")
def health() -> Dict:
    status = model_status()
    status.update({
        "status": "healthy" if status.get("model_available") else "no_model",
        "uptime_seconds": _uptime_seconds(),
        "server_time": _now_iso(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "pending_retrain_images": _pending_retrain_count(),
    })
    return status

@app.get("/metrics")
def metrics() -> Dict:
    meta = model_mod.load_metadata()
    status = model_status()
    return {
        "model_version": meta.get("version", "unknown"),
        "model_saved_at": meta.get("saved_at"),
        "metrics": meta.get("metrics", {}),
        "prediction_count": status.get("prediction_count", 0),
        "last_prediction_at": status.get("last_prediction_at"),
        "uptime_seconds": _uptime_seconds(),
        "class_names": CLASS_NAMES,
        "class_display": CLASS_DISPLAY,
    }

# Prediction endpoint 
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{file.content_type}'. Allowed: {sorted(ALLOWED_IMAGE_TYPES)}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        result = predict_from_bytes(data)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="No model loaded. Train and save a model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result["filename"] = file.filename
    return JSONResponse(result)

# Retraining data upload 
def _pending_retrain_count() -> int:
    try:
        return count_uploaded_images(RETRAIN_DIR)
    except Exception:
        return 0

def _extract_zip_to_retrain(data: bytes) -> Dict[str, int]:
    """Extract uploaded ZIP into RETRAIN_DIR, keeping only valid class folders."""
    from src.preprocessing import _canonical_class
    added = {c: 0 for c in CLASS_NAMES}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            canonical = None
            for p in parts[:-1]:
                canonical = _canonical_class(p)
                if canonical:
                    break
            if canonical is None:
                continue
            ext = os.path.splitext(parts[-1])[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            dest_dir = os.path.join(RETRAIN_DIR, canonical)
            os.makedirs(dest_dir, exist_ok=True)
            idx = len(os.listdir(dest_dir))
            dest = os.path.join(dest_dir, f"{canonical}_up_{idx:05d}{ext}")
            with zf.open(info) as src_f, open(dest, "wb") as out_f:
                shutil.copyfileobj(src_f, out_f)
            added[canonical] += 1
    return added

@app.post("/upload-retrain")
async def upload_retrain(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict:
    """Upload ZIP of images for retraining. Auto-triggers if threshold exceeded."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="Please upload a .zip file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ZIP too large.")

    try:
        added = _extract_zip_to_retrain(data)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Corrupt or invalid ZIP.")

    total_added = sum(added.values())
    pending = _pending_retrain_count()
    auto_triggered = False
    if pending >= RETRAIN_THRESHOLD and _retrain_state["status"] != "running":
        background_tasks.add_task(_run_retraining_job, "auto")
        auto_triggered = True

    return {
        "added_per_class": added,
        "total_added": total_added,
        "pending_retrain_images": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "auto_retrain_triggered": auto_triggered,
        "message": "Automatic retraining triggered." if auto_triggered else f"{pending}/{RETRAIN_THRESHOLD} images toward auto-retrain.",
    }

# Retraining job
def _run_retraining_job(trigger: str = "manual") -> None:
    """
    Background retraining:
      1. load uploaded images from RETRAIN_DIR
      2. light_retrain() warm-started from production model
      3. evaluate on held-out test set
      4. promote iff macro F1 improves; hot-swap the live model
    """
    with _retrain_lock:
        if _retrain_state["status"] == "running":
            return
        _retrain_state.update({
            "status": "running", "started_at": _now_iso(),
            "finished_at": None, "message": f"Retraining ({trigger}) started.",
            "result": None,
        })
 
    try:
        X, y, files = load_images_from_folder(RETRAIN_DIR)
        if len(y) < 8:
            with _retrain_lock:
                _retrain_state.update({
                    "status": "rejected", "finished_at": _now_iso(),
                    "message": f"Not enough images to retrain ({len(y)} < 8).",
                })
            return
 
        # Train (light, warm-started)
        new_model, hist = model_mod.light_retrain(X, y)
 
        # Evaluate incumbent vs new on the test set, if available
        current_meta = model_mod.load_metadata()
        current_metrics = current_meta.get("metrics", {})
        new_metrics = {}
        promoted = False
        promote_msg = "No test set available; promoted new model by default."
 
        if os.path.isdir(TEST_DIR) and any(os.scandir(TEST_DIR)):
            from src.preprocessing import build_datasets_from_directory
            test_ds = build_datasets_from_directory(
                TEST_DIR, batch_size=32, shuffle=False, augment=False)
            new_metrics = model_mod.evaluate_model(new_model, test_ds)
            promoted, promote_msg = model_mod.should_promote(
                new_metrics, current_metrics)
        else:
            promoted = True
 
        if promoted:
            model_mod.save_model(
                new_model,
                metrics=new_metrics or None,
                notes=f"Retrained ({trigger}) on {len(y)} uploaded images.",
            )
            reload_model()   # hot-swap the live singleton
            # clear the retrain bucket after a successful promotion
            _clear_retrain_dir()
 
        with _retrain_lock:
            _retrain_state.update({
                "status": "completed",
                "finished_at": _now_iso(),
                "message": promote_msg,
                "result": {
                    "trigger": trigger,
                    "n_images": int(len(y)),
                    "promoted": promoted,
                    "elapsed_seconds": hist.get("elapsed_seconds"),
                    "new_macro_f1": (new_metrics or {}).get("macro_f1"),
                    "previous_macro_f1": current_metrics.get("macro_f1"),
                },
            })
    except Exception as e:
        with _retrain_lock:
            _retrain_state.update({
                "status": "failed", "finished_at": _now_iso(),
                "message": f"Retraining failed: {e}",
            })
 
 
def _clear_retrain_dir() -> None:
    for c in CLASS_NAMES:
        d = os.path.join(RETRAIN_DIR, c)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
 
 
@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks) -> Dict:
    """Manually trigger retraining on whatever is currently in the retrain bucket."""
    pending = _pending_retrain_count()
    if pending < 8:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 8 uploaded images to retrain (have {pending}). "
                   f"Upload more via /upload-retrain.",
        )
    if _retrain_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A retraining job is already running.")
 
    background_tasks.add_task(_run_retraining_job, "manual")
    return {
        "message": "Retraining scheduled.",
        "pending_retrain_images": pending,
        "check_status_at": "/retrain/status",
    }
 
 
@app.get("/retrain/status")
def retrain_status() -> Dict:
    with _retrain_lock:
        state = dict(_retrain_state)
    state["pending_retrain_images"] = _pending_retrain_count()
    return state
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")), reload=False)