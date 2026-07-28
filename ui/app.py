"""
Streamlit UI for the corn leaf disease classifier.

Implements four assignment requirements:
1. Model uptime ........ health panel (version, uptime, prediction count)
2. Data visualizations .. class distribution + per-class stats + samples
3. Prediction interface .. upload one image -> class + confidence
4. Retraining interface .. upload ZIP of class folders -> trigger retrain

The UI talks to the FastAPI service via HTTP (API_URL env var).
If the API is unreachable, the dashboard degrades gracefully.
"""

import os, io, time, json, zipfile
from collections import Counter
import requests, numpy as np, pandas as pd, streamlit as st
from PIL import Image   # Pillow for local image stats

# Config 
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DATA_TRAIN_DIR = os.environ.get(
    "DATA_TRAIN_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train"),
)

CLASS_DISPLAY = {
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Common_Rust": "Common Rust",
    "Healthy": "Healthy",
    "Northern_Leaf_Blight": "Northern Leaf Blight",
}
CLASS_ORDER = ["Gray_Leaf_Spot", "Common_Rust", "Healthy", "Northern_Leaf_Blight"]

st.set_page_config(page_title="Corn Leaf Disease MLOps", page_icon="🌽", layout="wide")

# API helpers (wrapped to avoid crashes if API is down)
def api_get(path: str, timeout: float = 5.0):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post_file(path: str, file_tuple, field: str = "file", timeout: float = 120.0):
    try:
        r = requests.post(f"{API_URL}{path}", files={field: file_tuple}, timeout=timeout)
        if r.status_code >= 400:
            try:
                return None, r.json().get("detail", r.text)
            except Exception:
                return None, r.text
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path: str, timeout: float = 30.0):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=timeout)
        if r.status_code >= 400:
            try:
                return None, r.json().get("detail", r.text)
            except Exception:
                return None, r.text
        return r.json(), None
    except Exception as e:
        return None, str(e)

# Sidebar: connection + uptime panel 
def render_sidebar():
    st.sidebar.title("🌽 Corn Disease MLOps")
    st.sidebar.caption("MobileNetV2 - 4 corn leaf classes")

    st.sidebar.markdown("### API connection")
    api_url_input = st.sidebar.text_input("API URL", value=API_URL)
    globals()["API_URL"] = api_url_input.rstrip("/")

    health, err = api_get("/health")
    if err:
        st.sidebar.error("API offline")
        st.sidebar.caption(f"Could not reach {API_URL}")
        st.sidebar.caption(f"{err[:80]}")
        return None

    status = health.get("status", "unknown")
    if status == "healthy":
        st.sidebar.success("API online - model ready")
    else:
        st.sidebar.warning(f"API online - {status}")

    st.sidebar.markdown("### Model uptime")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Version", health.get("model_version", "?"))
    up = health.get("uptime_seconds", 0)
    col2.metric("Uptime", f"{up/60:.1f} min" if up >= 60 else f"{up:.0f} s")

    st.sidebar.metric("Predictions served", health.get("prediction_count", 0))
    pending = health.get("pending_retrain_images", 0)
    thresh = health.get("retrain_threshold", 0)
    if thresh:
        st.sidebar.progress(min(pending / thresh, 1.0),
                            text=f"Auto-retrain: {pending}/{thresh} images")
    last = health.get("last_prediction_at")
    if last:
        st.sidebar.caption(f"Last prediction: {last[:19].replace('T',' ')} UTC")
    return health

# Tab 1: Prediction 
def render_prediction_tab(health):
    st.header("Single-image prediction")
    st.write("Upload a corn leaf image; the model returns the predicted disease class with confidence.")

    up = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "bmp"], key="predict_uploader")
    if up is None:
        st.info("Upload an image to get a prediction.")
        return

    col_img, col_res = st.columns([1, 1.3])
    with col_img:
        st.image(up, caption=up.name, use_container_width=True)

    with col_res:
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                data = up.getvalue()
                result, err = api_post_file("/predict", (up.name, io.BytesIO(data), up.type or "image/jpeg"))
            if err:
                st.error(f"Prediction failed: {err}")
                return

            pred = result["predicted_display_name"]
            conf = result["confidence"]
            healthy = result.get("is_healthy", False)

            st.success(f"### {pred}") if healthy else st.warning(f"### {pred}")
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.caption(f"Inference time: {result.get('inference_time_ms','?')} ms")

            # Probability distribution chart
            probs = result["all_probabilities"]
            dfp = pd.DataFrame([{"Class": p["display_name"], "Probability": p["probability"]} for p in probs]).set_index("Class")
            st.bar_chart(dfp, height=240)

# Tab 2: Data visualizations
st.cache_data(show_spinner=False)
def scan_local_dataset(train_dir: str):
    """Compute class counts + basic per-class image stats from local data/train."""
    if not os.path.isdir(train_dir):
        return None
    rows = []
    counts = {}
    for cname in CLASS_ORDER:
        cdir = os.path.join(train_dir, cname)
        if not os.path.isdir(cdir):
            counts[cname] = 0
            continue
        files = [f for f in os.listdir(cdir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        counts[cname] = len(files)
        # sample up to 40 images per class for brightness/color stats
        for f in files[:40]:
            try:
                arr = np.asarray(
                    Image.open(os.path.join(cdir, f)).convert("RGB"),
                    dtype=np.float32)
                rows.append({
                    "class": CLASS_DISPLAY[cname],
                    "brightness": float(arr.mean()),
                    "R": float(arr[:, :, 0].mean()),
                    "G": float(arr[:, :, 1].mean()),
                    "B": float(arr[:, :, 2].mean()),
                })
            except Exception:
                continue
    return {"counts": counts, "stats": pd.DataFrame(rows)}

def render_visualization_tab():
    st.header("Dataset visualizations")
 
    scan = scan_local_dataset(DATA_TRAIN_DIR)
    if scan is None or sum(scan["counts"].values()) == 0:
        st.warning(
            f"No local training data found at `{DATA_TRAIN_DIR}`. "
            "Run the data-acquisition cells in the notebook first, or set "
            "the DATA_TRAIN_DIR environment variable.")
        return
 
    counts = scan["counts"]
    total = sum(counts.values())
 
    # Feature 1: class distribution 
    st.subheader("1. Class distribution")
    dfc = pd.DataFrame(
        {"Class": [CLASS_DISPLAY[c] for c in CLASS_ORDER],
         "Images": [counts[c] for c in CLASS_ORDER]}
    ).set_index("Class")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.bar_chart(dfc, height=300)
    with c2:
        st.metric("Total images", total)
        largest = max(counts, key=counts.get)
        smallest = min(counts, key=counts.get)
        ratio = counts[largest] / max(counts[smallest], 1)
        st.metric("Imbalance ratio", f"{ratio:.2f}x")
        st.caption(f"Largest: {CLASS_DISPLAY[largest]} ({counts[largest]})")
        st.caption(f"Smallest: {CLASS_DISPLAY[smallest]} ({counts[smallest]})")
    st.info("**Interpretation:** the classes are imbalanced, so the model uses "
            "inverse-frequency class weights and is judged on per-class recall / "
            "macro F1 rather than raw accuracy, to protect the smallest class.")
 
    stats = scan["stats"]
    if not stats.empty:
        # Feature 2: brightness by class 
        st.subheader("2. Brightness distribution by class")
        bright = stats.groupby("class")["brightness"].mean().reindex(
            [CLASS_DISPLAY[c] for c in CLASS_ORDER])
        st.bar_chart(bright, height=260)
        st.info("**Interpretation:** brightness overlaps across classes, so the "
                "model cannot separate diseases by exposure alone - it must learn "
                "genuine lesion texture and colour.")
 
        # Feature 3: mean RGB by class 
        st.subheader("3. Mean colour channels by class")
        rgb = stats.groupby("class")[["R", "G", "B"]].mean().reindex(
            [CLASS_DISPLAY[c] for c in CLASS_ORDER])
        st.bar_chart(rgb, height=260)
        st.info("**Interpretation:** healthy leaves show the strongest green; "
                "diseased leaves lose green and gain red/brown as tissue dies. "
                "Colour carries real diagnostic signal, so hue augmentation is "
                "deliberately avoided.")
 
    # Sample images 
    st.subheader("Sample images per class")
    cols = st.columns(len(CLASS_ORDER))
    for col, cname in zip(cols, CLASS_ORDER):
        cdir = os.path.join(DATA_TRAIN_DIR, cname)
        if os.path.isdir(cdir):
            files = [f for f in os.listdir(cdir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                col.image(os.path.join(cdir, files[0]),
                          caption=CLASS_DISPLAY[cname],
                          use_container_width=True)

# --- Tab 3: Retraining (Requirement 4) ---
def render_retrain_tab(health):
    st.header("Upload data & retrain")
    st.write("Upload a **ZIP** of new labelled images to improve the model. "
             "Inside the ZIP, put images in folders named by class:")
    st.code("Common_Rust/img1.jpg\nGray_Leaf_Spot/img2.jpg\n"
            "Healthy/img3.jpg\nNorthern_Leaf_Blight/img4.jpg\n"
            "(a folder named 'Blight' is accepted as Northern Leaf Blight)")
 
    zip_up = st.file_uploader("Upload retraining ZIP", type=["zip"],
                              key="retrain_uploader")
    col_a, col_b = st.columns(2)
 
    with col_a:
        if zip_up is not None and st.button("Upload data", use_container_width=True):
            with st.spinner("Uploading and extracting..."):
                res, err = api_post_file(
                    "/upload-retrain",
                    (zip_up.name, io.BytesIO(zip_up.getvalue()),
                     "application/zip"))
            if err:
                st.error(f"Upload failed: {err}")
            else:
                st.success(f"Added {res['total_added']} images.")
                st.json(res["added_per_class"])
                if res.get("auto_retrain_triggered"):
                    st.info("Threshold reached - automatic retraining started.")
                else:
                    st.caption(res.get("message", ""))
 
    with col_b:
        if st.button("Trigger retraining now", type="primary",
                     use_container_width=True):
            res, err = api_post("/retrain")
            if err:
                st.error(f"Could not start retraining: {err}")
            else:
                st.success(res.get("message", "Retraining scheduled."))
                st.caption(f"Pending images: {res.get('pending_retrain_images')}")
 
    st.divider()
    st.subheader("Retraining status")
    if st.button("Refresh status"):
        pass  # triggers a rerun
    state, err = api_get("/retrain/status")
    if err:
        st.caption("Status unavailable (API offline).")
        return
    status = state.get("status", "idle")
    badge = {"idle": "\u26AA", "running": "\U0001F7E1",
             "completed": "\U0001F7E2", "failed": "\U0001F534",
             "rejected": "\U0001F7E0"}.get(status, "\u26AA")
    st.write(f"{badge} **{status.upper()}** - {state.get('message','')}")
    if state.get("result"):
        st.json(state["result"])
 

# Tab 4: Metrics 
def render_metrics_tab():
    st.header("Model metrics")
    metrics, err = api_get("/metrics")
    if err:
        st.warning("Metrics unavailable (API offline).")
        return
    m = metrics.get("metrics", {})
    if not m:
        st.info("No evaluation metrics recorded yet. Train and save a model.")
    else:
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
        cols[1].metric("Macro F1", f"{m.get('macro_f1', 0):.3f}")
        cols[2].metric("Macro Recall", f"{m.get('macro_recall', 0):.3f}")
        auc = m.get("roc_auc_ovr_macro")
        cols[3].metric("ROC-AUC", f"{auc:.3f}" if auc else "-")
    st.caption(f"Model version: {metrics.get('model_version','?')} | "
               f"Predictions served: {metrics.get('prediction_count',0)}")

# Main 
def main():
    health = render_sidebar()
    st.title("Corn Leaf Disease Classification")
    st.caption("End-to-end MLOps dashboard - prediction, monitoring, retraining")

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Visualizations", "Retrain", "Metrics"])
    with tab1: render_prediction_tab(health)
    with tab2: render_visualization_tab()
    with tab3: render_retrain_tab(health)
    with tab4: render_metrics_tab()

if __name__ == "__main__":
    main()
