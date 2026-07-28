# Corn Leaf Disease Classification: End-to-End MLOps Pipeline

An end-to-end ML pipeline that classifies corn (maize) leaf diseases from images, deployed and scaled on Google Cloud Run with monitoring, load testing, and on-demand retraining.

The model distinguishes four classes of corn leaf condition: **Gray Leaf Spot**, **Common Rust**, **Healthy**, and **Northern Leaf Blight**; and is served through a REST API and an interactive dashboard, both running as independent containerized services in the cloud.

---

## Useful Links For Project

| Resource | Link |
|---|---|
| **Live API** | https://corn-api-185814802944.us-central1.run.app |
| **API interactive docs** | https://corn-api-185814802944.us-central1.run.app/docs |
| **Live UI (dashboard)** | https://corn-ui-185814802944.us-central1.run.app |
| **YouTube demo video** | _[]_ |
| **GitHub repository** | https://github.com/ai-ngabo/classification_model_mlop.git |

> The services scale to zero when idle, so the first request after a period of inactivity may take a few seconds to respond (a cold start while the model loads).

---

## Project Description

This project demonstrates the complete machine learning lifecycle for a non-tabular (image) classification problem:

1. **Offline model development:** data acquisition, exploratory analysis, training, and evaluation in a Jupyter notebook.
2. **Modular, production-ready code:** the notebook logic refactored into importable `src/` modules shared by the API and the retraining job.
3. **A serving API:** a FastAPI service exposing prediction, retraining, health, and metrics endpoints.
4. **A monitoring dashboard:** a Streamlit UI for predictions, dataset visualisations, uptime monitoring, and triggering retraining.
5. **Containerisation:** separate Docker images for the API and UI, orchestrated with Docker Compose.
6. **Load testing*:* a Locust flood test measuring latency and throughput across 1, 2, and 4 API container replicas behind an nginx load balancer.
7. **Cloud deployment:** both services deployed publicly on Google Cloud Platform Run.
8. **Retraining:** a user can upload new labelled images and retrain the model, with both a manual trigger and an automatic threshold-based trigger.

### Dataset

The model is trained on the **corn subset of the PlantVillage dataset** (available on Kaggle as `smaranjitghose/corn-or-maize-leaf-disease-dataset`), containing roughly 4,188 images across the four corn classes. The corn subset was chosen because its four classes are visually distinct enough to produce an interpretable confusion matrix while remaining light enough to retrain quickly on free-tier hardware.

---

## Model Performance

The production model is a **MobileNetV2** backbone (pretrained on ImageNet) with a custom classification head, trained using a **two-stage transfer-learning strategy**: a frozen-backbone warm-up followed by fine-tuning of the top layers. Training uses inverse-frequency class weights to protect recall on the smaller classes, and model promotion during retraining is gated on **macro F1** rather than raw accuracy.

Evaluated on a held-out test set of **840 images**:

| Metric | Value |
|---|---|
| Accuracy | 93.8% |
| Macro Precision | 0.923 |
| Macro Recall | 0.916 |
| Macro F1 | 0.919 |
| Weighted F1 | 0.937 |
| ROC-AUC (one-vs-rest, macro) | 0.990 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Gray Leaf Spot | 0.84 | 0.77 | 0.81 | 115 |
| Common Rust | 0.97 | 0.99 | 0.98 | 262 |
| Healthy | 1.00 | 1.00 | 1.00 | 233 |
| Northern Leaf Blight | 0.89 | 0.90 | 0.89 | 230 |

### Model progression

The two-stage transfer-learning approach was validated against a from-scratch baseline:

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Baseline CNN (from scratch) | 90.5% | 0.885 | 0.986 |
| MobileNetV2 (stage 1, head only) | 91.9% | 0.894 | 0.986 |
| MobileNetV2 (fine-tuned, final) | **93.8%** | **0.919** | **0.990** |

Fine-tuning the backbone added **+1.9 percentage points** of accuracy over head-only transfer learning, with the gain concentrated in the harder lesion classes. The dominant remaining confusion is between Gray Leaf Spot and Northern Leaf Blight; expected, since both present as elongated grey-brown lesions.

---

## Architecture

```
                    ┌──────────────────────┐
   User browser ───>│  Streamlit UI        │ (Cloud Run 
                    │  prediction, uptime, │service: corn-ui)
                    │  visualisations,     │
                    │  retrain controls    │
                    └──────────┬───────────┘
                               │ HTTP (API_URL)
                               v
                    ┌──────────────────────┐
                    │  FastAPI API         │ (Cloud Run 
                    │  /predict /retrain   │service: corn-api)
                    │  /health /metrics    │
                    └──────────┬───────────┘
                               │
                               v
                    ┌───────────────────── ┐
                    │  MobileNetV2 model   │ (bundled .keras 
                    └──────────────────────┘artifact)

Local load-testing setup (Locust experiment):
   Locust ──> nginx load balancer ──> N × API replicas  (docker-compose.scale.yml)
```

The UI and API are **independent services**. This separation is what makes the multi-container load-testing experiment meaningful and mirrors a realistic production deployment.

---

## Repository Structure

```
classification_model_mlop/
├── README.md
├── notebook/
│   └── corn_disease_classification.ipynb   # EDA, training, evaluation
├── src/
│   ├── preprocessing.py                     # data loading, transforms, retrain ingestion
│   ├── model.py                             # architecture, two-stage training, evaluation
│   └── prediction.py                        # inference wrapper (model singleton)
├── api/
│   ├── main.py                              # FastAPI service
│   └── requirements.txt
├── ui/
│   ├── app.py                               # Streamlit dashboard
│   └── requirements.txt
├── locust/
│   └── locustfile.py                        # load-test definition
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   ├── Dockerfile.nginx
│   └── nginx.conf
├── docker-compose.yml                       # local run: API + UI
├── docker-compose.scale.yml                 # load-balanced setup for Locust
├── data/
│   ├── train/                               # created by the notebook
│   ├── test/                                # created by the notebook
│   └── retrain/                             # uploaded retraining data
└── models/
    └── corn_disease_model.keras             # trained model + metadata
```

---

## Setup Instructions

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- (For the notebook) TensorFlow 2.21 / Keras 3

### 1. Clone and install

```bash
git clone <your-repo-url>
cd classification_model_mlop
```

### 2. Prepare data and train the model

Download the dataset from Kaggle (`smaranjitghose/corn-or-maize-leaf-disease-dataset`) and place the four class folders under `data/plant_village/`. Then open and run the notebook top to bottom:

```bash
jupyter notebook notebook/corn_disease_classification.ipynb
```

This performs data acquisition, splits the data into `data/train/` and `data/test/`, trains the model, evaluates it, and saves the artifact to `models/`.

> The model must exist in `models/` before building the Docker images, since the API image bundles it.

### 3. Run locally with Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API: http://localhost:8000 (docs at http://localhost:8000/docs)

### 4. Run locally without Docker (development)

```bash
# Terminal 1: for API
uvicorn api.main:app --port 8000

# Terminal 2: for UI
streamlit run ui/app.py
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model status, version, uptime, prediction count |
| `/metrics` | GET | Current model performance metrics |
| `/predict` | POST | Upload one image -> predicted class + confidence |
| `/upload-retrain` | POST | Upload a ZIP of class-foldered images for retraining |
| `/retrain` | POST | Trigger a background retraining job |
| `/retrain/status` | GET | Status of the latest retraining job |

Example prediction request:

```bash
curl -X POST https://corn-api-185814802944.us-central1.run.app/predict \
  -F "file=@path/to/leaf.jpg"
```

---

## Retraining

The pipeline supports retraining the deployed model on new data, satisfying the requirement for a retraining trigger "when the need arises":

- **Manual trigger:** a user uploads a ZIP of new labelled images through the UI (organised into class folders) and presses **Trigger retraining**. Inside the ZIP, a folder named `Blight` is automatically treated as `Northern_Leaf_Blight`.
- **Automatic trigger:** once the number of newly uploaded images crosses a configurable threshold (`RETRAIN_THRESHOLD`, default 50), a retraining job fires automatically.

Retraining runs as a background task (using FastAPI `BackgroundTasks`, no external broker required). It warm-starts from the current production model, trains briefly on the new data, evaluates on the held-out test set, and **only promotes the new model if it beats the incumbent on macro F1**. On promotion, the running model is hot-swapped without a restart. This protects production from being overwritten by a worse model.

---

## Load Testing Results

Load testing was performed with **Locust**, flooding the `/predict` endpoint with corn leaf images. The API was scaled to 1, 2, and 4 container replicas behind an nginx load balancer (`docker-compose.scale.yml`), with each run using 30 concurrent users for 60 seconds.

To reproduce:

```bash
# Start the load-balanced stack with N replicas
docker compose -f docker-compose.scale.yml up --build --scale api=1

# In another terminal, run the flood
python3 -m locust -f locust/locustfile.py --host http://localhost:8080 \
        --headless -u 30 -r 5 -t 60s --csv results_1container
```

Repeat with `--scale api=2` and `--scale api=4`.

### Results

| API Containers | Median (p50) | p95 | p99 | Throughput | Failures |
|---|---|---|---|---|---|
| 1 | 4,200 ms | 26,000 ms | 27,000 ms | 2.76 req/s | 0% |
| 2 | 2,300 ms | 22,000 ms | 23,000 ms | 4.44 req/s | 0% |
| 4 | 740 ms | 26,000 ms | 38,000 ms | 3.31 req/s | 0% |

### Analysis

Median latency improved **~5.7× from 1 to 4 containers** (4,200 ms -> 740 ms), confirming that horizontal scaling distributes load and reduces per-request latency. Throughput improved from 1 to 2 containers (2.76 -> 4.44 req/s) but plateaued at 4 containers.

This plateau is expected and instructive: TensorFlow inference is CPU-bound, and the test machine has a fixed number of physical cores. Once the container count exceeds the available cores, additional replicas compete for the same CPU rather than adding compute; so median latency continues to fall (requests are distributed more evenly) while total throughput stops rising. This illustrates that horizontal scaling delivers diminishing returns once the underlying hardware becomes the bottleneck, and that on constrained hardware there is an optimal replica count beyond which more containers do not help throughput.

All runs completed with a **0% failure rate**, demonstrating the service remains reliable under sustained concurrent load.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras, MobileNetV2 (transfer learning) |
| API | FastAPI, Uvicorn |
| UI | Streamlit |
| Load balancing | nginx |
| Containerisation | Docker, Docker Compose |
| Load testing | Locust |
| Cloud | Google Cloud Run |
| Dataset | PlantVillage corn subset (Kaggle) |

---

## Cloud Deployment (Google Cloud Run)

Both services are deployed on Cloud Run from source. To reproduce:

```bash
# Enable required APIs (one-time)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Deploy the API (uses docker/Dockerfile.api copied to ./Dockerfile)
cp docker/Dockerfile.api Dockerfile
gcloud run deploy corn-api \
  --source . --region us-central1 --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 300 --port 8000

# Deploy the UI, pointing it at the API's URL
cp docker/Dockerfile.ui Dockerfile
gcloud run deploy corn-ui \
  --source . --region us-central1 --allow-unauthenticated \
  --memory 1Gi --port 8501 \
  --set-env-vars API_URL=<your-api-url>
```

A `.gcloudignore` file excludes the training images from the upload, keeping only the trained model that the API needs.

---

## Notes and Limitations

- The model is trained on studio-style PlantVillage images with uniform backgrounds. Predictions on real-world field photographs (varied lighting, multiple leaves, complex backgrounds) may be less reliable; which is precisely the scenario the retraining pipeline is designed to address.
- Cloud Run's filesystem is ephemeral; a model retrained on a deployed instance persists only for that instance's lifetime. For durable retraining in production, the model artifacts would be written to Cloud Storage (the model and data paths are already environment-configurable to support this).
- Free-tier cloud hardware means inference latency is higher than it would be on GPU-backed infrastructure; the load-test numbers reflect CPU-only serving.

---
@ Prepared by Alain Ishimwe Ngabo,  ALU 2026
