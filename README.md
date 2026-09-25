# 🚗 Swedish Auto Insurance Analytics Ecosystem

A fully decoupled, production-grade Machine Learning microservice ecosystem that performs **Simple Linear Regression** on the classic Swedish Auto Insurance dataset. 

This repository has been architected from the ground up to separate mathematical modeling, web api routing protocols, data presentation layouts, and automated testing hooks into isolated, self-contained service layers.

---

## 📐 Completed Architecture & Core Components

The application is structured into distinct, modular functional tiers:

### 1. Core ML Processing Layer (`model.py`)
*   **Pipeline Logic:** Manages training and testing data splits dynamically using `scikit-learn` configurations.
*   **Offline Resilience Framework:** Attempts to stream the latest dataset dynamically via a remote repository endpoint. If network drops occur, it smoothly intercepts the failure using built-in network handlers and falls back onto a local, packaged `auto-insurance.csv` file without dropping the initialization loop.
*   **Prediction Isolation:** Wraps 1D prediction output matrices securely, parsing data types safely into native Python float metrics to protect against structural type exceptions.

### 2. FastAPI Microservice Web Engine (`api.py`)
*   **Operational Control:** Intercepts payload matrices and handles remote orchestration routing.
*   **Strict Pydantic Input Guards:** Protects against erratic variables. The engine locks `test_size` properties strictly to a `[0.05, 0.95]` distribution window and maps `claims` metrics to rules requiring parameters `>= 0.0`. Out-of-bound arguments trigger clean `422 Unprocessable Content` structures.
*   **Global Failure Middleware:** Intercepts unhandled app breakdowns cleanly, isolating raw stack trace vulnerabilities from front-facing client networks.
*   **System Health Telemetry (`/health`):** Features a dedicated monitoring gateway checking external API endpoints and local file storage paths in real time.
*   **Structured Logs:** Generates persistent diagnostic execution records saved locally into an `api_service.log` matrix.

### 3. Interactive Streamlit Client Dashboard (`app.py`)
*   **Reactive KPI Cards:** Formats model outputs (\(R^2\), \(MSE\), \(RMSE\)) using clean scorecard containers.
*   **Dynamic Visual Plots:** Renders linear lines of best fit against source scatter data utilizing raw-string string-literal structures (`fr"..."`) to cleanly handle LaTeX math formats.
*   **Live Health Trackers:** Features a persistent sidebar diagnostic component displaying connection modes (`🟢 Online / Live Data` vs `🟡 Degraded / Local Backup Mode`).

### 4. Automated Testing Suite (`test_api.py`)
*   **Continuous Verification:** Runs synchronous boundary audits using `pytest` and `httpx` to systematically check validation parameters, payload layouts, and failure responses.

### 5. Dockerization & CI/CD Deployment Orchestration
*   **`Dockerfile.api` & `Dockerfile.app`:** Builds minimal Linux container footprints.
*   **`docker-compose.yml`:** Links the application tiers into an isolated container network loop using internal DNS routing.
*   **`.dockerignore`:** Strips heavy local execution environments (`venv/`) and cache indices (`__pycache__/`), accelerating build upload operations.
*   **GitHub Actions Workflow (`.github/workflows/ci-pipeline.yml`):** Automatically boots an Ubuntu server runner on code changes, verifies unit test coverage parameters, and builds container configurations natively.

---

## 📂 Completed Project Hierarchy

```text
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml   # CI system verification pipeline
├── model.py                  # Operational machine learning core class
├── api.py                    # FastAPI web server, validations, & health check checks
├── app.py                    # Streamlit visual frontend client
├── test_api.py               # Automated unit testing suite
├── requirements.txt          # Frozen application dependency matrix
├── auto-insurance.csv        # Packed local fallback offline dataset matrix
├── Dockerfile.api            # Microservice environment composition script
├── Dockerfile.app            # Client dashboard environment composition script
├── docker-compose.yml        # Multi-container cluster orchestration manager
├── .dockerignore             # Cache and virtual environment build exclusions
└── api_service.log           # Generated backend service operational runtime logs
```

---

## 🛠️ Execution & Deployment Guide

### Option A: Local Execution (Bare Metal)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt pip install pytest httpx
   ```
2. **Launch the API Service Backend (Terminal 1):**
   ```bash
   python api.py
   ```
   * *Interactive Endpoint Documentation:* Accessible via [http://127.0.0](http://127.0.0).
3. **Launch the Frontend Client Dashboard (Terminal 2):**
   ```bash
   streamlit run app.py
   ```
   * Access the dashboard interface at [http://localhost:8501](http://localhost:8501).

### Option B: Unified Containerized Build (Docker Compose)

Launch the entire multi-service microservice layer instantly with a single instruction from your project workspace root:
```bash
docker compose up --build
```
* **Streamlit UI Interface:** Open browser context to [http://localhost:8501](http://localhost:8501).
* **FastAPI Backend WebDocs:** Open browser context to [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🧪 Testing Protocol

Run automated regression performance and schema validation checks using:
```bash
pytest -v
```
