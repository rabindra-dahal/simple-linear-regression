# 🚗 Swedish Auto Insurance Analytics Pipeline

A decoupled, production-ready Machine Learning application that performs **Simple Linear Regression** on the classic Swedish Auto Insurance dataset. The architecture separates the system into a core mathematical engine, a **FastAPI** microservice backend, and an interactive **Streamlit** user interface.

## 📐 Project Architecture

The codebase is split into three decoupled operational layers to ensure scalability and ease of testing:

- **`model.py` (Core ML Layer)**: Handles structural data loading via Pandas, manages train/test matrix segmentation via Scikit-Learn, fits the model coefficients, and outputs mathematical metric structures.
- **`api.py` (FastAPI Web Service)**: Exposes REST network endpoints, wraps runtime inputs in strict Pydantic validation frames, captures error handling via global interceptors, and writes structured logs to `api_service.log`.
- **`app.py` (Streamlit Frontend)**: A visual client dashboard that requests metrics and inference predictions via HTTP from the FastAPI microservice and renders interactive Matplotlib graphs.

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.8+** configured on your local machine.

### 1. Clone the Project & Install Dependencies

Clone this workspace folder repository and install the comprehensive framework matrix package requirements:

```bash
pip install fastapi uvicorn requests streamlit pandas scikit-learn matplotlib pytest httpx
```

### 2. Launch the Web API Service Backend

Fire up the Uvicorn deployment engine server to host the endpoint routing rules locally on `port 8000`:

```bash
python api.py
```

- **Interactive API Docs**: You can explore and test the endpoints directly by opening your browser to [http://127.0.0](http://127.0.0).

### 3. Initialize the Streamlit Frontend Client

Open a second terminal window instance in the same directory context and start the presentation UI engine:

```bash
streamlit run app.py
```

The application will automatically launch your default browser to [http://localhost:8501](http://localhost:8501).

---

## 🧪 Automated Testing

We use `pytest` alongside `TestClient` frameworks to verify input schema constraints automatically. The suite validates positive calculations, out-of-bound structural inputs (e.g., negative claim data points), and fallback HTTP status responses.

To execute the unit tests, run:

```bash
pytest -v
```

---

## 🛡️ Input Validation Boundaries

The FastAPI layer relies on **Pydantic** constraints to secure inference calculations:

- **`test_size`**: Must sit strictly within the logical range of `[0.05, 0.95]`. Passing values outside this scope generates an immediate `422 Unprocessable Entity` response.
- **`claims`**: Bound strictly to numbers `>= 0.0`. Negative historical claim inputs are immediately intercepted and rejected at the gateway.

## 📂 Project Structure

```swedish_insurance_ml
|
├── model.py          # Core Machine Learning regression pipeline class
├── api.py            # FastAPI service layer with structured loggers & handlers
├── app.py            # Streamlit dashboard client and plot builder
├── test_api.py       # Pytest suite targeting endpoint structural resilience
└── api_service.log   # Persisted operational system logs (generated on run)
```
