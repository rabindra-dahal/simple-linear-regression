# test_api.py
from fastapi.testclient import TestClient
from api import app  # Imports your FastAPI instance

# Initialize the TestClient using our FastAPI application context
client = TestClient(app)


def test_train_and_metrics_success():
    """Verifies successful data loading and structural consistency on valid configurations."""
    payload = {"test_size": 0.25}
    response = client.post("/train-and-metrics", json=payload)

    assert response.status_code == 200

    data = response.json()
    # Confirm presence of the primary mathematical indicators
    assert "r2" in data
    assert "mse" in data
    assert "rmse" in data
    assert "slope" in data
    assert "intercept" in data
    assert "raw_data" in data

    # Verify return types are correctly extracted as floats
    assert isinstance(data["r2"], float)
    assert isinstance(data["intercept"], float)


def test_train_and_metrics_validation_failure():
    """Ensures Pydantic blocks out-of-bounds test sizes with a 422 error."""
    # Test split ratio cannot exceed 0.95 based on Field parameters
    payload = {"test_size": 0.99}
    response = client.post("/train-and-metrics", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert "details" in data


def test_predict_success():
    """Verifies that the inference engine accurately serves numeric target calculations."""
    payload = {"claims": 42.0, "test_size": 0.20}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_payment" in data
    assert data["claims"] == 42.0
    assert isinstance(data["predicted_payment"], float)


def test_predict_validation_failure():
    """Ensures Pydantic blocks structurally impossible input contexts like negative claims."""
    payload = {"claims": -5.0, "test_size": 0.20}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
