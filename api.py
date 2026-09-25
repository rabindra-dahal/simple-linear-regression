# api.py
import http.client
import logging
import os
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from model import InsuranceRegressionModel
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api_service.log")],
)
logger = logging.getLogger("InsuranceAPI")

app = FastAPI(
    title="Swedish Auto Insurance ML API",
    description="Microservice backend with robust input parsing and logging overrides.",
)


class TrainRequest(BaseModel):
    test_size: float = Field(default=0.2, ge=0.05, le=0.95)


class PredictRequest(BaseModel):
    claims: float = Field(..., ge=0.0)
    test_size: float = Field(default=0.2, ge=0.05, le=0.95)


# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    logger.warning(
        f"Bad Payload Request: {exc.errors()} | Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "Validation Error",
            "message": "The inputs provided fail business validation rules.",
            "details": jsonable_encoder(exc.errors()),
        },
    )
    


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Critical Unhandled Error on path {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while computing linear regression pipelines.",
        },
    )


# --- NEW: HEALTH CHECK ENDPOINT ---
@app.get("/health", tags=["System Health"])
def health_check():
    """Checks service status and verifies if the remote data dependency is reachable."""
    health_status = "healthy"
    remote_reachable = False
    dataset_url = "https://githubusercontent.com"

    # 1. Check if the remote GitHub dataset URL is reachable (Timeout in 2 seconds)
    try:
        parsed_url = urlparse(dataset_url)
        conn = http.client.HTTPSConnection(parsed_url.netloc, timeout=2)
        conn.request("HEAD", parsed_url.path)
        response = conn.getresponse()
        if response.status == 200:
            remote_reachable = True
        conn.close()
    except Exception:
        remote_reachable = False

    # 2. Check if local backup exists
    local_backup_exists = os.path.exists("auto-insurance.csv")

    # 3. Determine overall system health state
    if not remote_reachable:
        if local_backup_exists:
            # Degraded: Internet is down, but app works fine using local file
            health_status = "degraded"
        else:
            # Unhealthy: Internet is down AND local file is missing (App will crash)
            health_status = "unhealthy"

    return {
        "status": health_status,
        "dependencies": {
            "remote_dataset_reachable": remote_reachable,
            "local_backup_found": local_backup_exists,
        },
    }


# Core Endpoints
@app.post("/train-and-metrics")
def train_and_metrics(payload: TrainRequest):
    logger.info(f"Received training request with test_size={payload.test_size}")
    try:
        engine = InsuranceRegressionModel(test_size=payload.test_size)
        metrics = engine.load_and_train()
        logger.info("Regression pipeline successfully optimized.")
        return metrics
    except FileNotFoundError as fnf:
        logger.error(f"Missing Data Error: {str(fnf)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except Exception as e:
        raise e


@app.post("/predict")
def predict(payload: PredictRequest):
    logger.info(
        f"Inference called for claims={payload.claims} using test_size={payload.test_size}"
    )
    try:
        engine = InsuranceRegressionModel(test_size=payload.test_size)
        engine.load_and_train()
        prediction = engine.predict(payload.claims)
        logger.info(f"Successful prediction: {prediction:.4f}")
        return {"claims": payload.claims, "predicted_payment": prediction}
    except Exception as e:
        raise e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
