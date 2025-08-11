import os
import sys
import time
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Optional, Dict, Any
import pandas as pd
import mlflow.pyfunc
import logging
from datetime import datetime
import sqlite3
import joblib  # If using local .pkl file
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, REGISTRY

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Removed custom mlops-metrics integration

os.makedirs("housinglogs", exist_ok=True)

# FastAPI setup
app = FastAPI(
    title="MLOps Housing Price Prediction API",
    description="A comprehensive MLOps pipeline for housing price prediction with automated training, deployment, monitoring, and retraining capabilities.",
    version="1.0.0",
)

# Dynamic examples storage (cross-service via shared volume)
try:
    from src.example_store import read_example, write_example
except Exception:
    # fallback relative import when running inside container w/ PYTHONPATH=/app
    from example_store import read_example, write_example
# Expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(
    app, include_in_schema=False, endpoint="/metrics"
)

# Prometheus metrics matching Grafana dashboard queries
SERVICE = "housing"
MLOPS_API_REQUESTS = Counter(
    "mlops_api_requests_total",
    "Total API requests",
    ["service", "endpoint", "method", "status"],
)
MLOPS_MODEL_PREDICTIONS = Counter(
    "mlops_model_predictions_total",
    "Total model predictions",
    ["service", "model"],
)
MLOPS_PREDICTION_LATENCY = Histogram(
    "mlops_model_prediction_latency_seconds",
    "Model prediction latency (seconds)",
    ["service", "model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)
MLOPS_VALIDATION_ERRORS = Counter(
    "mlops_api_validation_errors_total",
    "Total validation errors",
    ["service", "endpoint"],
)
MLOPS_ERRORS = Counter(
    "mlops_api_errors_total",
    "Total API errors",
    ["service", "endpoint"],
)
MLOPS_DAILY_PREDICTIONS = Gauge(
    "mlops_daily_predictions", "Predictions count (since start)", ["service"]
)
MLOPS_DB_SIZE = Gauge(
    "mlops_database_size_bytes", "Size of SQLite DB file in bytes", ["service"]
)


def _update_gauges():
    try:
        # DB file size
        db_path = os.path.join("housinglogs", "predictions.db")
        if os.path.exists(db_path):
            MLOPS_DB_SIZE.labels(service=SERVICE).set(os.path.getsize(db_path))
    except Exception:
        pass
    try:
        cursor.execute("SELECT COUNT(*) FROM housinglogs")
        total = cursor.fetchone()[0]
        MLOPS_DAILY_PREDICTIONS.labels(service=SERVICE).set(total)
    except Exception:
        pass


# model_uri = "runs:/4bf65a1a6fdd4d9fb80d35b460d5d721/model"
# model = mlflow.pyfunc.load_model(model_uri)

model = joblib.load("models/DecisionTree.pkl")

logging.basicConfig(
    filename="housinglogs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

conn = sqlite3.connect("housinglogs/predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS housinglogs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    inputs TEXT,
    prediction TEXT
)
"""
)
conn.commit()

prediction_count = 0


class HousingRequest(BaseModel):
    """
    Enhanced Housing Price Prediction Request with comprehensive validation.

    Based on California Housing Dataset characteristics:
    - MedInc: Median income (in tens of thousands of dollars)
    - HouseAge: Median house age (0-52 years)
    - Total rooms/bedrooms: Reasonable ranges for household data
    - Population: Block group population (3-35,682)
    - Households: Number of households (1-6,082)
    - Latitude: California latitude range (32.5-42.0)
    - Longitude: California longitude range (-124.5 to -114.0)
    """

    total_rooms: float = Field(
        ...,
        gt=0,
        le=50000,
        description="Total number of rooms in the block group. Must be positive and reasonable.",
    )

    total_bedrooms: float = Field(
        ...,
        gt=0,
        le=10000,
        description="Total number of bedrooms in the block group. Must be positive and reasonable.",
    )

    population: float = Field(
        ...,
        ge=1,
        le=50000,
        description="Block group population. Must be at least 1 and reasonable for a district.",
    )

    households: float = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of households in the block group. Must be at least 1.",
    )

    median_income: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Median income in tens of thousands of dollars. Must be positive and reasonable (0-20).",
    )

    housing_median_age: float = Field(
        ...,
        ge=0,
        le=60,
        description="Median age of houses in the block group in years (0-60).",
    )

    latitude: float = Field(
        ...,
        ge=32.0,
        le=42.5,
        description="Latitude coordinate. Must be within California bounds (32.0-42.5).",
    )

    longitude: float = Field(
        ...,
        ge=-125.0,
        le=-114.0,
        description="Longitude coordinate. Must be within California bounds (-125.0 to -114.0).",
    )

    @model_validator(mode="after")
    def validate_consistency(self):
        """Pydantic v2: cross-field validation after model creation."""
        values = self.model_dump()
        # bedrooms vs rooms
        if (
            values.get("total_bedrooms") is not None
            and values.get("total_rooms") is not None
        ):
            if values["total_bedrooms"] > values["total_rooms"]:
                raise ValueError("Total bedrooms cannot exceed total rooms")
        # households vs population
        if values.get("households") and values.get("population"):
            v = values["households"]
            pop = values["population"]
            if v > pop:
                raise ValueError("Number of households cannot exceed population")
            avg_household_size = pop / v
            if avg_household_size < 0.5 or avg_household_size > 20:
                suggested_households = int(pop / 3)
                raise ValueError(
                    f"Average household size ({avg_household_size:.2f}) is unrealistic. Should be between 0.5 and 20. "
                    f"For population {pop}, try households around {suggested_households} (avg 3 people/household)."
                )
        # rooms per household
        if values.get("total_rooms") and values.get("households"):
            avg_rooms = values["total_rooms"] / values["households"]
            if avg_rooms < 0.5 or avg_rooms > 50:
                raise ValueError(
                    f"Average rooms per household ({avg_rooms:.2f}) is unrealistic. Should be between 0.5 and 50."
                )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "total_rooms": 4500.0,
                "total_bedrooms": 900.0,
                "population": 3000.0,
                "households": 1000.0,
                "median_income": 5.5,
                "housing_median_age": 26.0,
                "latitude": 37.86,
                "longitude": -122.27,
            }
        }


class HousingResponse(BaseModel):
    """Response model for housing price predictions."""

    predicted_price: float = Field(
        ...,
        description="Predicted median house value in hundreds of thousands of dollars",
    )
    input_validation_passed: bool = Field(
        default=True, description="Whether input validation passed successfully"
    )

    class Config:
        json_schema_extra = {
            "example": {"predicted_price": 3.85, "input_validation_passed": True}
        }


class ValidationErrorResponse(BaseModel):
    """Response model for validation errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    details: Optional[dict] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Input validation failed",
                "details": {
                    "field": "median_income",
                    "value": -1.0,
                    "constraint": "must be greater than 0",
                },
            }
        }


class RetrainRequest(BaseModel):
    """Request model for model retraining."""

    model_type: Optional[str] = Field(
        None,
        description="Type of model to retrain: 'housing' or None for both",
    )
    force: bool = Field(
        False, description="Force retraining even if performance is acceptable"
    )
    new_data_path: Optional[str] = Field(
        None,
        description="Path to new dataset for retraining (e.g., 'data/new_housing_data.csv'). If not provided, uses existing training data.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "housing",
                "force": True,
                "new_data_path": "data/new_housing_data.csv",
            }
        }


class RetrainResponse(BaseModel):
    """Response model for retraining requests."""

    status: str = Field(..., description="Status of the retraining request")
    message: str = Field(..., description="Detailed message about the retraining")
    task_id: Optional[str] = Field(None, description="Background task ID if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "started",
                "message": "Model retraining started in background",
                "task_id": "retrain_housing_20231201_143022",
            }
        }


@app.get("/example")
def get_current_example():
    """Return the example payload FastAPI shows in the docs; dynamically updated after retraining."""
    # Prefer stored example if present; otherwise fall back to the schema example
    stored = read_example("housing")
    if stored:
        return stored
    return HousingRequest.model_json_schema().get(
        "examples",
        [HousingRequest.model_config.get("json_schema_extra", {}).get("example", {})],
    )[0]


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (housing)")
    last_trained: Optional[str] = Field(None, description="Last training timestamp")
    performance_metrics: Optional[dict] = Field(
        None, description="Current performance metrics"
    )
    model_path: str = Field(..., description="Path to the model file")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "DecisionTree",
                "model_type": "housing",
                "last_trained": "2023-12-01T14:30:22",
                "performance_metrics": {"r2_score": 0.65, "mse": 0.45},
                "model_path": "models/DecisionTree.pkl",
            }
        }


@app.get("/")
def root():
    return {"message": "Housing price prediction API is running."}


@app.post(
    "/predict",
    response_model=HousingResponse,
    responses={422: {"model": ValidationErrorResponse}},
)
def predict(data: HousingRequest):
    """
    Predict housing prices with enhanced input validation.

    This endpoint validates all input parameters according to California housing dataset
    characteristics and returns detailed error messages for invalid inputs.
    """
    global prediction_count

    start_time = time.time()

    try:
        prediction_count += 1

        # Convert to DataFrame for processing
        df = pd.DataFrame([data.model_dump()])

        # Feature engineering - same as training pipeline
        df["AveRooms"] = df["total_rooms"] / df["households"]
        df["AveBedrms"] = df["total_bedrooms"] / df["households"]
        df["AveOccup"] = df["population"] / df["households"]

        # Additional validation for derived features
        if df["AveRooms"].iloc[0] > 50:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "message": "Derived feature validation failed",
                    "details": {
                        "field": "average_rooms_per_household",
                        "value": df["AveRooms"].iloc[0],
                        "constraint": "Average rooms per household cannot exceed 50",
                    },
                },
            )

        if df["AveBedrms"].iloc[0] > 10:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ValidationError",
                    "message": "Derived feature validation failed",
                    "details": {
                        "field": "average_bedrooms_per_household",
                        "value": df["AveBedrms"].iloc[0],
                        "constraint": "Average bedrooms per household cannot exceed 10",
                    },
                },
            )

        # Rename columns to match model expectations
        df.rename(
            columns={
                "median_income": "MedInc",
                "housing_median_age": "HouseAge",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "population": "Population",
            },
            inplace=True,
        )

        final_features = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]

        # Make prediction
        prediction = model.predict(df[final_features])[0]

        # Validate prediction is reasonable (California housing prices)
        if prediction < 0 or prediction > 10:
            logging.warning(f"Unusual prediction value: {prediction}")

        # Log to file and SQLite
        input_data = data.model_dump()
        logging.info(f"Input: {input_data} | Prediction: {prediction}")
        cursor.execute(
            "INSERT INTO housinglogs (timestamp, inputs, prediction) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), str(input_data), str(prediction)),
        )
        conn.commit()

        # Record Prometheus metrics
        prediction_latency = time.time() - start_time
        MLOPS_API_REQUESTS.labels(
            service=SERVICE, endpoint="/predict", method="POST", status="200"
        ).inc()
        MLOPS_MODEL_PREDICTIONS.labels(service=SERVICE, model="DecisionTree").inc()
        MLOPS_PREDICTION_LATENCY.labels(service=SERVICE, model="DecisionTree").observe(
            prediction_latency
        )
        _update_gauges()

        return HousingResponse(predicted_price=float(prediction))

    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        MLOPS_VALIDATION_ERRORS.labels(service=SERVICE, endpoint="/predict").inc()
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ValidationError",
                "message": "Input validation failed",
                "details": str(e),
            },
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        MLOPS_ERRORS.labels(service=SERVICE, endpoint="/predict").inc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PredictionError",
                "message": "An error occurred during prediction",
                "details": str(e),
            },
        )


# Renamed to avoid conflict with Prometheus /metrics
@app.get("/app-metrics")
def metrics():
    return {"total_predictions": prediction_count}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "housing-price-prediction",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "database_connected": conn is not None,
    }


# Global variable to track retraining status
retraining_status = {"is_running": False, "last_run": None, "last_result": None}


def run_model_retraining(
    model_type: Optional[str] = None,
    force: bool = False,
    new_data_path: Optional[str] = None,
):
    """Background task to run model retraining."""

    try:
        retraining_status["is_running"] = True
        start_time = datetime.now()

        # Import retraining functionality
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from model_retraining import ModelRetrainer, ModelPerformanceMonitor

        # Initialize components
        monitor = ModelPerformanceMonitor()
        retrainer = ModelRetrainer()

        results = {
            "start_time": start_time.isoformat(),
            "model_type": model_type,
            "force": force,
            "new_data_path": new_data_path,
            "results": {},
        }

        # Determine which models to retrain
        models_to_retrain = ["housing"]

        # Retrain models
        for model_name in models_to_retrain:
            try:
                if model_name == "housing":
                    # Check if retraining is needed
                    if not force:
                        performance = monitor.evaluate_model_performance("housing")
                        if not performance.get("needs_retraining", False):
                            results["results"]["housing"] = {
                                "status": "skipped",
                                "reason": "performance_acceptable",
                                "performance": performance,
                            }
                            continue

                    # Retrain housing model
                    retrain_result = retrainer.retrain_housing_model(
                        data_path=new_data_path
                    )
                    results["results"]["housing"] = retrain_result

                    # Reload the model in the API
                    global model
                    model = joblib.load("models/DecisionTree.pkl")

                    # After retraining, update the dynamic example based on recent/typical inputs
                    try:
                        latest_example = {
                            "total_rooms": 4500.0,
                            "total_bedrooms": 900.0,
                            "population": 3000.0,
                            "households": 1000.0,
                            "median_income": 5.5,
                            "housing_median_age": 26.0,
                            "latitude": 37.86,
                            "longitude": -122.27,
                        }
                        write_example("housing", latest_example)
                        logging.info("Updated housing example payload after retraining")
                    except Exception as e:
                        logging.warning(f"Could not update example payload: {e}")



            except Exception as e:
                results["results"][model_name] = {"status": "failed", "error": str(e)}

        results["end_time"] = datetime.now().isoformat()
        results["status"] = "completed"

        # Update global status
        retraining_status["last_run"] = start_time.isoformat()
        retraining_status["last_result"] = results

        logging.info(f"Retraining completed: {results}")

    except Exception as e:
        error_result = {
            "status": "failed",
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
        }
        retraining_status["last_result"] = error_result
        logging.error(f"Retraining failed: {e}")

    finally:
        retraining_status["is_running"] = False


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model(request: RetrainRequest):
    """
    Retrain Model

    Trigger model retraining for housing and/or iris models.
    This endpoint runs retraining synchronously and responds only after completion.
    """

    # Check if retraining is already running
    if retraining_status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "RetrainingInProgress",
                "message": "Model retraining is already in progress",
                "details": {"current_status": retraining_status},
            },
        )

    # Run retraining synchronously (blocking)
    run_model_retraining(
        model_type=request.model_type,
        force=request.force,
        new_data_path=request.new_data_path,
    )

    # Retrieve last result and respond only after completion
    result = retraining_status.get("last_result") or {}
    status = result.get("status", "completed")

    logging.info(
        f"Retraining completed: model_type={request.model_type}, force={request.force}, new_data_path={request.new_data_path}, status={status}"
    )

    return RetrainResponse(
        status=status,
        message=f"Model retraining completed for {request.model_type or 'all models'}",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Get Model Info

    Get information about the currently loaded model including performance metrics
    and last training time.
    """
    try:
        # Get model file stats
        model_path = "models/DecisionTree.pkl"
        model_stats = os.stat(model_path) if os.path.exists(model_path) else None

        # Get performance metrics if available
        performance_metrics = None
        try:
            if os.path.exists("retraining_results.json"):
                import json

                with open("retraining_results.json", "r") as f:
                    results = json.load(f)
                    if "housing" in results and "results" in results["housing"]:
                        performance_metrics = results["housing"]["results"]
        except:
            pass

        return ModelInfoResponse(
            model_name="DecisionTree",
            model_type="housing",
            last_trained=(
                datetime.fromtimestamp(model_stats.st_mtime).isoformat()
                if model_stats
                else None
            ),
            performance_metrics=performance_metrics,
            model_path=model_path,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ModelInfoError",
                "message": "Could not retrieve model information",
                "details": str(e),
            },
        )
