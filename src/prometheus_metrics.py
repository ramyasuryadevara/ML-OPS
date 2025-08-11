#!/usr/bin/env python3
"""
Enhanced Prometheus Metrics Collection
Comprehensive metrics for MLOps monitoring including model performance,
data drift, system health, and business metrics.
"""

import time
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
)
import logging

logger = logging.getLogger(__name__)


class MLOpsMetrics:
    """Comprehensive MLOps metrics collector."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""

        # === API Metrics ===
        self.api_requests_total = Counter(
            "mlops_api_requests_total",
            "Total number of API requests",
            ["endpoint", "method", "status_code", "model_type"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "mlops_api_request_duration_seconds",
            "API request duration in seconds",
            ["endpoint", "model_type"],
            registry=self.registry,
        )

        self.api_validation_errors = Counter(
            "mlops_api_validation_errors_total",
            "Total number of validation errors",
            ["model_type", "error_type"],
            registry=self.registry,
        )

        # === Model Performance Metrics ===
        self.model_predictions_total = Counter(
            "mlops_model_predictions_total",
            "Total number of model predictions",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        self.model_prediction_latency = Histogram(
            "mlops_model_prediction_latency_seconds",
            "Model prediction latency in seconds",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        self.model_accuracy = Gauge(
            "mlops_model_accuracy",
            "Current model accuracy",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        self.model_mse = Gauge(
            "mlops_model_mse",
            "Current model Mean Squared Error",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        self.model_r2_score = Gauge(
            "mlops_model_r2_score",
            "Current model R² score",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        # === Data Quality Metrics ===
        self.data_drift_score = Gauge(
            "mlops_data_drift_score",
            "Data drift detection score",
            ["model_type", "feature"],
            registry=self.registry,
        )

        self.input_feature_stats = Gauge(
            "mlops_input_feature_stats",
            "Input feature statistics",
            ["model_type", "feature", "stat_type"],
            registry=self.registry,
        )

        self.prediction_distribution = Histogram(
            "mlops_prediction_distribution",
            "Distribution of model predictions",
            ["model_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")],
            registry=self.registry,
        )

        # === Business Metrics ===
        self.daily_predictions = Gauge(
            "mlops_daily_predictions",
            "Number of predictions made today",
            ["model_type"],
            registry=self.registry,
        )

        self.prediction_confidence = Histogram(
            "mlops_prediction_confidence",
            "Model prediction confidence scores",
            ["model_type"],
            registry=self.registry,
        )

        # === System Health Metrics ===
        self.model_load_time = Gauge(
            "mlops_model_load_time_seconds",
            "Time taken to load model",
            ["model_type", "model_name"],
            registry=self.registry,
        )

        self.database_size = Gauge(
            "mlops_database_size_bytes",
            "Size of prediction database",
            ["database_type"],
            registry=self.registry,
        )

        self.retraining_status = Info(
            "mlops_retraining_status",
            "Current retraining status information",
            registry=self.registry,
        )

        self.last_retraining_time = Gauge(
            "mlops_last_retraining_timestamp",
            "Timestamp of last model retraining",
            ["model_type"],
            registry=self.registry,
        )

        # === Error Tracking ===
        self.model_errors_total = Counter(
            "mlops_model_errors_total",
            "Total number of model errors",
            ["model_type", "error_type"],
            registry=self.registry,
        )

        self.system_errors_total = Counter(
            "mlops_system_errors_total",
            "Total number of system errors",
            ["component", "error_type"],
            registry=self.registry,
        )

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        model_type: str,
        duration: float,
    ):
        """Record API request metrics."""
        self.api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
            model_type=model_type,
        ).inc()

        self.api_request_duration.labels(
            endpoint=endpoint, model_type=model_type
        ).observe(duration)

    def record_validation_error(self, model_type: str, error_type: str):
        """Record validation error."""
        self.api_validation_errors.labels(
            model_type=model_type, error_type=error_type
        ).inc()

    def record_prediction(
        self, model_type: str, model_name: str, prediction_value: float, latency: float
    ):
        """Record model prediction metrics."""
        self.model_predictions_total.labels(
            model_type=model_type, model_name=model_name
        ).inc()

        self.model_prediction_latency.labels(
            model_type=model_type, model_name=model_name
        ).observe(latency)

        self.prediction_distribution.labels(model_type=model_type).observe(
            prediction_value
        )

    def update_model_performance(self, model_type: str, model_name: str, metrics: Dict):
        """Update model performance metrics."""
        if "accuracy" in metrics:
            self.model_accuracy.labels(
                model_type=model_type, model_name=model_name
            ).set(metrics["accuracy"])

        if "mse" in metrics:
            self.model_mse.labels(model_type=model_type, model_name=model_name).set(
                metrics["mse"]
            )

        if "r2_score" in metrics:
            self.model_r2_score.labels(
                model_type=model_type, model_name=model_name
            ).set(metrics["r2_score"])

    def update_data_drift(self, model_type: str, feature: str, drift_score: float):
        """Update data drift metrics."""
        self.data_drift_score.labels(model_type=model_type, feature=feature).set(
            drift_score
        )

    def update_feature_stats(self, model_type: str, feature_stats: Dict):
        """Update input feature statistics."""
        for feature, stats in feature_stats.items():
            for stat_type, value in stats.items():
                self.input_feature_stats.labels(
                    model_type=model_type, feature=feature, stat_type=stat_type
                ).set(value)

    def update_daily_predictions(self):
        """Update daily prediction counts."""
        try:
            # Housing predictions
            housing_count = self._get_daily_prediction_count(
                "housinglogs/predictions.db", "housinglogs"
            )
            self.daily_predictions.labels(model_type="housing").set(housing_count)

            # Iris predictions
            iris_count = self._get_daily_prediction_count(
                "irislogs/predictions.db", "irislogs"
            )
            self.daily_predictions.labels(model_type="iris").set(iris_count)

        except Exception as e:
            logger.error(f"Error updating daily predictions: {e}")

    def _get_daily_prediction_count(self, db_path: str, table_name: str) -> int:
        """Get prediction count for today."""
        try:
            import os

            if not os.path.exists(db_path):
                return 0

            conn = sqlite3.connect(db_path)
            today = datetime.now().date().isoformat()

            query = f"""
            SELECT COUNT(*) FROM {table_name} 
            WHERE DATE(timestamp) = ?
            """

            cursor = conn.execute(query, [today])
            count = cursor.fetchone()[0]
            conn.close()

            return count
        except Exception as e:
            logger.error(f"Error getting daily prediction count: {e}")
            return 0

    def update_database_sizes(self):
        """Update database size metrics."""
        try:
            import os

            # Housing database
            if os.path.exists("housinglogs/predictions.db"):
                size = os.path.getsize("housinglogs/predictions.db")
                self.database_size.labels(database_type="housing").set(size)

            # Iris database
            if os.path.exists("irislogs/predictions.db"):
                size = os.path.getsize("irislogs/predictions.db")
                self.database_size.labels(database_type="iris").set(size)

        except Exception as e:
            logger.error(f"Error updating database sizes: {e}")

    def update_retraining_status(self, status_info: Dict):
        """Update retraining status information."""
        try:
            # Convert dict to string format for Info metric
            status_str = {k: str(v) for k, v in status_info.items()}
            self.retraining_status.info(status_str)

            # Update last retraining timestamps
            if "housing" in status_info and "last_retrain" in status_info["housing"]:
                timestamp = datetime.fromisoformat(
                    status_info["housing"]["last_retrain"]
                ).timestamp()
                self.last_retraining_time.labels(model_type="housing").set(timestamp)

            if "iris" in status_info and "last_retrain" in status_info["iris"]:
                timestamp = datetime.fromisoformat(
                    status_info["iris"]["last_retrain"]
                ).timestamp()
                self.last_retraining_time.labels(model_type="iris").set(timestamp)

        except Exception as e:
            logger.error(f"Error updating retraining status: {e}")

    def record_model_error(self, model_type: str, error_type: str):
        """Record model error."""
        self.model_errors_total.labels(
            model_type=model_type, error_type=error_type
        ).inc()

    def record_system_error(self, component: str, error_type: str):
        """Record system error."""
        self.system_errors_total.labels(
            component=component, error_type=error_type
        ).inc()

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics instance
mlops_metrics = MLOpsMetrics()


def get_metrics_handler():
    """FastAPI handler for metrics endpoint."""
    # Update dynamic metrics before serving
    mlops_metrics.update_daily_predictions()
    mlops_metrics.update_database_sizes()

    return mlops_metrics.get_metrics()


# Decorator for timing functions
def time_function(metric_name: str, labels: Dict = None):
    """Decorator to time function execution."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Record timing metric (you can customize this based on your needs)
                logger.debug(f"{metric_name} took {duration:.3f} seconds")

        return wrapper

    return decorator
