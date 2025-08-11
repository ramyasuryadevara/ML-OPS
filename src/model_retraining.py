#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline
Monitors model performance and triggers retraining when needed.
"""

import os
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("retraining.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Ensure MLflow logs locally inside the repo (works in CI and scripts)
mlflow.set_tracking_uri("file:./mlruns")


class ModelPerformanceMonitor:
    """Monitor model performance and detect when retraining is needed."""

    def __init__(
        self,
        housing_db_path: str = "housinglogs/predictions.db",
        iris_db_path: str = "irislogs/predictions.db",
    ):
        self.housing_db_path = housing_db_path
        self.iris_db_path = iris_db_path
        self.performance_thresholds = {
            "housing": {
                "min_r2": 0.5,  # Minimum R² score
                "max_mse": 1.0,  # Maximum MSE
                "min_predictions": 100,  # Minimum predictions before evaluation
            },
            "iris": {
                "min_accuracy": 0.85,  # Minimum accuracy
                "min_f1": 0.85,  # Minimum F1 score
                "min_predictions": 50,  # Minimum predictions before evaluation
            },
        }

    def get_recent_predictions(self, model_type: str, days: int = 7) -> pd.DataFrame:
        """Get recent predictions from the database."""
        db_path = self.housing_db_path if model_type == "housing" else self.iris_db_path
        table_name = "housinglogs" if model_type == "housing" else "irislogs"

        if not os.path.exists(db_path):
            logger.warning(f"Database {db_path} not found")
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(db_path)
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            query = f"""
            SELECT timestamp, inputs, prediction
            FROM {table_name}
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(query, conn, params=[cutoff_date])
            conn.close()

            logger.info(f"Retrieved {len(df)} recent predictions for {model_type}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving predictions for {model_type}: {e}")
            return pd.DataFrame()

    def evaluate_model_performance(self, model_type: str) -> Dict:
        """Evaluate current model performance based on recent predictions."""
        predictions_df = self.get_recent_predictions(model_type)

        if (
            len(predictions_df)
            < self.performance_thresholds[model_type]["min_predictions"]
        ):
            return {
                "status": "insufficient_data",
                "message": f'Need at least {self.performance_thresholds[model_type]["min_predictions"]} predictions',
                "prediction_count": len(predictions_df),
            }

        # For now, we'll simulate performance evaluation
        # In a real scenario, you'd need ground truth labels to compare against

        if model_type == "housing":
            # Simulate housing model performance
            predictions = [
                float(eval(row["prediction"])) for _, row in predictions_df.iterrows()
            ]

            # Simulate some performance metrics (in real scenario, compare with actual values)
            simulated_r2 = np.random.uniform(0.4, 0.7)  # Simulate R² between 0.4-0.7
            simulated_mse = np.random.uniform(0.3, 1.2)  # Simulate MSE between 0.3-1.2

            performance = {
                "r2_score": simulated_r2,
                "mse": simulated_mse,
                "prediction_count": len(predictions),
                "avg_prediction": np.mean(predictions),
                "std_prediction": np.std(predictions),
            }

            needs_retraining = (
                simulated_r2 < self.performance_thresholds["housing"]["min_r2"]
                or simulated_mse > self.performance_thresholds["housing"]["max_mse"]
            )

        else:  # iris
            # Simulate iris model performance
            predictions = [
                int(eval(row["prediction"])) for _, row in predictions_df.iterrows()
            ]

            # Simulate some performance metrics
            simulated_accuracy = np.random.uniform(0.8, 0.98)
            simulated_f1 = np.random.uniform(0.8, 0.98)

            performance = {
                "accuracy": simulated_accuracy,
                "f1_score": simulated_f1,
                "prediction_count": len(predictions),
                "class_distribution": {str(i): predictions.count(i) for i in range(3)},
            }

            needs_retraining = (
                simulated_accuracy < self.performance_thresholds["iris"]["min_accuracy"]
                or simulated_f1 < self.performance_thresholds["iris"]["min_f1"]
            )

        performance.update(
            {
                "status": "needs_retraining" if needs_retraining else "performing_well",
                "needs_retraining": needs_retraining,
                "evaluation_time": datetime.now().isoformat(),
            }
        )

        logger.info(f"{model_type} model performance: {performance}")
        return performance


class ModelRetrainer:
    """Handle automated model retraining."""

    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def retrain_housing_model(self, data_path: Optional[str] = None) -> Dict:
        """Retrain the housing price prediction model."""
        logger.info(f"Starting housing model retraining with data_path: {data_path}")

        try:
            # Determine data path
            if data_path and os.path.exists(data_path):
                logger.info(f"Using custom data path: {data_path}")
                df = pd.read_csv(data_path)
            else:
                # Use default data
                default_path = "data/housing.csv"
                if not os.path.exists(default_path):
                    logger.error("Housing data not found. Running data loading...")
                    from load_data import load_and_save

                    load_and_save()

                logger.info(f"Using default data path: {default_path}")
                df = pd.read_csv(default_path)
            X = df.drop("MedHouseVal", axis=1)
            y = df["MedHouseVal"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train models
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
            }

            best_model = None
            best_score = -float("inf")
            best_name = None

            results = {}

            # Train each candidate in parallel to speed up retraining
            def _train_one(name_model):
                name, mdl = name_model
                with mlflow.start_run(
                    run_name=f"Retrain_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ) as run:
                    mdl.fit(X_train, y_train)
                    predictions = mdl.predict(X_test)

                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    mlflow.log_param("model_name", name)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2_score", r2)

                    signature = infer_signature(X_test, predictions)
                    input_example = X_test.head(2)
                    mlflow.sklearn.log_model(
                        sk_model=mdl,
                        name="model",
                        input_example=input_example,
                        signature=signature,
                    )

                    model_path = f"{self.models_dir}/{name}.pkl"
                    joblib.dump(mdl, model_path)

                    return (
                        name,
                        mdl,
                        {
                            "mse": mse,
                            "r2_score": r2,
                            "model_path": model_path,
                            "run_id": run.info.run_id,
                        },
                    )

            with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
                futures = [executor.submit(_train_one, item) for item in models.items()]
                for fut in futures:
                    name, mdl, res = fut.result()
                    results[name] = res
                    if res["r2_score"] > best_score:
                        best_score = res["r2_score"]
                        best_model = mdl
                        best_name = name
                    logger.info(
                        f"Retrained {name}: MSE={res['mse']:.3f}, R²={res['r2_score']:.3f}"
                    )

            # Update the main model with the best performing one
            if best_model:
                main_model_path = f"{self.models_dir}/DecisionTree.pkl"
                joblib.dump(best_model, main_model_path)
                logger.info(f"Updated main housing model with {best_name}")

                # Try to register best model in MLflow
                try:
                    registered = mlflow.register_model(
                        model_uri=f"runs:/{results[best_name]['run_id']}/model",
                        name="HousingPricePredictor",
                    )
                    logger.info(
                        f"Registered HousingPricePredictor version {registered.version}"
                    )
                except Exception as e:
                    logger.warning(f"Model registry step skipped or failed: {e}")

            return {
                "status": "success",
                "best_model": best_name,
                "best_score": best_score,
                "results": results,
                "retrain_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Housing model retraining failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "retrain_time": datetime.now().isoformat(),
            }

    def retrain_iris_model(self, data_path: Optional[str] = None) -> Dict:
        """Retrain the iris classification model."""
        logger.info(f"Starting iris model retraining with data_path: {data_path}")

        try:
            # Load iris data
            if data_path and os.path.exists(data_path):
                logger.info(f"Using custom iris data path: {data_path}")
                df = pd.read_csv(data_path)
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            else:
                # Use default iris dataset
                logger.info("Using default iris dataset from sklearn")
                from sklearn.datasets import load_iris

                data = load_iris(as_frame=True)
                df = data.frame
                X = df.drop("target", axis=1)
                y = df["target"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Train models
            models = {
                "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
            }

            best_model = None
            best_score = -float("inf")
            best_name = None

            results = {}

            # Train each candidate in parallel to speed up retraining
            def _train_one_iris(name_model):
                name, mdl = name_model
                with mlflow.start_run(
                    run_name=f"Retrain_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ) as run:
                    mdl.fit(X_train, y_train)
                    predictions = mdl.predict(X_test)

                    accuracy = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average="weighted")

                    mlflow.log_param("model_name", name)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("f1_score", f1)

                    signature = infer_signature(X_test, predictions)
                    input_example = X_test.head(2)
                    mlflow.sklearn.log_model(
                        sk_model=mdl,
                        name="model",
                        input_example=input_example,
                        signature=signature,
                    )

                    model_path = f"{self.models_dir}/{name}.pkl"
                    joblib.dump(mdl, model_path)

                    return (
                        name,
                        mdl,
                        {
                            "accuracy": accuracy,
                            "f1_score": f1,
                            "model_path": model_path,
                            "run_id": run.info.run_id,
                        },
                    )

            with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
                futures = [
                    executor.submit(_train_one_iris, item) for item in models.items()
                ]
                for fut in futures:
                    name, mdl, res = fut.result()
                    results[name] = res
                    if res["accuracy"] > best_score:
                        best_score = res["accuracy"]
                        best_model = mdl
                        best_name = name
                    logger.info(
                        f"Retrained {name}: Accuracy={res['accuracy']:.3f}, F1={res['f1_score']:.3f}"
                    )

            # Update the main model with the best performing one
            if best_model:
                main_model_path = f"{self.models_dir}/RandomForest.pkl"
                joblib.dump(best_model, main_model_path)
                logger.info(f"Updated main iris model with {best_name}")

                # Try to register best model in MLflow
                try:
                    registered = mlflow.register_model(
                        model_uri=f"runs:/{results[best_name]['run_id']}/model",
                        name="IrisClassifier",
                    )
                    logger.info(
                        f"Registered IrisClassifier version {registered.version}"
                    )
                except Exception as e:
                    logger.warning(f"Model registry step skipped or failed: {e}")

            return {
                "status": "success",
                "best_model": best_name,
                "best_score": best_score,
                "results": results,
                "retrain_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Iris model retraining failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "retrain_time": datetime.now().isoformat(),
            }


def main():
    """Main retraining pipeline."""
    logger.info("Starting automated model retraining pipeline...")

    monitor = ModelPerformanceMonitor()
    retrainer = ModelRetrainer()

    results = {"pipeline_start": datetime.now().isoformat(), "housing": {}, "iris": {}}

    # Evaluate and retrain housing model if needed
    housing_performance = monitor.evaluate_model_performance("housing")
    results["housing"]["performance"] = housing_performance

    if housing_performance.get("needs_retraining", False):
        logger.info("Housing model needs retraining...")
        housing_retrain_result = retrainer.retrain_housing_model()
        results["housing"]["retraining"] = housing_retrain_result
    else:
        logger.info("Housing model is performing well, no retraining needed")
        results["housing"]["retraining"] = {
            "status": "skipped",
            "reason": "performance_acceptable",
        }

    # Evaluate and retrain iris model if needed
    iris_performance = monitor.evaluate_model_performance("iris")
    results["iris"]["performance"] = iris_performance

    if iris_performance.get("needs_retraining", False):
        logger.info("Iris model needs retraining...")
        iris_retrain_result = retrainer.retrain_iris_model()
        results["iris"]["retraining"] = iris_retrain_result
    else:
        logger.info("Iris model is performing well, no retraining needed")
        results["iris"]["retraining"] = {
            "status": "skipped",
            "reason": "performance_acceptable",
        }

    results["pipeline_end"] = datetime.now().isoformat()

    # Save results
    with open("retraining_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Retraining pipeline completed")
    return results


if __name__ == "__main__":
    main()
