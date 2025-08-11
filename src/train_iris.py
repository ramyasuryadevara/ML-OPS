import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import joblib
import os

# Ensure MLflow logs locally inside the repo (works in CI)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_classification")

# Load data
data = load_iris(as_frame=True)
df = data.frame
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Store model performance for comparison
model_performance = {}


def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_test, preds)
        input_example = X_test.head(2)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        # Save locally
        joblib.dump(model, f"models/{model_name}.pkl")

        print(
            f"[OK] {model_name} | Accuracy: {acc:.3f} | F1 Score: {f1:.3f} | Saved to models/{model_name}.pkl"
        )

        # Store performance for comparison
        model_performance[model_name] = {
            "accuracy": acc,
            "f1": f1,
            "run_id": run.info.run_id,
        }


# Train models
train_and_log_model(LogisticRegression(max_iter=200), "LogisticRegression")
train_and_log_model(RandomForestClassifier(n_estimators=100), "RandomForest")

# Register the best model based on performance
print("\n[INFO] Model Performance Comparison:")
print("=" * 40)
for model_name, metrics in model_performance.items():
    print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

# Find the best model (higher accuracy and F1)
best_model_name = max(
    model_performance.keys(),
    key=lambda x: (model_performance[x]["accuracy"], model_performance[x]["f1"]),
)
best_metrics = model_performance[best_model_name]

print(f"\n[INFO] Best Model: {best_model_name}")
print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
print(f"   F1 Score: {best_metrics['f1']:.3f}")

# Register the best model
try:
    registered_model = mlflow.register_model(
        model_uri=f"runs:/{best_metrics['run_id']}/model", name="IrisClassifier"
    )
    print(
        f"[OK] Successfully registered 'IrisClassifier' model (version {registered_model.version})"
    )
except Exception as e:
    print(f"[WARN] IrisClassifier already registered or error: {e}")
