import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ✅ Set tracking URI for MLflow UI (local directory)
mlflow.set_tracking_uri("file:///app/mlruns")  # or change to http://<tracking-server>:5000 for remote

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
signature = infer_signature(X_train, model.predict(X_train))

# 🔁 Start MLflow run
with mlflow.start_run() as run:
    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # ✅ Log and register model to the model registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="LinearRegressionModel"  # This registers to MLflow Model Registry
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model logged and registered with MSE: {mse}")
