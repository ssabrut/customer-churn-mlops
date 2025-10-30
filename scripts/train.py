import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory (your project root)
project_root = os.path.dirname(script_dir)

# Add the project root to the system path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import mlflow.xgboost
import time
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from core import constant

MLFLOW_S3_ENDPOINT_URL = os.environ.get(
    "MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9090"
)

timestamp = time.time()

mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment(f"churn_prediction_{timestamp}")

def run_training():
    print("Loading data...")
    df = pd.read_csv("data/preprocessed/train.csv")

    X = df.drop(constant.TARGET, axis=1)
    y = df[constant.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=24
    )

    print("Building model...")
    xgb = XGBClassifier()

    with mlflow.start_run() as run:
        print("Training model...")
        xgb.fit(X_train, y_train)

        print("Evaluating model...")
        yhat = xgb.predict(X_test)
        f1 = f1_score(y_test, yhat)

        print(f"F1 Score: {f1:.4f}")

        print("Logging parameters...")
        mlflow.log_params(xgb.get_params())

        print("Logging metrics...")
        mlflow.log_metric("f1_score", f1)

        print("Logging XGBoost model...")
        mlflow.xgboost.log_model(
            xgb_model=xgb,
            name="model",
            registered_model_name="churn-predictor-xgb",
            input_example=X_train.values[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5]
        )

        print(f"Run ID: {run.info.run_id}")
        print("Training complete.")


if __name__ == "__main__":
    run_training()
