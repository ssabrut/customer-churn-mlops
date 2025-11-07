import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from loguru import logger

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from core import constant

MLFLOW_S3_ENDPOINT_URL = os.environ.get(
    "MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9002"
)

timestamp = time.time()

mlflow.xgboost.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5050")
mlflow.set_experiment(f"churn_prediction_{timestamp}")


def run_training():
    logger.info("Loading data...")
    df = pd.read_csv("data/preprocessed/train.csv")

    X = df.drop(constant.TARGET, axis=1)
    y = df[constant.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    with mlflow.start_run() as run:
        params = {"objective": "binary:logistic", "random_state": 42}

        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}

        logger.info("Building model...")
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=eval_results,
            verbose_eval=5
        )

        logger.info("Eval result:")
        logger.debug(str(eval_results))

        logger.info("Logging training metric...")
        for epoch, (train_metrics, test_metrics) in enumerate(
            zip(eval_results["train"]["logloss"], eval_results["test"]["logloss"])
        ):
            logger.debug(f"Epoch: {epoch} - train_logloss: {train_metrics} - test_logloss: {test_metrics}")
            mlflow.log_metrics(
                {"train_logloss": train_metrics, "test_logloss": test_metrics}, step=epoch
            )

        logger.info("Evaluating model...")
        yhat_proba = model.predict(dtest)
        yhat = (yhat_proba > 0.5).astype(int)

        final_metrics = {
            "accuracy": accuracy_score(y_test, yhat),
            "f1_score": f1_score(y_test, yhat),
            "roc_auc": roc_auc_score(y_test, yhat_proba),
        }

        logger.info("Evaluating final metric...")
        mlflow.log_metrics(final_metrics)

        logger.info("Logging model...")
        signature = infer_signature(X_train, yhat_proba)

        logger.info("Signature:")
        logger.debug(signature)
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            registered_model_name="XGBoostChurnModel",
            input_example=X_train[:5],
            model_format="json"
        )


if __name__ == "__main__":
    run_training()
