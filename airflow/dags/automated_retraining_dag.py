from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.monitor_drift import monitor_model_performance
from src.train_model import execute_training


def check_drift():
    accuracy = monitor_model_performance()

    if accuracy < 0.80:
        print("Drift detected. Retraining will start.")
    else:
        print("No drift detected.")


def retrain_model():
    execute_training(
        "data/incoming_batch.csv",
        "models/staging_model.pkl"
    )


with DAG(
    dag_id="mlops_exp7_retraining_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "retraining"],
) as dag:

    monitor_task = PythonOperator(
        task_id="monitor_model",
        python_callable=check_drift,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    monitor_task >> retrain_task