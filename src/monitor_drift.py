import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def monitor_model_performance(
    model_path="models/champion_model.pkl",
    data_path="data/incoming_batch.csv"
):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    print(f"Current Production Accuracy: {accuracy:.4f}")

    if accuracy < 0.80:
        print("Drift detected. Retraining required.")
    else:
        print("Model performance is stable.")

    return accuracy


if __name__ == "__main__":
    monitor_model_performance()