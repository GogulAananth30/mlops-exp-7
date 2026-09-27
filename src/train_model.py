import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def execute_training(data_path, output_model_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Retrained Model Production Accuracy evaluated: {acc:.4f}")

    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)

    return acc


if __name__ == "__main__":
    execute_training(
        "data/production_data.csv",
        "models/champion_model.pkl"
    )