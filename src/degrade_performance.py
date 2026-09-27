import pandas as pd
import numpy as np

def inject_drift(file_path, noise_level=1.5):
    """Corrupts feature matrix values to force validation pipeline failure."""
    df = pd.read_csv(file_path)

    feature_cols = [col for col in df.columns if col != 'target']

    # Inject intense Gaussian noise to shift data distributions
    noise = np.random.normal(
        0,
        noise_level,
        size=(df.shape[0], len(feature_cols))
    )

    df[feature_cols] += noise

    df.to_csv("data/incoming_batch.csv", index=False)

    print("Successfully injected synthetic performance degradation into dataset.")

if __name__ == "__main__":
    inject_drift("data/production_data.csv")