import os
import pandas as pd
import joblib

from ml.data import process_data
from ml.model import inference, compute_model_metrics

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "census.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "slice_output.txt")


def main():
    data = pd.read_csv(DATA_PATH)

    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    lb = joblib.load(os.path.join(MODEL_DIR, "lb.pkl"))

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open(OUTPUT_PATH, "w") as f:
        for feature in cat_features:
            f.write(f"\n### Slice by {feature}\n")
            for value in data[feature].unique():
                slice_df = data[data[feature] == value]

                X, y, _, _ = process_data(
                    slice_df,
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )

                preds = inference(model, X)
                precision, recall, fbeta = compute_model_metrics(y, preds)

                f.write(
                    f"{feature} = {value} | "
                    f"Precision: {precision:.3f}, "
                    f"Recall: {recall:.3f}, "
                    f"F1: {fbeta:.3f}\n"
                )


if __name__ == "__main__":
    main()
