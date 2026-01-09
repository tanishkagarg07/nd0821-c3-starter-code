import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "census.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def main():
    # Load data
    data = pd.read_csv(DATA_PATH)

    # Define categorical features
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

    # Split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    precision, recall, fbeta = compute_model_metrics(
        y_test, inference(model, X_test)
    )

    print("Model performance on test set:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {fbeta:.3f}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(lb, os.path.join(MODEL_DIR, "lb.pkl"))


if __name__ == "__main__":
    main()

