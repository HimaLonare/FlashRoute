"""
model_training.py
-------------------
Trains and compares three regression models for delivery ETA prediction:
Linear Regression, Random Forest, and XGBoost.

Design choice: a single `train_and_evaluate_all()` function returns a
results dict + fitted models, so this module can be imported both by a
notebook (for exploration/plots) and by the Flask app (to retrain on
demand) without duplicating logic.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

TARGET_COLUMN = "Time_taken(min)"

# Columns that must NOT be used as model input.
#   - raw lat/lon: their signal is already captured by Distance_km;
#     feeding raw coordinates would let the model memorize specific
#     restaurant locations instead of learning general distance effects.
#   - is_valid_location: a data-quality flag, not a real-world predictor
#     of delivery time.
LEAKY_OR_UNUSED_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "is_valid_location",
]


def load_model_ready_data(path: str):
    """Load the featured dataset and split into X (features) / y (target)."""
    df = pd.read_csv(path)
    drop_cols = [c for c in LEAKY_OR_UNUSED_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def get_models():
    """Return the three candidate models with sensible, non-tuned defaults.

    We deliberately do NOT hyperparameter-tune yet -- comparing models on
    reasonable defaults first tells us which algorithm family is worth
    investing tuning effort into. Tuning a weak algorithm first is a
    common beginner mistake (wasted effort).
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R² for a set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate_all(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Train all candidate models and return fitted models + metrics.

    Returns
    -------
    fitted_models : dict[str, estimator]
    results : dict[str, dict]  -- metrics per model name
    split : tuple -- (X_train, X_test, y_train, y_test) for downstream use
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    fitted_models = {}
    results = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = evaluate(y_test, preds)
        fitted_models[name] = model

    return fitted_models, results, (X_train, X_test, y_train, y_test)


def save_best_model(fitted_models: dict, results: dict, out_path: str) -> str:
    """Save the model with the lowest RMSE (our primary selection metric).

    Why RMSE as the tie-breaker over MAE or R²: it penalizes large
    errors more, and for a delivery ETA product, being wildly wrong on
    a few orders (customer sees "5 min" and waits 40) is worse than
    being consistently a little off -- RMSE reflects that cost better.
    """
    best_name = min(results, key=lambda name: results[name]["RMSE"])
    joblib.dump(fitted_models[best_name], out_path)
    return best_name


if __name__ == "__main__":
    X, y = load_model_ready_data("data/processed/featured_delivery_data.csv")
    fitted_models, results, split = train_and_evaluate_all(X, y)

    print("Model comparison (on held-out test set):")
    for name, metrics in results.items():
        print(
            f"  {name:20s}  MAE={metrics['MAE']:.3f}  "
            f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}"
        )

    best_name = save_best_model(fitted_models, results, "models/best_model.joblib")
    print(f"\nBest model by RMSE: {best_name} -> saved to models/best_model.joblib")

    # Persist the exact feature column order -- the Flask app must build
    # its input row with these columns in this exact order at inference time.
    joblib.dump(list(X.columns), "models/feature_columns.joblib")
