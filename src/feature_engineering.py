"""
feature_engineering.py
-----------------------
Builds model-ready features from the cleaned FlashRoute delivery data.

Assumes the input DataFrame has already been through
data_cleaning.clean_delivery_data() (sign-fixed coordinates, stripped
categoricals, clipped ratings).
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/lon points.

    Why Haversine and not Euclidean distance on raw lat/lon:
    latitude and longitude are angles on a sphere, not points on a flat
    plane. One degree of longitude covers a different physical distance
    depending on latitude (it shrinks toward the poles). Haversine
    accounts for the Earth's curvature, so it stays accurate anywhere.

    Formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2 · atan2(√a, √(1−a))
        distance = R · c
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add Distance_km computed via Haversine formula."""
    df = df.copy()
    df["Distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    return df


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket Distance_km into short / medium / long.

    Why: tree-based models (Random Forest, XGBoost) can already learn
    non-linear thresholds on their own, but explicitly surfacing a
    bucket can help a *linear* model capture non-linearity it otherwise
    couldn't, and makes the EDA/interview story ("most deliveries are
    short-to-medium range") easier to communicate.
    """
    df = df.copy()
    df["Distance_bucket"] = pd.cut(
        df["Distance_km"],
        bins=[0, 5, 10, np.inf],
        labels=["short", "medium", "long"],
    )
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket driver age into groups.

    Why: age's relationship to delivery time may not be perfectly
    linear (e.g. very new/young drivers and much older drivers might
    both be slightly slower for different reasons). Bucketing lets
    simpler models pick up on this without assuming linearity.
    """
    df = df.copy()
    df["Age_group"] = pd.cut(
        df["Delivery_person_Age"],
        bins=[0, 25, 35, np.inf],
        labels=["young", "mid", "senior"],
    )
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns used for modeling.

    Why one-hot and not label/ordinal encoding: Type_of_order,
    Type_of_vehicle, Distance_bucket, and Age_group are NOMINAL
    categories -- there's no inherent order between 'motorcycle' and
    'bicycle'. Label-encoding them as 0/1/2/3 would incorrectly imply
    an ordinal relationship the model might learn to exploit.

    drop_first=True avoids the dummy variable trap (perfect
    multicollinearity) for linear models, at a small cost of losing a
    baseline category name -- acceptable since we're comparing
    against tree models too, which don't care about collinearity.
    """
    categorical_cols = [
        "Type_of_order",
        "Type_of_vehicle",
        "Distance_bucket",
        "Age_group",
    ]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in order."""
    df = add_distance_feature(df)
    df = add_distance_bucket(df)
    df = add_age_group(df)
    df = one_hot_encode(df)
    return df


if __name__ == "__main__":
    in_path = "data/processed/cleaned_delivery_data.csv"
    out_path = "data/processed/featured_delivery_data.csv"

    cleaned_df = pd.read_csv(in_path)
    featured_df = build_features(cleaned_df)
    featured_df.to_csv(out_path, index=False)

    print(f"Engineered features for {len(featured_df)} rows -> saved to {out_path}")
    print(f"Final columns: {list(featured_df.columns)}")
