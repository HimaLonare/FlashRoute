"""
data_cleaning.py
-----------------
Cleans the raw FlashRoute delivery dataset.

Responsibilities (and ONLY these — feature engineering lives in
feature_engineering.py, keeping each module single-purpose):
    1. Strip whitespace from categorical text columns.
    2. Fix sign-flipped GPS coordinates (negative lat/lon in India).
    3. Clip out-of-range delivery ratings to a valid 1.0-5.0 scale.
    4. Flag (but do not drop) rows with placeholder (0, 0) coordinates,
       so downstream steps can decide whether to use or exclude them.
    5. Drop identifier columns that don't generalize (Delivery_person_ID).

Design choice: every function takes a DataFrame and returns a DataFrame.
No function reads/writes files directly — that keeps these functions
unit-testable and reusable inside the Flask app later.
"""

import pandas as pd


CATEGORICAL_COLUMNS = ["Type_of_order", "Type_of_vehicle"]

COORDINATE_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

RATING_MIN, RATING_MAX = 1.0, 5.0


def strip_categorical_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from categorical text columns.

    Why: raw values like 'Snack ' and 'motorcycle ' would otherwise be
    treated as distinct categories from 'Snack' and 'motorcycle' during
    one-hot encoding, silently doubling category counts.
    """
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].str.strip()
    return df


def fix_coordinate_signs(df: pd.DataFrame) -> pd.DataFrame:
    """Take the absolute value of all latitude/longitude columns.

    Why: every delivery in this dataset originates in India, where valid
    latitude/longitude values are always positive. A negative sign is a
    GPS/geocoding bug, not a legitimate location on the opposite
    hemisphere. Using abs() recovers the row instead of discarding
    otherwise-valid data.
    """
    df = df.copy()
    for col in COORDINATE_COLUMNS:
        df[col] = df[col].abs()
    return df


def clip_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Clip Delivery_person_Ratings to the valid [1.0, 5.0] scale.

    Why: the rating system is defined as 1-5 stars. A handful of rows
    have 6.0, which is an impossible/typo value. Clipping (rather than
    dropping) preserves the row's other valid information.
    """
    df = df.copy()
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(
        lower=RATING_MIN, upper=RATING_MAX
    )
    return df


def flag_invalid_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean `is_valid_location` flag for placeholder (0, 0) coords.

    Why: (0, 0) is "null island" -- a common sentinel value from failed
    geocoding, not a real restaurant in India. The *relative* distance
    computed from these rows is still numerically plausible (so we keep
    them for ETA regression), but real-world mapping/route-optimization
    steps (OSMnx, Folium) must exclude them since they don't correspond
    to an actual place on the map.
    """
    df = df.copy()
    is_zero_island = (df["Restaurant_latitude"] == 0) & (
        df["Restaurant_longitude"] == 0
    )
    df["is_valid_location"] = ~is_zero_island
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-cardinality identifier columns not useful for modeling.

    Why: Delivery_person_ID has 1,320 unique values. One-hot encoding it
    would create 1,320 sparse columns and encourage the model to
    memorize individual drivers rather than learn generalizable
    patterns (overfitting). `ID` is a row identifier with no predictive
    signal at all.
    """
    df = df.copy()
    cols_to_drop = [c for c in ["ID", "Delivery_person_ID"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clean_delivery_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the correct order.

    Order matters: coordinate sign-fixing must happen before we compute
    any distance-based feature (in feature_engineering.py), and the
    invalid-location flag must be computed on already-sign-fixed
    coordinates so it isn't confused by a (0, 0) that used to be
    (-0, -0) or similar.
    """
    df = strip_categorical_whitespace(df)
    df = fix_coordinate_signs(df)
    df = clip_ratings(df)
    df = flag_invalid_locations(df)
    df = drop_identifier_columns(df)
    return df


if __name__ == "__main__":
    raw_path = "data/raw/deliverytime.csv"
    out_path = "data/processed/cleaned_delivery_data.csv"

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_delivery_data(raw_df)
    cleaned_df.to_csv(out_path, index=False)

    print(f"Cleaned {len(raw_df)} rows -> saved to {out_path}")
    print(f"Rows flagged invalid location: {(~cleaned_df['is_valid_location']).sum()}")
