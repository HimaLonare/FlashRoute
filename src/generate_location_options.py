"""
generate_location_options.py
------------------------------
Samples a small set of REAL restaurants and their REAL, paired delivery
locations from the cleaned dataset, for use as Flask dropdown options.

Why PAIRED sampling, not independent restaurant/delivery lists:
Independently sampling a restaurant coordinate and a delivery
coordinate from two unrelated dataset rows can pair a restaurant in
one city with a delivery point hundreds of kilometers away -- every
real delivery in this dataset is a single-city trip of 1.5-21 km. An
out-of-distribution input like that would force the model to
extrapolate wildly and produce a meaningless ETA. Restaurants repeat
heavily across rows (388 unique restaurants, many with 100+ orders),
so we instead sample N restaurants and, for each, list only the
delivery locations that were ACTUALLY paired with it in real orders.

Why this exists at all: the raw dataset has no restaurant/customer
names -- only GPS coordinates. Rather than fabricating fake business
names (which would violate the "no synthetic data" requirement), we
sample actual coordinate pairs and label them generically. A real
production system would replace this with a restaurant database +
geocoding service; this is documented as a known simplification.
"""

import json

import pandas as pd

N_RESTAURANTS = 10
MAX_DELIVERIES_PER_RESTAURANT = 4
RANDOM_STATE = 42


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_delivery_data.csv")
    df = df[df["is_valid_location"]]  # exclude placeholder (0,0) coordinates

    restaurant_groups = list(
        df.groupby(["Restaurant_latitude", "Restaurant_longitude"])
    )

    # Prefer restaurants with several distinct paired delivery points, so
    # the dropdown offers real variety instead of one option each.
    restaurant_groups.sort(key=lambda g: g[1]["Delivery_location_latitude"].nunique(), reverse=True)
    sampled_groups = restaurant_groups[:N_RESTAURANTS]

    restaurants = []
    for i, ((rest_lat, rest_lon), rows) in enumerate(sampled_groups, start=1):
        deliveries_for_this_restaurant = (
            rows[["Delivery_location_latitude", "Delivery_location_longitude"]]
            .drop_duplicates()
            .sample(n=min(MAX_DELIVERIES_PER_RESTAURANT, rows["Delivery_location_latitude"].nunique()),
                     random_state=RANDOM_STATE)
        )

        delivery_options = [
            {
                "label": f"Location {i}.{j} ({d_lat:.4f}, {d_lon:.4f})",
                "lat": round(float(d_lat), 6),
                "lon": round(float(d_lon), 6),
            }
            for j, (_, (d_lat, d_lon)) in enumerate(deliveries_for_this_restaurant.iterrows(), start=1)
        ]

        restaurants.append(
            {
                "label": f"Restaurant {i} ({rest_lat:.4f}, {rest_lon:.4f})",
                "lat": round(float(rest_lat), 6),
                "lon": round(float(rest_lon), 6),
                "delivery_options": delivery_options,
            }
        )

    with open("models/locations.json", "w") as f:
        json.dump({"restaurants": restaurants}, f, indent=2)

    total_deliveries = sum(len(r["delivery_options"]) for r in restaurants)
    print(f"Saved {len(restaurants)} restaurants with {total_deliveries} real paired "
          f"delivery locations -> models/locations.json")
