"""
app.py
-------
FlashRoute Flask application: predicts delivery ETA and computes an
optimized route between a restaurant and delivery location.

Design notes:
    - The model, feature-column order, average driver profile, and
      location dropdown options are all loaded ONCE at startup
      (module-level), not per-request -- re-deserializing a model from
      disk on every request would be a real performance bug.
    - Route optimization tries a live OSMnx street-graph lookup first;
      if that fails (no network, OSM servers down, rate limited), the
      app degrades gracefully to a straight-line Folium map rather than
      crashing the whole prediction.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

from src.feature_engineering import haversine_distance

app = Flask(__name__)

MODEL_DIR = "models"
STATIC_MAPS_DIR = os.path.join("static", "maps")
os.makedirs(STATIC_MAPS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load model artifacts ONCE at startup
# ---------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
FEATURE_COLUMNS = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
AVG_DRIVER_PROFILE = joblib.load(os.path.join(MODEL_DIR, "avg_driver_profile.joblib"))

with open(os.path.join(MODEL_DIR, "locations.json")) as f:
    LOCATIONS = json.load(f)

VEHICLE_TYPES = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
ORDER_TYPES = ["Snack", "Drinks", "Buffet", "Meal"]


# ---------------------------------------------------------------------
# Feature construction (must mirror src/feature_engineering.py exactly)
# ---------------------------------------------------------------------

def build_feature_row(distance_km: float, vehicle_type: str, order_type: str) -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's training schema.

    Why we rebuild features by hand instead of re-running the full
    feature_engineering.build_features() pipeline: that pipeline expects
    a full dataset with raw lat/lon columns to compute buckets/dummies
    across many rows. At inference time we only have ONE data point, so
    we replicate the same bucketing/encoding logic directly -- but the
    THRESHOLDS (5km/10km for distance, 25/35 for age) are kept in sync
    with feature_engineering.py so behavior never silently diverges.
    """
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Delivery_person_Age"] = AVG_DRIVER_PROFILE["Delivery_person_Age"]
    row["Delivery_person_Ratings"] = AVG_DRIVER_PROFILE["Delivery_person_Ratings"]
    row["Distance_km"] = distance_km

    # Type_of_order one-hot (Buffet is the dropped baseline category)
    order_col = f"Type_of_order_{order_type}"
    if order_col in row:
        row[order_col] = 1

    # Type_of_vehicle one-hot (bicycle is the dropped baseline category)
    vehicle_col = f"Type_of_vehicle_{vehicle_type}"
    if vehicle_col in row:
        row[vehicle_col] = 1

    # Distance_bucket one-hot (short <5km is the dropped baseline)
    if distance_km > 10:
        row["Distance_bucket_long"] = 1
    elif distance_km > 5:
        row["Distance_bucket_medium"] = 1

    # Age_group one-hot -- using the average driver's age bucket
    avg_age = AVG_DRIVER_PROFILE["Delivery_person_Age"]
    if avg_age > 35:
        row["Age_group_senior"] = 1
    elif avg_age > 25:
        row["Age_group_mid"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------
# Route optimization with graceful fallback
# ---------------------------------------------------------------------

def compute_route_map(restaurant, delivery, map_filename: str) -> dict:
    """Try a real OSMnx street-network route; fall back to a straight line.

    Returns a dict describing which method succeeded, so the UI can be
    transparent with the user about what they're seeing.
    """
    try:
        from src.route_optimization import (
            build_street_graph,
            get_nearest_node,
            compare_routes,
            plot_route_on_map,
        )

        center_lat = (restaurant["lat"] + delivery["lat"]) / 2
        center_lon = (restaurant["lon"] + delivery["lon"]) / 2
        graph = build_street_graph(center_lat, center_lon, dist_meters=5000)

        orig_node = get_nearest_node(graph, restaurant["lat"], restaurant["lon"])
        dest_node = get_nearest_node(graph, delivery["lat"], delivery["lon"])

        result = compare_routes(graph, orig_node, dest_node)
        plot_route_on_map(graph, result["astar_path"], os.path.join(STATIC_MAPS_DIR, map_filename))

        return {
            "method": "osmnx_street_network",
            "distance_m": result["astar_distance_m"],
            "dijkstra_time_ms": result["dijkstra_time_sec"] * 1000,
            "astar_time_ms": result["astar_time_sec"] * 1000,
        }

    except Exception as exc:
        # Graceful degradation: no network / OSMnx unavailable / OSM
        # server error should not crash the prediction request.
        import folium

        fallback_map = folium.Map(
            location=[
                (restaurant["lat"] + delivery["lat"]) / 2,
                (restaurant["lon"] + delivery["lon"]) / 2,
            ],
            zoom_start=13,
        )
        folium.Marker([restaurant["lat"], restaurant["lon"]], tooltip="Restaurant", icon=folium.Icon(color="red")).add_to(fallback_map)
        folium.Marker([delivery["lat"], delivery["lon"]], tooltip="Delivery location", icon=folium.Icon(color="green")).add_to(fallback_map)
        folium.PolyLine(
            [[restaurant["lat"], restaurant["lon"]], [delivery["lat"], delivery["lon"]]],
            color="#2a78d6",
            weight=4,
        ).add_to(fallback_map)
        fallback_map.save(os.path.join(STATIC_MAPS_DIR, map_filename))

        return {
            "method": "straight_line_fallback",
            "error": str(exc),
        }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        restaurants=LOCATIONS["restaurants"],
        vehicle_types=VEHICLE_TYPES,
        order_types=ORDER_TYPES,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    restaurant_idx = int(request.form["restaurant"])
    delivery_idx = int(request.form["delivery_location"])
    vehicle_type = request.form["vehicle_type"]
    order_type = request.form["order_type"]

    restaurant = LOCATIONS["restaurants"][restaurant_idx]
    delivery = restaurant["delivery_options"][delivery_idx]

    distance_km = haversine_distance(
        restaurant["lat"], restaurant["lon"], delivery["lat"], delivery["lon"]
    )

    feature_row = build_feature_row(distance_km, vehicle_type, order_type)
    predicted_eta = float(model.predict(feature_row)[0])

    map_filename = "route_map.html"
    route_info = compute_route_map(restaurant, delivery, map_filename)

    prediction = {
        "eta_minutes": round(predicted_eta, 1),
        "distance_km": round(distance_km, 2),
        "restaurant_label": restaurant["label"],
        "delivery_label": delivery["label"],
        "vehicle_type": vehicle_type,
        "order_type": order_type,
        "route_method": route_info["method"],
        "map_url": f"/{STATIC_MAPS_DIR}/{map_filename}",
    }

    return render_template(
        "index.html",
        restaurants=LOCATIONS["restaurants"],
        vehicle_types=VEHICLE_TYPES,
        order_types=ORDER_TYPES,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
