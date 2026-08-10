FROM python:3.12-slim

# build-essential: fallback in case any dependency needs to compile from
# source instead of using a prebuilt wheel (shapely/pyproj/pyogrio ship
# manylinux wheels with bundled GEOS/PROJ/GDAL, but this is a cheap safety net).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# data/processed/*.csv and models/locations.json are gitignored (generated
# artifacts, not raw data) -- regenerate them at build time from the raw
# CSV that IS committed (data/raw/deliverytime.csv). generate_location_options.py
# uses a fixed random_state, so this is fully deterministic.
RUN python src/data_cleaning.py && python src/generate_location_options.py

EXPOSE 5000

# Render/Railway/most PaaS inject $PORT; default to 5000 for local `docker run`.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120
