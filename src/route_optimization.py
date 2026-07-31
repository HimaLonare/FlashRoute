"""
route_optimization.py
-----------------------
Route optimization using Dijkstra's Algorithm and A* Search on a real
street network fetched via OSMnx, with results visualized on an
interactive Folium map.

All core functions (`dijkstra_route`, `astar_route`, `compare_routes`)
operate on any NetworkX graph that follows OSMnx's node/edge schema:
    - node attributes: 'y' (latitude), 'x' (longitude)
    - edge attribute:  'length' (meters)
This means they work identically whether the graph came from
`build_street_graph()` (real OSM data) or a hand-built test graph.
"""

import time

import networkx as nx

from src.feature_engineering import haversine_distance

# OSMnx is only needed for live map downloads and Folium plotting -- the
# core Dijkstra/A* logic below works on ANY networkx graph with the
# right node/edge schema, OSMnx-sourced or not. Importing it lazily
# (inside the functions that need it) keeps this module usable in
# environments where OSMnx/network access isn't available, and makes
# the dependency boundary explicit.


# ---------------------------------------------------------------------
# 1. Graph construction (real street network via OSMnx)
# ---------------------------------------------------------------------

def build_street_graph(center_lat: float, center_lon: float, dist_meters: int = 3000):
    """Download a real drivable street network around a center point.

    Why `network_type='drive'`: FlashRoute optimizes routes for delivery
    vehicles (motorcycle/scooter/bicycle-ish), so we want the drivable
    road graph, not footpaths or highways-only.

    `dist_meters` controls how large an area to fetch -- large enough to
    contain both the restaurant and delivery location, small enough to
    keep the graph fast to search. In production you'd compute this
    dynamically from the actual restaurant/delivery coordinates.
    """
    import osmnx as ox

    graph = ox.graph_from_point(
        (center_lat, center_lon), dist=dist_meters, network_type="drive"
    )
    return graph


def get_nearest_node(graph, lat: float, lon: float):
    """Snap a raw (lat, lon) coordinate to the nearest graph node.

    Why this is needed: a restaurant's GPS coordinate almost never lands
    exactly on a graph node (an intersection) -- we need the closest
    routable point on the road network to start/end the search from.
    """
    import osmnx as ox

    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)


# ---------------------------------------------------------------------
# 2. Shortest-path algorithms
# ---------------------------------------------------------------------

def _haversine_heuristic(graph):
    """Build an A* heuristic function bound to this graph's coordinates.

    Why Haversine and not Euclidean here too: same reasoning as the ETA
    feature -- node coordinates are lat/lon (spherical), so straight-line
    distance must account for Earth's curvature to stay an admissible
    (never-overestimating) heuristic.
    """

    def heuristic(node_a, node_b):
        y1, x1 = graph.nodes[node_a]["y"], graph.nodes[node_a]["x"]
        y2, x2 = graph.nodes[node_b]["y"], graph.nodes[node_b]["x"]
        # Haversine returns km; convert to meters to match edge 'length' units.
        return haversine_distance(y1, x1, y2, x2) * 1000

    return heuristic


def dijkstra_route(graph, source_node, target_node):
    """Find the shortest path by road distance using Dijkstra's algorithm.

    Returns (path_as_node_list, elapsed_seconds).
    """
    start = time.perf_counter()
    path = nx.dijkstra_path(graph, source_node, target_node, weight="length")
    elapsed = time.perf_counter() - start
    return path, elapsed


def astar_route(graph, source_node, target_node):
    """Find the shortest path using A* search with a Haversine heuristic.

    Returns (path_as_node_list, elapsed_seconds).
    """
    heuristic = _haversine_heuristic(graph)
    start = time.perf_counter()
    path = nx.astar_path(
        graph, source_node, target_node, heuristic=heuristic, weight="length"
    )
    elapsed = time.perf_counter() - start
    return path, elapsed


def path_length_meters(graph, path) -> float:
    """Total road distance (meters) along a path of nodes."""
    return sum(
        graph[u][v][0]["length"] for u, v in zip(path[:-1], path[1:])
    )


def compare_routes(graph, source_node, target_node) -> dict:
    """Run both algorithms and return a side-by-side comparison.

    Why compare paths for equality, not just timing: this is the crux
    interview point -- Dijkstra and A* should find the SAME optimal
    path (given an admissible heuristic); the only difference should be
    how many nodes each explores / how fast it runs. If the paths
    differ, either the heuristic is inadmissible (a bug) or the graph
    has multiple equal-length shortest paths (a legitimate tie).
    """
    dijkstra_path, dijkstra_time = dijkstra_route(graph, source_node, target_node)
    astar_path, astar_time = astar_route(graph, source_node, target_node)

    return {
        "dijkstra_path": dijkstra_path,
        "dijkstra_time_sec": dijkstra_time,
        "dijkstra_distance_m": path_length_meters(graph, dijkstra_path),
        "astar_path": astar_path,
        "astar_time_sec": astar_time,
        "astar_distance_m": path_length_meters(graph, astar_path),
        "paths_match": dijkstra_path == astar_path,
    }


# ---------------------------------------------------------------------
# 3. Visualization (Folium)
# ---------------------------------------------------------------------

def plot_route_on_map(graph, route, out_html_path: str, route_color: str = "#2a78d6"):
    """Render a route on an interactive Folium map and save it as HTML.

    Uses OSMnx's built-in Folium plotting helper, which draws the graph
    edges of the route as a polyline on an OpenStreetMap basemap.
    """
    import osmnx as ox

    route_map = ox.plot_route_folium(graph, route, route_color=route_color, route_width=5)
    route_map.save(out_html_path)
    return route_map


if __name__ == "__main__":
    # Example usage (requires network access to fetch OSM data):
    RESTAURANT = (22.745049, 75.892471)
    DELIVERY = (22.765049, 75.912471)

    center_lat = (RESTAURANT[0] + DELIVERY[0]) / 2
    center_lon = (RESTAURANT[1] + DELIVERY[1]) / 2

    G = build_street_graph(center_lat, center_lon, dist_meters=3000)
    orig_node = get_nearest_node(G, *RESTAURANT)
    dest_node = get_nearest_node(G, *DELIVERY)

    result = compare_routes(G, orig_node, dest_node)
    print(
        f"Dijkstra: {result['dijkstra_distance_m']:.0f} m in "
        f"{result['dijkstra_time_sec']*1000:.3f} ms"
    )
    print(
        f"A*:       {result['astar_distance_m']:.0f} m in "
        f"{result['astar_time_sec']*1000:.3f} ms"
    )
    print(f"Paths match: {result['paths_match']}")

    plot_route_on_map(G, result["astar_path"], "screenshots/route_map.html")
