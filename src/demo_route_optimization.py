"""
demo_route_optimization.py
----------------------------
Demonstrates and verifies route_optimization.py's Dijkstra/A* functions
on a synthetic street-grid graph, built in the exact schema OSMnx
produces (MultiGraph, node attrs 'y'/'x', edge attr 'length' in meters).

Why this file exists: build_street_graph() needs live internet access
to download real OpenStreetMap data. This script lets you verify the
ALGORITHM LOGIC is correct without that dependency -- useful for CI,
offline development, or (as here) a sandboxed environment. Swap in a
real graph from build_street_graph() and every downstream function
(dijkstra_route, astar_route, compare_routes) works unchanged.
"""

import random

import networkx as nx

from src.feature_engineering import haversine_distance
from src.route_optimization import compare_routes


def build_synthetic_grid(size: int, base_lat: float, base_lon: float, step_deg: float = 0.001):
    """Build a size x size street grid as a MultiGraph matching OSMnx's schema.

    A handful of random long-range edges are added so the grid isn't
    perfectly uniform (a pure grid makes every shortest path equally
    "easy", which understates how much A* helps on irregular real
    street networks).
    """
    grid = nx.MultiGraph(nx.grid_2d_graph(size, size))
    mapping = {(i, j): i * size + j for (i, j) in list(grid.nodes())}
    grid = nx.relabel_nodes(grid, mapping)

    for (i, j), node_id in mapping.items():
        grid.nodes[node_id]["y"] = base_lat + i * step_deg
        grid.nodes[node_id]["x"] = base_lon + j * step_deg

    for u, v, k in list(grid.edges(keys=True)):
        y1, x1 = grid.nodes[u]["y"], grid.nodes[u]["x"]
        y2, x2 = grid.nodes[v]["y"], grid.nodes[v]["x"]
        grid[u][v][k]["length"] = haversine_distance(y1, x1, y2, x2) * 1000

    random.seed(42)
    for _ in range(size):
        u, v = random.sample(list(grid.nodes()), 2)
        if not grid.has_edge(u, v):
            y1, x1 = grid.nodes[u]["y"], grid.nodes[u]["x"]
            y2, x2 = grid.nodes[v]["y"], grid.nodes[v]["x"]
            grid.add_edge(u, v, length=haversine_distance(y1, x1, y2, x2) * 1000 * 1.3)

    return grid, mapping


if __name__ == "__main__":
    # Centered on a real restaurant coordinate from the dataset.
    G, mapping = build_synthetic_grid(size=20, base_lat=22.745049, base_lon=75.892471)
    source, target = mapping[(0, 0)], mapping[(19, 19)]

    result = compare_routes(G, source, target)

    print(f"Graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(
        f"Dijkstra -> distance: {result['dijkstra_distance_m']:.1f} m, "
        f"time: {result['dijkstra_time_sec']*1000:.4f} ms"
    )
    print(
        f"A*       -> distance: {result['astar_distance_m']:.1f} m, "
        f"time: {result['astar_time_sec']*1000:.4f} ms"
    )
    print(f"Paths identical (correctness check): {result['paths_match']}")
