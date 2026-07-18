"""OSMnx-based geographic mapper for 4 Indian cities."""
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import DATA_DIR, GENERATED_DIR, SEED_TRAIN

logger = logging.getLogger(__name__)

CITIES = {
    "mumbai":    {"query": "Mumbai, Maharashtra, India",   "bbox": [18.87, 72.75, 19.27, 73.00]},
    "delhi":     {"query": "Delhi, India",                 "bbox": [28.40, 76.82, 28.89, 77.35]},
    "bangalore": {"query": "Bangalore, Karnataka, India",  "bbox": [12.83, 77.45, 13.14, 77.78]},
    "hyderabad": {"query": "Hyderabad, Telangana, India",  "bbox": [17.30, 78.40, 17.50, 78.55]},
}

GEO_CACHE_FILE = os.path.join(GENERATED_DIR, "city_geo_cache.json")


def _city_cache_path(city: str) -> str:
    return os.path.join(GENERATED_DIR, f"city_{city}.json")


def fetch_city_graph(city: str, use_cache: bool = True) -> dict:
    """
    Fetch road/water network graph for a city via OSMnx.
    Returns GeoJSON-style dict with nodes and edges.
    """
    cache_path = _city_cache_path(city)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        logger.info("Loaded city graph from cache: %s (%d nodes)", city, len(data["nodes"]))
        return data

    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("osmnx not installed. Run: pip install osmnx")

    city_conf = CITIES.get(city)
    if not city_conf:
        raise ValueError(f"Unknown city '{city}'. Available: {list(CITIES.keys())}")

    logger.info("Fetching OSMnx graph for %s...", city)
    start = time.time()
    try:
        G = ox.graph_from_place(city_conf["query"], network_type="drive", simplify=True)
    except Exception as e:
        logger.warning("graph_from_place failed for %s: %s. Falling back to bbox.", city, e)
        bb = city_conf["bbox"]
        G = ox.graph_from_bbox(bb[2], bb[0], bb[3], bb[1], network_type="drive")

    elapsed = time.time() - start
    logger.info("Fetched %s in %.1fs: %d nodes, %d edges", city, elapsed, len(G.nodes), len(G.edges))

    # Subsample to ≤500 nodes for performance
    nodes_list = list(G.nodes(data=True))
    if len(nodes_list) > 500:
        rng = np.random.default_rng(SEED_TRAIN)
        sampled_ids = set(rng.choice([n for n, _ in nodes_list], size=500, replace=False))
        G = G.subgraph(sampled_ids).copy()
        nodes_list = list(G.nodes(data=True))

    nodes_out = []
    for node_id, data in nodes_list:
        nodes_out.append({
            "id": str(node_id),
            "lat": data.get("y", 0.0),
            "lon": data.get("x", 0.0),
            "elevation": data.get("elevation", 0.0),
            "street_count": data.get("street_count", 0),
        })

    edges_out = []
    for u, v, data in G.edges(data=True):
        if str(u) in {n["id"] for n in nodes_out} and str(v) in {n["id"] for n in nodes_out}:
            edges_out.append({
                "source": str(u),
                "target": str(v),
                "length_m": data.get("length", 100.0),
                "name": data.get("name", "") if isinstance(data.get("name", ""), str) else "",
            })

    result = {
        "city": city,
        "n_nodes": len(nodes_out),
        "n_edges": len(edges_out),
        "bbox": city_conf["bbox"],
        "nodes": nodes_out,
        "edges": edges_out,
    }

    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Cached city graph: %s -> %s", city, cache_path)
    return result


def get_city_summary() -> dict:
    """Return cached city graph metadata without re-fetching."""
    summary = {}
    for city in CITIES:
        path = _city_cache_path(city)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            summary[city] = {
                "n_nodes": data["n_nodes"],
                "n_edges": data["n_edges"],
                "bbox": data["bbox"],
                "cached": True,
            }
        else:
            summary[city] = {
                "n_nodes": None,
                "n_edges": None,
                "bbox": CITIES[city]["bbox"],
                "cached": False,
            }
    return summary


def prefetch_all_cities() -> None:
    """Pre-fetch and cache all 4 cities."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    for city in CITIES:
        try:
            data = fetch_city_graph(city, use_cache=True)
            print(f"  {city}: {data['n_nodes']} nodes, {data['n_edges']} edges")
        except Exception as e:
            print(f"  {city}: ERROR - {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=list(CITIES.keys()) + ["all"], default="all")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.city == "all":
        prefetch_all_cities()
    else:
        data = fetch_city_graph(args.city, use_cache=not args.no_cache)
        print(f"{args.city}: {data['n_nodes']} nodes, {data['n_edges']} edges")
