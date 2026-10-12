"""
India Water Distribution Network generator.

Builds synthetic WDNs for Indian cities from OSMnx POIs.
Falls back to a deterministic template when OSMnx is unavailable.
Each network has at least `min_nodes` nodes covering all important place types.
"""
import heapq
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import GENERATED_DIR, MIN_PRESSURE_M, REQUIRED_PRESSURE_M, SEED_TRAIN, NODE_TYPE_WEIGHTS

logger = logging.getLogger(__name__)

# ── City catalogue ────────────────────────────────────────────────────────────
INDIA_CITIES = {
    "mumbai":     {"name": "Mumbai",     "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "radius": 2200, "district": "Mumbai City"},
    "delhi":      {"name": "New Delhi",  "state": "Delhi",       "lat": 28.6139, "lon": 77.2090, "radius": 2500, "district": "Central Delhi"},
    "bangalore":  {"name": "Bengaluru",  "state": "Karnataka",   "lat": 12.9716, "lon": 77.5946, "radius": 2000, "district": "Bengaluru Urban"},
    "hyderabad":  {"name": "Hyderabad",  "state": "Telangana",   "lat": 17.3850, "lon": 78.4867, "radius": 2000, "district": "Hyderabad"},
}

# Demand by node type (m³/s)
DEMANDS = {
    "hospital":      0.025,
    "fire_station":  0.040,
    "school":        0.012,
    "college":       0.015,
    "park":          0.002,
    "market":        0.008,
    "residential":   0.004,
    "junction":      0.003,
    "reservoir":     0.0,
    "water_tower":   0.0,
}

# Priority weights come from backend.config.NODE_TYPE_WEIGHTS (single source).
PRIORITY_WEIGHTS = NODE_TYPE_WEIGHTS

# Pipe attributes by class
PIPE_ATTRS = {
    "main":      {"diameter": 0.40, "roughness": 130, "C_hw": 130},
    "secondary": {"diameter": 0.25, "roughness": 120, "C_hw": 120},
    "local":     {"diameter": 0.15, "roughness": 110, "C_hw": 110},
    "branch":    {"diameter": 0.10, "roughness": 100, "C_hw": 100},
}

# Minimum guaranteed POIs per type in synthetic fallback
GUARANTEED_POIS = {
    "hospital":     2,
    "fire_station": 2,
    "school":       3,
    "college":      1,
    "market":       2,
    "park":         2,
}


def _cache_path(city: str, n_nodes: int = 0) -> str:
    suffix = f"_n{n_nodes}" if n_nodes > 0 else ""
    return os.path.join(GENERATED_DIR, f"india_wdn_{city}{suffix}.json")


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _assign_elevation(dlat: float, dlon: float, rng: np.random.Generator) -> float:
    return max(5.0, 30.0 + dlat * 220 + dlon * 110 + rng.uniform(-3, 3))


def get_city_list() -> list:
    result = []
    for key, cfg in INDIA_CITIES.items():
        cached = os.path.exists(_cache_path(key))
        result.append({
            "city": key,
            "name": cfg["name"],
            "state": cfg["state"],
            "district": cfg.get("district", cfg["name"]),
            "lat": cfg["lat"],
            "lon": cfg["lon"],
            "cached": cached,
        })
    return result


def fetch_city_pois(city: str, n_nodes: int = 20, force_regen: bool = False) -> dict:
    """
    Fetch or generate WDN for an Indian city.
    n_nodes: target number of demand nodes (min 15, max 60).
    """
    cfg = INDIA_CITIES.get(city)
    if not cfg:
        raise ValueError(f"Unknown city: {city}")

    n_nodes = max(15, min(60, n_nodes))
    cache = _cache_path(city, n_nodes)

    if not force_regen and os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        logger.info("Loaded %s WDN from cache (%d nodes)", city, len(data["nodes"]))
        return data

    try:
        import osmnx as ox
        logger.info("Fetching OSMnx POIs for %s (r=%dm)…", city, cfg["radius"])
        lat, lon, r = cfg["lat"], cfg["lon"], cfg["radius"]
        pois_by_type: dict = {}

        _osmnx_fetch = [
            ("hospital",     {"amenity": "hospital"}),
            ("fire_station", {"amenity": "fire_station"}),
            ("school",       {"amenity": "school"}),
            ("college",      {"amenity": ["college", "university"]}),
            ("park",         {"leisure": "park"}),
            ("market",       {"amenity": ["marketplace", "food_court"]}),
        ]
        for ptype, tags in _osmnx_fetch:
            time.sleep(0.3)  # rate-limit guard
            try:
                gdf = ox.features_from_point((lat, lon), dist=r, tags=tags)
                pois_by_type[ptype] = _extract_centroids(gdf, ptype)
            except Exception as e:
                logger.debug("OSMnx %s fetch skipped: %s", ptype, e)
                pois_by_type[ptype] = []

        return _build_wdn(city, cfg, pois_by_type, n_nodes)

    except ImportError:
        logger.warning("osmnx not installed — using synthetic WDN for %s", city)
    except Exception as e:
        logger.warning("OSMnx failed for %s: %s — using synthetic fallback", city, e)

    return _build_synthetic_wdn(city, cfg, n_nodes)


def _extract_centroids(gdf, node_type: str) -> list:
    results = []
    for _, row in gdf.iterrows():
        try:
            geom = row.geometry
            if geom.geom_type == "Point":
                lat_c, lon_c = geom.y, geom.x
            else:
                c = geom.centroid
                lat_c, lon_c = c.y, c.x
            name = str(row.get("name", node_type)).strip()
            if not name or name == "nan":
                name = node_type.replace("_", " ").title()
            results.append({"lat": lat_c, "lon": lon_c, "type": node_type, "name": name})
        except Exception:
            continue
    return results


def _build_wdn(city: str, cfg: dict, pois_by_type: dict, n_nodes: int) -> dict:
    """Build WDN from OSMnx POIs. Supplements with synthetic nodes to reach n_nodes."""
    rng = np.random.default_rng(SEED_TRAIN)
    lat, lon = cfg["lat"], cfg["lon"]
    r = cfg["radius"]
    nodes = []
    node_id = 0

    # ── Sources ──────────────────────────────────────────────────────────────
    sources = [
        {"id": "S0", "lat": lat + 0.008, "lon": lon - 0.005,
         "type": "reservoir",    "name": "Water Treatment Plant A",
         "demand": 0.0, "elevation": 55.0, "is_source": True, "priority": 0},
        {"id": "S1", "lat": lat - 0.010, "lon": lon + 0.007,
         "type": "water_tower",  "name": "Elevated Storage Tank",
         "demand": 0.0, "elevation": 48.0, "is_source": True, "priority": 0},
    ]
    nodes.extend(sources)
    node_id += len(sources)

    # ── POI demand nodes from OSMnx ──────────────────────────────────────────
    type_prefix = {"hospital": "H", "fire_station": "F", "school": "SC",
                   "college": "CL", "park": "P", "market": "M"}
    type_counts = {t: 0 for t in type_prefix}

    # Caps per type (ensure variety)
    caps = {"hospital": 3, "fire_station": 3, "school": 5, "college": 2, "park": 3, "market": 3}

    for ptype, poi_list in pois_by_type.items():
        if ptype not in type_prefix:
            continue
        for poi in poi_list[: caps.get(ptype, 3)]:
            pfx = type_prefix[ptype]
            nid = f"{pfx}{type_counts[ptype]}"
            type_counts[ptype] += 1
            elev_d = poi["lat"] - lat, poi["lon"] - lon
            nodes.append({
                "id": nid,
                "lat": poi["lat"], "lon": poi["lon"],
                "type": ptype, "name": poi["name"],
                "demand": DEMANDS.get(ptype, 0.004),
                "elevation": _assign_elevation(*elev_d, rng=rng),
                "is_source": False,
                "priority": PRIORITY_WEIGHTS.get(ptype, 1),
            })
            node_id += 1

    # Fill guaranteed POI minimums with synthetic offsets
    for ptype, guaranteed in GUARANTEED_POIS.items():
        have = type_counts.get(ptype, 0)
        if have < guaranteed:
            for i in range(guaranteed - have):
                pfx = type_prefix.get(ptype, "X")
                nid = f"{pfx}{have + i}"
                angle = 2 * math.pi * (have + i) / max(guaranteed, 1)
                dist = r * 0.4 * (1 + i * 0.2) / 111000
                dlat = dist * math.cos(angle)
                dlon = dist * math.sin(angle) / math.cos(math.radians(lat))
                nodes.append({
                    "id": nid, "lat": lat + dlat, "lon": lon + dlon,
                    "type": ptype,
                    "name": f"{ptype.replace('_',' ').title()} {have + i + 1}",
                    "demand": DEMANDS.get(ptype, 0.004),
                    "elevation": _assign_elevation(dlat, dlon, rng=rng),
                    "is_source": False, "priority": PRIORITY_WEIGHTS.get(ptype, 1),
                })
                node_id += 1

    # ── Residential junctions to reach n_nodes ───────────────────────────────
    current_demand_nodes = len(nodes) - len(sources)
    n_residential = max(0, n_nodes - current_demand_nodes)
    for i in range(n_residential):
        angle = 2 * math.pi * i / max(n_residential, 1)
        frac = 0.25 + 0.65 * ((i % 4) / 3)
        dlat = r * frac * math.cos(angle) / 111000
        dlon = r * frac * math.sin(angle) / (111000 * math.cos(math.radians(lat)))
        nodes.append({
            "id": f"R{i}", "lat": lat + dlat, "lon": lon + dlon,
            "type": "residential",
            "name": f"Residential Zone {i + 1}",
            "demand": float(rng.uniform(0.002, 0.007)),
            "elevation": _assign_elevation(dlat, dlon, rng=rng),
            "is_source": False, "priority": 1,
        })
        node_id += 1

    edges = _connect_nodes(nodes, lat, max_dist_m=1500, rng=rng)
    return _finalise_wdn(city, cfg, nodes, edges, n_nodes)


def _build_synthetic_wdn(city: str, cfg: dict, n_nodes: int) -> dict:
    """Full synthetic WDN — no OSMnx. Deterministic template scaled to n_nodes."""
    rng = np.random.default_rng(SEED_TRAIN)
    lat, lon = cfg["lat"], cfg["lon"]
    r = cfg["radius"]
    nodes = []

    # Sources
    nodes += [
        {"id": "S0", "lat": lat + 0.008, "lon": lon - 0.005,
         "type": "reservoir",   "name": "Water Treatment Plant",
         "demand": 0.0, "elevation": 55.0, "is_source": True, "priority": 0},
        {"id": "S1", "lat": lat - 0.010, "lon": lon + 0.008,
         "type": "water_tower", "name": "Elevated Water Tank",
         "demand": 0.0, "elevation": 48.0, "is_source": True, "priority": 0},
    ]

    # Guaranteed critical facilities (always present)
    templates = [
        # (id, type, name, dlat, dlon)
        ("H0",  "hospital",     "City General Hospital",     0.007,  0.002),
        ("H1",  "hospital",     "Government Medical Centre", -0.005, 0.010),
        ("H2",  "hospital",     "Primary Health Centre",      0.012, -0.003),
        ("F0",  "fire_station", "Central Fire Station",       0.003, -0.007),
        ("F1",  "fire_station", "North Fire Brigade",        -0.008, -0.003),
        ("F2",  "fire_station", "South Fire Post",            0.002,  0.013),
        ("SC0", "school",       "Government High School",     0.002,  0.006),
        ("SC1", "school",       "Municipal Primary School",  -0.004, -0.008),
        ("SC2", "school",       "Secondary School",           0.010,  0.001),
        ("CL0", "college",      "Government College",        -0.006,  0.012),
        ("M0",  "market",       "Central Market",             0.001, -0.004),
        ("M1",  "market",       "Wholesale Market",          -0.009,  0.005),
        ("P0",  "park",         "City Central Park",          0.006, -0.004),
        ("P1",  "park",         "Garden Area",               -0.003,  0.009),
    ]
    for nid, ntype, name, dlat, dlon in templates:
        nodes.append({
            "id": nid, "lat": lat + dlat, "lon": lon + dlon,
            "type": ntype, "name": name,
            "demand": DEMANDS.get(ntype, 0.004),
            "elevation": _assign_elevation(dlat, dlon, rng=rng),
            "is_source": False, "priority": PRIORITY_WEIGHTS.get(ntype, 1),
        })

    # Residential junctions to fill n_nodes target
    n_residential = max(0, n_nodes - (len(nodes) - 2))  # subtract sources
    for i in range(n_residential):
        angle = 2 * math.pi * i / max(n_residential, 1)
        frac = 0.2 + 0.6 * ((i % 5) / 4)
        dlat = r * frac * math.cos(angle) / 111000
        dlon = r * frac * math.sin(angle) / (111000 * math.cos(math.radians(lat)))
        nodes.append({
            "id": f"R{i}", "lat": lat + dlat, "lon": lon + dlon,
            "type": "residential", "name": f"Residential Zone {i + 1}",
            "demand": float(rng.uniform(0.002, 0.007)),
            "elevation": _assign_elevation(dlat, dlon, rng=rng),
            "is_source": False, "priority": 1,
        })

    edges = _connect_nodes(nodes, lat, max_dist_m=1600, rng=rng)
    return _finalise_wdn(city, cfg, nodes, edges, n_nodes)


def _finalise_wdn(city: str, cfg: dict, nodes: list, edges: list, n_nodes: int) -> dict:
    result = {
        "city": city,
        "name": cfg["name"],
        "state": cfg["state"],
        "district": cfg.get("district", cfg["name"]),
        "center": {"lat": cfg["lat"], "lon": cfg["lon"]},
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_pipes": len(edges),
        "base_head_m": 60.0,
        "node_type_counts": _type_counts(nodes),
    }
    os.makedirs(GENERATED_DIR, exist_ok=True)
    cache = _cache_path(city, n_nodes)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Built WDN for %s: %d nodes, %d pipes", city, len(nodes), len(edges))
    return result


def _type_counts(nodes: list) -> dict:
    counts: dict = {}
    for n in nodes:
        t = n["type"]
        counts[t] = counts.get(t, 0) + 1
    return counts


def _connect_nodes(nodes: list, center_lat: float, max_dist_m: float = 1500, rng: np.random.Generator = None) -> list:  # rng always provided by callers
    """Prim's MST + redundant edges for network resilience."""
    from scipy.spatial.distance import cdist

    coords = np.array([[n["lat"], n["lon"]] for n in nodes])
    n = len(nodes)
    edges = []
    edge_set = set()
    pipe_id = 0

    # Convert to approximate metres
    scale = np.array([111000, 111000 * math.cos(math.radians(center_lat))])
    xy = (coords - coords.mean(axis=0)) * scale
    dist_matrix = cdist(xy, xy)

    sources = [i for i, nd in enumerate(nodes) if nd.get("is_source")]
    if not sources:
        sources = [0]

    # ── Prim's MST from first source ─────────────────────────────────────────
    in_tree = {sources[0]}
    remaining = set(range(n)) - in_tree
    while remaining:
        best_d, best_e = float("inf"), None
        for u in in_tree:
            for v in remaining:
                d = dist_matrix[u, v]
                if d < best_d:
                    best_d, best_e = d, (u, v)
        if best_e is None:
            break
        u, v = best_e
        _add_edge(edges, edge_set, nodes, u, v, pipe_id, best_d)
        pipe_id += 1
        in_tree.add(v)
        remaining.discard(v)

    # ── Redundant edges for resilience ────────────────────────────────────────
    # Higher-priority nodes get more redundant connections
    priority_bonus = {5: 0.9, 4: 0.85, 3: 0.75, 2: 0.6, 1: 0.35, 0: 0.1}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set or (j, i) in edge_set:
                continue
            d = dist_matrix[i, j]
            if d > max_dist_m:
                continue
            pi = PRIORITY_WEIGHTS.get(nodes[i]["type"], 1)
            pj = PRIORITY_WEIGHTS.get(nodes[j]["type"], 1)
            p = priority_bonus.get(math.ceil(max(pi, pj)), 0.3) * (1 - d / max_dist_m)
            if rng.random() < p:
                _add_edge(edges, edge_set, nodes, i, j, pipe_id, d)
                pipe_id += 1

    return edges


def _add_edge(edges, edge_set, nodes, i, j, pipe_id, dist_m):
    u, v = nodes[i], nodes[j]
    edge_set.add((i, j))
    src_t, dst_t = u.get("type", "junction"), v.get("type", "junction")
    src_is_main = src_t in ("reservoir", "water_tower")
    dst_is_main = dst_t in ("reservoir", "water_tower")
    if src_is_main or dst_is_main:
        cls = "main"
    elif src_t in ("hospital", "fire_station") or dst_t in ("hospital", "fire_station"):
        cls = "secondary"
    elif src_t in ("school", "college", "market") or dst_t in ("school", "college", "market"):
        cls = "secondary"
    else:
        cls = "local" if dist_m < 600 else "branch"
    attrs = PIPE_ATTRS[cls]
    edges.append({
        "id":         f"P{pipe_id}",
        "source":     u["id"],
        "target":     v["id"],
        "length_m":   round(dist_m, 1),
        "diameter_m": attrs["diameter"],
        "roughness":  attrs["roughness"],
        "C_hw":       attrs["C_hw"],
        "class":      cls,
        "status":     "OPEN",
    })


# ── Hydraulic simulation ──────────────────────────────────────────────────────

MIN_HEAD_M    = 7.03   # minimum service pressure (m)
REQUIRED_HEAD = 14.06  # full-service pressure (m)


def run_simple_simulation(
    wdn: dict,
    failed_pipes: list = None,
    custom_sources: list = None,
    custom_priorities: dict = None,
    scenario: str = "burst",          # "burst" | "leak" | "surge"
    surge_nodes: list = None,
    surge_multiplier: float = 2.5,
    leak_pipe: str = None,
    leak_coefficient: float = 0.001,
) -> dict:
    """
    BFS-based pressure propagation from sources through the WDN.
    Supports burst (pipe closed), leak (partial loss), and surge (demand ×N).
    Returns per-node pressure heads and demand satisfaction.
    """
    nodes = {n["id"]: n for n in wdn["nodes"]}
    edges = wdn["edges"]
    base_head = wdn.get("base_head_m", 60.0)
    failed_set = set(failed_pipes or [])

    # Effective demand per node (modified for surge)
    surge_set = set(surge_nodes or [])
    effective_demand = {}
    for nid, n in nodes.items():
        d = n.get("demand", DEMANDS.get(n.get("type", "junction"), 0.003))
        if nid in surge_set:
            d *= surge_multiplier
        effective_demand[nid] = d

    # Adjacency — remove closed pipes; reduce capacity for leaked pipe
    adj: dict = {nid: [] for nid in nodes}
    for e in edges:
        if e["status"] != "OPEN":
            continue
        eid = e["id"]
        if eid in failed_set:
            continue   # burst: fully closed
        leak_factor = 1.0
        if scenario == "leak" and eid == leak_pipe:
            # Leak: pipe open but reduced flow
            leak_factor = max(0.0, 1.0 - leak_coefficient * 100)
        adj[e["source"]].append((e["target"], e, leak_factor))
        adj[e["target"]].append((e["source"], e, leak_factor))

    # Sources
    sources = [nid for nid, n in nodes.items() if n.get("is_source")]
    if custom_sources:
        sources = [nid for nid in custom_sources if nid in nodes] or sources

    # BFS pressure propagation
    pressures: dict = {}
    for s in sources:
        pressures[s] = base_head + nodes[s].get("elevation", 0) * 0.1

    visited = set(sources)
    queue = deque(sources)
    while queue:
        u = queue.popleft()
        h_u = pressures[u]
        for v, edge, lf in adj[u]:
            if v in visited:
                continue
            L = edge["length_m"]
            D = edge["diameter_m"]
            C = edge["C_hw"]
            Q = max(effective_demand.get(v, 0.003), 0.001)
            try:
                h_f = 10.67 * L * (Q ** 1.852) / ((C ** 1.852) * (D ** 4.87))
            except Exception:
                h_f = 2.0
            pressures[v] = h_u - h_f * lf
            visited.add(v)
            queue.append(v)

    for nid in nodes:
        if nid not in pressures:
            pressures[nid] = 0.0

    # Demand satisfaction
    demand_sat: dict = {}
    for nid, h in pressures.items():
        if nodes[nid].get("is_source"):
            demand_sat[nid] = 1.0
        elif h <= MIN_HEAD_M:
            demand_sat[nid] = 0.0
        elif h >= REQUIRED_HEAD:
            demand_sat[nid] = 1.0
        else:
            demand_sat[nid] = (h - MIN_HEAD_M) / (REQUIRED_HEAD - MIN_HEAD_M)

    priority_config = _build_priority_config(nodes, custom_priorities)
    cds = _compute_cds(demand_sat, priority_config)
    pvc = sum(1 for nid, h in pressures.items()
              if not nodes[nid].get("is_source") and h < MIN_HEAD_M and h > 0)

    return {
        "method": "none",
        "pressures_m": pressures,
        "pressure_method": "hw_bfs_heuristic",
        "demand_satisfaction": demand_sat,
        "CDS": cds,
        "PVC": pvc,
        "unreachable_nodes": [nid for nid, h in pressures.items() if h <= 0
                               and not nodes[nid].get("is_source")],
        "failed_pipes": list(failed_set),
        "active_sources": sources,
        "scenario": scenario,
    }


# Driving head above ground elevation for synthetic sources — a reservoir's
# water surface or a tower's tank sits above bare terrain elevation. These
# margins were originally 40m/25m, which combined with these sources sitting
# on naturally higher local ground (~48-55m) than the demand nodes they serve
# (~21-36m) produced 44-71m of baseline pressure against a 14.06m requirement
# — over 3x real design margin, meaning no plausible single-pipe head-loss
# increase could ever push a node below the service threshold. Recalibrated
# so the lowest-pressure node under baseline (no-failure) conditions sits at
# ~1.5-1.6x REQUIRED_PRESSURE_M (~21-22m) — a realistic operating margin per
# typical utility design practice, not an excessive one. Verified empirically
# against this generator's actual elevation data, not guessed.
_SOURCE_HEAD_MARGIN_M = {"reservoir": 5.0, "water_tower": 3.0}

# Design velocity used to derive Dinic's pipe capacity (capacity = v * area).
# AWWA M31 cites ~0.6-1.5 m/s as normal distribution-main design velocity.
_DINIC_MAX_VELOCITY_MS = 1.5


_BURST_EMITTER_COEFF_PER_AREA = 0.1


from backend.wntr_engine import add_leak_from_emitter  # noqa: E402


def _build_wntr_model(wdn: dict, closed_pipes: set, demand_scale: dict = None,
                       burst_pipes: set = None, leak_pipes: dict = None):
    """Build a real WNTR WaterNetworkModel from a synthetic India WDN dict.

    demand_scale: optional {node_id: ratio in [0,1]} applied to each
    junction's base_demand. Used to evaluate a specific flow-allocation
    decision (e.g. Dinic's max-flow output) under real hydraulics, instead
    of the WNTR run always using each node's full nominal demand.

    burst_pipes: pipes to close *and* model as actively discharging until
    isolated — closing a pipe alone represents an idealized, already-isolated
    outage. A real "burst" also loses water at the break point, which is an
    additional draw on the rest of the network, not just a routing change.
    Modeled as a WNTR emitter (Q = C * P^0.5, the standard orifice-flow
    relationship) split across the failed pipe's two endpoints, with C
    proportional to the pipe's cross-sectional area — a bigger pipe bursts
    a bigger hole. This is self-limiting (discharge tapers off as local
    pressure drops), unlike a fixed extra demand, so it can't produce a
    runaway/unstable solve.

    leak_pipes: {pipe_id: leak_coefficient in [0, 0.01]} — unlike a burst,
    a leaking pipe stays OPEN (still routes flow) but loses some water along
    the way; modeled the same way as a burst (an emitter at its endpoints)
    but scaled down by leak_coefficient instead of the full burst magnitude.
    """
    import wntr

    demand_scale = demand_scale or {}
    burst_pipes = burst_pipes or set()
    leak_pipes = leak_pipes or {}
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.demand_model = "PDA"
    wn.options.hydraulic.minimum_pressure = MIN_PRESSURE_M
    wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE_M
    wn.options.hydraulic.headloss = "H-W"

    for n in wdn["nodes"]:
        nid = n["id"]
        elev = float(n.get("elevation", 0.0) or 0.0)
        if n.get("is_source"):
            margin = _SOURCE_HEAD_MARGIN_M.get(n.get("type"), 30.0)
            wn.add_reservoir(nid, base_head=elev + margin)
        else:
            base_d = float(n.get("demand", 0.0) or 0.0) * demand_scale.get(nid, 1.0)
            wn.add_junction(nid, base_demand=max(base_d, 0.0), elevation=elev)

    edge_map = {e["id"]: e for e in wdn["edges"]}
    for e in wdn["edges"]:
        pid = e["id"]
        if e["source"] not in wn.node_name_list or e["target"] not in wn.node_name_list:
            continue
        wn.add_pipe(
            pid, e["source"], e["target"],
            length=max(float(e.get("length_m", 100.0)), 1.0),
            diameter=max(float(e.get("diameter_m", 0.1)), 0.01),
            roughness=float(e.get("roughness", e.get("C_hw", 120.0)) or 120.0),
        )
        # A leaking pipe stays open — it's not in closed_pipes — so this
        # only ever closes genuine bursts/outages.
        if pid in closed_pipes or e.get("status") != "OPEN":
            wn.get_link(pid).initial_status = wntr.network.LinkStatus.Closed

    # A junction can touch several failed pipes: accumulate, then add ONE leak per
    # node (WNTRSimulator ignores node.emitter_coefficient, so real leaks are used).
    leak_c: dict = {}

    def _add_emitter(pid: str, magnitude: float):
        e = edge_map.get(pid)
        if e is None or magnitude <= 0:
            return
        area = math.pi / 4.0 * float(e.get("diameter_m", 0.1)) ** 2
        emitter_c = magnitude * area / 2.0  # split across both ends
        for endpoint in (e["source"], e["target"]):
            if endpoint in wn.junction_name_list:
                leak_c[endpoint] = leak_c.get(endpoint, 0.0) + emitter_c

    for pid in burst_pipes:
        _add_emitter(pid, _BURST_EMITTER_COEFF_PER_AREA)
    for pid, leak_coefficient in leak_pipes.items():
        # Same leak_factor convention as run_simple_simulation: coefficient
        # of 0.01 (the UI's max) maps to the full burst magnitude; smaller
        # coefficients scale down proportionally.
        scale = min(max(leak_coefficient, 0.0), 0.01) / 0.01
        _add_emitter(pid, _BURST_EMITTER_COEFF_PER_AREA * scale)
    for _node, _c in leak_c.items():
        add_leak_from_emitter(wn, _node, _c)

    return wn


def _run_wntr_steady_state(wn) -> dict:
    """Single-period WNTR/EPANET hydraulic solve (no 24h demand pattern —
    this synthetic network has none, so a steady-state snapshot is the
    correct evaluation, not the 24-hour EPS loop used for Net3/Hanoi/L-Town)."""
    import wntr

    wn.options.time.duration = 0
    wn.options.time.hydraulic_timestep = 3600
    try:
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
        pressure_df = results.node["pressure"]
        demand_df = results.node["demand"]
        converged = not pressure_df.isnull().all(axis=None)
        return {"pressure_df": pressure_df, "demand_df": demand_df, "converged": converged}
    except Exception as exc:
        logger.warning("Dinic hydraulic evaluation did not converge: %s", exc)
        return {"pressure_df": None, "demand_df": None, "converged": False}


def run_dinic_simulation(
    wdn: dict,
    failed_pipes: list = None,
    custom_priorities: dict = None,
    scenario: str = "burst",
    leak_pipe: str = None,
    leak_coefficient: float = 0.001,
    surge_nodes: list = None,
    surge_multiplier: float = 2.5,
) -> dict:
    """
    Dinic's max-flow determines the flow-allocation *decision* — how much of
    each node's demand it believes is deliverable given diameter-based pipe
    capacity on the failed topology. That decision is unchanged from before.

    What's new: instead of reporting that graph-capacity ratio directly as
    "Demand Served," Dinic's allocation becomes each junction's demand
    target for a real WNTR/EPANET hydraulic solve of the same failed
    network. Demand Served, Critical Node Service, and PVC are computed
    from the resulting real pressures and real delivered flow — the same
    evaluation functions HARP's results already go through
    (_compute_cds) — not from the max-flow ratio itself. This can only
    hold Dinic's numbers steady or reduce them to what's physically
    achievable; it never inflates them, so Dinic isn't being penalized
    beyond what real hydraulics actually requires.

    scenario/leak_pipe/leak_coefficient/surge_nodes/surge_multiplier: same
    convention as run_simple_simulation(). Previously this function only
    ever accepted failed_pipes, so /india/simulate's "leak" and "surge"
    scenario types silently ran Dinic (and HARP) against an unmodified
    network regardless of which pipe or nodes the UI selected.
    """
    import time as _time
    import networkx as nx
    from backend.algorithms.dinic import Dinic

    t0 = _time.perf_counter()
    failed_set = set(failed_pipes or [])
    surge_set = set(surge_nodes or [])
    is_leak = scenario == "leak" and leak_pipe
    # Same convention as run_simple_simulation: coefficient of 0.01 (the
    # UI's max) fully chokes the pipe; smaller values scale down linearly.
    leak_capacity_factor = max(0.0, 1.0 - leak_coefficient * 100) if is_leak else 1.0
    nodes = {n["id"]: n for n in wdn["nodes"]}

    G = nx.DiGraph()
    for nid, n in nodes.items():
        d = float(n.get("demand", 0.004) or 0.0)
        if nid in surge_set:
            d *= surge_multiplier
        G.add_node(nid, base_demand=d, elevation=n.get("elevation", 0))

    for e in wdn["edges"]:
        pid = e["id"]
        if e["status"] != "OPEN":
            continue
        if pid in failed_set:
            continue  # burst/multi-failure: fully closed, no capacity at all
        D = e["diameter_m"]
        # Capacity = v_max * cross-sectional area, not diameter*10. AWWA M31
        # cites ~0.6-1.5 m/s as normal distribution-main design velocity;
        # _DINIC_MAX_VELOCITY_MS=1.5 sits at the top of that range (Dinic is
        # meant to represent the best achievable *without* real hydraulic
        # awareness, so a generous-but-real velocity, not an emergency one).
        # diameter*10 was 20-85x more generous than this at the diameters
        # this network uses (0.1-0.4m) — verified against these exact values.
        cap = _DINIC_MAX_VELOCITY_MS * (math.pi / 4.0) * (D ** 2)
        if is_leak and pid == leak_pipe:
            cap *= leak_capacity_factor  # open but throttled, not excluded
        # Both directions — same capacity (no * 0.5 bug)
        G.add_edge(e["source"], e["target"], diameter=D, capacity=cap)
        G.add_edge(e["target"], e["source"], diameter=D, capacity=cap)

    source_nodes = [nid for nid, n in nodes.items() if n.get("is_source")]
    sink_nodes   = [nid for nid, n in nodes.items() if not n.get("is_source")]

    SUPERSRC = "__SUPERSRC__"
    G.add_node(SUPERSRC)
    for s in source_nodes:
        G.add_edge(SUPERSRC, s, diameter=5.0)

    dinic = Dinic()
    result = dinic.run(G, SUPERSRC, sink_nodes)
    ct = (_time.perf_counter() - t0) * 1000

    dinic_allocation = {nid: 1.0 for nid in source_nodes}
    dinic_allocation.update(result.demand_satisfaction)

    # Evaluate Dinic's allocation under real hydraulics: same failed pipes,
    # each junction's demand scaled to what Dinic believes it can deliver
    # (and, under a demand-surge scenario, inflated first — Dinic's own
    # max-flow allocation above was already computed against the surged
    # demand via G's base_demand, so this keeps the two consistent).
    # Only the pipes explicitly failed *in this scenario* burst (discharge);
    # pipes that were already closed in the base topology are a pre-existing
    # condition, not an active break, so they're isolated only. A leak stays
    # open with a smaller, ongoing emitter instead of being closed/bursting.
    effective_demand_scale = {
        nid: ratio * (surge_multiplier if nid in surge_set else 1.0)
        for nid, ratio in dinic_allocation.items()
    }
    all_closed = set(failed_set) | {e["id"] for e in wdn["edges"] if e.get("status") != "OPEN"}
    wn = _build_wntr_model(
        wdn, closed_pipes=all_closed, demand_scale=effective_demand_scale,
        burst_pipes=set(failed_set),
        leak_pipes={leak_pipe: leak_coefficient} if is_leak else None,
    )
    hydraulic = _run_wntr_steady_state(wn)

    if hydraulic["converged"] and hydraulic["pressure_df"] is not None:
        pdf, ddf = hydraulic["pressure_df"], hydraulic["demand_df"]
        t = pdf.index[-1]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}
        demand_sat = {}
        for nid, n in nodes.items():
            if n.get("is_source"):
                demand_sat[nid] = 1.0
                continue
            base = float(n.get("demand", 0.0) or 0.0)
            if nid in surge_set:
                base *= surge_multiplier
            if base <= 0:
                demand_sat[nid] = 1.0
                continue
            actual = float(ddf.at[t, nid]) if nid in ddf.columns else 0.0
            demand_sat[nid] = min(max(actual / base, 0.0), 1.0)
        pvc = sum(1 for nid, p in pressures.items()
                  if nid in nodes and not nodes[nid].get("is_source") and p < MIN_PRESSURE_M)
        pressures_out = pressures
        hydraulically_evaluated = True
    else:
        # Real hydraulic solve didn't converge (can happen under severe
        # multi-failure demand deficits) — fall back to Dinic's own
        # graph-capacity estimate rather than silently reporting zero, but
        # say so explicitly rather than presenting it as hydraulically verified.
        demand_sat = dinic_allocation
        pressures_out = None
        pvc = None
        hydraulically_evaluated = False

    priority_config = _build_priority_config(nodes, custom_priorities)
    cds = _compute_cds(demand_sat, priority_config)

    return {
        "method": "dinic",
        "demand_satisfaction": demand_sat,
        "pressures_m": pressures_out,
        "pressure_method": "hw_bfs_heuristic" if pressures_out is not None else "unavailable",
        "CDS": cds,
        "PVC": pvc,
        "computation_time_ms": round(ct, 3),
        "total_flow": round(result.total_flow, 4),
        "dinic_allocation": dinic_allocation,
        "hydraulically_evaluated": hydraulically_evaluated,
        "failed_pipes": list(failed_set),
    }


def run_harp_simulation(
    wdn: dict,
    failed_pipes: list = None,
    custom_priorities: dict = None,
    k: int = 2,
    scenario: str = "burst",
    leak_pipe: str = None,
    leak_coefficient: float = 0.001,
    surge_nodes: list = None,
    surge_multiplier: float = 2.5,
) -> dict:
    """
    HARP on India WDN.
    Uses BFS pressure heads as proxy for GAT predictions.
    Capacity = hw_max_flow(L, D, C, dH) from those heads (HW-based, not diameter×10).

    scenario/leak_pipe/leak_coefficient/surge_nodes/surge_multiplier: same
    convention as run_simple_simulation()/run_dinic_simulation(). Previously
    this function only ever accepted failed_pipes, so selecting a "leak" or
    "surge" scenario in the UI silently ran HARP against an unmodified
    network — same pipe, same demand, regardless of what was selected.
    """
    import time as _time
    from backend.algorithms.harp import HARP

    t0 = _time.perf_counter()
    failed_set  = set(failed_pipes or [])
    surge_set   = set(surge_nodes or [])
    is_leak     = scenario == "leak" and leak_pipe
    leak_capacity_factor = max(0.0, 1.0 - leak_coefficient * 100) if is_leak else 1.0
    # HARP's k-hop subgraph localizes around one "anchor" pipe. For a burst
    # that's whichever pipe failed; for a leak (which doesn't populate
    # failed_pipes — the pipe stays open) it needs to be the leak location
    # instead, or a leak scenario would never localize/escalate correctly.
    failed_pipe = list(failed_set)[0] if failed_set else (leak_pipe if is_leak else None)
    nodes = {n["id"]: n for n in wdn["nodes"]}

    # Step 1: Run base simulation for demand satisfaction fallback
    sim = run_simple_simulation(
        wdn, failed_pipes=list(failed_set), scenario=scenario,
        leak_pipe=leak_pipe, leak_coefficient=leak_coefficient,
        surge_nodes=list(surge_set), surge_multiplier=surge_multiplier,
    )

    # Prefer a real WNTR/EPANET steady-state solve of the failed network (same
    # model the Dinic evaluation uses) as the pressure field HARP's hydraulic
    # gate reads. This is *not* a trained-GAT prediction: there is no GAT for
    # these synthetic networks. The earlier distance-gradient stand-in
    # (100 m - 0.05 m/m * distance) is kept only as a labelled fallback for
    # when the solve does not converge.
    _closed = set(failed_set) | {e["id"] for e in wdn["edges"] if e.get("status") != "OPEN"}
    _wn_real = _build_wntr_model(
        wdn, closed_pipes=_closed,
        demand_scale={nid: surge_multiplier for nid in surge_set},
        burst_pipes=set(failed_set),
        leak_pipes={leak_pipe: leak_coefficient} if is_leak else None,
    )
    _hyd = _run_wntr_steady_state(_wn_real)
    _source_head = {
        nid: float(n.get("elevation", 0.0) or 0.0) + _SOURCE_HEAD_MARGIN_M.get(n.get("type"), 30.0)
        for nid, n in nodes.items() if n.get("is_source")
    }
    if _hyd["converged"] and _hyd["pressure_df"] is not None:
        _pdf = _hyd["pressure_df"]
        _t = _pdf.index[-1]
        gat_pressures = {nid: float(_pdf.at[_t, nid]) for nid in _pdf.columns if nid in nodes}
        pressure_source = "wntr_steady_state"
    else:
        _adj: dict = {}
        for _e in wdn["edges"]:
            if _e["id"] in failed_set or _e.get("status") != "OPEN":
                continue
            _adj.setdefault(_e["source"], []).append((_e["target"], _e["length_m"]))
            _adj.setdefault(_e["target"], []).append((_e["source"], _e["length_m"]))
        _dist: dict = {s: 0.0 for s in _source_head}
        _pq = [(0.0, s) for s in _source_head]
        while _pq:
            _d, _u = heapq.heappop(_pq)
            if _d > _dist.get(_u, float("inf")):
                continue
            for _v, _w in _adj.get(_u, []):
                _nd = _d + _w
                if _nd < _dist.get(_v, float("inf")):
                    _dist[_v] = _nd
                    heapq.heappush(_pq, (_nd, _v))
        gat_pressures = {
            nid: max(5.0, 100.0 - _dist.get(nid, 2000.0) * 0.05) - nodes[nid].get("elevation", 0)
            for nid in nodes
        }
        _source_head = {nid: 100.0 for nid in _source_head}
        pressure_source = "distance_gradient_fallback"

    priority_config = _build_priority_config(nodes, custom_priorities)

    # Step 2: HARP reads the real WNTR model built above (junctions, reservoirs,
    # pipes, failed pipes closed, surge already applied to demands). It used to be
    # handed a hand-built SimpleNamespace that re-implemented the same lookups.
    # A leak keeps the pipe open but throttles its capacity: reduce its diameter
    # on the graph model only (the hydraulic solve above already ran).
    if is_leak and leak_pipe in _wn_real.pipe_name_list:
        _wn_real.get_link(leak_pipe).diameter *= leak_capacity_factor ** 0.5   # D enters HW capacity at ~2.6 power
    if pressure_source == "distance_gradient_fallback":
        for _s in _source_head:                      # keep heads consistent with the fallback gradient
            _wn_real.get_node(_s).base_head = _source_head[_s]

    # Step 3: also need priority info on graph nodes
    # Build supplemented priority_config with weights
    pc = {
        "hospital_nodes":      [nid for nid, n in nodes.items() if n.get("type") == "hospital"],
        "fire_station_nodes":  [nid for nid, n in nodes.items() if n.get("type") == "fire_station"],
        "school_nodes":        [nid for nid, n in nodes.items() if n.get("type") in ("school", "college")],
        "weights": dict(NODE_TYPE_WEIGHTS),
    }
    if custom_priorities:
        # custom_priorities maps node_id → type_string
        for nid, ntype in custom_priorities.items():
            if ntype == "hospital" and nid not in pc["hospital_nodes"]:
                pc["hospital_nodes"].append(nid)
            elif ntype == "fire_station" and nid not in pc["fire_station_nodes"]:
                pc["fire_station_nodes"].append(nid)

    harp_fell_back = False
    try:
        harp = HARP(safety_margin_m=1.0, q_estimate=0.005)
        result = harp.run(
            wn=_wn_real,
            failed_pipe_id=failed_pipe or "NONE",
            gat_pressures_dict=gat_pressures,
            priority_config=pc,
            k=k,
            use_subgraph=bool(failed_pipe),
        )
        demand_sat = result.demand_satisfaction
        ct = result.computation_time_ms
        escalation = result.escalation_flag
        accepted_paths = [[str(n) for n in p.nodes] for p in result.accepted_paths]
        rejected_count = len(result.rejected_paths)
    except Exception as e:
        logger.warning("HARP failed on India WDN: %s — falling back to BFS sim", e)
        harp_fell_back = True
        demand_sat = sim["demand_satisfaction"]
        ct = (_time.perf_counter() - t0) * 1000
        escalation = False
        accepted_paths = []
        rejected_count = 0

    cds = _compute_cds(demand_sat, priority_config)
    return {
        "method": "harp",
        "demand_satisfaction": demand_sat,
        "pressures_m": gat_pressures,
        "pressure_method": pressure_source,
        "harp_fell_back_to_bfs": harp_fell_back,
        "CDS": cds,
        "PVC": sum(1 for v in demand_sat.values() if v < 0.5),
        "computation_time_ms": round(ct, 3),
        "escalation_flag": escalation,
        "accepted_paths": accepted_paths,
        "rejected_paths_count": rejected_count,
        "failed_pipes": list(failed_set),
    }


def _build_priority_config(nodes: dict, custom: dict = None) -> dict:
    effective = {}
    for nid, n in nodes.items():
        t = n.get("type", "residential")
        if custom and nid in custom:
            t = custom[nid]
        effective[nid] = t
    return {
        "hospital_nodes":     [nid for nid, t in effective.items() if t == "hospital"],
        "fire_station_nodes": [nid for nid, t in effective.items() if t == "fire_station"],
        "school_nodes":       [nid for nid, t in effective.items() if t in ("school", "college")],
        "weights": dict(NODE_TYPE_WEIGHTS),
    }


def _compute_cds(demand_sat: dict, priority_config: dict) -> dict:
    def _mean(node_ids):
        vals = [demand_sat.get(n, 0) for n in node_ids if n in demand_sat]
        return float(sum(vals) / len(vals)) if vals else 0.0

    h = _mean(priority_config["hospital_nodes"])
    f = _mean(priority_config["fire_station_nodes"])
    s = _mean(priority_config["school_nodes"])
    # Exclude virtual nodes (supersource/supersink) from overall metric
    all_vals = [v for nid, v in demand_sat.items() if not nid.startswith("__SUPER")]
    overall  = float(sum(all_vals) / len(all_vals)) if all_vals else 0.0
    crit = priority_config["hospital_nodes"] + priority_config["fire_station_nodes"]
    critical = _mean(crit)
    return {
        "CDS_hospital": round(h, 4),
        "CDS_fire":     round(f, 4),
        "CDS_school":   round(s, 4),
        "CDS_critical": round(critical, 4),
        "CDS_overall":  round(overall, 4),
    }
