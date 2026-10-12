"""HARP-WDN FastAPI REST API."""
import asyncio
import collections
import hmac
import json
import logging
import math
import os
import random
import sqlite3
import sys
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.config import DATA_DIR, FIGURES_DIR, GENERATED_DIR, RESULTS_DIR, SNAPSHOT_HOUR, NODE_TYPE_WEIGHTS

logger = logging.getLogger(__name__)

app = FastAPI(title="HARP-WDN API", version="1.0.0")


def _server_error(exc: Exception) -> HTTPException:
    """Log the real exception server-side; give the client a generic 500."""
    logger.exception("Unhandled error: %s", exc)
    return HTTPException(status_code=500, detail="Internal server error")

# ---------------------------------------------------------------------------
# API key auth — enabled only when HARP_API_KEY env var is set.
# Compute endpoints require the key in the X-API-Key header.
# When the env var is not set (dev mode) all requests pass through.
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get("HARP_API_KEY", "")
_AUTH_PREFIXES = ("/simulate/", "/anomaly/detect", "/india/simulate", "/india/report")

# ---------------------------------------------------------------------------
# Rate limiting — fixed-window per client IP.
# Limits: 60 req/min for general endpoints, 10 req/min for heavy compute
# endpoints (/simulate/*, /anomaly/detect).  Override via env vars:
#   RATE_LIMIT_GENERAL   (default 60)
#   RATE_LIMIT_COMPUTE   (default 10)
# ---------------------------------------------------------------------------
_RL_WINDOW = 60.0  # seconds
_RL_GENERAL = int(os.environ.get("RATE_LIMIT_GENERAL", "60"))
_RL_COMPUTE = int(os.environ.get("RATE_LIMIT_COMPUTE", "10"))
_COMPUTE_PREFIXES = ("/simulate/", "/anomaly/detect", "/india/simulate", "/india/report")

_rate_counts: dict[str, list] = collections.defaultdict(list)  # ip → [timestamp, ...]


def _check_rate_limit(ip: str, limit: int) -> bool:
    """Return True if request is allowed, False if limit exceeded."""
    now = time.monotonic()
    window_start = now - _RL_WINDOW
    # Evict old timestamps; remove key entirely when list becomes empty
    # to prevent unbounded memory growth under many unique source IPs.
    active = [t for t in _rate_counts[ip] if t > window_start]
    if len(active) >= limit:
        _rate_counts[ip] = active
        return False
    active.append(now)
    if active:
        _rate_counts[ip] = active
    elif ip in _rate_counts:
        del _rate_counts[ip]
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    # Behind a reverse proxy (Render, Vercel) every request arrives from the
    # proxy's IP, so all users would share one rate-limit bucket. Opt in with
    # TRUST_PROXY=1 only when a trusted proxy sets X-Forwarded-For.
    if os.environ.get("TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            ip = fwd.split(",")[0].strip() or ip
    path = request.url.path
    is_compute = any(path.startswith(p) for p in _COMPUTE_PREFIXES)
    limit = _RL_COMPUTE if is_compute else _RL_GENERAL
    if not _check_rate_limit(ip, limit):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded: max {limit} requests per {int(_RL_WINDOW)}s"},
            headers={"Retry-After": str(int(_RL_WINDOW))},
        )
    return await call_next(request)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY and any(request.url.path.startswith(p) for p in _AUTH_PREFIXES):
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), _API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key. Pass X-API-Key header."})
    return await call_next(request)

# ALLOWED_ORIGINS: comma-separated list of production origins, e.g.
# "https://harp-wdn.onrender.com". When set, localhost:5173 is NOT added so
# the production deployment doesn't accept cross-origin requests from dev.
_extra_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
_cors_origins = _extra_origins if _extra_origins else ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

os.makedirs(FIGURES_DIR, exist_ok=True)
app.mount("/static/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

# In-memory priority overrides: {city: {node_id: {"type": str, "weight": float}}}
# Written by POST /india/priority/override, consumed by india/simulate.
# Persisted to SQLite so overrides survive server restarts.
_priority_overrides: dict[str, dict[str, dict]] = {}

_DB_PATH = os.path.join(os.path.dirname(__file__), "results", "harp_wdn.db")


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_overrides_table() -> None:
    with _db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS priority_overrides (
                city TEXT NOT NULL,
                node_id TEXT NOT NULL,
                new_type TEXT NOT NULL,
                new_weight REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (city, node_id)
            )
        """)


def _load_overrides_from_db() -> None:
    """Populate in-memory _priority_overrides from DB at startup."""
    try:
        _init_overrides_table()
        with _db_conn() as conn:
            rows = conn.execute("SELECT city, node_id, new_type, new_weight FROM priority_overrides").fetchall()
        for row in rows:
            _priority_overrides.setdefault(row["city"], {})[row["node_id"]] = {
                "type": row["new_type"],
                "weight": row["new_weight"],
            }
        if rows:
            logger.info("Loaded %d priority overrides from DB", len(rows))
    except Exception as e:
        logger.warning("Could not load priority overrides from DB: %s", e)


def _save_override_to_db(city: str, node_id: str, new_type: str, new_weight: float) -> None:
    try:
        _init_overrides_table()
        with _db_conn() as conn:
            conn.execute("""
                INSERT INTO priority_overrides (city, node_id, new_type, new_weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(city, node_id) DO UPDATE SET
                    new_type=excluded.new_type,
                    new_weight=excluded.new_weight,
                    updated_at=excluded.updated_at
            """, (city, node_id, new_type, new_weight, time.time()))
    except Exception as e:
        logger.warning("Could not save priority override to DB: %s", e)


def _delete_overrides_from_db(city: str) -> None:
    try:
        with _db_conn() as conn:
            conn.execute("DELETE FROM priority_overrides WHERE city = ?", (city,))
    except Exception as e:
        logger.warning("Could not delete priority overrides from DB: %s", e)


_load_overrides_from_db()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    network: str = "Net3"
    pipe_id: str
    method: str = "harp"  # "none" | "dinic" | "harp"
    k: int = 2


class AnomalyRequest(BaseModel):
    network: str = "Net3"
    pipe_id: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

@app.get("/config/priorities")
def priority_weights():
    """Authoritative node-type priority weights (the frontend mirrors these)."""
    return {"weights": NODE_TYPE_WEIGHTS}


@app.get("/networks")
def list_networks():
    """List available WDN networks and their metadata.

    Hanoi is a classic academic benchmark network (not a real deployed city WDN)
    and has no trained GAT model, so it is excluded from the public list.
    Net3 and L-Town are the two supported networks.
    """
    nets = ["Net3", "L-Town"]
    result = []
    for name in nets:
        inp_path = os.path.join(DATA_DIR, "networks", f"{name}.inp")
        model_path = os.path.join(DATA_DIR, "trained_models", f"{name}_gat.pt")
        result.append({
            "name": name,
            "available": os.path.exists(inp_path),
            "harp_available": os.path.exists(model_path),
        })
    return {"networks": result}


@app.get("/networks/{network}/info")
def network_info(network: str):
    """Return node/pipe counts for a network."""
    try:
        from backend.wntr_engine import load_network
        wn = load_network(network)
        nodes = wn.junction_name_list + wn.reservoir_name_list + wn.tank_name_list
        return {
            "network": network,
            "n_junctions": len(wn.junction_name_list),
            "n_reservoirs": len(wn.reservoir_name_list),
            "n_tanks": len(wn.tank_name_list),
            "n_pipes": len(wn.pipe_name_list),
            "n_total_nodes": len(nodes),
        }
    except Exception as e:
        raise _server_error(e)


@app.get("/networks/{network}/pipes")
def list_pipes(network: str):
    """List all pipe IDs for a network."""
    try:
        from backend.wntr_engine import load_network
        wn = load_network(network)
        pipes = []
        for pid in wn.pipe_name_list:
            p = wn.get_link(pid)
            pipes.append({
                "id": pid,
                "start": p.start_node_name,
                "end": p.end_node_name,
                "length_m": round(p.length, 2),
                "diameter_m": round(p.diameter, 4),
            })
        return {"network": network, "pipes": pipes}
    except Exception as e:
        raise _server_error(e)


@app.get("/networks/{network}/graph")
def network_graph(network: str):
    """Return node/edge graph for Leaflet rendering."""
    try:
        from backend.wntr_engine import load_network
        wn = load_network(network)
        priority_path = os.path.join(DATA_DIR, "priority_config.json")
        priority_map = {}
        if os.path.exists(priority_path):
            with open(priority_path) as f:
                cfg = json.load(f)
            pc = cfg.get(network, {})
            for nid in pc.get("hospital_nodes", []):
                priority_map[nid] = "hospital"
            for nid in pc.get("fire_station_nodes", []):
                priority_map[nid] = "fire_station"
            for nid in pc.get("school_nodes", []):
                priority_map[nid] = "school"

        nodes = []
        for nid in wn.junction_name_list + wn.reservoir_name_list + wn.tank_name_list:
            node = wn.get_node(nid)
            coords = getattr(node, "coordinates", (0.0, 0.0)) or (0.0, 0.0)
            nodes.append({
                "id": nid,
                "x": coords[0],
                "y": coords[1],
                "type": priority_map.get(nid, "junction"),
                "elevation": getattr(node, "elevation", 0.0),
            })

        edges = []
        for pid in wn.pipe_name_list:
            p = wn.get_link(pid)
            edges.append({
                "id": pid,
                "source": p.start_node_name,
                "target": p.end_node_name,
                "diameter_m": round(p.diameter, 4),
                "length_m": round(p.length, 2),
            })

        return {"network": network, "nodes": nodes, "edges": edges}
    except Exception as e:
        raise _server_error(e)


# ---------------------------------------------------------------------------
# Simulation / Scenarios
# ---------------------------------------------------------------------------

@app.post("/simulate/scenario")
def run_scenario(req: ScenarioRequest):
    """Run a single failure scenario with chosen recovery method.

    Delegates to experiments/runner.py's _run_no_recovery / _run_dinic /
    _run_harp rather than reimplementing the call sequence here. The old
    inline version read a "demand_satisfaction" key that run_simulation()
    never sets (always silently defaulting to 100% satisfied) and called
    predict_pressures() with the wrong signature (network name + a
    "features" key that doesn't exist), which crashed the "harp" branch
    on every request.
    """
    try:
        from backend.wntr_engine import load_network
        from backend.gat.inference import load_model
        from backend.experiments.runner import (
            _run_no_recovery, _run_dinic, _run_harp, _load_priority_config,
        )

        priority_config = _load_priority_config(req.network)
        wn_base = load_network(req.network)

        if req.pipe_id not in wn_base.pipe_name_list:
            raise HTTPException(status_code=400, detail=f"Unknown pipe '{req.pipe_id}' in network '{req.network}'")

        if req.method == "none":
            r = _run_no_recovery(wn_base, req.pipe_id, priority_config)
        elif req.method == "dinic":
            r = _run_dinic(wn_base, req.pipe_id, priority_config)
        elif req.method == "harp":
            model, norm_stats = load_model(req.network)
            r = _run_harp(wn_base, req.pipe_id, priority_config, model, norm_stats, k=req.k)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

        result = {
            "network": req.network,
            "pipe_id": req.pipe_id,
            "method": req.method,
            "pressures_m": r["pressures"],
            "demand_satisfaction": r["demand_satisfaction"],
            "PVC": r["PVC"],
            "CDS": r["cds"],
            "WLV_m3": r["WLV_m3"],
            "computation_time_ms": r["CT_ms"],
        }
        if req.method == "harp":
            result["escalation_flag"] = r.get("escalation_flag", False)
            result["k_used"] = r.get("k_used", req.k)
            harp_result = r.get("harp_result")
            if harp_result is not None:
                result["accepted_paths"] = [list(p.nodes) for p in harp_result.accepted_paths]
            # MC-Dropout uncertainty — run 20 stochastic forward passes
            try:
                from backend.gat.inference import predict_pressures_with_uncertainty
                from backend.wntr_engine import extract_graph_features, inject_failure
                wn_fail = inject_failure(wn_base, req.pipe_id)
                graph_data = extract_graph_features(
                    wn_fail, failed_pipe_id=req.pipe_id,
                    priority_config=priority_config, network_name=req.network,
                )
                _, std_m, _ = predict_pressures_with_uncertainty(model, norm_stats, graph_data)
                result["pressure_uncertainty_m"] = {
                    nid: round(float(s), 3)
                    for nid, s in zip(graph_data["node_ids"], std_m)
                }
            except Exception as ue:
                logger.warning("Uncertainty estimation failed: %s", ue)
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.warning("Missing model/network file: %s", e)
        raise HTTPException(status_code=404, detail="Trained model or network file not found")
    except Exception as e:
        logger.exception("Scenario error")
        raise _server_error(e)


# ---------------------------------------------------------------------------
# Experiment Results
# ---------------------------------------------------------------------------

@app.get("/results/experiments")
def list_experiment_results():
    """Return available experiment CSVs."""
    import glob
    csvs = glob.glob(os.path.join(RESULTS_DIR, "experiment_*.csv"))
    return {
        "experiments": [
            {"file": os.path.basename(c), "size_kb": os.path.getsize(c) // 1024}
            for c in sorted(csvs)
        ]
    }


@app.get("/results/experiment/{exp_id}")
def get_experiment_result(exp_id: int):
    """Return experiment CSV as JSON rows."""
    import csv
    path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"experiment_{exp_id}.csv not found")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {"experiment": exp_id, "n_rows": len(rows), "rows": rows}


@app.get("/results/figures")
def list_figures():
    """List generated figures."""
    from backend.config import FIGURES_DIR
    if not os.path.exists(FIGURES_DIR):
        return {"figures": []}
    figs = []
    for fname in sorted(os.listdir(FIGURES_DIR)):
        if fname.endswith(".png"):
            figs.append({
                "name": fname,
                "size_kb": os.path.getsize(os.path.join(FIGURES_DIR, fname)) // 1024,
            })
    return {"figures": figs}


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

@app.post("/anomaly/detect")
def detect_anomaly(req: AnomalyRequest):
    """Detect anomalies after pipe failure."""
    try:
        from backend.wntr_engine import inject_failure, load_network, run_simulation
        from backend.anomaly.detector import build_baseline, detect_anomalies

        wn = load_network(req.network)
        if req.pipe_id not in wn.pipe_name_list:
            raise HTTPException(status_code=400, detail=f"Unknown pipe '{req.pipe_id}' in network '{req.network}'")
        wn_fail = inject_failure(wn, req.pipe_id)
        sim = run_simulation(wn_fail)
        if not sim["converged"]:
            raise HTTPException(status_code=500, detail="Simulation did not converge")

        pdf = sim["pressure_df"]
        hour = min(SNAPSHOT_HOUR, len(pdf.index) - 1)
        t = pdf.index[hour]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}

        baseline = build_baseline(req.network)
        pipe_endpoints = {
            p: (wn.get_link(p).start_node_name, wn.get_link(p).end_node_name)
            for p in wn.pipe_name_list
        }
        result = detect_anomalies(pressures, baseline, timestep_hour=hour, pipe_endpoints=pipe_endpoints)
        return {
            "network": req.network,
            "pipe_failed": req.pipe_id,
            "n_anomalous": len(result["anomalous_nodes"]),
            "anomalous_nodes": result["anomalous_nodes"],
            "most_likely_failed_pipe": result["most_likely_failed_pipe"],
            "top_z_scores": dict(
                sorted(result["z_scores"].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _server_error(e)


# ---------------------------------------------------------------------------
# City Geo
# ---------------------------------------------------------------------------

@app.get("/geo/cities")
def list_cities():
    """List available city graphs with metadata."""
    from backend.geo.city_mapper import get_city_summary
    return {"cities": get_city_summary()}


@app.get("/geo/city/{city_name}")
def get_city(city_name: str):
    """Return city graph (nodes + edges) for Leaflet rendering."""
    from backend.geo.city_mapper import CITIES, fetch_city_graph
    if city_name not in CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city_name}")
    try:
        return fetch_city_graph(city_name, use_cache=True)
    except Exception as e:
        raise _server_error(e)


# ---------------------------------------------------------------------------
# India WDN API
# ---------------------------------------------------------------------------

class IndiaSimRequest(BaseModel):
    city: str
    failed_pipes: list = []
    method: str = "harp"       # "none" | "dinic" | "harp"
    k: int = 2
    n_nodes: int = 20
    custom_priorities: dict = {}   # node_id → type_string
    custom_sources: list = []
    scenario: str = "burst"        # "burst" | "leak" | "surge"
    surge_nodes: list = []
    surge_multiplier: float = 2.5
    leak_pipe: str = ""
    leak_coefficient: float = 0.001


class PriorityOverrideRequest(BaseModel):
    city: str
    node_id: str
    new_type: str
    new_weight: float = 0.0   # 0 = use default for type


@app.get("/india/cities")
def india_cities():
    from backend.geo.india_wdn import get_city_list
    return {"cities": get_city_list()}


@app.get("/india/wdn/{city}")
def india_wdn(city: str, n_nodes: int = 20, regen: bool = False):
    """Return WDN graph for an Indian city (generates if not cached)."""
    from backend.geo.india_wdn import fetch_city_pois, INDIA_CITIES
    if city not in INDIA_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
    try:
        return fetch_city_pois(city, n_nodes=n_nodes, force_regen=regen)
    except Exception as e:
        logger.exception("India WDN fetch error")
        raise _server_error(e)


@app.post("/india/simulate")
def india_simulate(req: IndiaSimRequest):
    """Run simulation on an Indian city WDN. method=compare returns all three."""
    from backend.geo.india_wdn import (
        fetch_city_pois, run_simple_simulation,
        run_dinic_simulation, run_harp_simulation,
        INDIA_CITIES,
    )
    if req.city not in INDIA_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {req.city}")
    try:
        wdn = fetch_city_pois(req.city, n_nodes=req.n_nodes)
        fp = req.failed_pipes
        # Merge session-level overrides (set via /india/priority/override) under
        # request-level custom_priorities so the API caller can still override them.
        stored = {nid: v["type"] for nid, v in _priority_overrides.get(req.city, {}).items()}
        cp = {**stored, **(req.custom_priorities or {})}

        sim_kwargs = dict(
            failed_pipes=fp,
            custom_priorities=cp,
            scenario=req.scenario,
            surge_nodes=req.surge_nodes or [],
            surge_multiplier=req.surge_multiplier,
            leak_pipe=req.leak_pipe or None,
            leak_coefficient=req.leak_coefficient,
        )

        scenario_kwargs = dict(
            scenario=req.scenario,
            leak_pipe=req.leak_pipe or None,
            leak_coefficient=req.leak_coefficient,
            surge_nodes=req.surge_nodes or [],
            surge_multiplier=req.surge_multiplier,
        )

        if req.method == "none":
            return run_simple_simulation(wdn, custom_sources=req.custom_sources or [], **sim_kwargs)

        if req.method == "dinic":
            return run_dinic_simulation(wdn, failed_pipes=fp, custom_priorities=cp, **scenario_kwargs)

        if req.method == "harp":
            return run_harp_simulation(wdn, failed_pipes=fp, custom_priorities=cp, k=req.k, **scenario_kwargs)

        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("India simulate error")
        raise _server_error(e)


@app.post("/india/wdn/{city}/regenerate")
def india_regenerate_wdn(city: str, n_nodes: int = 20):
    from backend.geo.india_wdn import INDIA_CITIES, fetch_city_pois
    if city not in INDIA_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
    try:
        wdn = fetch_city_pois(city, n_nodes=n_nodes, force_regen=True)
        return {"message": f"Regenerated WDN for {city}", "n_nodes": wdn["n_nodes"], "n_pipes": wdn["n_pipes"]}
    except Exception as e:
        raise _server_error(e)


@app.post("/india/priority/override")
def india_priority_override(req: PriorityOverrideRequest):
    """Override node type/priority — persisted to SQLite, survives restarts."""
    weight = req.new_weight if req.new_weight > 0 else NODE_TYPE_WEIGHTS.get(req.new_type, 1.0)
    _priority_overrides.setdefault(req.city, {})[req.node_id] = {
        "type": req.new_type,
        "weight": weight,
    }
    _save_override_to_db(req.city, req.node_id, req.new_type, weight)
    return {
        "city": req.city,
        "node_id": req.node_id,
        "new_type": req.new_type,
        "new_weight": weight,
        "acknowledged": True,
        "total_overrides_for_city": len(_priority_overrides.get(req.city, {})),
    }


@app.get("/india/priority/overrides/{city}")
def india_get_priority_overrides(city: str):
    """Return all active priority overrides for a city."""
    return {"city": city, "overrides": _priority_overrides.get(city, {})}


@app.delete("/india/priority/overrides/{city}")
def india_clear_priority_overrides(city: str):
    """Clear all priority overrides for a city — removes from DB too."""
    removed = len(_priority_overrides.pop(city, {}))
    _delete_overrides_from_db(city)
    return {"city": city, "cleared": removed}


# ---------------------------------------------------------------------------
# Scenario endpoints (WNTR networks)
# ---------------------------------------------------------------------------

class LeakRequest(BaseModel):
    network: str = "Net3"
    pipe_id: str
    emitter_coefficient: float = 0.001


class SurgeRequest(BaseModel):
    network: str = "Net3"
    node_ids: list
    multiplier: float = 2.5


class MultiFailRequest(BaseModel):
    network: str = "Net3"
    pipe_ids: list
    method: str = "harp"
    k: int = 2


@app.post("/simulate/leak")
def simulate_leak(req: LeakRequest):
    """Pipe stays OPEN but loses flow via emitter at split midpoint."""
    try:
        from backend.wntr_engine import inject_leak, load_network, run_simulation
        from backend.experiments.metrics import pressure_violation_count
        wn = load_network(req.network)
        if req.pipe_id not in wn.pipe_name_list:
            raise HTTPException(status_code=400, detail=f"Unknown pipe '{req.pipe_id}' in network '{req.network}'")
        wn_leak = inject_leak(wn, req.pipe_id, req.emitter_coefficient)
        sim = run_simulation(wn_leak)
        if not sim["converged"]:
            raise HTTPException(status_code=500, detail="Simulation did not converge")
        pdf = sim["pressure_df"]
        t = pdf.index[min(SNAPSHOT_HOUR, len(pdf.index) - 1)]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}
        return {
            "network": req.network, "pipe_id": req.pipe_id, "scenario": "leak",
            "emitter_coefficient": req.emitter_coefficient,
            "pressures_m": pressures,
            "PVC": pressure_violation_count(pressures),
            "demand_satisfaction": sim.get("demand_satisfaction", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _server_error(e)


@app.post("/simulate/surge")
def simulate_surge(req: SurgeRequest):
    """Demand surge at selected nodes."""
    try:
        from backend.wntr_engine import inject_demand_surge, load_network, run_simulation
        from backend.experiments.metrics import pressure_violation_count
        wn = load_network(req.network)
        unknown = [nid for nid in req.node_ids if nid not in wn.junction_name_list]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown node(s) {unknown} in network '{req.network}'")
        wn_surge = inject_demand_surge(wn, req.node_ids, req.multiplier)
        sim = run_simulation(wn_surge)
        if not sim["converged"]:
            raise HTTPException(status_code=500, detail="Simulation did not converge")
        pdf = sim["pressure_df"]
        t = pdf.index[min(SNAPSHOT_HOUR, len(pdf.index) - 1)]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}
        return {
            "network": req.network, "node_ids": req.node_ids,
            "multiplier": req.multiplier, "scenario": "surge",
            "pressures_m": pressures,
            "PVC": pressure_violation_count(pressures),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _server_error(e)


@app.post("/simulate/multi_failure")
def simulate_multi_failure(req: MultiFailRequest):
    """Simultaneous multi-pipe failure + HARP/Dinic recovery."""
    try:
        from backend.wntr_engine import (
            compute_demand_satisfaction, extract_graph_features,
            inject_multi_failure, load_network, run_simulation,
        )
        from backend.algorithms.harp import HARP
        from backend.gat.inference import load_model, predict_pressures
        from backend.experiments.metrics import critical_demand_satisfaction, pressure_violation_count
        from backend.experiments.runner import _load_priority_config

        priority_config = _load_priority_config(req.network)

        wn = load_network(req.network)
        unknown = [pid for pid in req.pipe_ids if pid not in wn.pipe_name_list]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown pipe(s) {unknown} in network '{req.network}'")
        wn_fail = inject_multi_failure(wn, req.pipe_ids)
        sim = run_simulation(wn_fail)
        if not sim["converged"]:
            raise HTTPException(status_code=500, detail="Simulation did not converge")

        pdf = sim["pressure_df"]
        t = pdf.index[min(SNAPSHOT_HOUR, len(pdf.index) - 1)]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}
        demand_sat = compute_demand_satisfaction(wn_fail, sim)["hour_12_snapshot"]

        if req.method == "harp":
            model, norm_stats = load_model(req.network)
            # wn_fail already has every pipe in req.pipe_ids closed; build_wdn_graph
            # and extract_graph_features now read that real status directly, so a
            # single GAT/HARP pass over wn_fail already reflects the full
            # multi-pipe failure (the failed_pipe_id argument just needs to name
            # one of them for the k-hop subgraph anchor).
            graph_data = extract_graph_features(
                wn_fail, failed_pipe_id=req.pipe_ids[0] if req.pipe_ids else None,
                priority_config=priority_config, network_name=req.network,
            )
            pressures_arr, gat_ms = predict_pressures(model, norm_stats, graph_data)
            gat_pressures = dict(zip(graph_data["node_ids"], pressures_arr.tolist()))

            harp = HARP(safety_margin_m=1.0, q_estimate=0.005)
            harp_result = harp.run(
                wn=wn_fail, failed_pipe_id=req.pipe_ids[0] if req.pipe_ids else None,
                gat_pressures_dict=gat_pressures,
                priority_config=priority_config, k=req.k,
            )
            cds = critical_demand_satisfaction(harp_result.demand_satisfaction, priority_config)
            return {
                "network": req.network, "pipe_ids": req.pipe_ids,
                "method": "harp", "CDS": cds,
                "demand_satisfaction": harp_result.demand_satisfaction,
                "pressures_m": gat_pressures,
                "PVC": pressure_violation_count(gat_pressures),
                "computation_time_ms": harp_result.computation_time_ms,
                "n_failed": len(req.pipe_ids),
            }

        cds_base = critical_demand_satisfaction(demand_sat, priority_config)
        return {
            "network": req.network, "pipe_ids": req.pipe_ids,
            "method": "none", "pressures_m": pressures,
            "demand_satisfaction": demand_sat,
            "PVC": pressure_violation_count(pressures),
            "CDS": cds_base,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Multi-failure error")
        raise _server_error(e)


# ---------------------------------------------------------------------------
# Real-time IoT Sensor Streaming (Server-Sent Events)
# ---------------------------------------------------------------------------

async def _iot_sensor_stream(city: str, n_nodes: int = 20, tick_ms: float = 2500.0):
    """
    Async generator yielding Server-Sent Events for simulated SCADA/IoT sensor
    reads on an Indian city WDN.  Uses asyncio.sleep so the event loop is free
    to serve other requests between ticks (fixes H4 — synchronous time.sleep
    blocked the thread pool under concurrent clients).
    """
    try:
        from backend.geo.india_wdn import fetch_city_pois, INDIA_CITIES
        if city not in INDIA_CITIES:
            yield f"event: error\ndata: {{\"error\": \"Unknown city: {city}\"}}\n\n"
            return
        wdn = fetch_city_pois(city, n_nodes=n_nodes)
    except Exception as e:
        logger.warning("IoT stream setup failed for %s: %s", city, e)
        yield 'event: error\ndata: {"error": "Could not start sensor stream"}\n\n'
        return

    # Per-stream entropy-seeded RNG so concurrent clients see different series.
    np_rng = np.random.default_rng()

    nodes = wdn["nodes"]
    node_ids   = [n["id"]   for n in nodes]
    node_types = [n.get("type", "junction") for n in nodes]

    # Base pressures derived from node type
    BASE_P = {
        "reservoir": 80, "water_tower": 75, "hospital": 55,
        "fire_station": 50, "school": 45, "college": 45,
        "market": 40, "park": 38, "residential": 35, "junction": 32,
    }
    base = np.array([BASE_P.get(t, 35.0) for t in node_types], dtype=float)

    tick = 0
    while True:
        elapsed = tick * (tick_ms / 1000.0)

        # Sine-wave pressure fluctuation + Gaussian noise
        fluctuation = 2.0 * np.sin(elapsed * 0.05 + np_rng.random(len(base)) * 0.5)
        noise       = np_rng.normal(0, 0.4, len(base))
        pressures   = (base + fluctuation + noise).clip(5, 100).tolist()

        warnings = [node_ids[i] for i, p in enumerate(pressures) if p < 14]

        event_data = {
            "t": round(elapsed, 1),
            "tick": tick,
            "city": city,
            "pressures": {nid: round(p, 2) for nid, p in zip(node_ids, pressures)},
            "warnings": warnings,
            "n_warnings": len(warnings),
            "timestamp": time.time(),
        }
        yield f"data: {json.dumps(event_data)}\n\n"
        tick += 1
        await asyncio.sleep(tick_ms / 1000.0)


@app.get("/india/events/{city}")
async def india_iot_events(city: str, n_nodes: int = 20, tick_ms: float = 2500.0):
    """
    Server-Sent Events stream of simulated IoT sensor readings for a city WDN.

    Connect with:
        const es = new EventSource('/india/events/bangalore')
        es.onmessage = e => console.log(JSON.parse(e.data))

    Each event contains: t (seconds), pressures (node→head_m), warnings ([node_ids]).
    Streams indefinitely; client disconnects to stop.
    Tick interval: tick_ms milliseconds (default 2500ms = 2.5s SCADA rate).
    """
    return StreamingResponse(
        _iot_sensor_stream(city, n_nodes=n_nodes, tick_ms=tick_ms),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_report_status: dict = {"state": "idle", "started_at": None, "finished_at": None, "error": None}


def _run_report_generation() -> None:
    """Background task: generate India WDN figures without blocking the event loop."""
    _report_status.update({"state": "running", "started_at": time.time(), "error": None})
    try:
        from backend.experiments.india_visualizer import generate_all
        generate_all()
        _report_status.update({"state": "done", "finished_at": time.time()})
    except Exception as e:
        logger.exception("Report generation error")
        _report_status.update({"state": "error", "error": str(e), "finished_at": time.time()})


@app.get("/india/report/generate")
def india_generate_report_graphs(background_tasks: BackgroundTasks):
    """
    Trigger generation of all India WDN report figures as a background task.
    Returns immediately with status=generating. Poll /india/report/status to
    check completion, then /results/figures to list the generated files.
    Running while already generating is a no-op (returns current status).
    """
    if _report_status["state"] == "running":
        return {"status": "already_running", "started_at": _report_status["started_at"]}
    background_tasks.add_task(_run_report_generation)
    return {
        "status": "generating",
        "message": "Report generation started in background. Poll /india/report/status for completion.",
        "check_figures_at": "/results/figures",
    }


@app.get("/india/report/status")
def india_report_status():
    """Return current state of background report generation."""
    figs = []
    if _report_status["state"] == "done" and os.path.exists(FIGURES_DIR):
        figs = sorted(f for f in os.listdir(FIGURES_DIR) if f.startswith("india_") and f.endswith(".png"))
    return {**_report_status, "figures": figs}


# Serve built React frontend (production). Must be registered AFTER all API
# routes so the SPA catch-all doesn't shadow them.
_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):  # noqa: ARG001
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render (and similar PaaS) inject PORT
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
