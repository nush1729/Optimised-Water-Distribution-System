import copy
import logging
import math
import os
import pickle
from collections import deque

import numpy as np
import wntr
import wntr.morph  # noqa: F401  (split_pipe)

from backend.config import (
    SNAPSHOT_HOUR,
    GENERATED_DIR,
    MIN_PRESSURE_M,
    NETWORK_FILES,
    REQUIRED_PRESSURE_M,
)

logger = logging.getLogger(__name__)


LEAK_DISCHARGE_COEFF = 0.75   # orifice discharge coefficient used by WNTR's leak model


def add_leak_from_emitter(wn, node_name: str, emitter_coefficient: float) -> None:
    """Make `node_name` leak like an emitter Q = C * sqrt(P), in the way
    WNTRSimulator understands.

    WNTRSimulator silently ignores Junction.emitter_coefficient (that attribute
    only exists for the EPANET solver); leaks must be added with add_leak(area,
    discharge_coeff), Q = Cd * A * sqrt(2 g P). Equating the two gives
    A = C / (Cd * sqrt(2 g)).
    """
    if emitter_coefficient <= 0:
        return
    area = emitter_coefficient / (LEAK_DISCHARGE_COEFF * (2.0 * 9.81) ** 0.5)
    # start_time=0 adds a control that switches the leak on at t=0 (add_leak()
    # alone creates it switched off, and leak_status is read-only).
    wn.get_node(node_name).add_leak(wn, area=area, discharge_coeff=LEAK_DISCHARGE_COEFF, start_time=0)


def scale_node_demand(node, factor: float) -> None:
    """Multiply a junction's demand by `factor`.

    In WNTR >= 1.x `Junction.base_demand` is read-only (assigning raises), so the
    demand has to be changed on its demand timeseries. Older code assigned to
    `base_demand` inside `try/except Exception: pass`, which silently did
    nothing: demand surges and the demand noise used for the GAT training data
    and the anomaly baseline were never applied.
    """
    for ts in node.demand_timeseries_list:
        ts.base_value = float(ts.base_value) * factor


def load_network(name: str) -> wntr.network.WaterNetworkModel:
    """Load network from .inp file. Configure for PDA simulation."""
    path = NETWORK_FILES.get(name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Network file not found for '{name}': {path}")
    wn = wntr.network.WaterNetworkModel(path)
    # PDA settings: pressure-dependent demand
    wn.options.hydraulic.demand_model = "PDA"
    wn.options.hydraulic.minimum_pressure = MIN_PRESSURE_M
    wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE_M
    wn.options.hydraulic.headloss = "H-W"
    return wn


def run_simulation(wn: wntr.network.WaterNetworkModel) -> dict:
    """Run 24-hour extended period simulation, 1-hour timestep."""
    wn_copy = copy.deepcopy(wn)
    wn_copy.options.time.duration = 24 * 3600
    wn_copy.options.time.hydraulic_timestep = 3600
    wn_copy.options.time.report_timestep = 3600

    try:
        sim = wntr.sim.WNTRSimulator(wn_copy)
        results = sim.run_sim()

        # WNTR returns: index=timesteps_seconds, columns=node/link names
        pressure_df = results.node["pressure"]
        flow_df = results.link["flowrate"]
        demand_df = results.node["demand"]

        # Convert index from seconds to hours
        pressure_df.index = [int(t / 3600) for t in pressure_df.index]
        flow_df.index = [int(t / 3600) for t in flow_df.index]
        demand_df.index = [int(t / 3600) for t in demand_df.index]

        # Check convergence: if all pressures are NaN → failed
        converged = not pressure_df.isnull().all(axis=None)

        return {
            "pressure_df": pressure_df,
            "flow_df": flow_df,
            "demand_df": demand_df,
            "converged": converged,
        }
    except Exception as exc:
        logger.warning("WNTR simulation failed: %s", exc)
        return {
            "pressure_df": None,
            "flow_df": None,
            "demand_df": None,
            "converged": False,
        }


def inject_failure(
    wn: wntr.network.WaterNetworkModel, pipe_id: str
) -> wntr.network.WaterNetworkModel:
    """Set pipe_id status to CLOSED. Returns modified deepcopy."""
    if pipe_id not in wn.pipe_name_list:
        raise ValueError(f"Pipe '{pipe_id}' not found in network")
    wn_failed = copy.deepcopy(wn)
    pipe = wn_failed.get_link(pipe_id)
    pipe.initial_status = wntr.network.LinkStatus.Closed
    return wn_failed


def _get_priority_features(node_id: str, priority_config: dict, network_name: str) -> list:
    """Return one-hot [hospital, school, fire, residential] for a node."""
    # priority_config is already the network-specific dict (e.g. the value of
    # config["Net3"]), so no second .get(network_name) lookup is needed.
    cfg = priority_config
    if node_id in cfg.get("hospital_nodes", []):
        return [1, 0, 0, 0]
    if node_id in cfg.get("school_nodes", []):
        return [0, 1, 0, 0]
    if node_id in cfg.get("fire_station_nodes", []):
        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def extract_graph_features(
    wn: wntr.network.WaterNetworkModel,
    failed_pipe_id: str = None,
    demand_multipliers: dict = None,
    priority_config: dict = None,
    network_name: str = "Net3",
    reference_demands: dict = None,
    extended: bool = False,
) -> dict:
    """Extract PyG-compatible graph representation from WNTR network.

    extended=True appends two load features (10 node columns instead of 8):
      [8] demand ratio  = this node's demand / its reference (intact) demand
      [9] load factor   = total demand / total reference demand (same for every node)
    They tell the model how far today's demand is from normal, so it can separate
    demand-driven pressure shifts from failure-driven ones. reference_demands maps
    node id -> intact demand; when omitted the network's own demands are the
    reference, giving ratio = load = 1 (what inference sees for the base network).
    """
    if demand_multipliers is None:
        demand_multipliers = {}
    if priority_config is None:
        priority_config = {}

    # Ordered node list
    node_ids = sorted(
        wn.junction_name_list + wn.reservoir_name_list + wn.tank_name_list
    )
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    # Node features (N, 8)
    node_features = np.zeros((N, 10 if extended else 8), dtype=np.float32)
    failed_pipe_endpoints: list = []
    closed_pipe_ids: set = set()
    for i, nid in enumerate(node_ids):
        node = wn.get_node(nid)
        # WNTR's Reservoir class has no `elevation` attribute at all — a
        # reservoir's supply head is `base_head` instead — so this used to
        # silently feed the model elevation=0 for every reservoir, discarding
        # the exact signal (River ≈ 67m, Lake ≈ 51m here) that tells the
        # model "this is where high pressure comes from". Tanks do have
        # `elevation`, but that alone excludes the water level sitting on
        # top of it; true hydraulic head there is elevation + level.
        if nid in wn.reservoir_name_list:
            elev = getattr(node, "base_head", 0.0) or 0.0
        elif nid in wn.tank_name_list:
            elev = (getattr(node, "elevation", 0.0) or 0.0) + (getattr(node, "level", 0.0) or 0.0)
        else:
            elev = getattr(node, "elevation", 0.0) or 0.0
        is_res = 1.0 if nid in wn.reservoir_name_list or nid in wn.tank_name_list else 0.0
        base_d = 0.0
        if nid in wn.junction_name_list:
            # demand_multipliers is only for callers whose network still carries the
            # unscaled demand; the training generator scales the network itself.
            base_d = float(node.base_demand) * demand_multipliers.get(nid, 1.0)
        ph = _get_priority_features(nid, priority_config, network_name)
        # 8th feature (proximity_to_failure) filled in below once the open-
        # edge adjacency and failed-pipe endpoints are known.
        node_features[i, :8] = [elev, base_d, is_res] + ph + [0.0]

    # Edge list from pipes (bidirectional)
    pipe_ids_ordered = sorted(wn.pipe_name_list)
    src_list, dst_list, efeats = [], [], []
    open_adjacency: dict = {}

    for pid in pipe_ids_ordered:
        pipe = wn.get_link(pid)
        n1, n2 = pipe.start_node_name, pipe.end_node_name
        if n1 not in node_idx or n2 not in node_idx:
            continue
        L = pipe.length or 1.0
        D = pipe.diameter or 0.1
        C = pipe.roughness or 100.0
        # A pipe reads as failed if it's the explicit failed_pipe_id argument
        # or if `wn` already carries a closed status (e.g. built from
        # inject_multi_failure()) — otherwise a multi-pipe failure would only
        # ever mark one of the closed pipes as such in the GAT's input.
        _status = getattr(pipe, "initial_status", None)
        already_closed = _status is not None and _status == wntr.network.LinkStatus.Closed
        is_failed_edge = pid == failed_pipe_id or already_closed
        status = 0.0 if is_failed_edge else 1.0
        if is_failed_edge:
            closed_pipe_ids.add(pid)
            if pid == failed_pipe_id:
                failed_pipe_endpoints.extend([n1, n2])
        else:
            open_adjacency.setdefault(n1, []).append(n2)
            open_adjacency.setdefault(n2, []).append(n1)
        # log resistance factor
        log_res = math.log10(max(10.67 * L / ((C ** 1.852) * (D ** 4.87)), 1e-9))
        feat = [L, D, C, C, status, log_res]  # C_hw appears twice per design spec
        # Forward edge
        src_list.append(node_idx[n1])
        dst_list.append(node_idx[n2])
        efeats.append(feat)
        # Reverse edge
        src_list.append(node_idx[n2])
        dst_list.append(node_idx[n1])
        efeats.append(feat)

    # Proximity-to-failure node feature: 1 / (1 + hop_distance) from the
    # failed pipe's endpoints, BFS'd over the *open* (post-failure) topology.
    # Only 2 of ~234 directed edges actually change when a pipe fails (a
    # ~0.85% perturbation), and node features otherwise carry zero failure
    # information at all — the model has to rely entirely on 3 rounds of
    # message passing to propagate that single sparse edge bit outward.
    # Giving every node a direct, explicit "how close am I to the failure"
    # signal removes the dependency on receptive-field depth for that
    # propagation. Unreachable nodes (real disconnection) naturally get 0.0.
    if failed_pipe_endpoints:
        dist = {n: 0 for n in failed_pipe_endpoints if n in node_idx}
        frontier = deque(dist.keys())
        while frontier:
            u = frontier.popleft()
            for v in open_adjacency.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    frontier.append(v)
        for nid, d in dist.items():
            node_features[node_idx[nid], 7] = 1.0 / (1.0 + d)

    if extended:
        ref = reference_demands or {}
        tot = ref_tot = 0.0
        for i, nid in enumerate(node_ids):
            d = float(node_features[i, 1])
            r = float(ref.get(nid, d))
            tot += d
            ref_tot += r
            node_features[i, 8] = d / r if r > 0 else 1.0
        node_features[:, 9] = tot / ref_tot if ref_tot > 0 else 1.0

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_features = np.array(efeats, dtype=np.float32)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "node_ids": node_ids,
        "pipe_ids": pipe_ids_ordered * 2,  # each pipe appears twice
    }


def compute_demand_satisfaction(
    wn: wntr.network.WaterNetworkModel, sim_results: dict
) -> dict:
    """Compute demand satisfaction ratio per node at each timestep."""
    if sim_results["demand_df"] is None:
        return {"by_node": {}, "hour_12_snapshot": {}}

    demand_df = sim_results["demand_df"]
    junction_ids = wn.junction_name_list

    # demand_df: index=hours, columns=node_ids
    by_node = {}
    for nid in junction_ids:
        if nid not in demand_df.columns:
            continue
        node = wn.get_node(nid)
        base = getattr(node, "base_demand", 0.0) or 0.0
        series = demand_df[nid]
        if base <= 0:
            by_node[nid] = [1.0] * len(series)
        else:
            ratios = [min(max(float(v) / base, 0.0), 1.0) for v in series]
            by_node[nid] = ratios

    # Snapshot at hour 12
    hour_12_idx = min(SNAPSHOT_HOUR, len(demand_df.index) - 1)
    snapshot = {}
    for nid in by_node:
        snapshot[nid] = by_node[nid][hour_12_idx] if hour_12_idx < len(by_node[nid]) else 0.0
    return {"by_node": by_node, "hour_12_snapshot": snapshot}


def generate_baseline_distribution(
    network_name: str, n_runs: int = 100, seed: int = 42
) -> dict:
    """Run n_runs normal simulations with ±10% demand noise; compute per-node pressure stats."""
    np.random.seed(seed)
    wn_base = load_network(network_name)
    junction_ids = wn_base.junction_name_list
    n_hours = 25  # 0..24

    pressure_by_node = {nid: [[] for _ in range(n_hours)] for nid in junction_ids}

    for run_i in range(n_runs):
        wn = copy.deepcopy(wn_base)
        # Apply ±10% demand noise
        for nid in wn.junction_name_list:
            node = wn.get_node(nid)
            scale_node_demand(node, np.random.uniform(0.9, 1.1))
        result = run_simulation(wn)
        if not result["converged"] or result["pressure_df"] is None:
            continue
        pdf = result["pressure_df"]
        for nid in junction_ids:
            if nid not in pdf.columns:
                continue
            for t_idx, t in enumerate(pdf.index):
                val = float(pdf.at[t, nid])
                if t_idx < n_hours:
                    pressure_by_node[nid][t_idx].append(val)

        if (run_i + 1) % 20 == 0:
            logger.info("Baseline run %d/%d", run_i + 1, n_runs)

    baseline = {}
    for nid in junction_ids:
        mean_by_hour, std_by_hour = [], []
        for h in range(n_hours):
            vals = pressure_by_node[nid][h]
            mean_by_hour.append(float(np.mean(vals)) if vals else 0.0)
            std_by_hour.append(float(np.std(vals)) if vals else 1.0)
        baseline[nid] = {"mean_by_hour": mean_by_hour, "std_by_hour": std_by_hour}

    out_path = os.path.join(GENERATED_DIR, f"{network_name}_baseline.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(baseline, f)
    logger.info("Baseline saved to %s", out_path)
    return baseline


def inject_leak(
    wn: wntr.network.WaterNetworkModel,
    pipe_id: str,
    emitter_coefficient: float = 0.001,
) -> wntr.network.WaterNetworkModel:
    """Pipe stays OPEN but leaks: split it at its midpoint and put an emitter there.

    The leak flow is Q = C * sqrt(P) at the new midpoint junction (orifice flow),
    so it tapers off as local pressure drops (see add_leak_from_emitter). (The previous implementation
    assigned to Pipe.end_node_name, which WNTR 1.x no longer allows, so this
    scenario raised on every call.)
    """
    if pipe_id not in wn.pipe_name_list:
        raise ValueError(f"Pipe '{pipe_id}' not found")
    wn_leak = copy.deepcopy(wn)
    mid_name = f"_leak_{pipe_id}"
    wntr.morph.split_pipe(wn_leak, pipe_id, f"{pipe_id}_b", mid_name, return_copy=False)
    add_leak_from_emitter(wn_leak, mid_name, emitter_coefficient)
    return wn_leak


def inject_demand_surge(
    wn: wntr.network.WaterNetworkModel,
    node_ids: list,
    multiplier: float = 2.5,
) -> wntr.network.WaterNetworkModel:
    """Multiply base_demand by multiplier for selected nodes."""
    wn_surge = copy.deepcopy(wn)
    for nid in node_ids:
        if nid in wn_surge.junction_name_list:
            node = wn_surge.get_node(nid)
            scale_node_demand(node, multiplier)
    return wn_surge


def inject_multi_failure(
    wn: wntr.network.WaterNetworkModel, pipe_ids: list
) -> wntr.network.WaterNetworkModel:
    """Set all pipe_ids to CLOSED simultaneously."""
    wn_failed = copy.deepcopy(wn)
    for pid in pipe_ids:
        if pid in wn_failed.pipe_name_list:
            pipe = wn_failed.get_link(pid)
            pipe.initial_status = wntr.network.LinkStatus.Closed
    return wn_failed
