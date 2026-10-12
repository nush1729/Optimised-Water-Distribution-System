"""Experiment runner for HARP-WDN evaluation."""
import argparse
import csv
import json
import logging
import os
import random
import sqlite3
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import (
    load_priority_config,
    SNAPSHOT_HOUR,
    DEFAULT_K,
    GENERATED_DIR,
    RESULTS_DIR,
    SEED_TEST,
    SEED_TRAIN,
)
from backend.algorithms.dinic import Dinic
from backend.algorithms.harp import HARP
from backend.algorithms.subgraph import build_wdn_graph
from backend.experiments.metrics import (
    critical_demand_satisfaction,
    pressure_violation_count,
    water_loss_volume,
)
from backend.gat.inference import load_model, predict_pressures
from backend.wntr_engine import (
    extract_graph_features,
    inject_failure,
    load_network,
    run_simulation,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_priority_config(network_name: str) -> dict:
    return load_priority_config(network_name)


def _get_base_demands(wn) -> dict:
    return {
        nid: max(float(wn.get_node(nid).base_demand or 0.0), 0.0)
        for nid in wn.junction_name_list
    }


def _run_no_recovery(wn_base, pipe_id: str, priority_config: dict) -> dict:
    """Method A: just inject failure, measure resulting state."""
    wn_fail = inject_failure(wn_base, pipe_id)
    sim = run_simulation(wn_fail)
    if not sim["converged"] or sim["pressure_df"] is None:
        n = len(wn_base.junction_name_list)
        demand_satisfaction = {nid: 0.0 for nid in wn_base.junction_name_list}
        pressures = {nid: 0.0 for nid in wn_base.junction_name_list}
    else:
        pdf = sim["pressure_df"]
        ddf = sim["demand_df"]
        t12 = min(SNAPSHOT_HOUR, len(pdf.index) - 1)
        t = pdf.index[t12]
        pressures = {nid: float(pdf.at[t, nid]) for nid in pdf.columns}
        demand_satisfaction = {}
        for nid in wn_base.junction_name_list:
            node = wn_base.get_node(nid)
            base = float(node.base_demand or 0.0)
            if base <= 0:
                demand_satisfaction[nid] = 1.0
            else:
                actual = float(ddf.at[t, nid]) if nid in ddf.columns else 0.0
                demand_satisfaction[nid] = min(max(actual / base, 0.0), 1.0)

    cds = critical_demand_satisfaction(demand_satisfaction, priority_config)
    pvc = pressure_violation_count(pressures)
    wlv = water_loss_volume(demand_satisfaction, _get_base_demands(wn_base))
    return {"demand_satisfaction": demand_satisfaction, "pressures": pressures,
            "cds": cds, "PVC": pvc, "WLV_m3": wlv, "CT_ms": 0.0}


def _run_dinic(wn_base, pipe_id: str, priority_config: dict) -> dict:
    """Method B: Standard Dinic's (graph-theoretic, hydraulically blind).

    Dinic's has no hydraulic model, so it cannot know node pressures. Both
    fields are reported as None rather than a fabricated constant, since a
    made-up pressure would silently zero out PVC for every scenario.
    """
    t0 = time.perf_counter()
    G, _ = build_wdn_graph(wn_base, pipe_id, priority_config)
    reservoir_nodes = set(wn_base.reservoir_name_list + wn_base.tank_name_list)
    source_nodes = [n for n in G.nodes if n in reservoir_nodes]
    sink_nodes = [n for n in G.nodes
                  if G.nodes[n].get("base_demand", 0) > 0
                  and not G.nodes[n].get("is_reservoir", False)]

    flow_result = None
    demand_satisfaction = {}
    if source_nodes and sink_nodes:
        # Multi-source via a virtual supersource connected to every reservoir
        # (Net3 has 5: River, Lake, and three tanks). Previously this ran
        # Dinic from source_nodes[0] alone — an arbitrary single reservoir —
        # so any node unreachable from that one reservoir specifically read
        # as permanently unsatisfied regardless of which pipe failed,
        # independent of whether the other reservoirs could actually reach it.
        G_multi_source = G.copy()
        supersrc = "__DINIC_SUPERSRC__"
        G_multi_source.add_node(supersrc)
        for s in source_nodes:
            G_multi_source.add_edge(supersrc, s, diameter=100.0)
        dinic = Dinic()
        flow_result = dinic.run(G_multi_source, supersrc, sink_nodes)
        demand_satisfaction = flow_result.demand_satisfaction

    # Fill missing nodes
    for nid in wn_base.junction_name_list:
        if nid not in demand_satisfaction:
            demand_satisfaction[nid] = 0.0

    ct_ms = (time.perf_counter() - t0) * 1000.0
    cds = critical_demand_satisfaction(demand_satisfaction, priority_config)
    wlv = water_loss_volume(demand_satisfaction, _get_base_demands(wn_base))
    return {"demand_satisfaction": demand_satisfaction, "pressures": None,
            "cds": cds, "PVC": None, "WLV_m3": wlv, "CT_ms": ct_ms}


def _run_harp(wn_base, pipe_id: str, priority_config: dict,
              model, norm_stats, k: int = DEFAULT_K,
              use_hydraulic: bool = True, use_priority: bool = True,
              use_subgraph: bool = True) -> dict:
    """Method C: HARP + GAT."""
    # GAT inference
    wn_fail = inject_failure(wn_base, pipe_id)
    graph_data = extract_graph_features(
        wn_fail,
        failed_pipe_id=pipe_id,
        priority_config=priority_config,
        network_name=wn_base.name if hasattr(wn_base, 'name') else "Net3",
    )
    pressures_arr, gat_ms = predict_pressures(model, norm_stats, graph_data)
    node_ids = graph_data["node_ids"]
    gat_pressures = dict(zip(node_ids, pressures_arr.tolist()))

    harp = HARP()
    result = harp.run(
        wn=wn_base,
        failed_pipe_id=pipe_id,
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=k,
        use_hydraulic_check=use_hydraulic,
        use_priority=use_priority,
        use_subgraph=use_subgraph,
        gat_inference_time_ms=gat_ms,
    )

    demand_satisfaction = result.demand_satisfaction
    for nid in wn_base.junction_name_list:
        if nid not in demand_satisfaction:
            demand_satisfaction[nid] = 0.0

    cds = critical_demand_satisfaction(demand_satisfaction, priority_config)
    pvc = pressure_violation_count(gat_pressures)
    wlv = water_loss_volume(demand_satisfaction, _get_base_demands(wn_base))
    return {
        "demand_satisfaction": demand_satisfaction,
        "pressures": gat_pressures,
        "cds": cds,
        "PVC": pvc,
        "WLV_m3": wlv,
        "CT_ms": result.computation_time_ms,
        "gat_ms": gat_ms,
        "escalation_flag": result.escalation_flag,
        "k_used": result.k_used,
        "harp_result": result,
    }


def run_exp1(network_name: str = "Net3", n_scenarios: int = 20):
    """Experiment 1: Core Algorithm Comparison."""
    print(f"\n=== Experiment 1: Core Comparison ({network_name}, {n_scenarios} scenarios) ===")
    np.random.seed(SEED_TRAIN)
    priority_config = _load_priority_config(network_name)
    wn_base = load_network(network_name)
    model, norm_stats = load_model(network_name)

    pipes = list(wn_base.pipe_name_list)
    np.random.seed(SEED_TRAIN)
    selected = list(np.random.choice(pipes, min(n_scenarios, len(pipes)), replace=False))

    rows = []
    for i, pipe_id in enumerate(selected):
        print(f"  Scenario {i+1}/{len(selected)}: pipe={pipe_id}")
        try:
            rA = _run_no_recovery(wn_base, pipe_id, priority_config)
            rB = _run_dinic(wn_base, pipe_id, priority_config)
            rC = _run_harp(wn_base, pipe_id, priority_config, model, norm_stats)
        except Exception as e:
            logger.warning("Scenario %s failed: %s", pipe_id, e)
            continue

        for method, r in [("none", rA), ("dinic", rB), ("harp", rC)]:
            rows.append({
                "scenario_id": i,
                "pipe_failed": pipe_id,
                "method": method,
                "CDS_hospital": r["cds"]["CDS_hospital"],
                "CDS_school": r["cds"]["CDS_school"],
                "CDS_fire": r["cds"]["CDS_fire"],
                "CDS_residential": r["cds"]["CDS_residential"],
                "CDS_overall": r["cds"]["CDS_overall"],
                "PVC": r["PVC"],
                "WLV_m3": r["WLV_m3"],
                "CT_ms": r["CT_ms"],
            })

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_1.csv"))
    print(f"  Saved experiment_1.csv ({len(rows)} rows)")
    return rows


def run_exp2(network_name: str = "Net3"):
    """Experiment 2: GAT Accuracy Evaluation."""
    import pickle
    print(f"\n=== Experiment 2: GAT Accuracy ({network_name}) ===")
    model, norm_stats = load_model(network_name)

    split_path = os.path.join(GENERATED_DIR, f"{network_name}_split.json")
    with open(split_path) as f:
        split = json.load(f)
    test_ids = split.get("test", [])

    rows = []
    inf_times = []
    for idx in test_ids:
        pkl_path = os.path.join(GENERATED_DIR, f"{network_name}_{idx:04d}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            sc = pickle.load(f)

        pressures_arr, dt = predict_pressures(model, norm_stats, sc)
        inf_times.append(dt)
        true_heads = sc["pressure_heads_m"]
        node_ids = sc["node_ids"]

        for j, nid in enumerate(node_ids):
            true_h = float(true_heads[j])
            pred_h = float(pressures_arr[j])
            ae = abs(true_h - pred_h)
            # Only compute relative error for nodes with meaningful pressure (>1m)
            re = ae / true_h * 100 if true_h > 1.0 else float("nan")
            rows.append({
                "scenario_id": idx,
                "network": network_name,
                "node_id": nid,
                "true_head_m": true_h,
                "pred_head_m": pred_h,
                "abs_error_m": ae,
                "relative_error_pct": re,
            })

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_2.csv"))
    if rows:
        mae = np.mean([r["abs_error_m"] for r in rows])
        rmse = np.sqrt(np.mean([r["abs_error_m"]**2 for r in rows]))
        re_vals = [r["relative_error_pct"] for r in rows
                   if r["relative_error_pct"] == r["relative_error_pct"]]  # exclude NaN
        mre = np.mean(re_vals) if re_vals else float("nan")
        print(f"  MAE={mae:.3f}m RMSE={rmse:.3f}m MRE={mre:.1f}% InfTime={np.mean(inf_times):.1f}ms")
    return rows


def run_exp3(network_name: str = "Net3", n_scenarios: int = 20):
    """Experiment 3: Subgraph k Analysis."""
    print(f"\n=== Experiment 3: Subgraph k Analysis ({network_name}) ===")
    np.random.seed(SEED_TRAIN)
    priority_config = _load_priority_config(network_name)
    wn_base = load_network(network_name)
    model, norm_stats = load_model(network_name)

    pipes = list(wn_base.pipe_name_list)
    np.random.seed(SEED_TRAIN)
    selected = list(np.random.choice(pipes, min(n_scenarios, len(pipes)), replace=False))

    rows = []
    for i, pipe_id in enumerate(selected):
        for k in [1, 2, 3, 4, 99]:  # 99 = full graph
            use_sub = k != 99
            k_actual = k if k != 99 else DEFAULT_K
            try:
                r = _run_harp(wn_base, pipe_id, priority_config, model, norm_stats,
                              k=k_actual, use_subgraph=use_sub)
                cds = r["cds"]["CDS_critical"]
            except Exception as e:
                cds = 0.0
                r = {"CT_ms": 0.0, "escalation_flag": False}
            rows.append({
                "scenario_id": i,
                "pipe_failed": pipe_id,
                "k": "full" if k == 99 else k,
                "CDS_critical": cds,
                "CT_ms": r["CT_ms"],
                "escalation_flag": r.get("escalation_flag", False),
            })

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_3.csv"))
    print(f"  Saved experiment_3.csv ({len(rows)} rows)")
    return rows


def run_exp4(network_name: str = "Net3"):
    """Experiment 4: Priority Breakdown."""
    print(f"\n=== Experiment 4: Priority Breakdown ({network_name}) ===")
    exp1_path = os.path.join(RESULTS_DIR, "experiment_1.csv")
    if not os.path.exists(exp1_path):
        print("  experiment_1.csv not found, running exp1 first")
        run_exp1(network_name)

    with open(exp1_path) as f:
        reader = csv.DictReader(f)
        exp1_rows = list(reader)

    # Find 5 worst failures by WLV (from 'none' method)
    none_rows = [r for r in exp1_rows if r["method"] == "none"]
    none_rows.sort(key=lambda r: -float(r.get("WLV_m3", 0)))
    worst_pipes = [r["pipe_failed"] for r in none_rows[:5]]

    priority_config = _load_priority_config(network_name)
    wn_base = load_network(network_name)
    model, norm_stats = load_model(network_name)

    rows = []
    for pipe_id in worst_pipes:
        for method, run_fn in [("dinic", lambda p: _run_dinic(wn_base, p, priority_config)),
                                ("harp", lambda p: _run_harp(wn_base, p, priority_config, model, norm_stats))]:
            try:
                r = run_fn(pipe_id)
            except Exception as e:
                logger.warning("exp4 %s %s: %s", method, pipe_id, e)
                continue
            ds = r["demand_satisfaction"]
            p_cfg = priority_config
            h_nodes = p_cfg.get("hospital_nodes", [])
            f_nodes = p_cfg.get("fire_station_nodes", [])
            s_nodes = p_cfg.get("school_nodes", [])
            priority_set = set(h_nodes + f_nodes + s_nodes)
            type_map = {n: "hospital" for n in h_nodes}
            type_map.update({n: "fire_station" for n in f_nodes})
            type_map.update({n: "school" for n in s_nodes})
            for nid, sat in ds.items():
                rows.append({
                    "scenario_id": pipe_id,
                    "method": method,
                    "node_id": nid,
                    "node_type": type_map.get(nid, "residential"),
                    "demand_satisfaction": sat,
                })

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_4.csv"))
    print(f"  Saved experiment_4.csv ({len(rows)} rows)")
    return rows


def run_exp5(n_scenarios: int = 10):
    """Experiment 5: Scalability on L-Town."""
    print(f"\n=== Experiment 5: Scalability (L-Town, {n_scenarios} scenarios) ===")
    np.random.seed(SEED_TEST)
    network_name = "L-Town"
    priority_config = _load_priority_config(network_name)

    try:
        wn_base = load_network(network_name)
        model, norm_stats = load_model(network_name)
    except Exception as e:
        print(f"  L-Town model not available: {e}. Run L-Town GAT training first.")
        return []

    pipes = list(wn_base.pipe_name_list)
    np.random.seed(SEED_TEST)
    selected = list(np.random.choice(pipes, min(n_scenarios, len(pipes)), replace=False))

    rows = []
    for i, pipe_id in enumerate(selected):
        for method in ["dinic", "harp_k2"]:
            tracemalloc.start()
            try:
                t0 = time.perf_counter()
                if method == "dinic":
                    r = _run_dinic(wn_base, pipe_id, priority_config)
                else:
                    r = _run_harp(wn_base, pipe_id, priority_config, model, norm_stats, k=2)
                ct_ms = (time.perf_counter() - t0) * 1000.0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({
                    "scenario_id": i,
                    "pipe_failed": pipe_id,
                    "method": method,
                    "CT_ms": min(ct_ms, 30000),
                    "timeout": ct_ms >= 30000,
                    "CDS_critical": r["cds"].get("CDS_critical", 0),
                    "memory_mb": peak / 1e6,
                })
            except Exception as e:
                tracemalloc.stop()
                logger.warning("exp5 %s %s: %s", method, pipe_id, e)
                rows.append({
                    "scenario_id": i,
                    "pipe_failed": pipe_id,
                    "method": method,
                    "CT_ms": 30000,
                    "timeout": True,
                    "CDS_critical": 0.0,
                    "memory_mb": 0.0,
                })

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_5.csv"))
    print(f"  Saved experiment_5.csv ({len(rows)} rows)")
    return rows


def run_exp6(network_name: str = "Net3", n_scenarios: int = 20):
    """Experiment 6: Ablation Study."""
    print(f"\n=== Experiment 6: Ablation ({network_name}) ===")
    np.random.seed(SEED_TRAIN)
    priority_config = _load_priority_config(network_name)
    wn_base = load_network(network_name)
    model, norm_stats = load_model(network_name)

    pipes = list(wn_base.pipe_name_list)
    np.random.seed(SEED_TRAIN)
    selected = list(np.random.choice(pipes, min(n_scenarios, len(pipes)), replace=False))

    variants = {
        "HARP_full":         {"use_hydraulic": True,  "use_priority": True,  "use_subgraph": True},
        "HARP_no_hydraulic": {"use_hydraulic": False, "use_priority": True,  "use_subgraph": True},
        "HARP_no_priority":  {"use_hydraulic": True,  "use_priority": False, "use_subgraph": True},
        "HARP_no_subgraph":  {"use_hydraulic": True,  "use_priority": True,  "use_subgraph": False},
    }

    rows = []
    for i, pipe_id in enumerate(selected):
        for variant_name, flags in variants.items():
            try:
                r = _run_harp(
                    wn_base, pipe_id, priority_config, model, norm_stats,
                    k=DEFAULT_K,
                    use_hydraulic=flags["use_hydraulic"],
                    use_priority=flags["use_priority"],
                    use_subgraph=flags["use_subgraph"],
                )
                rows.append({
                    "scenario_id": i,
                    "pipe_failed": pipe_id,
                    "variant": variant_name,
                    "CDS_hospital": r["cds"]["CDS_hospital"],
                    "CDS_overall": r["cds"]["CDS_overall"],
                    "PVC": r["PVC"],
                    "CT_ms": r["CT_ms"],
                })
            except Exception as e:
                logger.warning("exp6 %s %s: %s", variant_name, pipe_id, e)

    _save_csv(rows, os.path.join(RESULTS_DIR, "experiment_6.csv"))
    print(f"  Saved experiment_6.csv ({len(rows)} rows)")
    return rows


def _save_csv(rows: list, path: str) -> None:
    if not rows:
        logger.warning("Empty rows for %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _init_db():
    """Initialize SQLite database schema."""
    db_path = os.path.join(RESULTS_DIR, "harp_wdn.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY,
        exp_number INTEGER NOT NULL,
        network TEXT NOT NULL,
        run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        n_scenarios INTEGER,
        seed INTEGER
    );
    CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY,
        exp_id INTEGER REFERENCES experiments(id),
        scenario_id INTEGER,
        pipe_failed TEXT NOT NULL,
        network TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS algorithm_runs (
        id INTEGER PRIMARY KEY,
        scenario_id INTEGER REFERENCES scenarios(id),
        algorithm TEXT NOT NULL,
        k_value INTEGER,
        escalation_flag BOOLEAN,
        CT_ms REAL,
        gat_ms REAL
    );
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY,
        algorithm_run_id INTEGER REFERENCES algorithm_runs(id),
        CDS_hospital REAL,
        CDS_school REAL,
        CDS_fire REAL,
        CDS_residential REAL,
        CDS_overall REAL,
        PVC INTEGER,
        WLV_m3 REAL
    );
    CREATE TABLE IF NOT EXISTS gat_accuracy (
        id INTEGER PRIMARY KEY,
        network TEXT,
        scenario_id INTEGER,
        node_id TEXT,
        true_head_m REAL,
        pred_head_m REAL,
        abs_error_m REAL,
        relative_error_pct REAL
    );
    """)
    conn.commit()
    return conn


def write_to_db(exp_num: int, rows: list, network_name: str) -> None:
    """Write experiment results to SQLite after CSV verified.

    exp1 rows are per-(scenario, algorithm) records that map onto the
    normalized scenarios/algorithm_runs/metrics tables; exp2 rows are
    per-(scenario, node) GAT predictions that map onto gat_accuracy.
    Other experiments' row shapes don't correspond 1:1 to those tables,
    so only the top-level experiments row is recorded for them.
    """
    csv_path = os.path.join(RESULTS_DIR, f"experiment_{exp_num}.csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.warning("CSV not verified non-empty for exp %d, skipping DB write", exp_num)
        return
    conn = _init_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO experiments (exp_number, network, n_scenarios, seed) VALUES (?,?,?,?)",
        (exp_num, network_name, len(rows), SEED_TRAIN),
    )
    exp_db_id = cur.lastrowid

    if exp_num == 1:
        # exp1: per-(scenario, algorithm) rows → scenarios + algorithm_runs + metrics
        scenario_db_id = {}
        for r in rows:
            key = r["scenario_id"]
            if key not in scenario_db_id:
                cur.execute(
                    "INSERT INTO scenarios (exp_id, scenario_id, pipe_failed, network) VALUES (?,?,?,?)",
                    (exp_db_id, key, r["pipe_failed"], network_name),
                )
                scenario_db_id[key] = cur.lastrowid
            cur.execute(
                "INSERT INTO algorithm_runs (scenario_id, algorithm, k_value, escalation_flag, CT_ms, gat_ms) "
                "VALUES (?,?,?,?,?,?)",
                (scenario_db_id[key], r["method"], None, None, r.get("CT_ms"), None),
            )
            run_db_id = cur.lastrowid
            cur.execute(
                "INSERT INTO metrics (algorithm_run_id, CDS_hospital, CDS_school, CDS_fire, "
                "CDS_residential, CDS_overall, PVC, WLV_m3) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_db_id, r["CDS_hospital"], r["CDS_school"], r["CDS_fire"],
                    r["CDS_residential"], r["CDS_overall"], r["PVC"], r["WLV_m3"],
                ),
            )
    elif exp_num == 2:
        # exp2: per-(scenario, node) GAT predictions → gat_accuracy
        for r in rows:
            re = r["relative_error_pct"]
            cur.execute(
                "INSERT INTO gat_accuracy (network, scenario_id, node_id, true_head_m, pred_head_m, "
                "abs_error_m, relative_error_pct) VALUES (?,?,?,?,?,?,?)",
                (
                    r["network"], r["scenario_id"], r["node_id"], r["true_head_m"],
                    r["pred_head_m"], r["abs_error_m"], None if re != re else re,
                ),
            )
    elif exp_num == 3:
        # exp3: k-sweep rows → scenarios + algorithm_runs (k_value, escalation_flag) + metrics (CDS_overall)
        scenario_db_id = {}
        for r in rows:
            key = r["scenario_id"]
            if key not in scenario_db_id:
                cur.execute(
                    "INSERT INTO scenarios (exp_id, scenario_id, pipe_failed, network) VALUES (?,?,?,?)",
                    (exp_db_id, key, r["pipe_failed"], network_name),
                )
                scenario_db_id[key] = cur.lastrowid
            k_val = None if r["k"] == "full" else int(r["k"])
            cur.execute(
                "INSERT INTO algorithm_runs (scenario_id, algorithm, k_value, escalation_flag, CT_ms, gat_ms) "
                "VALUES (?,?,?,?,?,?)",
                (scenario_db_id[key], "harp", k_val, bool(r.get("escalation_flag")), r.get("CT_ms"), None),
            )
            run_db_id = cur.lastrowid
            cur.execute(
                "INSERT INTO metrics (algorithm_run_id, CDS_hospital, CDS_school, CDS_fire, "
                "CDS_residential, CDS_overall, PVC, WLV_m3) VALUES (?,?,?,?,?,?,?,?)",
                (run_db_id, None, None, None, None, r.get("CDS_critical"), None, None),
            )
    elif exp_num == 4:
        # exp4: per-node priority breakdown → scenarios + algorithm_runs only
        # (metrics table is CDS aggregates; exp4 is node-level, so only top rows written)
        scenario_db_id = {}
        for r in rows:
            key = r["scenario_id"]
            if key not in scenario_db_id:
                cur.execute(
                    "INSERT INTO scenarios (exp_id, scenario_id, pipe_failed, network) VALUES (?,?,?,?)",
                    (exp_db_id, key, r["scenario_id"], network_name),
                )
                scenario_db_id[key] = cur.lastrowid
            cur.execute(
                "INSERT INTO algorithm_runs (scenario_id, algorithm, k_value, escalation_flag, CT_ms, gat_ms) "
                "VALUES (?,?,?,?,?,?)",
                (scenario_db_id[key], r["method"], None, None, None, None),
            )
    elif exp_num in (5, 6):
        # exp5 (scalability) and exp6 (ablation) → scenarios + algorithm_runs + partial metrics
        scenario_db_id = {}
        for r in rows:
            key = r["scenario_id"]
            if key not in scenario_db_id:
                cur.execute(
                    "INSERT INTO scenarios (exp_id, scenario_id, pipe_failed, network) VALUES (?,?,?,?)",
                    (exp_db_id, key, r["pipe_failed"], network_name),
                )
                scenario_db_id[key] = cur.lastrowid
            algo = r.get("method") or r.get("variant", "harp")
            cur.execute(
                "INSERT INTO algorithm_runs (scenario_id, algorithm, k_value, escalation_flag, CT_ms, gat_ms) "
                "VALUES (?,?,?,?,?,?)",
                (scenario_db_id[key], algo, None, bool(r.get("escalation_flag")), r.get("CT_ms"), None),
            )
            run_db_id = cur.lastrowid
            cur.execute(
                "INSERT INTO metrics (algorithm_run_id, CDS_hospital, CDS_school, CDS_fire, "
                "CDS_residential, CDS_overall, PVC, WLV_m3) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_db_id,
                    r.get("CDS_hospital"), None, None, None,
                    r.get("CDS_overall") or r.get("CDS_critical"),
                    r.get("PVC"), None,
                ),
            )

    conn.commit()
    conn.close()
    logger.info("DB write: experiment %d inserted as id=%d", exp_num, exp_db_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--network", default="Net3")
    args = parser.parse_args()

    if args.exp == 1:
        rows = run_exp1(args.network)
    elif args.exp == 2:
        rows = run_exp2(args.network)
    elif args.exp == 3:
        rows = run_exp3(args.network)
    elif args.exp == 4:
        rows = run_exp4(args.network)
    elif args.exp == 5:
        rows = run_exp5()
    elif args.exp == 6:
        rows = run_exp6(args.network)

    # DB write after CSV
    write_to_db(args.exp, rows or [], args.network)

    # Print first 5 rows
    if rows:
        print("\nFirst 5 rows:")
        for r in rows[:5]:
            print(" ", {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
