"""Generate WNTR simulation scenarios for GAT training."""
import csv
import json
import logging
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import GENERATED_DIR, RESULTS_DIR, SEED_TRAIN
from backend.wntr_engine import (
    extract_graph_features,
    inject_failure,
    load_network,
    run_simulation,
)

np.random.seed(SEED_TRAIN)
random.seed(SEED_TRAIN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_priority_config(network_name: str) -> dict:
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "priority_config.json"
    )
    with open(cfg_path) as f:
        all_cfg = json.load(f)
    return all_cfg.get(network_name, {})


def generate_scenarios(
    network_name: str,
    n_scenarios: int = 500,
    seed_start: int = 0,
) -> None:
    """Generate n_scenarios hydraulic simulation scenarios for GAT training."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    errors_path = os.path.join(RESULTS_DIR, "generation_errors.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    priority_config = _load_priority_config(network_name)
    wn_base = load_network(network_name)
    all_pipes = wn_base.pipe_name_list

    # Exclude very small pipes and reservoir-connected pipes
    reservoir_nodes = set(wn_base.reservoir_name_list + wn_base.tank_name_list)
    eligible_pipes = []
    for pid in all_pipes:
        pipe = wn_base.get_link(pid)
        if pipe.diameter < 0.05:
            continue
        if (
            pipe.start_node_name in reservoir_nodes
            or pipe.end_node_name in reservoir_nodes
        ):
            continue
        eligible_pipes.append(pid)

    if not eligible_pipes:
        eligible_pipes = list(all_pipes)

    # Derive a physical-plausibility envelope for hour-12 pressure labels from
    # the network's own intact baseline, rather than a hardcoded bound (Net3
    # has pumps that legitimately boost pressure above reservoir head, so a
    # simple "can't exceed source head" check is wrong; deriving the bound
    # from the network itself keeps this generic across Net3/Hanoi/L-Town).
    #
    # This matters because run_simulation()'s "converged" check only verifies
    # the solver didn't return all-NaN — a numerically unstable PDA solve
    # under severe demand deficit can still return finite but unphysical
    # pressures (large negative values, or spikes far past any legitimate
    # pump-boosted head). Those labels were silently included in training
    # data and, worse, set the global pressure_min/pressure_range used to
    # normalize *every* label (see dataset.py), compressing the 80% of valid
    # scenarios into a fraction of the model's output range.
    _baseline_pressure_min, _baseline_pressure_max = None, None
    try:
        _base_sim = run_simulation(wn_base)
        if _base_sim["converged"] and _base_sim["pressure_df"] is not None:
            _bpdf = _base_sim["pressure_df"]
            _bhour = min(12, len(_bpdf.index) - 1)
            _bvals = _bpdf.loc[_bpdf.index[_bhour]].values
            _baseline_pressure_min = float(np.nanmin(_bvals))
            _baseline_pressure_max = float(np.nanmax(_bvals))
    except Exception as exc:
        logger.warning("Could not compute baseline plausibility envelope: %s", exc)

    if _baseline_pressure_min is not None:
        PLAUSIBLE_MIN = _baseline_pressure_min - 10.0
        PLAUSIBLE_MAX = _baseline_pressure_max * 1.5
    else:
        # Fallback if baseline itself doesn't converge: a permissive but
        # still-finite bound so pathological solves are still rejected.
        PLAUSIBLE_MIN = -5.0
        PLAUSIBLE_MAX = 500.0
    logger.info(
        "Plausible hour-12 pressure envelope for %s: [%.2f, %.2f] m",
        network_name, PLAUSIBLE_MIN, PLAUSIBLE_MAX,
    )

    n_generated = 0
    n_skipped = 0
    error_rows = []

    for i in range(n_scenarios):
        scenario_seed = seed_start + i
        np.random.seed(scenario_seed)
        random.seed(scenario_seed)

        # Fresh network per scenario
        import wntr as _wntr
        wn = _wntr.network.WaterNetworkModel(
            os.path.join(os.path.dirname(__file__), "..", "data", "networks",
                         f"{network_name}.inp")
        )
        wn.options.hydraulic.demand_model = "PDA"
        from backend.config import MIN_PRESSURE_M, REQUIRED_PRESSURE_M
        wn.options.hydraulic.minimum_pressure = MIN_PRESSURE_M
        wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE_M

        # Lognormal demand multipliers clipped to [0.3, 3.0]
        demand_multipliers = {}
        for nid in wn.junction_name_list:
            mult = float(np.random.lognormal(mean=0, sigma=0.2))
            mult = max(0.3, min(3.0, mult))
            demand_multipliers[nid] = mult
            node = wn.get_node(nid)
            try:
                node.base_demand = float(node.base_demand) * mult
            except Exception:
                pass

        # Random pipe failure
        failed_pipe_id = random.choice(eligible_pipes)
        pipe = wn.get_link(failed_pipe_id)
        pipe.initial_status = _wntr.network.LinkStatus.Closed

        # Run simulation
        wn.options.time.duration = 24 * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600
        try:
            sim = _wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
            pressure_df = results.node["pressure"]
            demand_df = results.node["demand"]
            pressure_df.index = [int(t / 3600) for t in pressure_df.index]
            demand_df.index = [int(t / 3600) for t in demand_df.index]
            converged = not pressure_df.isnull().all(axis=None)
        except Exception as exc:
            converged = False
            logger.warning("Scenario %d convergence error: %s", i, exc)

        if not converged:
            n_skipped += 1
            error_rows.append({"scenario": i, "network": network_name,
                                "pipe": failed_pipe_id, "error": "convergence_failure"})
            continue

        # Extract features from the modified wn (failure already set)
        wn_for_features = wn  # failure already applied
        graph_data = extract_graph_features(
            wn_for_features,
            failed_pipe_id=failed_pipe_id,
            demand_multipliers=demand_multipliers,
            priority_config=priority_config,
            network_name=network_name,
        )

        # Hour-12 pressure heads as labels
        node_ids = graph_data["node_ids"]
        hour_12 = min(12, len(pressure_df.index) - 1)
        t12 = pressure_df.index[hour_12]
        pressure_heads = np.array(
            [float(pressure_df.at[t12, nid]) if nid in pressure_df.columns else 0.0
             for nid in node_ids],
            dtype=np.float32,
        )

        # Reject numerically unstable PDA solves that "converged" (not
        # all-NaN) but produced physically implausible pressures — these
        # would otherwise corrupt the global label normalization (see the
        # PLAUSIBLE_MIN/MAX derivation above).
        if pressure_heads.min() < PLAUSIBLE_MIN or pressure_heads.max() > PLAUSIBLE_MAX:
            n_skipped += 1
            error_rows.append({
                "scenario": i, "network": network_name, "pipe": failed_pipe_id,
                "error": f"implausible_pressure(min={pressure_heads.min():.2f},max={pressure_heads.max():.2f})",
            })
            continue

        # Demand met ratios at hour 12
        demand_met = np.zeros(len(node_ids), dtype=np.float32)
        for j, nid in enumerate(node_ids):
            if nid not in wn.junction_name_list:
                demand_met[j] = 1.0
                continue
            base = float(wn.get_node(nid).base_demand) / demand_multipliers.get(nid, 1.0)
            if base <= 0:
                demand_met[j] = 1.0
            else:
                actual = float(demand_df.at[t12, nid]) if nid in demand_df.columns else 0.0
                demand_met[j] = min(max(actual / base, 0.0), 1.0)

        scenario_data = {
            "scenario_id": i,
            "network_name": network_name,
            "seed": scenario_seed,
            "failed_pipe_id": failed_pipe_id,
            "demand_multipliers": demand_multipliers,
            "node_features": graph_data["node_features"],
            "edge_index": graph_data["edge_index"],
            "edge_features": graph_data["edge_features"],
            "node_ids": node_ids,
            "pipe_ids": graph_data["pipe_ids"],
            "pressure_heads_m": pressure_heads,
            "demand_met_ratios": demand_met,
            "wntr_converged": True,
        }

        out_path = os.path.join(GENERATED_DIR, f"{network_name}_{i:04d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(scenario_data, f)
        n_generated += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_scenarios} scenarios | generated: {n_generated}, skipped: {n_skipped}")

    # Save errors
    if error_rows:
        with open(errors_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario", "network", "pipe", "error"])
            writer.writeheader()
            writer.writerows(error_rows)

    # Create split file
    generated_ids = [
        i for i in range(n_scenarios)
        if os.path.exists(os.path.join(GENERATED_DIR, f"{network_name}_{i:04d}.pkl"))
    ]
    n_total = len(generated_ids)
    if n_total >= 10:
        n_test = max(1, int(n_total * 0.1))
        n_val = max(1, int(n_total * 0.1))
        n_train = n_total - n_val - n_test
        split = {
            "train": generated_ids[:n_train],
            "val": generated_ids[n_train:n_train + n_val],
            "test": generated_ids[n_train + n_val:],
        }
    else:
        split = {"train": generated_ids, "val": [], "test": []}

    split_path = os.path.join(GENERATED_DIR, f"{network_name}_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f)

    print(f"\n=== {network_name} generation complete ===")
    print(f"  Generated: {n_generated} | Skipped: {n_skipped} | Error rate: {n_skipped/n_scenarios*100:.1f}%")
    print(f"  Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    return {"generated": n_generated, "skipped": n_skipped}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="Net3")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    generate_scenarios(args.network, args.n)
