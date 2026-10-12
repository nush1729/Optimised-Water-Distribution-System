"""Z-score anomaly detection on pressure readings."""
import copy
import csv
import logging
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import GENERATED_DIR, RESULTS_DIR, SEED_TRAIN, Z_THRESHOLD, SNAPSHOT_HOUR
from backend.wntr_engine import inject_failure, load_network, run_simulation, scale_node_demand

logger = logging.getLogger(__name__)


def build_baseline(network_name: str) -> dict:
    """Load pre-computed baseline from pkl, generating it on first call if absent."""
    path = os.path.join(GENERATED_DIR, f"{network_name}_baseline.pkl")
    if not os.path.exists(path):
        logger.info("Baseline missing for %s — generating now (one-time cost)", network_name)
        from backend.wntr_engine import generate_baseline_distribution
        return generate_baseline_distribution(network_name, n_runs=100, seed=SEED_TRAIN)
    with open(path, "rb") as f:
        return pickle.load(f)


def detect_anomalies(
    current_pressures: dict,
    baseline: dict,
    timestep_hour: int,
    z_threshold: float = Z_THRESHOLD,
    pipe_endpoints: dict = None,
) -> dict:
    """
    Z-score anomaly detection.
    Returns: anomalous_nodes, z_scores, candidate_failure_zone, most_likely_failed_pipe

    pipe_endpoints: optional {pipe_id: (node_a, node_b)} — when provided, ranks pipes
    by combined |z| of their two endpoints and returns the top candidate.
    """
    z_scores = {}
    anomalous_nodes = []

    for nid, pressure in current_pressures.items():
        if nid not in baseline:
            continue
        node_baseline = baseline[nid]
        t = min(timestep_hour, len(node_baseline["mean_by_hour"]) - 1)
        mean = node_baseline["mean_by_hour"][t]
        std = node_baseline["std_by_hour"][t]
        if std < 1e-6:
            std = 1.0  # avoid div by zero
        z = (pressure - mean) / std
        z_scores[nid] = float(z)
        if abs(z) > z_threshold:
            anomalous_nodes.append(nid)

    most_likely = None
    if pipe_endpoints and anomalous_nodes:
        anomaly_set = set(anomalous_nodes)
        best_score = -1.0
        for pid, (na, nb) in pipe_endpoints.items():
            # Only consider pipes touching the anomaly zone
            if na not in anomaly_set and nb not in anomaly_set:
                continue
            score = abs(z_scores.get(na, 0.0)) + abs(z_scores.get(nb, 0.0))
            if score > best_score:
                best_score = score
                most_likely = pid

    return {
        "anomalous_nodes": anomalous_nodes,
        "z_scores": z_scores,
        "candidate_failure_zone": anomalous_nodes,
        "most_likely_failed_pipe": most_likely,
    }


def evaluate_detector(
    network_name: str,
    n_test_scenarios: int = 20,
    n_negative_scenarios: int = 20,
    seed: int = 42,
) -> dict:
    """Evaluate anomaly detection on known failure scenarios (positive class)
    AND on normal, no-failure operation (negative class).

    Previously this only ever ran positive-class (real-failure) scenarios,
    so fp/tn could never be incremented and precision read as 1.0 by
    construction whenever there was at least one detection — not because
    the detector was accurate, but because there was no scenario in which
    it *could* be wrong. This also now counts a detection during a real
    failure as a false positive contributor if it flags nodes nowhere near
    the actual failed pipe, rather than only tracking localization separately.
    """
    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        baseline = build_baseline(network_name)
    except FileNotFoundError:
        logger.warning("No baseline found; generating with 20 runs")
        from backend.wntr_engine import generate_baseline_distribution
        baseline = generate_baseline_distribution(network_name, n_runs=20, seed=seed)

    wn_base = load_network(network_name)
    all_pipes = wn_base.pipe_name_list
    np.random.seed(seed)
    test_pipes = list(np.random.choice(all_pipes, min(n_test_scenarios, len(all_pipes)), replace=False))

    tp, fp, fn, tn = 0, 0, 0, 0
    localized = 0
    rows = []

    # --- Positive class: known pipe failures ---
    for pid in test_pipes:
        wn_fail = inject_failure(wn_base, pid)
        sim = run_simulation(wn_fail)
        if not sim["converged"] or sim["pressure_df"] is None:
            continue

        pdf = sim["pressure_df"]
        hour = min(SNAPSHOT_HOUR, len(pdf.index) - 1)
        t12 = pdf.index[hour]
        current_pressures = {
            nid: float(pdf.at[t12, nid])
            for nid in pdf.columns
            if nid in baseline
        }
        # Add Gaussian noise σ=0.5m (synthetic IoT)
        for nid in current_pressures:
            current_pressures[nid] += float(np.random.normal(0, 0.5))

        result = detect_anomalies(current_pressures, baseline, timestep_hour=hour)
        anomalous = set(result["anomalous_nodes"])

        # For each anomalous node check if adjacent to failed pipe
        pipe = wn_base.get_link(pid)
        failed_endpoints = {pipe.start_node_name, pipe.end_node_name}
        adjacent_to_failure = failed_endpoints & anomalous
        spurious = anomalous - failed_endpoints

        detected = len(anomalous) > 0
        localized_correct = bool(adjacent_to_failure)

        if detected:
            tp += 1
            if localized_correct:
                localized += 1
            if spurious:
                fp += 1
        else:
            fn += 1

        rows.append({
            "scenario_type": "failure",
            "pipe_failed": pid,
            "n_anomalous": len(anomalous),
            "detected": detected,
            "localized": localized_correct,
            "n_spurious": len(spurious),
            "z_max": max((abs(v) for v in result["z_scores"].values()), default=0),
        })

    n_eval = tp + fn

    # --- Negative class: normal operation, no failure injected ---
    # Same ±10% demand-noise model used by generate_baseline_distribution(),
    # so this is testing against realistic day-to-day fluctuation rather
    # than a trivially quiet network.
    np.random.seed(seed + 1)
    for _ in range(n_negative_scenarios):
        wn_normal = copy.deepcopy(wn_base)
        for nid in wn_normal.junction_name_list:
            node = wn_normal.get_node(nid)
            scale_node_demand(node, np.random.uniform(0.9, 1.1))
        sim = run_simulation(wn_normal)
        if not sim["converged"] or sim["pressure_df"] is None:
            continue

        pdf = sim["pressure_df"]
        hour = min(SNAPSHOT_HOUR, len(pdf.index) - 1)
        t12 = pdf.index[hour]
        current_pressures = {
            nid: float(pdf.at[t12, nid])
            for nid in pdf.columns
            if nid in baseline
        }
        for nid in current_pressures:
            current_pressures[nid] += float(np.random.normal(0, 0.5))

        result = detect_anomalies(current_pressures, baseline, timestep_hour=hour)
        anomalous = set(result["anomalous_nodes"])
        false_alarm = len(anomalous) > 0

        if false_alarm:
            fp += 1
        else:
            tn += 1

        rows.append({
            "scenario_type": "normal",
            "pipe_failed": None,
            "n_anomalous": len(anomalous),
            "detected": false_alarm,
            "localized": False,
            "n_spurious": len(anomalous),
            "z_max": max((abs(v) for v in result["z_scores"].values()), default=0),
        })

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    loc_acc = localized / max(n_eval, 1)
    false_alarm_rate = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0

    summary = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "localization_accuracy": loc_acc,
        "false_alarm_rate": false_alarm_rate,
        "n_evaluated": n_eval,
        "n_negative_evaluated": tn + fp,
    }

    out_path = os.path.join(RESULTS_DIR, "anomaly_evaluation.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    logger.info(
        "Anomaly detector (%s): precision=%.3f recall=%.3f f1=%.3f localization=%.3f far=%.3f n=%d",
        network_name, precision, recall, f1, loc_acc, false_alarm_rate, n_eval,
    )
    return summary
