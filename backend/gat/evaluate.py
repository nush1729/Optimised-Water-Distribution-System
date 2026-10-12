"""Held-out evaluation for a trained PressureGAT.

Answers the questions that decide whether a model is good enough for HARP:

1. How big is the error, and does it beat a naive "predict the mean" model?
2. Is the error small where it matters - scenarios where the failure actually
   changes pressures (high-impact) - not just on easy ones?
3. Does MC-Dropout uncertainty track real error (is it usable as a confidence)?
4. Does the model make the same hydraulic-gate decisions HARP would make with
   the true WNTR pressures (the decision HARP actually depends on)?

Run:  python -m backend.gat.evaluate --network Net3
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import (  # noqa: E402
    GENERATED_DIR,
    HW_SAFETY_MARGIN_M,
    MIN_PRESSURE_M,
    RESULTS_DIR,
)
from backend.gat.inference import (  # noqa: E402
    load_model,
    predict_pressures,
    predict_pressures_with_uncertainty,
)


def _load_test_scenarios(network_name: str, limit: int = None) -> list:
    with open(os.path.join(GENERATED_DIR, f"{network_name}_split.json")) as f:
        ids = json.load(f).get("test", [])
    if limit:
        ids = ids[:limit]
    out = []
    for i in ids:
        with open(os.path.join(GENERATED_DIR, f"{network_name}_{i:04d}.pkl"), "rb") as f:
            out.append(pickle.load(f))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_model(network_name: str, mc_samples: int = 30, mc_scenarios: int = 40) -> dict:
    """Accuracy, impact-stratified error, naive baseline, gate agreement, MC-Dropout calibration."""
    model, norm_stats = load_model(network_name)
    scenarios = _load_test_scenarios(network_name)
    if not scenarios:
        raise RuntimeError(f"No test scenarios for {network_name}; generate data first.")

    # Naive reference: predict the mean of the *test* labels' per-node average is
    # leaky, so use the training-set mean pressure recorded in norm_stats.
    train_mean = norm_stats["pressure_min"] + 0.5 * norm_stats["pressure_range"]

    y_true, y_pred, per_scn_mean = [], [], []
    gate_thr = MIN_PRESSURE_M + HW_SAFETY_MARGIN_M
    gate_agree, gate_n, gate_false_ok, gate_false_block = 0, 0, 0, 0
    for sc in scenarios:
        pred, _ = predict_pressures(model, norm_stats, sc)
        true = sc["pressure_heads_m"]
        y_true.append(true)
        y_pred.append(pred)
        per_scn_mean.append(float(true.mean()))
        # Gate decision agreement, on demand nodes only (the nodes HARP serves).
        # node_features[:, 1] is demand; node_features[:, 0] is elevation(+head for sources).
        demand_mask = (sc["node_features"][:, 1] > 0)
        t_ok = true[demand_mask] >= gate_thr
        p_ok = pred[demand_mask] >= gate_thr
        gate_n += int(demand_mask.sum())
        gate_agree += int((t_ok == p_ok).sum())
        gate_false_ok += int((p_ok & ~t_ok).sum())      # model says servable, truth says not
        gate_false_block += int((~p_ok & t_ok).sum())   # model blocks a servable node

    flat_t = np.concatenate(y_true)
    flat_p = np.concatenate(y_pred)
    err = flat_p - flat_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((flat_t - flat_t.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    naive_mae = float(np.mean(np.abs(flat_t - train_mean)))

    # Impact stratification: scenarios whose mean pressure sits furthest below the
    # median are the ones where the failure really bit.
    means = np.array(per_scn_mean)
    cut = np.quantile(means, 0.25)
    hi = means <= cut
    mae_hi = float(np.mean([np.abs(y_pred[i] - y_true[i]).mean() for i in np.where(hi)[0]])) if hi.any() else float("nan")
    mae_lo = float(np.mean([np.abs(y_pred[i] - y_true[i]).mean() for i in np.where(~hi)[0]])) if (~hi).any() else float("nan")

    # MC-Dropout calibration: does predicted std rank real error?
    stds, abserr = [], []
    for sc in scenarios[:mc_scenarios]:
        mean_p, std_p, _ = predict_pressures_with_uncertainty(model, norm_stats, sc, n_samples=mc_samples)
        stds.append(std_p)
        abserr.append(np.abs(mean_p - sc["pressure_heads_m"]))
    mc_spearman = _spearman(np.concatenate(stds), np.concatenate(abserr))

    result = {
        "network": network_name,
        "n_test_scenarios": len(scenarios),
        "MAE_m": mae,
        "RMSE_m": rmse,
        "R2": r2,
        "naive_MAE_m": naive_mae,
        "improvement_over_naive_pct": 100.0 * (1.0 - mae / naive_mae) if naive_mae > 0 else float("nan"),
        "MAE_high_impact_quartile_m": mae_hi,
        "MAE_other_m": mae_lo,
        "gate_decision_agreement": gate_agree / max(gate_n, 1),
        "gate_false_servable": gate_false_ok / max(gate_n, 1),
        "gate_false_blocked": gate_false_block / max(gate_n, 1),
        "mc_dropout_spearman_std_vs_error": mc_spearman,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"gat_eval_{network_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def harp_end_to_end(network_name: str, n_scenarios: int = 20, seed: int = 99) -> dict:
    """Run HARP on real failures twice - with GAT pressures and with the true WNTR
    pressures - and compare. This is the check that retraining actually matters
    for the system, not just for a pressure-error metric."""
    from backend.algorithms.harp import HARP
    from backend.experiments.metrics import critical_demand_satisfaction
    from backend.experiments.runner import _load_priority_config, _run_no_recovery
    from backend.wntr_engine import (
        extract_graph_features,
        inject_failure,
        load_network,
        run_simulation,
    )

    model, norm_stats = load_model(network_name)
    priority_config = _load_priority_config(network_name)
    wn = load_network(network_name)
    rng = np.random.default_rng(seed)
    pipes = list(rng.choice(wn.pipe_name_list, min(n_scenarios, len(wn.pipe_name_list)), replace=False))

    rows = []
    for pid in pipes:
        wf = inject_failure(wn, str(pid))
        sim = run_simulation(wf)
        if not sim["converged"]:
            continue
        pdf = sim["pressure_df"]
        t = pdf.index[min(12, len(pdf.index) - 1)]
        true_p = {n: float(pdf.at[t, n]) for n in pdf.columns}
        gd = extract_graph_features(wf, failed_pipe_id=str(pid), priority_config=priority_config,
                                    network_name=network_name)
        pa, _ = predict_pressures(model, norm_stats, gd)
        gat_p = dict(zip(gd["node_ids"], pa.tolist()))

        def cds(pressures):
            r = HARP().run(wn, str(pid), pressures, priority_config, k=0, use_subgraph=False)
            return critical_demand_satisfaction(r.demand_satisfaction, priority_config)["CDS_overall"]

        rows.append({
            "pipe": str(pid),
            "CDS_no_recovery": _run_no_recovery(wn, str(pid), priority_config)["cds"]["CDS_overall"],
            "CDS_harp_true_pressure": cds(true_p),
            "CDS_harp_gat_pressure": cds(gat_p),
        })

    if not rows:
        raise RuntimeError("No converged scenarios")
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "pipe"}
    out["max_abs_gap_gat_vs_true"] = float(max(abs(r["CDS_harp_gat_pressure"] - r["CDS_harp_true_pressure"]) for r in rows))
    out["n_scenarios"] = len(rows)
    out["network"] = network_name
    with open(os.path.join(RESULTS_DIR, f"harp_end_to_end_{network_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="Net3")
    parser.add_argument("--skip-harp", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.network), indent=2))
    if not args.skip_harp:
        print(json.dumps(harp_end_to_end(args.network), indent=2))
