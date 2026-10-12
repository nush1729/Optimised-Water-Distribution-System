"""Experiment metrics for HARP-WDN evaluation."""
from backend.config import NODE_TYPE_WEIGHTS


def critical_demand_satisfaction(
    demand_satisfaction: dict,
    priority_config: dict,
    critical_types: list = None,
) -> dict:
    """CDS: % of full demand met at critical nodes."""
    if critical_types is None:
        critical_types = ["hospital", "fire_station", "school"]

    hospital_nodes = priority_config.get("hospital_nodes", [])
    school_nodes = priority_config.get("school_nodes", [])
    fire_nodes = priority_config.get("fire_station_nodes", [])
    weights = priority_config.get("weights", {})

    def _mean(nodes):
        vals = [demand_satisfaction.get(n, 0.0) for n in nodes if n in demand_satisfaction]
        return float(sum(vals) / len(vals)) if vals else 0.0

    cds_hospital = _mean(hospital_nodes)
    cds_school = _mean(school_nodes)
    cds_fire = _mean(fire_nodes)

    # Weighted CDS across all critical nodes
    all_critical = []
    all_critical_weights = []
    for n in hospital_nodes:
        if n in demand_satisfaction:
            all_critical.append(demand_satisfaction[n])
            all_critical_weights.append(weights.get("hospital", NODE_TYPE_WEIGHTS["hospital"]))
    for n in fire_nodes:
        if n in demand_satisfaction:
            all_critical.append(demand_satisfaction[n])
            all_critical_weights.append(weights.get("fire_station", NODE_TYPE_WEIGHTS["fire_station"]))
    for n in school_nodes:
        if n in demand_satisfaction:
            all_critical.append(demand_satisfaction[n])
            all_critical_weights.append(weights.get("school", NODE_TYPE_WEIGHTS["school"]))

    if all_critical:
        cds_critical = float(
            sum(v * w for v, w in zip(all_critical, all_critical_weights))
            / sum(all_critical_weights)
        )
    else:
        cds_critical = 0.0

    all_nodes = list(demand_satisfaction.values())
    cds_overall = float(sum(all_nodes) / len(all_nodes)) if all_nodes else 0.0

    # Residential = all nodes not in any priority group
    priority_set = set(hospital_nodes + school_nodes + fire_nodes)
    res_vals = [v for k, v in demand_satisfaction.items() if k not in priority_set]
    cds_residential = float(sum(res_vals) / len(res_vals)) if res_vals else 0.0

    return {
        "CDS_hospital": cds_hospital,
        "CDS_school": cds_school,
        "CDS_fire": cds_fire,
        "CDS_critical": cds_critical,
        "CDS_residential": cds_residential,
        "CDS_overall": cds_overall,
    }


def pressure_violation_count(
    pressures: dict,
    min_head_m: float = 7.03,
) -> int:
    """PVC: Number of nodes with head < min_head_m (default 10 PSI = 7.03m)."""
    return sum(1 for h in pressures.values() if h < min_head_m)


def recovery_time(
    demand_satisfaction_by_timestep: dict,
    priority_config: dict,
    target_CDS: float = 0.8,
) -> int:
    """RT: timesteps until CDS_critical >= target. Returns -1 if never reached."""
    timesteps = sorted(demand_satisfaction_by_timestep.keys())
    for t in timesteps:
        ds = demand_satisfaction_by_timestep[t]
        cds = critical_demand_satisfaction(ds, priority_config)
        if cds["CDS_critical"] >= target_CDS:
            return t
    return -1


def water_loss_volume(
    demand_satisfaction: dict,
    base_demands: dict,
    simulation_hours: float = 1.0,
) -> float:
    """WLV: Unserved demand volume in m³."""
    total = 0.0
    for nid, ratio in demand_satisfaction.items():
        base = base_demands.get(nid, 0.0)
        if base > 0:
            unserved = max(1.0 - ratio, 0.0) * base
            total += unserved * simulation_hours * 3600
    return total


def computation_time_breakdown(harp_result) -> dict:
    """Extract timing from HARPResult."""
    return {
        "total_ms": harp_result.computation_time_ms,
        "gat_inference_ms": harp_result.gat_inference_time_ms,
        "algorithm_ms": harp_result.algorithm_time_ms,
        "escalation_flag": harp_result.escalation_flag,
    }
