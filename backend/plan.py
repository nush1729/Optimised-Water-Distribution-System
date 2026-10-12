"""Turn a HARP result into an operator-readable recovery plan.

HARP *recommends*; it never actuates anything. The plan lists what to isolate,
which pipes carry the rerouted flow (largest first) and which demand nodes remain
unserved, so an operator can review and approve valve actions.
"""
from backend.config import NODE_TYPE_WEIGHTS

M3S_TO_LPS = 1000.0


def build_recovery_plan(wn, failed_pipes, harp_result, priority_config: dict) -> dict:
    type_of = {}
    for key, t in (("hospital_nodes", "hospital"), ("fire_station_nodes", "fire_station"),
                   ("school_nodes", "school")):
        for n in priority_config.get(key, []):
            type_of[n] = t

    def ends(pid):
        link = wn.get_link(pid)
        return link.start_node_name, link.end_node_name

    isolate = []
    for pid in failed_pipes or []:
        a, b = ends(pid)
        isolate.append({"pipe_id": pid, "from": a, "to": b, "action": "isolate (close adjacent valves)"})

    routes = []
    for pid, q in sorted(harp_result.flow_allocation.items(), key=lambda kv: kv[1], reverse=True):
        a, b = ends(pid)
        routes.append({"pipe_id": pid, "from": a, "to": b, "flow_lps": round(q * M3S_TO_LPS, 3)})

    unserved = []
    for node, sat in harp_result.demand_satisfaction.items():
        if node.startswith("__SUPER"):
            continue
        try:
            demand = float(wn.get_node(node).base_demand)
        except Exception:  # noqa: BLE001 - non-junction nodes have no demand
            continue
        if demand > 0 and sat < 0.99:
            t = type_of.get(node, "residential")
            unserved.append({"node": node, "type": t, "priority": NODE_TYPE_WEIGHTS.get(t, 1.0),
                             "satisfaction": round(float(sat), 3)})
    unserved.sort(key=lambda u: (-u["priority"], u["satisfaction"]))

    return {
        "requires_operator_approval": True,
        "isolate": isolate,
        "routes": routes,
        "unserved": unserved,
        "summary": {
            "pipes_carrying_flow": len(routes),
            "nodes_not_fully_served": len(unserved),
            "critical_not_fully_served": sum(1 for u in unserved if u["type"] in ("hospital", "fire_station")),
        },
    }
