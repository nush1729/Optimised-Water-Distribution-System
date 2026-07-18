"""Hazen-Williams hydraulic calculations."""


def hw_head_loss(Q_m3s: float, L_m: float, D_m: float, C_hw: float) -> float:
    """
    Hazen-Williams head loss.
    h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.87)
    Units: Q m³/s, L m, D m → h_f meters
    Raises ValueError on non-positive inputs.
    """
    if Q_m3s <= 0 or D_m <= 0 or L_m <= 0 or C_hw <= 0:
        raise ValueError(
            f"Invalid inputs: Q={Q_m3s}, D={D_m}, L={L_m}, C={C_hw}. All must be > 0."
        )
    return 10.67 * L_m * (Q_m3s ** 1.852) / ((C_hw ** 1.852) * (D_m ** 4.87))


def hw_max_flow(L_m: float, D_m: float, C_hw: float, dH_m: float) -> float:
    """
    Compute maximum hydraulic flow capacity using inverse Hazen-Williams.
    Given an available head difference dH_m (metres), pipe geometry and roughness,
    returns Q_max in m³/s.

    Inverse HW:  Q = (dH × C^1.852 × D^4.87 / (10.67 × L))^(1/1.852)

    This is used in HARP to scale residual edge capacities from GAT-predicted
    pressure heads — the key differentiation vs. Dinic's (diameter×10 proxy).
    """
    if L_m <= 0 or D_m <= 0 or C_hw <= 0:
        return D_m * 0.001 if D_m > 0 else 1e-6
    if dH_m <= 0:
        return 1e-6  # no head → negligible capacity
    inside = (dH_m * (C_hw ** 1.852) * (D_m ** 4.87)) / (10.67 * L_m + 1e-9)
    if inside <= 0:
        return 1e-6
    return inside ** (1.0 / 1.852)


def path_head_loss(
    path_edges: list,
    Q_estimate_m3s: float = 0.005,
) -> float:
    """
    Cumulative HW head loss along a path.
    path_edges: list of dicts with keys 'length', 'diameter', 'C_hw'
    Q_estimate_m3s: flow estimate (default 5 L/s typical junction demand)
    """
    total = 0.0
    for edge in path_edges:
        L = edge.get("length", 100.0)
        D = edge.get("diameter", 0.1)
        C = edge.get("C_hw", 100.0)
        if Q_estimate_m3s > 0 and D > 0 and L > 0 and C > 0:
            total += hw_head_loss(Q_estimate_m3s, L, D, C)
    return total


def path_hydraulically_feasible(
    path_edges: list,
    gat_head_source: float,
    gat_head_sink: float,
    Q_estimate_m3s: float = 0.005,
    safety_margin_m: float = 1.0,
) -> tuple:
    """
    Check if a path is hydraulically feasible.
    Condition: (head_source - head_sink) >= path_head_loss + safety_margin
    Returns (is_feasible: bool, head_margin_m: float)
    """
    available_head = gat_head_source - gat_head_sink
    h_f = path_head_loss(path_edges, Q_estimate_m3s)
    margin = available_head - h_f - safety_margin_m
    return (margin >= 0, margin)
