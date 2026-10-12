"""k-hop subgraph extraction around failed pipe endpoints."""
import networkx as nx
import wntr

from backend.config import DEFAULT_PIPE_DIAMETER_M, DEFAULT_PIPE_LENGTH_M, DEFAULT_PIPE_ROUGHNESS


def extract_failure_subgraph(
    G: nx.DiGraph,
    failed_pipe_id: str,
    pipe_to_nodes: dict,
    k: int = 2,
) -> nx.DiGraph:
    """
    Extract k-hop neighborhood around failed pipe endpoints.
    Returns subgraph with failed pipe removed and escalation metadata.
    """
    endpoints = pipe_to_nodes.get(failed_pipe_id)
    if endpoints is None:
        # Return full graph if pipe not found in map
        G_sub = G.copy()
        G_sub.graph["needs_escalation"] = False
        G_sub.graph["failed_pipe_id"] = failed_pipe_id
        return G_sub

    node_u, node_v = endpoints
    G_undirected = G.to_undirected()

    # k-hop ego graphs
    nodes_u = set(nx.ego_graph(G_undirected, node_u, radius=k).nodes())
    nodes_v = set(nx.ego_graph(G_undirected, node_v, radius=k).nodes())
    subgraph_nodes = nodes_u | nodes_v

    G_sub = G.subgraph(subgraph_nodes).copy()

    # Remove failed pipe edges
    edges_to_remove = [
        (u, v) for u, v, d in G_sub.edges(data=True)
        if d.get("pipe_id") == failed_pipe_id
    ]
    G_sub.remove_edges_from(edges_to_remove)

    G_sub.graph["needs_escalation"] = False
    G_sub.graph["failed_pipe_id"] = failed_pipe_id
    return G_sub


def escalation_needed(
    demand_satisfaction: dict,
    priority_config: dict,
    critical_threshold: float = 0.6,
) -> bool:
    """
    Returns True if any high-priority node has demand satisfaction < threshold.
    """
    critical_nodes = (
        priority_config.get("hospital_nodes", [])
        + priority_config.get("fire_station_nodes", [])
    )
    for nid in critical_nodes:
        if demand_satisfaction.get(nid, 1.0) < critical_threshold:
            return True
    return False


def build_wdn_graph(
    wn,
    failed_pipe_id: str = None,
    priority_config: dict = None,
) -> tuple:
    """
    Build NetworkX DiGraph from WNTR network.
    Returns (G, pipe_to_nodes_dict).
    """
    if priority_config is None:
        priority_config = {}

    G = nx.DiGraph()
    pipe_to_nodes = {}

    # Add nodes
    all_nodes = (
        wn.junction_name_list
        + wn.reservoir_name_list
        + wn.tank_name_list
    )
    for nid in all_nodes:
        node = wn.get_node(nid)
        elev = getattr(node, "elevation", 0.0) or 0.0
        try:
            base_demand = float(getattr(node, "base_demand", 0.0) or 0.0)
        except Exception:
            base_demand = 0.0
        is_res = nid in wn.reservoir_name_list or nid in wn.tank_name_list

        node_type = "residential"
        if nid in priority_config.get("hospital_nodes", []):
            node_type = "hospital"
        elif nid in priority_config.get("fire_station_nodes", []):
            node_type = "fire_station"
        elif nid in priority_config.get("school_nodes", []):
            node_type = "school"
        elif is_res:
            node_type = "reservoir"

        weights = priority_config.get("weights", {})
        priority_weight = weights.get(node_type, 1.0)

        G.add_node(
            nid,
            elevation=elev,
            base_demand=base_demand,
            node_type=node_type,
            priority_weight=priority_weight,
            is_reservoir=is_res,
        )

    # Add edges from pipes
    for pid in wn.pipe_name_list:
        pipe = wn.get_link(pid)
        n1 = pipe.start_node_name
        n2 = pipe.end_node_name
        if n1 not in G or n2 not in G:
            continue
        pipe_to_nodes[pid] = (n1, n2)
        L = pipe.length or DEFAULT_PIPE_LENGTH_M
        D = pipe.diameter or DEFAULT_PIPE_DIAMETER_M
        C = pipe.roughness or DEFAULT_PIPE_ROUGHNESS
        # A pipe is closed if it's the explicit failed_pipe_id argument, or if
        # the network object already carries a closed status (e.g. built from
        # inject_multi_failure()). Previously only the single failed_pipe_id
        # argument was honored, so a multi-pipe failure would have every pipe
        # but one silently treated as open here.
        _status = getattr(pipe, "initial_status", None)
        already_closed = _status is not None and _status == wntr.network.LinkStatus.Closed
        status = "CLOSED" if (pid == failed_pipe_id or already_closed) else "OPEN"
        if status == "OPEN":
            G.add_edge(n1, n2, pipe_id=pid, length=L, diameter=D, C_hw=C, status=status)
            G.add_edge(n2, n1, pipe_id=pid, length=L, diameter=D, C_hw=C, status=status)

    # Valves and pumps are real links that carry flow. Leaving them out splits
    # the graph: L-Town's reservoirs R1 and R2 reach the network only through
    # pressure-reducing valves, so with a pipes-only graph 2 of its 3 sources
    # were disconnected and HARP could route to just 86 of 701 demand nodes.
    # Both link types are one-way (a pump pushes start->end; a PRV does not pass
    # reverse flow) and are modelled as short pipes for capacity purposes.
    _pumps = list(getattr(wn, "pump_name_list", []))
    for lid in list(getattr(wn, "valve_name_list", [])) + _pumps:
        link = wn.get_link(lid)
        n1, n2 = link.start_node_name, link.end_node_name
        if n1 not in G or n2 not in G:
            continue
        _st = getattr(link, "initial_status", None)
        if _st is not None and _st == wntr.network.LinkStatus.Closed:
            continue
        D = getattr(link, "diameter", None) or 0.3
        G.add_edge(n1, n2, pipe_id=lid, length=1.0, diameter=D, C_hw=130.0,
                   status="OPEN", link_type="pump" if lid in _pumps else "valve")

    return G, pipe_to_nodes
