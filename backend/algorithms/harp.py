"""HARP: Hydraulics-Aware Resilient Prioritization Algorithm."""
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from backend.algorithms.hw_hydraulics import hw_max_flow, path_head_loss
from backend.algorithms.subgraph import (
    build_wdn_graph,
    escalation_needed,
    extract_failure_subgraph,
)
from backend.config import (
    DEFAULT_K,
    DEFAULT_PIPE_DIAMETER_M,
    DEFAULT_PIPE_LENGTH_M,
    DEFAULT_PIPE_ROUGHNESS,
    MIN_SINK_CAP_M3S,
    PDA_PRESSURE_EXPONENT,
    SOURCE_SUPPLY_CAP_M3S,
    ESCALATION_THRESHOLD,
    GAT_Q_ESTIMATE_M3S,
    HW_DESIGN_GRADIENT,
    HW_SAFETY_MARGIN_M,
    MAX_DESIGN_VELOCITY_MS,
    MIN_PRESSURE_M,
    REQUIRED_PRESSURE_M,
)

logger = logging.getLogger(__name__)


def _pressures_to_heads(pressures: dict, wn) -> dict:
    """Convert WNTR-convention pressure (head − elevation) to true head.

    Reservoirs have no elevation at all in WNTR — they define head directly
    via `base_head` — and always report pressure=0 by convention, so they're
    handled as a special case rather than by the elevation formula.
    """
    reservoir_ids = set(wn.reservoir_name_list)
    heads = {}
    for nid, p in pressures.items():
        try:
            node = wn.get_node(nid)
        except Exception:
            heads[nid] = p
            continue
        if nid in reservoir_ids:
            heads[nid] = getattr(node, "base_head", 0.0) or 0.0
        else:
            heads[nid] = p + (getattr(node, "elevation", 0.0) or 0.0)
    return heads


@dataclass
class PathRecord:
    nodes: list
    flow: float
    priority_score: float
    head_margin_m: float


@dataclass
class RejectedPathRecord:
    nodes: list
    rejection_reason: str
    head_deficit_m: float


@dataclass
class HARPResult:
    flow_allocation: dict = field(default_factory=dict)
    accepted_paths: list = field(default_factory=list)
    rejected_paths: list = field(default_factory=list)
    demand_satisfaction: dict = field(default_factory=dict)
    gat_predicted_pressures: dict = field(default_factory=dict)
    escalation_flag: bool = False
    k_used: int = DEFAULT_K
    computation_time_ms: float = 0.0
    gat_inference_time_ms: float = 0.0
    algorithm_time_ms: float = 0.0


class HARP:
    """
    Hydraulics-Aware Resilient Prioritization Algorithm.
    Modifies Dinic's blocking flow with:
      1. GAT-based hydraulic feasibility check (Hazen-Williams)
      2. Priority-weighted path scoring
      3. k-hop subgraph localization with escalation
    """

    def __init__(
        self,
        safety_margin_m: float = HW_SAFETY_MARGIN_M,
        q_estimate: float = GAT_Q_ESTIMATE_M3S,
        escalation_threshold: float = ESCALATION_THRESHOLD,
    ):
        self.safety_margin_m = safety_margin_m
        self.q_estimate = q_estimate
        self.escalation_threshold = escalation_threshold
        self._sink_demands: dict = {}

    def _get_priority_score(self, path_nodes: list, G: nx.DiGraph, demand_satisfaction: dict) -> float:
        """Priority score = sum of (priority_weight × unmet_demand) for real sink nodes in path."""
        score = 0.0
        for node in path_nodes[1:]:  # skip source
            if node.startswith("__SUPER"):
                continue
            if node not in G.nodes:
                continue
            pw = G.nodes[node].get("priority_weight", 1.0)
            unmet = 1.0 - demand_satisfaction.get(node, 1.0)
            score += pw * unmet
        return score

    def _bfs_level_graph(self, G_residual: nx.DiGraph, source: str, supersink: str) -> dict:
        """BFS to build level graph. Returns level dict."""
        level = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in G_residual.successors(u):
                edge_data = G_residual[u][v]
                if v not in level and edge_data.get("residual_cap", 0) > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def _hydraulic_feasible_path(
        self, path: list, G: nx.DiGraph, heads: dict, flow: float = None
    ) -> tuple:
        """Terminal-pressure feasibility check for a complete source->sink path.

        The path is feasible if the sink would still be left with at least
        MIN_PRESSURE_M + safety_margin_m of residual pressure after paying the
        Hazen-Williams head loss along the path:

            residual = h_supply - sum(h_f) - elevation(sink)
            feasible = min(residual, predicted_sink_pressure)
                       >= MIN_PRESSURE_M + safety_margin_m

        h_supply is the highest head found along the path. Taking the max
        (rather than the source node's head) accounts for pump lift: in a
        pumped network the source reservoir sits *below* the heads it feeds.

        The previous gate required a head drop of >= safety_margin_m between
        the source and every prefix of the path. That cannot hold in a
        looped, near-equilibrium network (adjacent nodes share almost the
        same head), so it rejected every first hop out of every source and
        HARP delivered zero flow. Virtual supersource/supersink nodes are
        ignored. Returns (feasible, margin_m, service_fraction), where
        service_fraction in [0, 1] is the pressure-dependent share of demand
        the sink can receive (0 at MIN_PRESSURE_M, 1 at REQUIRED_PRESSURE_M —
        the same pressure-dependent-demand law WNTR uses, exponent 0.5).
        """
        real_path = [n for n in path if not n.startswith("__SUPER")]
        if len(real_path) < 2:
            return True, 0.0, 1.0

        sink = real_path[-1]
        path_edges = []
        for i in range(len(real_path) - 1):
            u, v = real_path[i], real_path[i + 1]
            if G.has_edge(u, v):
                edata = G[u][v]
                path_edges.append({
                    "length": edata.get("length", DEFAULT_PIPE_LENGTH_M),
                    "diameter": edata.get("diameter", DEFAULT_PIPE_DIAMETER_M),
                    "C_hw": edata.get("C_hw", DEFAULT_PIPE_ROUGHNESS),
                })
        if not path_edges:
            return True, 0.0, 1.0

        h_supply = max(heads.get(n, 0.0) for n in real_path)
        elev_sink = G.nodes[sink].get("elevation", 0.0) if sink in G.nodes else 0.0
        # Head loss at the flow this path would actually carry; the fixed
        # q_estimate is only the fallback when no flow is known.
        q = flow if flow is not None and flow > 0 else self.q_estimate
        h_f = path_head_loss(path_edges, q)
        residual_pressure = h_supply - h_f - elev_sink
        # The sink must also already be predicted to hold adequate pressure in
        # the post-failure state; a node the model says is below the service
        # minimum cannot be served at full demand just because a high-head
        # node exists somewhere upstream on the path.
        predicted_sink_pressure = heads.get(sink, 0.0) - elev_sink
        pressure_eff = min(residual_pressure, predicted_sink_pressure)
        margin = pressure_eff - MIN_PRESSURE_M - self.safety_margin_m
        band = (pressure_eff - MIN_PRESSURE_M) / (REQUIRED_PRESSURE_M - MIN_PRESSURE_M)
        fraction = max(0.0, min(1.0, band)) ** PDA_PRESSURE_EXPONENT
        return margin >= 0, margin, fraction

    @staticmethod
    def _retire_sink(G_residual: nx.DiGraph, sink: str, supersink: str) -> None:
        """Remove a sink's edge to the supersink from the residual graph.

        Without this, a sink that cannot (or can no longer) be served keeps a
        positive-capacity edge, so every Dinic BFS phase re-discovers it as the
        shortest route to the supersink, the DFS rejects it again, and the
        longer paths to the *other* sinks are never explored. Delivered demand
        is tracked live in demand_satisfaction, not read back from this edge.
        """
        if G_residual.has_edge(sink, supersink):
            G_residual[sink][supersink]["residual_cap"] = 0.0

    def _harp_dfs(
        self,
        G_sub: nx.DiGraph,
        G_residual: nx.DiGraph,
        level: dict,
        gat_pressures: dict,
        ptr: dict,
        u: str,
        supersink: str,
        pushed: float,
        current_path: list,
        demand_satisfaction: dict,
        accepted_paths: list,
        rejected_paths: list,
        use_hydraulic_check: bool = True,
    ) -> float:
        if u == supersink:
            # Terminal hydraulic check on the complete path (see
            # _hydraulic_feasible_path for why this is not done per-prefix).
            margin = 0.0
            if use_hydraulic_check:
                _snk = current_path[-2]
                _dem = self._sink_demands.get(_snk)
                _q = min(pushed, _dem) if _dem else pushed
                feasible, margin, fraction = self._hydraulic_feasible_path(
                    current_path, G_sub, gat_pressures, flow=_q)
                if not feasible:
                    rejected_paths.append(RejectedPathRecord(
                        nodes=list(current_path),
                        rejection_reason="head_loss_exceeded",
                        head_deficit_m=abs(margin),
                    ))
                    self._retire_sink(G_residual, current_path[-2], supersink)
                    return 0.0
                # Pressure-dependent delivery: a sink below full service
                # pressure can only receive `fraction` of its demand in total.
                sink = current_path[-2]
                demand = self._sink_demands.get(sink)
                if demand:
                    allowed = (fraction - demand_satisfaction.get(sink, 0.0)) * demand
                    pushed = min(pushed, allowed)
                    if pushed <= 1e-12:
                        rejected_paths.append(RejectedPathRecord(
                            nodes=list(current_path),
                            rejection_reason="pressure_limited",
                            head_deficit_m=0.0,
                        ))
                        self._retire_sink(G_residual, sink, supersink)
                        return 0.0
            # Accept only if the path still serves some unmet demand.
            score = self._get_priority_score(current_path, G_sub, demand_satisfaction)
            if score <= 0:
                rejected_paths.append(RejectedPathRecord(
                    nodes=list(current_path),
                    rejection_reason="no_demand_benefit",
                    head_deficit_m=0.0,
                ))
                self._retire_sink(G_residual, current_path[-2], supersink)
                return 0.0
            accepted_paths.append(PathRecord(
                nodes=list(current_path),
                flow=pushed,
                priority_score=score,
                head_margin_m=margin,
            ))
            # Track delivered demand live so later paths see up-to-date unmet
            # demand (previously demand_satisfaction stayed 0 until the very
            # end, so the score gate could never fire).
            sink = current_path[-2]
            demand = self._sink_demands.get(sink)
            if demand:
                demand_satisfaction[sink] = min(demand_satisfaction.get(sink, 0.0) + pushed / demand, 1.0)
            return pushed

        # Explore higher-priority neighbors first so that, when capacity is
        # contended, blocking flow reaches high-priority sinks (hospitals,
        # fire stations) before low-priority ones. Previously priority_weight
        # only gated accept/reject of a completed path (score <= 0), never
        # traversal order, so it had no effect whenever every sink had some
        # unmet demand — which is nearly always.
        def _priority_key(v):
            if v.startswith("__SUPER") or v not in G_sub.nodes:
                return 0.0
            return G_sub.nodes[v].get("priority_weight", 1.0)

        neighbors = sorted(G_residual.successors(u), key=_priority_key, reverse=True)
        while ptr[u] < len(neighbors):
            v = neighbors[ptr[u]]
            edge_data = G_residual[u][v]
            rcap = edge_data.get("residual_cap", 0)
            if rcap > 0 and level.get(v, -1) == level[u] + 1:
                path_to_v = current_path + [v]
                flow = self._harp_dfs(
                    G_sub, G_residual, level, gat_pressures, ptr,
                    v, supersink, min(pushed, rcap), path_to_v,
                    demand_satisfaction, accepted_paths, rejected_paths,
                    use_hydraulic_check,
                )
                if flow > 0:
                    G_residual[u][v]["residual_cap"] -= flow
                    if G_residual.has_edge(v, u):
                        G_residual[v][u]["residual_cap"] = G_residual[v][u].get("residual_cap", 0) + flow
                    return flow
            ptr[u] += 1
        return 0.0

    def _run_on_subgraph(
        self,
        G_sub: nx.DiGraph,
        gat_pressures: dict,
        priority_config: dict,
        source_nodes: list,
        sink_nodes: list,
        use_hydraulic_check: bool = True,
    ) -> tuple:
        """Run HARP on the given subgraph. Returns (demand_satisfaction, accepted_paths, rejected_paths)."""
        accepted_paths = []
        rejected_paths = []
        supersink = "__SUPERSINK_HARP__"
        supersource = "__SUPERSOURCE_HARP__"

        # Build residual graph. With use_hydraulic_check=True this is HW-based
        # capacity from GAT pressures — the core differentiator vs. Dinic's
        # (diameter×10 proxy). With it False, fall back to that same
        # diameter×10 proxy so the ablation actually removes the hydraulic
        # signal instead of starving every edge via a synthetic zero head-diff.
        G_residual = G_sub.copy()
        for u, v in list(G_residual.edges()):
            edata = G_residual[u][v]
            L = edata.get("length", DEFAULT_PIPE_LENGTH_M)
            D = edata.get("diameter", DEFAULT_PIPE_DIAMETER_M)
            C = edata.get("C_hw", DEFAULT_PIPE_ROUGHNESS)
            if use_hydraulic_check:
                h_u = gat_pressures.get(u, 0.0)
                h_v = gat_pressures.get(v, 0.0)
                dH = max(h_u - h_v, 0.0)
                # Capacity = Hazen-Williams flow at the larger of the predicted
                # head difference and the design gradient, capped by the
                # maximum design velocity (see config.HW_DESIGN_GRADIENT).
                dH_eff = max(dH, HW_DESIGN_GRADIENT * L)
                hw_cap = hw_max_flow(L, D, C, dH_eff)
                vel_cap = MAX_DESIGN_VELOCITY_MS * 3.141592653589793 * D * D / 4.0
                G_residual[u][v]["residual_cap"] = max(min(hw_cap, vel_cap), D * 0.001)
            else:
                G_residual[u][v]["residual_cap"] = D * 10.0

        # Add reverse edges for residual (zero initial capacity)
        for u, v in list(G_sub.edges()):
            if not G_residual.has_edge(v, u):
                G_residual.add_edge(v, u, residual_cap=0.0, **G_sub[u][v])

        # Add supersource → all source nodes
        G_residual.add_node(supersource)
        for s in source_nodes:
            cap = SOURCE_SUPPLY_CAP_M3S  # a reservoir is unlimited supply
            G_residual.add_edge(supersource, s, residual_cap=cap)
            G_residual.add_edge(s, supersource, residual_cap=0.0)

        # Add sink nodes → supersink
        G_residual.add_node(supersink)
        sink_demands = {}
        for snk in sink_nodes:
            demand = G_sub.nodes[snk].get("base_demand", MIN_SINK_CAP_M3S)
            sink_demands[snk] = max(demand, 1e-9)
            G_residual.add_edge(snk, supersink, residual_cap=max(demand, MIN_SINK_CAP_M3S))
            G_residual.add_edge(supersink, snk, residual_cap=0.0)

        # Initial demand satisfaction
        demand_satisfaction = {n: 0.0 for n in sink_nodes}
        self._sink_demands = dict(sink_demands)

        # Dinic's with HARP modifications
        total_flow = 0.0
        # Dinic's needs at most |V| BFS phases (each lengthens the shortest
        # augmenting path by >= 1). A fixed cap of 50 silently truncated networks
        # with deeper paths.
        for _ in range(len(G_residual.nodes) + 1):
            level = self._bfs_level_graph(G_residual, supersource, supersink)
            if supersink not in level:
                break
            ptr = {n: 0 for n in G_residual.nodes}
            phase_flow = 0.0
            n_rejected_before = len(rejected_paths)
            while True:
                f = self._harp_dfs(
                    G_sub, G_residual, level, gat_pressures, ptr,
                    supersource, supersink, float("inf"),
                    [supersource], demand_satisfaction,
                    accepted_paths, rejected_paths,
                    use_hydraulic_check,
                )
                if f <= 0:
                    break
                total_flow += f
                phase_flow += f
            # A phase that only rejected paths still made progress: each rejection
            # retires a sink, so the next BFS can reach sinks that were hidden
            # behind it. Stop only when nothing was pushed and nothing retired.
            if phase_flow <= 0 and len(rejected_paths) == n_rejected_before:
                break

        # demand_satisfaction was accumulated live as paths were accepted.
        for snk in sink_nodes:
            demand_satisfaction[snk] = min(max(demand_satisfaction.get(snk, 0.0), 0.0), 1.0)

        return demand_satisfaction, accepted_paths, rejected_paths

    def run(
        self,
        wn,
        failed_pipe_id: str,
        gat_pressures_dict: dict,
        priority_config: dict,
        k: int = DEFAULT_K,
        use_hydraulic_check: bool = True,
        use_priority: bool = True,
        use_subgraph: bool = True,
        gat_inference_time_ms: float = 0.0,
    ) -> HARPResult:
        """Full HARP execution."""
        t_algo_start = time.perf_counter()

        # Build full graph
        G_full, pipe_to_nodes = build_wdn_graph(wn, failed_pipe_id, priority_config)

        # Identify sources and sinks in full graph
        reservoir_nodes = set(wn.reservoir_name_list + wn.tank_name_list)
        all_sink_nodes = [
            n for n in G_full.nodes
            if G_full.nodes[n].get("base_demand", 0) > 0
            and not G_full.nodes[n].get("is_reservoir", False)
        ]

        # Effective pressures always reflect the real GAT prediction; the
        # use_hydraulic_check flag disables the *feasibility check and
        # HW-derived capacity* (see _run_on_subgraph), not the pressures
        # themselves — substituting flat pressures here used to make the
        # feasibility check reject nearly every path outright, the opposite
        # of what "disable hydraulics" is supposed to test.
        #
        # gat_pressures_dict holds *pressure* (WNTR/hydraulic-engineering
        # convention: head − elevation — the quantity a real gauge or IoT
        # sensor reads, and what the GAT is trained to predict). Everything
        # below this point (_hydraulic_feasible_path's available_head, and
        # _run_on_subgraph's dH-based HW capacity) needs true *head* — the
        # elevation-independent driving potential between two points — not
        # raw pressure. Comparing raw pressures directly is wrong whenever
        # two nodes sit at different elevations, and it's *always* wrong at
        # a reservoir: WNTR reports pressure=0 there by definition (a
        # reservoir has no elevation, it defines head directly via
        # base_head), so treating that 0 as "low head" made every path look
        # infeasible from the moment it left the source, regardless of which
        # pipe failed. Converting to head here — once, for every caller —
        # is deliberately inside HARP rather than duplicated at each call site.
        gat_pressures = _pressures_to_heads(gat_pressures_dict, wn)
        if not use_priority:
            # Override priority weights to 1.0
            for n in G_full.nodes:
                G_full.nodes[n]["priority_weight"] = 1.0

        # Choose subgraph
        if use_subgraph and k > 0:
            G_work = extract_failure_subgraph(G_full, failed_pipe_id, pipe_to_nodes, k)
        else:
            G_work = G_full

        # Sources: reservoir/tank nodes present in the k-hop subgraph.
        # If the subgraph is too small to contain any source, fall back to
        # the full graph so HARP doesn't silently return zero-flow results.
        source_nodes = [n for n in G_work.nodes if n in reservoir_nodes]
        if not source_nodes:
            logger.info("No source in k-hop subgraph — falling back to full graph for source nodes")
            source_nodes = [n for n in G_full.nodes if n in reservoir_nodes]
            G_work = G_full
        sink_nodes = [
            n for n in G_work.nodes
            if G_work.nodes[n].get("base_demand", 0) > 0
            and not G_work.nodes[n].get("is_reservoir", False)
        ]

        demand_satisfaction, accepted_paths, rejected_paths = self._run_on_subgraph(
            G_work, gat_pressures, priority_config, source_nodes, sink_nodes,
            use_hydraulic_check,
        )

        # Check escalation
        escalation = False
        if use_subgraph and k > 0:
            critical_sat = {
                n: demand_satisfaction.get(n, 0.0)
                for n in priority_config.get("hospital_nodes", [])
                + priority_config.get("fire_station_nodes", [])
            }
            if any(v < self.escalation_threshold for v in critical_sat.values()):
                logger.info("Escalating to full graph (CDS_critical < %.1f)", self.escalation_threshold)
                source_nodes_full = [n for n in G_full.nodes if n in reservoir_nodes]
                sink_nodes_full = [
                    n for n in G_full.nodes
                    if G_full.nodes[n].get("base_demand", 0) > 0
                    and not G_full.nodes[n].get("is_reservoir", False)
                ]
                demand_satisfaction, accepted_paths, rejected_paths = self._run_on_subgraph(
                    G_full, gat_pressures, priority_config, source_nodes_full, sink_nodes_full,
                    use_hydraulic_check,
                )
                escalation = True
                k = len(G_full.nodes)

        # Fill in non-sink nodes
        for n in G_full.nodes:
            if n not in demand_satisfaction:
                demand_satisfaction[n] = 1.0

        algo_time_ms = (time.perf_counter() - t_algo_start) * 1000.0
        total_time_ms = algo_time_ms + gat_inference_time_ms

        # Aggregate flow per pipe from accepted paths
        flow_allocation: dict = {}
        for rec in accepted_paths:
            nodes_seq = [n for n in rec.nodes if not n.startswith("__SUPER")]
            for i in range(len(nodes_seq) - 1):
                u, v = nodes_seq[i], nodes_seq[i + 1]
                pid = None
                if G_full.has_edge(u, v):
                    pid = G_full[u][v].get("pipe_id")
                elif G_full.has_edge(v, u):
                    pid = G_full[v][u].get("pipe_id")
                if pid:
                    flow_allocation[pid] = round(flow_allocation.get(pid, 0.0) + rec.flow, 6)

        return HARPResult(
            flow_allocation=flow_allocation,
            accepted_paths=accepted_paths,
            rejected_paths=rejected_paths,
            demand_satisfaction=demand_satisfaction,
            gat_predicted_pressures=gat_pressures_dict,
            escalation_flag=escalation,
            k_used=k,
            computation_time_ms=total_time_ms,
            gat_inference_time_ms=gat_inference_time_ms,
            algorithm_time_ms=algo_time_ms,
        )
