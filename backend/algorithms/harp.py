"""HARP: Hydraulics-Aware Resilient Prioritization Algorithm."""
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from backend.algorithms.hw_hydraulics import path_hydraulically_feasible, hw_max_flow
from backend.algorithms.subgraph import (
    build_wdn_graph,
    escalation_needed,
    extract_failure_subgraph,
)
from backend.config import (
    DEFAULT_K,
    ESCALATION_THRESHOLD,
    GAT_Q_ESTIMATE_M3S,
    HW_SAFETY_MARGIN_M,
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
        self, path: list, G: nx.DiGraph, gat_pressures: dict
    ) -> tuple:
        """Check if a path from path[0] to path[-1] is hydraulically feasible.
        Skips virtual supersource/supersink nodes in pressure evaluation.
        """
        # Filter out virtual nodes
        real_path = [n for n in path if not n.startswith("__SUPER")]
        if len(real_path) < 2:
            return True, 0.0

        src = real_path[0]
        snk = real_path[-1]
        h_src = gat_pressures.get(src, 0.0)
        h_snk = gat_pressures.get(snk, 0.0)

        path_edges = []
        for i in range(len(real_path) - 1):
            u, v = real_path[i], real_path[i + 1]
            if G.has_edge(u, v):
                edata = G[u][v]
                path_edges.append({
                    "length": edata.get("length", 100.0),
                    "diameter": edata.get("diameter", 0.1),
                    "C_hw": edata.get("C_hw", 100.0),
                })

        if not path_edges:
            return True, 0.0

        feasible, margin = path_hydraulically_feasible(
            path_edges, h_src, h_snk, self.q_estimate, self.safety_margin_m
        )
        return feasible, margin

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
            # Accept if priority score > 0
            score = self._get_priority_score(current_path, G_sub, demand_satisfaction)
            if score <= 0:
                rejected_paths.append(RejectedPathRecord(
                    nodes=list(current_path),
                    rejection_reason="no_demand_benefit",
                    head_deficit_m=0.0,
                ))
                return 0.0
            accepted_paths.append(PathRecord(
                nodes=list(current_path),
                flow=pushed,
                priority_score=score,
                head_margin_m=0.0,
            ))
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
                # Hydraulic check (skip for supersink edges, and entirely
                # when use_hydraulic_check=False rather than faking flat
                # pressures that would reject every path outright)
                if v != supersink and use_hydraulic_check:
                    feasible, margin = self._hydraulic_feasible_path(path_to_v, G_sub, gat_pressures)
                    if not feasible:
                        rejected_paths.append(RejectedPathRecord(
                            nodes=path_to_v,
                            rejection_reason="head_loss_exceeded",
                            head_deficit_m=abs(margin),
                        ))
                        ptr[u] += 1
                        continue

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
            L = edata.get("length", 100.0)
            D = edata.get("diameter", 0.1)
            C = edata.get("C_hw", 100.0)
            if use_hydraulic_check:
                h_u = gat_pressures.get(u, 0.0)
                h_v = gat_pressures.get(v, 0.0)
                dH = max(h_u - h_v, 0.0)
                # HW capacity: Q_max = (dH * C^1.852 * D^4.87 / (10.67 * L))^(1/1.852)
                hw_cap = hw_max_flow(L, D, C, dH)
                # Floor: allow at least diameter-based token capacity so the graph
                # stays connected for paths that will be rejected by the feasibility check.
                G_residual[u][v]["residual_cap"] = max(hw_cap, D * 0.001)
            else:
                G_residual[u][v]["residual_cap"] = D * 10.0

        # Add reverse edges for residual (zero initial capacity)
        for u, v in list(G_sub.edges()):
            if not G_residual.has_edge(v, u):
                G_residual.add_edge(v, u, residual_cap=0.0, **G_sub[u][v])

        # Add supersource → all source nodes
        G_residual.add_node(supersource)
        for s in source_nodes:
            cap = 1e6  # unlimited from reservoir
            G_residual.add_edge(supersource, s, residual_cap=cap)
            G_residual.add_edge(s, supersource, residual_cap=0.0)

        # Add sink nodes → supersink
        G_residual.add_node(supersink)
        sink_demands = {}
        for snk in sink_nodes:
            demand = G_sub.nodes[snk].get("base_demand", 0.001)
            sink_demands[snk] = max(demand, 1e-9)
            G_residual.add_edge(snk, supersink, residual_cap=max(demand, 0.001))
            G_residual.add_edge(supersink, snk, residual_cap=0.0)

        # Initial demand satisfaction
        demand_satisfaction = {n: 0.0 for n in sink_nodes}

        # Dinic's with HARP modifications
        total_flow = 0.0
        for _ in range(50):  # max iterations
            level = self._bfs_level_graph(G_residual, supersource, supersink)
            if supersink not in level:
                break
            ptr = {n: 0 for n in G_residual.nodes}
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

        # Compute demand satisfaction from flow allocated to each sink
        for snk in sink_nodes:
            if G_residual.has_edge(snk, supersink):
                orig_cap = sink_demands.get(snk, 0.001)
                remaining_cap = G_residual[snk][supersink].get("residual_cap", orig_cap)
                flow_delivered = max(orig_cap - remaining_cap, 0.0)
                demand_satisfaction[snk] = min(flow_delivered / max(orig_cap, 1e-9), 1.0)

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
