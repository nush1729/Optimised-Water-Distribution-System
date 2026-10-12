"""Standard Dinic's max-flow algorithm for WDN baseline."""
import time
from collections import deque

import networkx as nx

from backend.config import MIN_SINK_CAP_M3S


class FlowResult:
    def __init__(self, flow_dict, total_flow, demand_satisfaction, computation_time_ms):
        self.flow_dict = flow_dict
        self.total_flow = total_flow
        self.demand_satisfaction = demand_satisfaction
        self.computation_time_ms = computation_time_ms
        self.algorithm = "dinic_standard"


class Dinic:
    """
    Standard Dinic's max-flow.
    Multi-sink handled via virtual supersink.
    Edge capacity proxy: pipe_diameter × 10 (intentionally hydraulics-blind).
    """

    def __init__(self):
        self._graph = {}   # node → {neighbor: [capacity, rev_index]}
        self._level = {}
        self._ptr = {}

    def _add_edge(self, u, v, cap):
        if u not in self._graph:
            self._graph[u] = []
        if v not in self._graph:
            self._graph[v] = []
        self._graph[u].append([cap, len(self._graph[v])])
        self._graph[v].append([0, len(self._graph[u]) - 1])

    def _bfs(self, source, supersink):
        self._level = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for i, (cap, _) in enumerate(self._graph.get(u, [])):
                v = self._nbr(u, i)
                if v not in self._level and cap > 0:
                    self._level[v] = self._level[u] + 1
                    q.append(v)
        return supersink in self._level

    def _nbr(self, u, edge_idx):
        # Reconstruct neighbor from edge list
        # We store neighbors separately for speed
        return self._nbrs[u][edge_idx]

    def _dfs(self, u, supersink, pushed):
        if u == supersink:
            return pushed
        edges = self._graph.get(u, [])
        while self._ptr[u] < len(edges):
            i = self._ptr[u]
            cap, rev_i = edges[i]
            v = self._nbrs[u][i]
            if cap > 0 and self._level.get(v, -1) == self._level[u] + 1:
                d = self._dfs(v, supersink, min(pushed, cap))
                if d > 0:
                    edges[i][0] -= d
                    self._graph[v][rev_i][0] += d
                    return d
            self._ptr[u] += 1
        return 0

    def run(self, G: nx.DiGraph, source: str, sinks: list) -> FlowResult:
        t0 = time.perf_counter()

        # Build internal graph
        self._graph = {}
        self._nbrs = {}
        supersink = "__SUPERSINK__"

        # Add all G edges. Capacity defaults to diameter * 10 (the original
        # hydraulics-blind proxy, kept as-is for callers that don't supply
        # anything better — e.g. the Net3/Hanoi/L-Town pipeline). Callers can
        # instead set an explicit "capacity" edge attribute — e.g. a
        # velocity-based capacity (v_max * pipe cross-sectional area) — which
        # is what backend/geo/india_wdn.py now does, since diameter*10 turned
        # out to be 20-85x more generous than realistic pipe capacity at the
        # diameters used there (see run_dinic_simulation's docstring).
        for u, v, data in G.edges(data=True):
            cap = data.get("capacity", data.get("diameter", 0.1) * 10.0)
            if u not in self._nbrs:
                self._nbrs[u] = []
            if v not in self._nbrs:
                self._nbrs[v] = []
            # Forward
            if u not in self._graph:
                self._graph[u] = []
            if v not in self._graph:
                self._graph[v] = []
            fwd_idx = len(self._graph[u])
            rev_idx = len(self._graph[v])
            self._graph[u].append([cap, rev_idx])
            self._graph[v].append([0, fwd_idx])
            self._nbrs[u].append(v)
            self._nbrs[v].append(u)

        # Add supersink edges with capacity = base_demand of sink node
        self._nbrs[supersink] = []
        self._graph[supersink] = []
        for sink in sinks:
            if sink not in G.nodes:
                continue
            demand = G.nodes[sink].get("base_demand", MIN_SINK_CAP_M3S)
            cap = max(demand, MIN_SINK_CAP_M3S)
            if sink not in self._nbrs:
                self._nbrs[sink] = []
            if sink not in self._graph:
                self._graph[sink] = []
            fwd_idx = len(self._graph[sink])
            rev_idx = len(self._graph[supersink])
            self._graph[sink].append([cap, rev_idx])
            self._graph[supersink].append([0, fwd_idx])
            self._nbrs[sink].append(supersink)
            self._nbrs[supersink].append(sink)

        if source not in self._graph:
            self._graph[source] = []
            self._nbrs[source] = []

        # Run Dinic's
        total_flow = 0.0
        while self._bfs(source, supersink):
            self._ptr = {n: 0 for n in self._graph}
            while True:
                f = self._dfs(source, supersink, float("inf"))
                if f == 0:
                    break
                total_flow += f

        # Compute demand satisfaction
        demand_satisfaction = {}
        for sink in sinks:
            if sink not in G.nodes:
                continue
            demand = G.nodes[sink].get("base_demand", MIN_SINK_CAP_M3S)
            # Find flow into sink from supersink edges
            flow_to_sink = 0.0
            for i, (cap, _) in enumerate(self._graph.get(sink, [])):
                v = self._nbrs[sink][i]
                if v == supersink:
                    # Original cap was demand, remaining cap = demand - flow
                    orig_cap = max(demand, MIN_SINK_CAP_M3S)
                    flow_to_sink = orig_cap - cap
            demand_satisfaction[sink] = min(flow_to_sink / max(demand, 1e-9), 1.0)

        # For non-sink nodes
        for node in G.nodes:
            if node not in demand_satisfaction:
                demand_satisfaction[node] = 1.0  # no demand → satisfied

        ct_ms = (time.perf_counter() - t0) * 1000.0
        return FlowResult(
            flow_dict={},
            total_flow=total_flow,
            demand_satisfaction=demand_satisfaction,
            computation_time_ms=ct_ms,
        )
