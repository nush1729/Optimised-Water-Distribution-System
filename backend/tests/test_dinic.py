import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import MagicMock

import networkx as nx
from backend.algorithms.dinic import Dinic
from backend.experiments.runner import _run_dinic


def _make_graph(edges):
    G = nx.DiGraph()
    for u, v, cap, diam in edges:
        G.add_edge(u, v, capacity=cap, diameter=diam)
    return G


def test_dinic_parallel_paths():
    """4-node graph: A→B(10), A→C(10), B→D(10), C→D(10). Max flow = 20."""
    G = nx.DiGraph()
    G.add_edge('A', 'B', diameter=1.0)
    G.add_edge('A', 'C', diameter=1.0)
    G.add_edge('B', 'D', diameter=1.0)
    G.add_edge('C', 'D', diameter=1.0)
    # D is the sink with large demand
    G.nodes['D']['base_demand'] = 20.0
    for n in ['A', 'B', 'C']:
        G.nodes[n]['base_demand'] = 0.0

    dinic = Dinic()
    result = dinic.run(G, source='A', sinks=['D'])
    assert result.total_flow > 0, "Expected positive flow"
    assert result.computation_time_ms >= 0


def test_dinic_single_path():
    """Single path A→B→C, capacity limited by diameter."""
    G = nx.DiGraph()
    G.add_edge('A', 'B', diameter=0.5)
    G.add_edge('B', 'C', diameter=0.5)
    G.nodes['C']['base_demand'] = 5.0
    for n in ['A', 'B']:
        G.nodes[n]['base_demand'] = 0.0

    dinic = Dinic()
    result = dinic.run(G, source='A', sinks=['C'])
    assert result.total_flow > 0
    assert result.total_flow <= 0.5 * 10 + 0.01  # capacity = diameter * 10


def test_dinic_no_path():
    """Disconnected graph: no flow possible."""
    G = nx.DiGraph()
    G.add_node('A')
    G.add_node('B')
    G.nodes['A']['base_demand'] = 0.0
    G.nodes['B']['base_demand'] = 5.0

    dinic = Dinic()
    result = dinic.run(G, source='A', sinks=['B'])
    assert result.total_flow == 0.0, f"Expected 0 flow but got {result.total_flow}"


def _make_mock_wn(nodes, pipes, reservoir_nodes):
    wn = MagicMock()
    wn.junction_name_list = [n for n in nodes if n not in reservoir_nodes]
    wn.reservoir_name_list = list(reservoir_nodes)
    wn.tank_name_list = []
    wn.pipe_name_list = list(pipes.keys())

    def get_node(nid):
        node = MagicMock()
        node.elevation = nodes[nid].get('elevation', 0.0)
        node.base_demand = nodes[nid].get('base_demand', 0.0)
        node.initial_status = None
        return node

    def get_link(pid):
        pipe_data = pipes[pid]
        link = MagicMock()
        link.start_node_name = pipe_data['start']
        link.end_node_name = pipe_data['end']
        link.length = pipe_data.get('length', 100.0)
        link.diameter = pipe_data.get('diameter', 0.3)
        link.roughness = pipe_data.get('roughness', 120.0)
        link.initial_status = None
        return link

    wn.get_node.side_effect = get_node
    wn.get_link.side_effect = get_link
    return wn


def test_run_dinic_uses_every_reservoir_not_just_the_first():
    """_run_dinic() used to run Dinic's from source_nodes[0] alone — an
    arbitrary single reservoir. A sink reachable from a *second* reservoir
    but not the first would then always read as unsatisfied, regardless of
    which pipe failed. Two reservoirs, each with its own dedicated pipe to
    a shared sink; only reservoir_b's pipe is intact.
    """
    nodes = {
        'reservoir_a': {'elevation': 0.0, 'base_demand': 0.0},
        'reservoir_b': {'elevation': 0.0, 'base_demand': 0.0},
        'sink': {'elevation': 0.0, 'base_demand': 1.0},
    }
    pipes = {
        # reservoir_a's only route to sink is via 'unrelated_pipe', which we
        # mark as the failed pipe below, so 'sink' is unreachable from
        # reservoir_a specifically.
        'unrelated_pipe': {'start': 'reservoir_a', 'end': 'sink', 'diameter': 0.3},
        'good_pipe': {'start': 'reservoir_b', 'end': 'sink', 'diameter': 0.3},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir_a', 'reservoir_b'])
    priority_config = {"hospital_nodes": [], "fire_station_nodes": [], "school_nodes": [],
                        "weights": {"residential": 1.0}}

    r = _run_dinic(wn, 'unrelated_pipe', priority_config)
    assert r["demand_satisfaction"]["sink"] > 0.0, (
        "sink should be reachable via reservoir_b even though reservoir_a's "
        "route is cut off by the failed pipe"
    )
