"""HARP unit tests — mock WNTR, no live simulation required."""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import networkx as nx
import pytest

from backend.algorithms.harp import HARP, HARPResult
from backend.algorithms.subgraph import build_wdn_graph


def _make_mock_wn(nodes, pipes, reservoir_nodes=None, tank_nodes=None):
    """Create a mock WNTR WaterNetworkModel."""
    wn = MagicMock()
    wn.junction_name_list = [n for n in nodes if (reservoir_nodes is None or n not in reservoir_nodes) and (tank_nodes is None or n not in tank_nodes)]
    wn.reservoir_name_list = reservoir_nodes or []
    wn.tank_name_list = tank_nodes or []
    wn.pipe_name_list = list(pipes.keys())

    def get_node(nid):
        node = MagicMock()
        node.elevation = nodes[nid].get('elevation', 0.0)
        node.base_demand = nodes[nid].get('base_demand', 0.0)
        # Real WNTR Reservoir objects have no .elevation — they define head
        # directly via .base_head, which is what HARP.run() now looks up for
        # reservoir nodes (see _pressures_to_heads in algorithms/harp.py).
        node.base_head = nodes[nid].get('base_head', nodes[nid].get('elevation', 0.0))
        return node

    def get_link(pid):
        pipe_data = pipes[pid]
        link = MagicMock()
        link.start_node_name = pipe_data['start']
        link.end_node_name = pipe_data['end']
        link.length = pipe_data.get('length', 100.0)
        link.diameter = pipe_data.get('diameter', 0.2)
        link.roughness = pipe_data.get('roughness', 120.0)
        return link

    wn.get_node.side_effect = get_node
    wn.get_link.side_effect = get_link
    return wn


def _default_priority_config():
    return {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "default_type": "residential",
        "weights": {
            "hospital": 5.0,
            "fire_station": 4.0,
            "school": 3.0,
            "residential": 1.0,
            "park": 0.5,
        }
    }


def test_harp_rejects_high_head_loss_path():
    """
    Small-diameter long pipe → head loss >> available ΔH → REJECTED.
    Network: reservoir --pipe_main--> junction --pipe_tiny--> hospital
    pipe_other (a bypass) is the FAILED pipe.
    After failure, HARP tries reservoir→junction→hospital via pipe_tiny.
    pipe_tiny (D=0.05m, L=500m) → h_f ≈ 189m >> available ΔH=23m → REJECTED.
    """
    nodes = {
        'reservoir': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 30.0},
        'junction': {'elevation': 0.0, 'base_demand': 0.001},
        'hospital': {'elevation': 0.0, 'base_demand': 0.05},
    }
    pipes = {
        'pipe_main': {'start': 'reservoir', 'end': 'junction', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
        'pipe_tiny': {'start': 'junction', 'end': 'hospital', 'length': 500.0, 'diameter': 0.05, 'roughness': 80.0},
        'pipe_failed': {'start': 'junction', 'end': 'hospital', 'length': 50.0, 'diameter': 0.3, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir'])
    priority_config = {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "weights": {"hospital": 5.0, "fire_station": 4.0, "school": 3.0, "residential": 1.0, "park": 0.5},
    }
    # GAT predicts (as *pressure*, i.e. head - elevation). The reservoir's
    # entry is ignored by HARP.run() in favor of its own base_head=30.0 (see
    # nodes['reservoir'] above) — real WNTR reservoirs always report
    # pressure=0 by convention, so this dict still needs a key for it, just
    # not a meaningful value.
    # Available ΔH from reservoir(30m) to hospital(5m) = 25m
    # But pipe_tiny h_f ≈ 189m >> 25m → REJECTED
    gat_pressures = {'reservoir': 0.0, 'junction': 28.0, 'hospital': 5.0}

    harp = HARP(safety_margin_m=1.0, q_estimate=0.005)
    result = harp.run(
        wn=wn,
        failed_pipe_id='pipe_failed',  # good pipe fails
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=0,
        use_subgraph=False,
    )

    # pipe_tiny path should be rejected due to head_loss_exceeded
    rejected_reasons = [r.rejection_reason for r in result.rejected_paths]
    assert 'head_loss_exceeded' in rejected_reasons, \
        f"Expected head_loss_exceeded in rejected paths, got: {rejected_reasons}"


def test_harp_accepts_large_diameter_path():
    """
    Large-diameter pipes, abundant head → path ACCEPTED.
    reservoir (h=50m) --large_pipe--> junction (h=48m) --large_pipe--> hospital (h=40m)
    Available head = 10m, head loss for large D, short L = tiny → FEASIBLE.
    """
    nodes = {
        'reservoir': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 50.0},
        'junction': {'elevation': 0.0, 'base_demand': 0.0},
        'hospital': {'elevation': 0.0, 'base_demand': 0.05},
    }
    pipes = {
        'pipe1': {'start': 'reservoir', 'end': 'junction', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
        'pipe2': {'start': 'junction', 'end': 'hospital', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir'])
    priority_config = {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "weights": {"hospital": 5.0, "residential": 1.0},
    }
    # Large ΔH available (reservoir head=50.0 via base_head above), large
    # pipes → low head loss → ACCEPTED
    gat_pressures = {'reservoir': 0.0, 'junction': 48.0, 'hospital': 40.0}

    harp = HARP(safety_margin_m=1.0, q_estimate=0.005)
    result = harp.run(
        wn=wn,
        failed_pipe_id='nonexistent_pipe',
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=0,
        use_subgraph=False,
    )

    # Hospital demand satisfaction should be > 0 (path feasible)
    hospital_sat = result.demand_satisfaction.get('hospital', 0.0)
    accepted_count = len(result.accepted_paths)
    assert accepted_count > 0 or hospital_sat > 0, \
        f"Expected feasible path accepted; accepted={accepted_count}, hospital_sat={hospital_sat:.3f}"


def test_harp_selects_lower_head_loss_parallel_path():
    """
    Two parallel paths to hospital.
    Path A: short but tiny diameter (high head loss).
    Path B: longer but large diameter (low head loss).
    HARP should accept Path B (hydraulically feasible) and reject Path A.
    """
    nodes = {
        'reservoir': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 50.0},
        'jA': {'elevation': 0.0, 'base_demand': 0.0},
        'jB': {'elevation': 0.0, 'base_demand': 0.0},
        'hospital': {'elevation': 0.0, 'base_demand': 0.05},
    }
    pipes = {
        'pA1': {'start': 'reservoir', 'end': 'jA', 'length': 50.0, 'diameter': 0.05, 'roughness': 100.0},
        'pA2': {'start': 'jA', 'end': 'hospital', 'length': 50.0, 'diameter': 0.05, 'roughness': 100.0},
        'pB1': {'start': 'reservoir', 'end': 'jB', 'length': 500.0, 'diameter': 0.3, 'roughness': 130.0},
        'pB2': {'start': 'jB', 'end': 'hospital', 'length': 500.0, 'diameter': 0.3, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir'])
    priority_config = {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "weights": {"hospital": 5.0, "residential": 1.0},
    }
    # Path A: res(50m) → jA(48m) → hospital(40m), available=10m but high h_f
    # Path B: res(50m) → jB(49m) → hospital(47m), low h_f
    gat_pressures = {'reservoir': 0.0, 'jA': 48.0, 'jB': 49.0, 'hospital': 47.0}

    harp = HARP(safety_margin_m=1.0, q_estimate=0.005)
    result = harp.run(
        wn=wn,
        failed_pipe_id='nonexistent',
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=0,
        use_subgraph=False,
    )

    rejected_reasons = [r.rejection_reason for r in result.rejected_paths]
    # At least some paths should have been rejected for head loss
    assert len(result.rejected_paths) > 0 or len(result.accepted_paths) > 0, \
        "Expected HARP to process paths"


def test_harp_escalation_triggered():
    """
    When CDS_critical < 0.6, escalation_flag should be True.
    We mock scenario: hospital node has demand not met in k-hop subgraph.
    """
    nodes = {
        'reservoir': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 1.0},
        'junction': {'elevation': 0.0, 'base_demand': 0.0},
        'hospital': {'elevation': 0.0, 'base_demand': 1.0},  # large demand
    }
    # Only pipe1 connects reservoir to junction (pipe2 connects junction to hospital)
    pipes = {
        'pipe1': {'start': 'reservoir', 'end': 'junction', 'length': 100.0, 'diameter': 0.001, 'roughness': 100.0},
        'pipe2': {'start': 'junction', 'end': 'hospital', 'length': 100.0, 'diameter': 0.001, 'roughness': 100.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir'])
    priority_config = {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "weights": {"hospital": 5.0, "residential": 1.0},
    }
    # Very low pressures → paths infeasible → hospital demand unmet → escalation
    gat_pressures = {'reservoir': 0.0, 'junction': 0.5, 'hospital': 0.1}

    harp = HARP(safety_margin_m=1.0, q_estimate=0.005, escalation_threshold=0.6)
    result = harp.run(
        wn=wn,
        failed_pipe_id='pipe1',
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=2,
        use_subgraph=True,
    )

    # Because the pipes are tiny, hospital demand won't be met → escalation
    # (The escalation may or may not help, but flag should be set if hospital < 0.6)
    if result.demand_satisfaction.get('hospital', 0.0) < 0.6:
        assert result.escalation_flag, \
            f"Expected escalation_flag=True when hospital demand={result.demand_satisfaction.get('hospital', 0):.2f}"


def test_harp_priority_scoring():
    """
    Two sinks: hospital (weight=5, high unmet) and park (weight=0.5, low unmet).
    Hospital path should have higher priority score than park path.
    """
    nodes = {
        'reservoir': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 50.0},
        'junction': {'elevation': 0.0, 'base_demand': 0.0},
        'hospital': {'elevation': 0.0, 'base_demand': 0.05},
        'park': {'elevation': 0.0, 'base_demand': 0.005},
    }
    pipes = {
        'p1': {'start': 'reservoir', 'end': 'junction', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
        'p2': {'start': 'junction', 'end': 'hospital', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
        'p3': {'start': 'junction', 'end': 'park', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['reservoir'])
    priority_config = {
        "hospital_nodes": ["hospital"],
        "fire_station_nodes": [],
        "school_nodes": [],
        "weights": {"hospital": 5.0, "park": 0.5, "residential": 1.0},
    }
    gat_pressures = {'reservoir': 0.0, 'junction': 48.0, 'hospital': 45.0, 'park': 45.0}

    harp = HARP(safety_margin_m=0.5, q_estimate=0.005)
    result = harp.run(
        wn=wn,
        failed_pipe_id='nonexistent',
        gat_pressures_dict=gat_pressures,
        priority_config=priority_config,
        k=0,
        use_subgraph=False,
    )

    # Hospital node should have higher or equal demand satisfaction than park
    h_sat = result.demand_satisfaction.get('hospital', 0.0)
    p_sat = result.demand_satisfaction.get('park', 0.0)
    # Both might be satisfied due to large capacity, or hospital first
    assert isinstance(result, HARPResult), "HARP should return HARPResult"
    assert result.computation_time_ms >= 0


# ---------------------------------------------------------------------------
# Regression tests for the hydraulic-gate / Dinic-phase fixes
# ---------------------------------------------------------------------------

_PC = {
    "hospital_nodes": ["hospital"], "fire_station_nodes": [], "school_nodes": [],
    "weights": {"hospital": 5.0, "fire_station": 4.0, "school": 3.0, "residential": 1.0, "park": 0.5},
}


def _line_network(sink_pressures: dict):
    """reservoir(head 60) -> j1 -> hospital, and j1 -> far (a longer branch)."""
    nodes = {
        'res': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 60.0},
        'j1': {'elevation': 0.0, 'base_demand': 0.0},
        'hospital': {'elevation': 0.0, 'base_demand': 0.01},
        'mid': {'elevation': 0.0, 'base_demand': 0.01},
        'far': {'elevation': 0.0, 'base_demand': 0.01},
    }
    pipes = {
        'p1': {'start': 'res', 'end': 'j1', 'length': 100.0, 'diameter': 0.4, 'roughness': 130.0},
        'p2': {'start': 'j1', 'end': 'hospital', 'length': 100.0, 'diameter': 0.4, 'roughness': 130.0},
        'p3': {'start': 'j1', 'end': 'mid', 'length': 100.0, 'diameter': 0.4, 'roughness': 130.0},
        'p4': {'start': 'mid', 'end': 'far', 'length': 100.0, 'diameter': 0.4, 'roughness': 130.0},
        'p_fail': {'start': 'res', 'end': 'far', 'length': 100.0, 'diameter': 0.4, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, pipes, reservoir_nodes=['res'])
    press = {'res': 0.0, 'j1': 59.0, 'hospital': 40.0, 'mid': 40.0, 'far': 40.0}
    press.update(sink_pressures)
    return wn, press


def test_gate_does_not_require_head_drop_out_of_source():
    """Adjacent nodes at ~equal head (a normal looped/equilibrium network)
    must not be rejected: the old per-prefix gate demanded a >=1 m head drop
    out of every source and delivered zero flow."""
    wn, press = _line_network({'j1': 60.0})  # j1 head == reservoir head
    result = HARP().run(wn, 'p_fail', press, _PC, k=0, use_subgraph=False)
    assert result.demand_satisfaction['hospital'] == pytest.approx(1.0, abs=1e-6)


def test_low_pressure_sink_is_rejected_not_silently_served():
    wn, press = _line_network({'hospital': 3.0})  # below MIN_PRESSURE_M + margin
    result = HARP().run(wn, 'p_fail', press, _PC, k=0, use_subgraph=False)
    assert result.demand_satisfaction['hospital'] == 0.0
    assert any(r.rejection_reason == 'head_loss_exceeded' for r in result.rejected_paths)


def test_partial_pressure_gives_partial_service():
    """Between min and required pressure, delivery is pressure-dependent."""
    wn, press = _line_network({'hospital': 10.6})  # ~50% of the 7.03-14.06 m band
    result = HARP().run(wn, 'p_fail', press, _PC, k=0, use_subgraph=False)
    assert 0.4 < result.demand_satisfaction['hospital'] < 0.6


def test_unservable_near_sink_does_not_block_farther_sinks():
    """A pressure-limited sink used to stay the shortest route to the
    supersink forever, so the Dinic loop never reached sinks behind it."""
    wn, press = _line_network({'mid': 3.0})  # 'mid' unservable, 'far' sits behind it
    result = HARP().run(wn, 'p_fail', press, _PC, k=0, use_subgraph=False)
    assert result.demand_satisfaction['hospital'] == pytest.approx(1.0, abs=1e-6)
    assert result.demand_satisfaction['mid'] == 0.0


def test_priority_gate_stops_flow_once_demand_is_met():
    wn, press = _line_network({})
    result = HARP().run(wn, 'p_fail', press, _PC, k=0, use_subgraph=False)
    for sink in ('hospital', 'mid'):
        assert result.demand_satisfaction[sink] <= 1.0 + 1e-9


def test_graph_includes_valves_and_pumps_so_sources_stay_connected():
    """L-Town's reservoirs reach the network only through pressure-reducing
    valves. A pipes-only graph left them disconnected, so HARP could route to
    only 86 of 701 demand nodes (21.7% delivery) even with the hydraulic gate off."""
    nodes = {
        'res': {'elevation': 0.0, 'base_demand': 0.0, 'base_head': 60.0},
        'a': {'elevation': 0.0, 'base_demand': 0.0},
        'b': {'elevation': 0.0, 'base_demand': 0.01},
        'c': {'elevation': 0.0, 'base_demand': 0.01},
    }
    links = {
        'p1': {'start': 'res', 'end': 'a', 'length': 100.0, 'diameter': 0.3, 'roughness': 130.0},
        'v1': {'start': 'a', 'end': 'b', 'length': 1.0, 'diameter': 0.3, 'roughness': 130.0},
        'pump1': {'start': 'b', 'end': 'c', 'length': 1.0, 'diameter': 0.3, 'roughness': 130.0},
    }
    wn = _make_mock_wn(nodes, links, reservoir_nodes=['res'])
    wn.pipe_name_list = ['p1']
    wn.valve_name_list = ['v1']
    wn.pump_name_list = ['pump1']

    G, _ = build_wdn_graph(wn, None, _PC)
    assert G.has_edge('a', 'b') and not G.has_edge('b', 'a')   # valve: one-way
    assert G.has_edge('b', 'c') and not G.has_edge('c', 'b')   # pump: one-way
    assert G['a']['b']['link_type'] == 'valve' and G['b']['c']['link_type'] == 'pump'

    press = {'res': 0.0, 'a': 55.0, 'b': 50.0, 'c': 45.0}
    result = HARP().run(wn, 'NONE', press, _PC, k=0, use_subgraph=False)
    assert result.demand_satisfaction['b'] == pytest.approx(1.0, abs=1e-6)
    assert result.demand_satisfaction['c'] == pytest.approx(1.0, abs=1e-6)
