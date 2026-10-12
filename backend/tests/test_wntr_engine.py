"""wntr_engine: failure injection, simulation and graph-feature extraction on Net3."""
import os
import sys

import numpy as np
import pytest
import wntr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.config import SNAPSHOT_HOUR
from backend.wntr_engine import (
    extract_graph_features,
    generate_baseline_distribution,
    scale_node_demand,
    inject_demand_surge,
    inject_failure,
    inject_leak,
    inject_multi_failure,
    load_network,
    run_simulation,
)

CLOSED = wntr.network.LinkStatus.Closed


@pytest.fixture(scope="module")
def net3():
    return load_network("Net3")


def test_load_network_uses_pressure_dependent_demand(net3):
    assert net3.options.hydraulic.demand_model == "PDA"
    assert net3.options.hydraulic.headloss == "H-W"


def test_inject_failure_closes_only_that_pipe_and_does_not_mutate_the_original(net3):
    pid = net3.pipe_name_list[7]
    failed = inject_failure(net3, pid)
    assert failed.get_link(pid).initial_status == CLOSED
    assert net3.get_link(pid).initial_status != CLOSED          # original untouched
    closed_before = {p for p in net3.pipe_name_list if net3.get_link(p).initial_status == CLOSED}
    closed_after = {p for p in failed.pipe_name_list if failed.get_link(p).initial_status == CLOSED}
    assert closed_after == closed_before | {pid}


def test_inject_failure_rejects_unknown_pipe(net3):
    with pytest.raises(ValueError):
        inject_failure(net3, "no-such-pipe")


def test_inject_multi_failure_closes_every_listed_pipe(net3):
    pids = net3.pipe_name_list[:3]
    failed = inject_multi_failure(net3, pids)
    assert all(failed.get_link(p).initial_status == CLOSED for p in pids)


def test_inject_demand_surge_scales_only_selected_nodes(net3):
    nid = next(n for n in net3.junction_name_list if float(net3.get_node(n).base_demand) > 0)
    other = next(n for n in net3.junction_name_list if n != nid and float(net3.get_node(n).base_demand) > 0)
    base, base_other = float(net3.get_node(nid).base_demand), float(net3.get_node(other).base_demand)
    surged = inject_demand_surge(net3, [nid], 2.5)
    assert float(surged.get_node(nid).base_demand) == pytest.approx(base * 2.5)
    assert float(surged.get_node(other).base_demand) == pytest.approx(base_other)
    assert float(net3.get_node(nid).base_demand) == pytest.approx(base)


def test_inject_leak_splits_the_pipe_and_loses_water(net3):
    pid = net3.pipe_name_list[10]
    leaky = inject_leak(net3, pid, 0.05)
    assert f"_leak_{pid}" in leaky.junction_name_list and f"{pid}_b" in leaky.pipe_name_list
    assert leaky.get_link(pid).initial_status != CLOSED          # leak keeps the pipe open
    sim = run_simulation(leaky)
    assert sim["converged"]
    base = run_simulation(net3)["pressure_df"].loc[SNAPSHOT_HOUR]
    after = sim["pressure_df"].loc[SNAPSHOT_HOUR]
    common = [n for n in base.index if n in after.index]
    assert float((base[common] - after[common]).max()) > 0.01   # somewhere pressure dropped


def test_run_simulation_covers_24_hours_hourly(net3):
    sim = run_simulation(net3)
    assert sim["converged"]
    assert list(sim["pressure_df"].index) == list(range(25))
    assert SNAPSHOT_HOUR < len(sim["pressure_df"].index)


def test_extract_graph_features_shapes_and_failure_marking(net3):
    pid = net3.pipe_name_list[20]
    wf = inject_failure(net3, pid)
    g = extract_graph_features(wf, failed_pipe_id=pid, priority_config={}, network_name="Net3")
    n = len(g["node_ids"])
    assert g["node_features"].shape == (n, 8)
    assert g["edge_features"].shape[1] == 6
    assert g["edge_index"].shape[0] == 2 and g["edge_index"].shape[1] == g["edge_features"].shape[0]
    assert np.isfinite(g["node_features"]).all() and np.isfinite(g["edge_features"]).all()
    # exactly the failed pipe's edge(s) carry the closed flag (feature 4)
    assert g["edge_features"][:, 4].sum() >= 1


def test_scale_node_demand_really_changes_the_demand(net3):
    """Regression: WNTR's base_demand is read-only, so assigning to it (inside a
    swallow-everything try/except) silently left every demand unchanged."""
    import copy
    wn = copy.deepcopy(net3)
    nid = next(n for n in wn.junction_name_list if float(wn.get_node(n).base_demand) > 0)
    before = float(wn.get_node(nid).base_demand)
    scale_node_demand(wn.get_node(nid), 2.0)
    assert float(wn.get_node(nid).base_demand) == pytest.approx(before * 2.0)


def test_demand_noise_actually_moves_pressures(net3):
    import copy
    wn = copy.deepcopy(net3)
    for nid in wn.junction_name_list:
        scale_node_demand(wn.get_node(nid), 1.5)
    base = run_simulation(net3)["pressure_df"].loc[SNAPSHOT_HOUR]
    heavy = run_simulation(wn)["pressure_df"].loc[SNAPSHOT_HOUR]
    assert float((base - heavy).abs().max()) > 0.01


def test_baseline_distribution_has_real_spread(net3, tmp_path, monkeypatch):
    """The anomaly baseline needs a nonzero std per node; with the demand noise
    silently disabled every run was identical and std collapsed to zero."""
    import backend.wntr_engine as engine
    monkeypatch.setattr(engine, "GENERATED_DIR", str(tmp_path))   # do not touch the real cache
    base = generate_baseline_distribution("Net3", n_runs=4, seed=1)
    stds = [b["std_by_hour"][SNAPSHOT_HOUR] for b in base.values()]
    assert max(stds) > 0.01
