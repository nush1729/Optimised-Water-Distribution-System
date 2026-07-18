"""Regression tests for the India WDN simulation path (geo/india_wdn.py).

This is a separate implementation from the WNTR/Net3 pipeline in
experiments/runner.py — it's what the app's only routed "Simulation" page
(IndiaMapView.tsx) actually calls via /india/simulate. It reproduced the same
"100% accuracy" symptom the WNTR pipeline had, for a different reason: HARP's
synthetic pressure proxy (no trained GAT exists for arbitrary India cities)
used raw BFS hop-count as a stand-in for pressure, which gives many node
pairs in a moderately meshed graph the *exact same* hop-count — any edge
connecting two same-hop-count nodes then has zero head differential, which
collapses HARP's per-edge capacity to a near-zero floor. In this network that
starved flow through a large fraction of undamaged pipes, capping demand
satisfaction at ~68% even with zero pipes failed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.geo.india_wdn import fetch_city_pois, run_dinic_simulation, run_harp_simulation, run_simple_simulation

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(
        os.path.dirname(__file__), "..", "data", "generated", "india_wdn_delhi_n20.json"
    )),
    reason="Cached Delhi WDN not present — run fetch_city_pois('delhi', n_nodes=20) once first",
)


def _wdn():
    return fetch_city_pois("delhi", n_nodes=20)


def test_harp_baseline_is_not_severely_degraded_with_no_failures():
    """With zero pipes failed, HARP should deliver most demand — not the
    ~68% ceiling the hop-count-tie bug used to impose regardless of any
    actual failure."""
    wdn = _wdn()
    result = run_harp_simulation(wdn, failed_pipes=[])
    assert result["CDS"]["CDS_overall"] > 0.75, (
        f"Expected healthy baseline satisfaction with no failures, got {result['CDS']['CDS_overall']}"
    )


def test_harp_responds_to_a_real_bottleneck_pipe():
    """P6 is a known structural bottleneck in the cached Delhi WDN — both
    the real BFS simulation and Dinic's independently agree it drops CDS to
    ~0.68. HARP should show a real, different-from-baseline drop too, not
    the same flat number it returns for a non-bottleneck pipe."""
    wdn = _wdn()
    baseline = run_harp_simulation(wdn, failed_pipes=[])
    bottleneck = run_harp_simulation(wdn, failed_pipes=["P6"])
    non_bottleneck = run_harp_simulation(wdn, failed_pipes=["P0"])

    assert bottleneck["CDS"]["CDS_overall"] < baseline["CDS"]["CDS_overall"] - 0.05, (
        "A known bottleneck pipe failure should measurably reduce HARP's CDS_overall"
    )
    # A non-bottleneck pipe (confirmed by both 'none' and 'dinic' staying at
    # full satisfaction) should stay close to the no-failure baseline.
    assert abs(non_bottleneck["CDS"]["CDS_overall"] - baseline["CDS"]["CDS_overall"]) < 0.05


def test_harp_output_varies_across_multiple_different_pipes():
    """Regression test for the original flat/invariant bug: sweep several
    pipes and require more than one distinct CDS_overall value."""
    wdn = _wdn()
    values = set()
    for pid in ["P0", "P6", "P17", "P18", "P19"]:
        r = run_harp_simulation(wdn, failed_pipes=[pid])
        values.add(round(r["CDS"]["CDS_overall"], 3))
    assert len(values) > 1, f"Expected HARP to respond differently to different pipe failures, got {values}"


def test_dinic_and_none_agree_on_reachability_for_a_bottleneck_pipe():
    """Sanity check unaffected by this fix: Dinic's and the real BFS
    simulation should independently agree when a pipe genuinely disconnects
    part of the network (both use the same underlying graph topology)."""
    wdn = _wdn()
    none_r = run_simple_simulation(wdn, failed_pipes=["P6"])
    dinic_r = run_dinic_simulation(wdn, failed_pipes=["P6"])
    assert abs(none_r["CDS"]["CDS_overall"] - dinic_r["CDS"]["CDS_overall"]) < 0.01


def test_dinic_is_hydraulically_evaluated_not_pure_maxflow():
    """Dinic's max-flow still decides the flow allocation (unchanged), but
    the final CDS/PVC must come from a real WNTR solve of that allocation,
    not the raw graph-capacity ratio. Regression test for the fix in
    run_dinic_simulation() — previously PVC was always 0/near-0 because it
    was derived from the same generous diameter×10 max-flow ratio instead
    of real pressure."""
    wdn = _wdn()
    r = run_dinic_simulation(wdn, failed_pipes=["P6"])  # a known bottleneck (see test above)
    assert r["hydraulically_evaluated"] is True
    assert r["pressures_m"] is not None, "Real pressures should be populated once hydraulically evaluated"
    assert r["PVC"] is not None
    # The raw max-flow allocation and the real-hydraulics-evaluated CDS are
    # tracked separately — real hydraulics should never claim *more* than
    # Dinic's own optimistic graph-capacity estimate.
    import numpy as np
    alloc_mean = float(np.mean(list(r["dinic_allocation"].values())))
    assert r["CDS"]["CDS_overall"] <= alloc_mean + 1e-6, (
        "Real hydraulic evaluation should never inflate Dinic's result above its own max-flow allocation"
    )


def test_dinic_pvc_reflects_real_pressure_violations():
    """PVC should be a real count derived from simulated pressures below
    MIN_PRESSURE_M, not the old proxy (demand_satisfaction < 0.5, which
    was almost never true given the generous diameter×10 capacity)."""
    from backend.config import MIN_PRESSURE_M
    wdn = _wdn()
    r = run_dinic_simulation(wdn, failed_pipes=["P6"])
    assert r["hydraulically_evaluated"] is True
    violations = sum(1 for p in r["pressures_m"].values() if p < MIN_PRESSURE_M)
    # PVC excludes source nodes; violations count includes them, so PVC <= violations.
    assert r["PVC"] <= violations
    assert r["PVC"] > 0, "P6 is a known bottleneck — expected at least one real pressure violation"


def test_dinic_multi_failure_remains_hydraulically_evaluated_under_stress():
    """Severe multi-pipe failures shouldn't silently fall back to the
    unevaluated graph estimate — the WNTR solve should still converge."""
    wdn = _wdn()
    edge_ids = [e["id"] for e in wdn["edges"]]
    r = run_dinic_simulation(wdn, failed_pipes=edge_ids[:8])
    assert r["hydraulically_evaluated"] is True
    assert r["PVC"] is not None
