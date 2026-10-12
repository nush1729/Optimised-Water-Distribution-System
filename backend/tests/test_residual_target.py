"""Residual-target GAT: the label is (pressure - intact baseline) and inference adds it back."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import backend.gat.dataset as ds_mod
import backend.gat.dataset_generator as dg
from backend.gat.inference import _residual_baseline, predict_pressures


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("resid")
    mp = pytest.MonkeyPatch()
    mp.setattr(dg, "GENERATED_DIR", str(d))
    mp.setattr(ds_mod, "GENERATED_DIR", str(d))
    dg._generate_chunk("Net3", 14, seed_start=0)
    dg._write_split("Net3", 0, 14)
    train = ds_mod.WDNScenarioDataset("Net3", split="train", residual=True)
    yield train, d
    mp.undo()


def test_stats_carry_the_baseline_and_absolute_range(tiny_dataset):
    train, _ = tiny_dataset
    ns = train.norm_stats
    assert ns["residual"] is True
    assert len(ns["baseline"]) == len(ns["baseline_node_ids"])
    assert ns["abs_pressure_range"] > ns["pressure_range"]      # residuals span far less than absolute pressure


def test_label_is_the_normalised_residual(tiny_dataset):
    train, _ = tiny_dataset
    ns = train.norm_stats
    data = train.get(0)
    expected = (data.pressure_raw.numpy() - data.baseline.numpy() - ns["pressure_min"]) / ns["pressure_range"]
    assert np.allclose(data.y.numpy(), expected, atol=1e-5)


def test_inference_adds_the_baseline_back(tiny_dataset):
    """A 'perfect' model (outputs the true normalised label) must reproduce the true pressures."""
    train, d = tiny_dataset
    import pickle
    sc = pickle.load(open(os.path.join(str(d), f"Net3_{train._scenario_indices[0]:04d}.pkl"), "rb"))
    y = train.get(0).y

    class Perfect(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr):
            return y

    pred, _ = predict_pressures(Perfect(), train.norm_stats, sc)
    assert np.allclose(pred, sc["pressure_heads_m"], atol=1e-3)


def test_baseline_is_realigned_when_node_order_differs():
    ns = {"residual": True, "baseline_node_ids": ["a", "b", "c"], "baseline": [1.0, 2.0, 3.0]}
    assert list(_residual_baseline(ns, ["c", "a"])) == [3.0, 1.0]
    with pytest.raises(ValueError):
        _residual_baseline(ns, ["a", "zzz"])
    assert _residual_baseline({"residual": False}, ["a"]) is None


# ---------------- load features (node inputs 8 -> 10) ----------------

def test_extended_features_default_to_ratio_one_and_load_one():
    from backend.wntr_engine import extract_graph_features, load_network
    wn = load_network("Net3")
    g = extract_graph_features(wn, priority_config={}, network_name="Net3", extended=True)
    nf = g["node_features"]
    assert nf.shape[1] == 10
    assert np.allclose(nf[:, 8], 1.0) and np.allclose(nf[:, 9], 1.0)
    assert extract_graph_features(wn, priority_config={}, network_name="Net3")["node_features"].shape[1] == 8


def test_extended_features_measure_deviation_from_the_reference_demand():
    import copy
    from backend.wntr_engine import extract_graph_features, load_network, scale_node_demand
    wn = load_network("Net3")
    ref = {n: float(wn.get_node(n).base_demand) for n in wn.junction_name_list}
    hot = copy.deepcopy(wn)
    for n in hot.junction_name_list:
        scale_node_demand(hot.get_node(n), 1.5)
    g = extract_graph_features(hot, priority_config={}, network_name="Net3", reference_demands=ref, extended=True)
    demand_nodes = [i for i, n in enumerate(g["node_ids"]) if ref.get(n, 0) > 0]
    assert np.allclose(g["node_features"][demand_nodes, 8], 1.5, atol=1e-4)
    assert np.allclose(g["node_features"][:, 9], 1.5, atol=1e-4)


def test_inference_pads_missing_load_features_with_ones():
    class Spy(torch.nn.Module):
        node_in = 10
        seen = None
        def forward(self, x, edge_index, edge_attr):
            Spy.seen = x.shape[1]
            return torch.zeros(x.shape[0])
    from backend.wntr_engine import extract_graph_features, load_network
    g = extract_graph_features(load_network("Net3"), priority_config={}, network_name="Net3")   # 8 columns
    ns = {"pressure_min": 0.0, "pressure_range": 1.0}
    predict_pressures(Spy(), ns, g)
    assert Spy.seen == 10
