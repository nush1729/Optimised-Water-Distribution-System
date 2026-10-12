"""Config that several modules depend on must stay consistent."""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.config import DATA_DIR, NODE_TYPE_WEIGHTS, load_priority_config
from backend.main import app
from backend.wntr_engine import load_network

client = TestClient(app)


@pytest.mark.parametrize("network", ["Net3", "L-Town"])
def test_priority_node_ids_exist_in_the_network(network):
    """The hospital/fire/school ids are hand-picked in priority_config.json; a typo
    would silently turn that node into an ordinary residential node."""
    cfg = load_priority_config(network)
    wn = load_network(network)
    nodes = set(wn.node_name_list)
    for key in ("hospital_nodes", "fire_station_nodes", "school_nodes"):
        missing = [n for n in cfg.get(key, []) if n not in nodes]
        assert not missing, f"{network}.{key} references unknown nodes {missing}"


def test_weights_are_not_duplicated_in_the_json():
    raw = json.load(open(os.path.join(DATA_DIR, "priority_config.json")))
    assert all("weights" not in v for v in raw.values())


def test_loaded_config_uses_the_central_weights():
    assert load_priority_config("Net3")["weights"] == NODE_TYPE_WEIGHTS


def test_sources_have_zero_priority_and_critical_types_rank_highest():
    assert NODE_TYPE_WEIGHTS["reservoir"] == NODE_TYPE_WEIGHTS["water_tower"] == 0
    assert NODE_TYPE_WEIGHTS["hospital"] > NODE_TYPE_WEIGHTS["fire_station"] > NODE_TYPE_WEIGHTS["school"] \
        > NODE_TYPE_WEIGHTS["residential"] > NODE_TYPE_WEIGHTS["park"]


def test_api_exposes_the_same_weights():
    r = client.get("/config/priorities")
    assert r.status_code == 200
    assert r.json()["weights"] == NODE_TYPE_WEIGHTS
