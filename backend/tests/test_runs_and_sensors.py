"""Run history, sensor ingest and the recovery plan."""
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend import runs, sensors
from backend.main import app
from backend.plan import build_recovery_plan
from backend.wntr_engine import load_network

client = TestClient(app)


# ---------------- run history ----------------

def test_record_and_read_back_a_run():
    rid = runs.record_run("scenario", "Net3", method="harp", scenario="burst", failed_pipes=["215"],
                          cds_overall=0.99, cds_critical=1.0, pvc=2, runtime_ms=12.5, escalated=True,
                          extra={"k": 2})
    assert rid is not None
    r = runs.get_run(rid)
    assert r["network"] == "Net3" and r["failed_pipes"] == ["215"]
    assert r["escalated"] is True and r["extra"] == {"k": 2}
    assert runs.list_runs(limit=1)[0]["id"] == rid          # newest first


def test_api_simulation_is_recorded_and_listed():
    before = len(client.get("/runs", params={"limit": 500}).json()["runs"])
    r = client.post("/simulate/scenario", json={"network": "Net3", "pipe_id": "215", "method": "none"})
    assert r.status_code == 200
    listing = client.get("/runs", params={"network": "Net3", "kind": "scenario"}).json()["runs"]
    assert len(client.get("/runs", params={"limit": 500}).json()["runs"]) == before + 1
    assert listing[0]["method"] == "none" and listing[0]["failed_pipes"] == ["215"]
    assert client.get(f"/runs/{listing[0]['id']}").status_code == 200


def test_unknown_run_is_404():
    assert client.get("/runs/99999999").status_code == 404


# ---------------- sensors ----------------

def test_store_readings_merges_and_drops_garbage():
    sensors.clear("TestNet")
    sensors.store_readings("TestNet", {"a": 40.0, "b": "41.5", "c": float("nan"), "d": "oops"})
    snap = sensors.store_readings("TestNet", {"a": 39.0})
    assert snap["readings"] == {"a": 39.0, "b": 41.5}        # merged, NaN and non-numeric dropped
    assert sensors.latest("nope") is None


def test_mqtt_message_parsing():
    assert sensors.parse_mqtt_message("harp/Net3/pressure", b'{"10": 41.2}') == ("Net3", {"10": 41.2})
    assert sensors.parse_mqtt_message("harp/Net3/flow", b"{}") is None
    assert sensors.parse_mqtt_message("harp/Net3/pressure", b"not json") is None
    assert sensors.parse_mqtt_message("harp/Net3/pressure", b"[1,2]") is None


def test_post_readings_validates_nodes_and_serves_latest():
    sensors.clear("Net3")
    assert client.get("/sensors/latest/Net3").status_code == 404
    r = client.post("/sensors/readings", json={"network": "Net3", "readings": {"10": 40.0, "BOGUS": 1.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1 and body["unknown_nodes"] == ["BOGUS"]
    assert client.get("/sensors/latest/Net3").json()["readings"] == {"10": 40.0}
    assert client.post("/sensors/readings", json={"network": "NoSuchNet", "readings": {"x": 1.0}}).status_code == 404


# ---------------- recovery plan ----------------

def test_recovery_plan_lists_isolation_routes_and_unserved_by_priority():
    wn = load_network("Net3")
    pids = wn.pipe_name_list
    nodes = [n for n in wn.junction_name_list if float(wn.get_node(n).base_demand) > 0][:3]
    pc = {"hospital_nodes": [nodes[0]], "fire_station_nodes": [], "school_nodes": []}
    harp = SimpleNamespace(
        flow_allocation={pids[1]: 0.002, pids[2]: 0.005},
        demand_satisfaction={nodes[0]: 0.4, nodes[1]: 1.0, nodes[2]: 0.5, "__SUPERSINK_HARP__": 1.0},
    )
    plan = build_recovery_plan(wn, [pids[0]], harp, pc)
    assert plan["requires_operator_approval"] is True
    assert plan["isolate"][0]["pipe_id"] == pids[0]
    assert [r["pipe_id"] for r in plan["routes"]] == [pids[2], pids[1]]          # largest flow first
    assert plan["routes"][0]["flow_lps"] == pytest.approx(5.0)
    assert [u["node"] for u in plan["unserved"]] == [nodes[0], nodes[2]]          # hospital outranks residential
    assert plan["summary"]["critical_not_fully_served"] == 1
