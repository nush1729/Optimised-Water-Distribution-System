"""Sensor readings -> anomaly detection -> localisation -> HARP recommendation."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import backend.anomaly.detector as detector
import backend.wntr_engine as engine
from backend import sensors
from backend.config import MODELS_DIR, SNAPSHOT_HOUR
from backend.main import app

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS_DIR, "Net3_gat.pt")),
    reason="Net3 GAT checkpoint not present",
)


@pytest.fixture(scope="module")
def small_baseline(tmp_path_factory):
    """A real (but small) baseline so the test does not need the 100-run cache."""
    d = tmp_path_factory.mktemp("baseline")
    mp = pytest.MonkeyPatch()
    mp.setattr(engine, "GENERATED_DIR", str(d))
    base = engine.generate_baseline_distribution("Net3", n_runs=6, seed=7)
    mp.setattr(detector, "build_baseline", lambda network_name: base)
    yield base
    mp.undo()


def _failure_pressures(pipe_id):
    wn = engine.load_network("Net3")
    sim = engine.run_simulation(engine.inject_failure(wn, pipe_id))
    pdf = sim["pressure_df"]
    return {n: float(pdf.at[pdf.index[SNAPSHOT_HOUR], n]) for n in wn.junction_name_list}


def test_failure_readings_are_detected_localised_and_answered(small_baseline):
    pressures = _failure_pressures("183")           # a failure the detector is known to see
    r = client.post("/anomaly/auto_respond", json={"network": "Net3", "pressures": pressures})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detected"] is True and body["requires_operator_approval"] is True
    wn = engine.load_network("Net3")
    link = wn.get_link(body["suspected_pipe"])
    assert {link.start_node_name, link.end_node_name} & set(body["anomalous_nodes"])   # suspect touches the anomaly
    rec = body["recommendation"]
    assert rec["recovery_plan"]["isolate"][0]["pipe_id"] == body["suspected_pipe"]
    assert 0.0 <= rec["CDS"]["CDS_overall"] <= 1.0


def test_normal_readings_raise_no_alarm(small_baseline):
    normal = {n: v["mean_by_hour"][SNAPSHOT_HOUR] for n, v in small_baseline.items()}
    body = client.post("/anomaly/auto_respond", json={"network": "Net3", "pressures": normal}).json()
    assert body["detected"] is False and body["suspected_pipe"] is None


def test_uses_ingested_readings_when_no_pressures_are_supplied(small_baseline):
    sensors.clear("Net3")
    assert client.post("/anomaly/auto_respond", json={"network": "Net3"}).status_code == 400   # nothing to analyse
    sensors.store_readings("Net3", _failure_pressures("183"))
    assert client.post("/anomaly/auto_respond", json={"network": "Net3"}).json()["detected"] is True
    sensors.clear("Net3")
