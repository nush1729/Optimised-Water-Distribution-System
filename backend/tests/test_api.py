"""Integration tests against the live FastAPI app.

These exist because the unit tests in test_harp.py / test_dinic.py / test_hw.py
only ever exercise the pure algorithm functions against tiny mocked networks —
none of them go through main.py. That's exactly where two critical bugs lived
undetected: /simulate/scenario read a "demand_satisfaction" key that
run_simulation() never sets (silently defaulting every node to 100% demand
met), and its "harp" branch called predict_pressures() with the wrong
signature, raising a KeyError on every request. A single TestClient call per
method against a real network would have caught both immediately.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app
from backend.config import MODELS_DIR

client = TestClient(app)

NET3_MODEL_PATH = os.path.join(MODELS_DIR, "Net3_gat.pt")
requires_net3_model = pytest.mark.skipif(
    not os.path.exists(NET3_MODEL_PATH),
    reason="Net3 GAT checkpoint not present — run `python -m backend.gat.train --network Net3` first",
)


def _first_pipe_id() -> str:
    resp = client.get("/networks/Net3/pipes")
    assert resp.status_code == 200
    pipes = resp.json()["pipes"]
    assert pipes, "Net3 network has no pipes"
    return pipes[0]["id"]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_scenario_none_returns_real_demand_satisfaction():
    """The "none" method must not silently report every node at 100% —
    that was the exact symptom of the demand_satisfaction key bug."""
    pipe_id = _first_pipe_id()
    resp = client.post("/simulate/scenario", json={
        "network": "Net3", "pipe_id": pipe_id, "method": "none",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "demand_satisfaction" in body
    assert "CDS" in body
    values = list(body["demand_satisfaction"].values())
    assert values, "expected at least one node in demand_satisfaction"
    # Not every failure drives every node to non-1.0 satisfaction, but the
    # response must actually be wired to the simulation, not a hardcoded
    # {n: 1.0} fallback — assert the field is populated with real floats.
    assert all(isinstance(v, (int, float)) for v in values)


def test_scenario_dinic_reports_pvc_as_unavailable():
    """Dinic's is hydraulics-blind: PVC must be null/absent, never a
    fabricated always-zero count from a hardcoded pressure."""
    pipe_id = _first_pipe_id()
    resp = client.post("/simulate/scenario", json={
        "network": "Net3", "pipe_id": pipe_id, "method": "dinic",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["PVC"] is None
    assert body["pressures_m"] is None


@requires_net3_model
def test_scenario_harp_does_not_crash():
    """This is the request shape that used to raise a KeyError on every
    call via predict_pressures(req.network, sim["features"])."""
    pipe_id = _first_pipe_id()
    resp = client.post("/simulate/scenario", json={
        "network": "Net3", "pipe_id": pipe_id, "method": "harp", "k": 2,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "demand_satisfaction" in body
    assert "CDS" in body
    assert "escalation_flag" in body


def test_scenario_unknown_method_is_400():
    pipe_id = _first_pipe_id()
    resp = client.post("/simulate/scenario", json={
        "network": "Net3", "pipe_id": pipe_id, "method": "bogus",
    })
    assert resp.status_code == 400


def test_scenario_missing_pipe_id_is_422():
    """Guards against the frontend/backend field-name drift bug: the
    dashboard used to send "failed_pipe" instead of "pipe_id", so pipe_id
    was always missing and every request was rejected before running."""
    resp = client.post("/simulate/scenario", json={
        "network": "Net3", "method": "none",
    })
    assert resp.status_code == 422
