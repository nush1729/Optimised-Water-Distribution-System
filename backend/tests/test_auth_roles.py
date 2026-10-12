"""API key roles: compute key vs admin key."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import backend.main as main

client = TestClient(main.app)
BODY = {"city": "delhi", "node_id": "H0", "new_type": "hospital"}


def test_dev_mode_is_open(monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "")
    monkeypatch.setattr(main, "_ADMIN_KEY", "")
    assert client.post("/india/priority/override", json=BODY).status_code == 200
    assert client.delete("/india/priority/overrides/delhi").status_code == 200


def test_compute_endpoints_need_the_api_key(monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "compute-key")
    monkeypatch.setattr(main, "_ADMIN_KEY", "admin-key")
    req = {"network": "Net3", "pipe_id": "no-such-pipe", "method": "none"}
    assert client.post("/simulate/scenario", json=req).status_code == 401
    assert client.post("/simulate/scenario", json=req, headers={"X-API-Key": "wrong"}).status_code == 401
    # past auth the request reaches validation (400 for the bad pipe), proving the key was accepted
    assert client.post("/simulate/scenario", json=req, headers={"X-API-Key": "compute-key"}).status_code == 400
    assert client.post("/simulate/scenario", json=req, headers={"X-API-Key": "admin-key"}).status_code == 400
    assert client.get("/health").status_code == 200                      # public


def test_admin_operations_need_the_admin_key(monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "compute-key")
    monkeypatch.setattr(main, "_ADMIN_KEY", "admin-key")
    assert client.post("/india/priority/override", json=BODY).status_code == 401
    assert client.post("/india/priority/override", json=BODY, headers={"X-API-Key": "compute-key"}).status_code == 401
    assert client.post("/india/priority/override", json=BODY, headers={"X-API-Key": "admin-key"}).status_code == 200
    assert client.delete("/india/priority/overrides/delhi", headers={"X-API-Key": "compute-key"}).status_code == 401
    assert client.delete("/india/priority/overrides/delhi", headers={"X-API-Key": "admin-key"}).status_code == 200
    # reading overrides stays public
    assert client.get("/india/priority/overrides/delhi").status_code == 200


def test_api_key_alone_guards_admin_ops_when_no_admin_key_is_set(monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "only-key")
    monkeypatch.setattr(main, "_ADMIN_KEY", "")
    assert client.post("/india/priority/override", json=BODY).status_code == 401
    assert client.post("/india/priority/override", json=BODY, headers={"X-API-Key": "only-key"}).status_code == 200
    client.delete("/india/priority/overrides/delhi", headers={"X-API-Key": "only-key"})
