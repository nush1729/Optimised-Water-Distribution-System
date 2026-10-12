"""Z-score anomaly detector: pure logic, no WNTR needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.anomaly.detector import detect_anomalies

BASELINE = {n: {"mean_by_hour": [40.0] * 25, "std_by_hour": [1.0] * 25} for n in "ABCD"}
ENDPOINTS = {"p_ab": ("A", "B"), "p_bc": ("B", "C"), "p_cd": ("C", "D")}


def test_normal_readings_are_not_flagged():
    r = detect_anomalies({"A": 40.2, "B": 39.5, "C": 40.8, "D": 40.0}, BASELINE, 12)
    assert r["anomalous_nodes"] == []
    assert r["most_likely_failed_pipe"] is None


def test_pressure_drop_is_flagged_and_localised_to_the_adjacent_pipe():
    # B and C both drop hard: the pipe between them is the best suspect
    r = detect_anomalies({"A": 40.0, "B": 30.0, "C": 28.0, "D": 40.0}, BASELINE, 12,
                         pipe_endpoints=ENDPOINTS)
    assert set(r["anomalous_nodes"]) == {"B", "C"}
    assert r["most_likely_failed_pipe"] == "p_bc"
    assert r["z_scores"]["C"] < r["z_scores"]["B"] < 0


def test_zero_std_baseline_does_not_divide_by_zero():
    flat = {"A": {"mean_by_hour": [40.0] * 25, "std_by_hour": [0.0] * 25}}
    r = detect_anomalies({"A": 41.0}, flat, 3)
    assert r["z_scores"]["A"] == 1.0   # falls back to std = 1


def test_unknown_nodes_are_ignored():
    r = detect_anomalies({"A": 40.0, "ZZZ": 0.0}, BASELINE, 12)
    assert "ZZZ" not in r["z_scores"]


def test_hour_beyond_the_baseline_is_clamped_not_an_index_error():
    r = detect_anomalies({"A": 40.0}, BASELINE, 999)
    assert "A" in r["z_scores"]


def test_threshold_is_respected():
    readings = {"A": 36.0}   # z = -4
    assert detect_anomalies(readings, BASELINE, 12, z_threshold=3.0)["anomalous_nodes"] == ["A"]
    assert detect_anomalies(readings, BASELINE, 12, z_threshold=5.0)["anomalous_nodes"] == []
