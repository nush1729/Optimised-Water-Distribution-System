import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.algorithms.hw_hydraulics import (
    hw_head_loss,
    path_head_loss,
    path_hydraulically_feasible,
)


def test_hw_head_loss_basic():
    """hw_head_loss(Q=0.05, L=100, D=0.2, C=120) should be approx 2.18m."""
    result = hw_head_loss(Q_m3s=0.05, L_m=100, D_m=0.2, C_hw=120)
    # Verify within 5% of hand-calculated value
    expected = 10.67 * 100 * (0.05 ** 1.852) / ((120 ** 1.852) * (0.2 ** 4.87))
    assert abs(result - expected) < 1e-6
    # Should be around 2 meters for these inputs
    assert 1.0 < result < 5.0


def test_hw_head_loss_zero_Q_raises():
    """Q <= 0 should raise ValueError."""
    with pytest.raises(ValueError):
        hw_head_loss(Q_m3s=0, L_m=100, D_m=0.2, C_hw=120)


def test_hw_head_loss_negative_Q_raises():
    """Negative Q should raise ValueError."""
    with pytest.raises(ValueError):
        hw_head_loss(Q_m3s=-0.01, L_m=100, D_m=0.2, C_hw=120)


def test_hw_head_loss_linear_in_L():
    """Head loss doubles when L doubles (all else equal)."""
    h1 = hw_head_loss(Q_m3s=0.05, L_m=100, D_m=0.2, C_hw=120)
    h2 = hw_head_loss(Q_m3s=0.05, L_m=200, D_m=0.2, C_hw=120)
    assert abs(h2 / h1 - 2.0) < 1e-6, f"Expected 2x but got {h2/h1}"


def test_hw_head_loss_smaller_diameter_higher_loss():
    """Smaller diameter yields dramatically higher head loss."""
    h_large = hw_head_loss(Q_m3s=0.05, L_m=100, D_m=0.2, C_hw=120)
    h_small = hw_head_loss(Q_m3s=0.05, L_m=100, D_m=0.1, C_hw=120)
    assert h_small > h_large * 10, f"Expected much higher loss for smaller D: {h_small} vs {h_large}"


def test_path_feasible_insufficient_head():
    """Path is infeasible when available head < head loss + safety margin."""
    edges = [{"length": 1000, "diameter": 0.05, "C_hw": 80}]
    # With small D and large L, head loss will be very high
    feasible, margin = path_hydraulically_feasible(
        path_edges=edges,
        gat_head_source=10.0,
        gat_head_sink=8.0,  # Only 2m available
        Q_estimate_m3s=0.005,
        safety_margin_m=1.0,
    )
    assert not feasible, f"Expected infeasible but got feasible with margin={margin}"
    assert margin < 0


def test_path_feasible_sufficient_head():
    """Path is feasible when available head >> head loss."""
    edges = [{"length": 100, "diameter": 0.3, "C_hw": 140}]
    # Large diameter, short pipe → very small head loss
    feasible, margin = path_hydraulically_feasible(
        path_edges=edges,
        gat_head_source=30.0,
        gat_head_sink=5.0,  # 25m available
        Q_estimate_m3s=0.005,
        safety_margin_m=1.0,
    )
    assert feasible, f"Expected feasible but got infeasible with margin={margin}"
    assert margin > 0
