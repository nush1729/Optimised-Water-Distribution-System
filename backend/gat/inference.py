"""GAT inference for real-time pressure prediction."""
import copy
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import MODELS_DIR
from backend.gat.model import PressureGAT

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def load_model(network_name: str):
    """Load trained model and normalization stats. Caches in memory."""
    if network_name in _MODEL_CACHE:
        return _MODEL_CACHE[network_name]

    model_path = os.path.join(MODELS_DIR, f"{network_name}_gat.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {network_name}: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = PressureGAT(node_in=int(checkpoint.get("node_in", 8)))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm_stats = checkpoint.get("norm_stats", {})
    _MODEL_CACHE[network_name] = (model, norm_stats)
    return model, norm_stats


def _residual_baseline(norm_stats: dict, node_ids: list):
    """Intact-network pressure per node when the model predicts the change from it,
    else None. Node order must match what the model was trained on."""
    if not norm_stats.get("residual"):
        return None
    ids = norm_stats["baseline_node_ids"]
    if list(node_ids) != list(ids):
        lookup = dict(zip(ids, norm_stats["baseline"]))
        missing = [n for n in node_ids if n not in lookup]
        if missing:
            raise ValueError(f"No intact baseline for nodes {missing[:3]}...")
        return np.array([lookup[n] for n in node_ids], dtype=np.float32)
    return np.asarray(norm_stats["baseline"], dtype=np.float32)


def _pad_node_features(nf: np.ndarray, model) -> np.ndarray:
    """Models trained with the load features expect more columns than a plain
    extract_graph_features() call provides; the missing ones are ratio = load = 1
    (demands equal the reference, which is what the base network has)."""
    need = int(getattr(model, "node_in", nf.shape[1]))
    if nf.shape[1] < need:
        nf = np.concatenate([nf, np.ones((nf.shape[0], need - nf.shape[1]), dtype=nf.dtype)], axis=1)
    return nf


def predict_pressures(
    model: PressureGAT,
    norm_stats: dict,
    graph_data: dict,
) -> tuple:
    """
    Run GAT forward pass, denormalize, return (pressure_heads_m, inference_time_ms).
    Warns if inference time > 100ms.
    """
    nf = _pad_node_features(graph_data["node_features"].copy(), model)
    ef = graph_data["edge_features"].copy()
    ei = graph_data["edge_index"]

    # Apply same normalization as training
    elev_mean = norm_stats.get("elev_mean", 0.0)
    elev_std = max(norm_stats.get("elev_std", 1.0), 1e-8)
    demand_max = max(norm_stats.get("demand_max", 1.0), 1e-8)
    log_len_mean = norm_stats.get("log_length_mean", 0.0)
    log_len_std = max(norm_stats.get("log_length_std", 1.0), 1e-8)
    diam_max = max(norm_stats.get("diam_max", 1.0), 1e-8)
    p_min = norm_stats.get("pressure_min", 0.0)
    p_range = max(norm_stats.get("pressure_range", 1.0), 1e-8)
    abs_min = norm_stats.get("abs_pressure_min", p_min)
    abs_range = max(norm_stats.get("abs_pressure_range", p_range), 1e-8)
    roughness_max = max(norm_stats.get("roughness_max", 1.0), 1e-8)
    log_res_mean = norm_stats.get("log_res_mean", 0.0)
    log_res_std = max(norm_stats.get("log_res_std", 1.0), 1e-8)

    nf[:, 0] = (nf[:, 0] - elev_mean) / elev_std
    nf[:, 1] = nf[:, 1] / demand_max

    import math
    log_lens = np.log10(np.maximum(ef[:, 0], 1e-6))
    ef[:, 0] = (log_lens - log_len_mean) / log_len_std
    ef[:, 1] = ef[:, 1] / diam_max
    # ef[:, 2] stays raw (unused at inference; kept for parity with training).
    # ef[:, 3] and ef[:, 5] must match the same normalization used in
    # dataset.py._normalize, or a model trained on normalized values will see
    # out-of-distribution inputs at serve time.
    ef[:, 3] = ef[:, 3] / roughness_max
    ef[:, 5] = (ef[:, 5] - log_res_mean) / log_res_std

    x = torch.tensor(nf, dtype=torch.float32)
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_attr = torch.tensor(ef, dtype=torch.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if dt_ms > 100.0:
        logger.warning("GAT inference slow: %.1f ms (>100ms threshold)", dt_ms)

    # Denormalize
    pressures = out.numpy() * p_range + p_min
    base = _residual_baseline(norm_stats, graph_data.get("node_ids", []))
    if base is not None:
        pressures = pressures + base

    # Clamp physically impossible predictions (raw model outputs outside ±2×range
    # produce pressures that violate hydraulic plausibility; they arise when
    # the network is under-represented in the training set, e.g. L-Town with
    # only 160 training scenarios).
    p_lo = abs_min - abs_range
    p_hi = abs_min + 2.0 * abs_range
    n_outliers = int(((pressures < p_lo) | (pressures > p_hi)).sum())
    if n_outliers > 0:
        logger.warning(
            "GAT: %d node predictions outside physical bounds [%.1f, %.1f] — clamping",
            n_outliers, p_lo, p_hi,
        )
        pressures = pressures.clip(p_lo, p_hi)

    return pressures, dt_ms


def predict_pressures_with_uncertainty(
    model: PressureGAT,
    norm_stats: dict,
    graph_data: dict,
    n_samples: int = 20,
) -> tuple:
    """
    MC-Dropout inference: run the model n_samples times with dropout active,
    return (mean_pressures_m, std_pressures_m, inference_time_ms).
    std is the spread of the stochastic passes. Evaluation shows it is NOT calibrated against real error (Spearman std-vs-error is ~0 on Net3 and negative on L-Town), so treat it as a diagnostic, not a confidence.
    Typical interpretation: std < 2m = confident, 2–5m = moderate, >5m = uncertain.
    """
    nf = _pad_node_features(graph_data["node_features"].copy(), model)
    ef = graph_data["edge_features"].copy()
    ei = graph_data["edge_index"]

    elev_mean    = norm_stats.get("elev_mean", 0.0)
    elev_std     = max(norm_stats.get("elev_std", 1.0), 1e-8)
    demand_max   = max(norm_stats.get("demand_max", 1.0), 1e-8)
    log_len_mean = norm_stats.get("log_length_mean", 0.0)
    log_len_std  = max(norm_stats.get("log_length_std", 1.0), 1e-8)
    diam_max     = max(norm_stats.get("diam_max", 1.0), 1e-8)
    p_min        = norm_stats.get("pressure_min", 0.0)
    p_range      = max(norm_stats.get("pressure_range", 1.0), 1e-8)
    abs_min      = norm_stats.get("abs_pressure_min", p_min)
    abs_range    = max(norm_stats.get("abs_pressure_range", p_range), 1e-8)
    roughness_max = max(norm_stats.get("roughness_max", 1.0), 1e-8)
    log_res_mean = norm_stats.get("log_res_mean", 0.0)
    log_res_std  = max(norm_stats.get("log_res_std", 1.0), 1e-8)

    nf[:, 0] = (nf[:, 0] - elev_mean) / elev_std
    nf[:, 1] = nf[:, 1] / demand_max
    log_lens = np.log10(np.maximum(ef[:, 0], 1e-6))
    ef[:, 0] = (log_lens - log_len_mean) / log_len_std
    ef[:, 1] = ef[:, 1] / diam_max
    ef[:, 3] = ef[:, 3] / roughness_max
    ef[:, 5] = (ef[:, 5] - log_res_mean) / log_res_std

    x          = torch.tensor(nf, dtype=torch.float32)
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_attr  = torch.tensor(ef, dtype=torch.float32)

    # Enable dropout on a private copy. The cached model is shared by every
    # request; flipping it to train mode in place would let a concurrent
    # deterministic predict_pressures() call run with dropout active.
    mc_model = copy.deepcopy(model)
    mc_model.train()
    t0 = time.perf_counter()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = mc_model(x, edge_index, edge_attr)
            samples.append(out.numpy() * p_range + p_min)
    base = _residual_baseline(norm_stats, graph_data.get("node_ids", []))
    if base is not None:
        samples = [x + base for x in samples]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    samples    = np.stack(samples, axis=0)  # (n_samples, n_nodes)
    mean_p     = samples.mean(axis=0)
    std_p      = samples.std(axis=0)

    # Clamp mean to physical bounds
    p_lo = abs_min - abs_range
    p_hi = abs_min + 2.0 * abs_range
    mean_p = mean_p.clip(p_lo, p_hi)

    return mean_p, std_p, dt_ms
