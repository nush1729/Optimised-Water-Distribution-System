"""PyTorch Geometric dataset for WDN pressure prediction."""
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import GENERATED_DIR


class WDNScenarioDataset(Dataset):
    """Loads pre-generated .pkl scenarios and returns PyG Data objects."""

    def __init__(self, network_name: str, split: str = "train", norm_stats: dict = None,
                 residual: bool = False):
        """residual=True trains on the *change* from the intact network's pressure
        (pressure - baseline) instead of absolute pressure. A single failed pipe
        moves pressures by a few metres out of tens, so the absolute target mostly
        asks the model to re-learn the intact pressure field every time. Only used
        when computing stats (train split); later splits follow norm_stats."""
        super().__init__()
        self.network_name = network_name
        self.split = split
        self._residual = residual

        split_path = os.path.join(GENERATED_DIR, f"{network_name}_split.json")
        with open(split_path) as f:
            split_data = json.load(f)
        self._scenario_indices = split_data.get(split, [])

        if norm_stats is None and split == "train":
            self.norm_stats = self._compute_norm_stats()
        elif norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            ns_path = os.path.join(GENERATED_DIR, f"{network_name}_norm_stats.json")
            if os.path.exists(ns_path):
                with open(ns_path) as f:
                    self.norm_stats = json.load(f)
            else:
                self.norm_stats = None

    def _load_scenario(self, idx: int) -> dict:
        path = os.path.join(GENERATED_DIR, f"{self.network_name}_{idx:04d}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _intact_baseline(self, node_ids: list) -> list:
        """Hour-12 pressure of the intact network (no failure, base demands), per node."""
        from backend.config import SNAPSHOT_HOUR
        from backend.wntr_engine import load_network, run_simulation
        sim = run_simulation(load_network(self.network_name))
        if not sim["converged"] or sim["pressure_df"] is None:
            raise RuntimeError(f"Intact {self.network_name} did not converge; cannot build a residual baseline")
        pdf = sim["pressure_df"]
        t = pdf.index[min(SNAPSHOT_HOUR, len(pdf.index) - 1)]
        return [float(pdf.at[t, n]) if n in pdf.columns else 0.0 for n in node_ids]

    def _compute_norm_stats(self) -> dict:
        """Compute normalization statistics from training split only."""
        elevations, demands, pressures, lengths, diameters = [], [], [], [], []
        roughness, log_resistances = [], []
        baseline, baseline_ids = None, None
        abs_pressures = []

        for idx in self._scenario_indices:
            try:
                sc = self._load_scenario(idx)
            except Exception as exc:
                logging.getLogger(__name__).warning("Skipping unreadable scenario %s: %s", idx, exc)
                continue
            nf = sc["node_features"]
            ef = sc["edge_features"]
            ph = sc["pressure_heads_m"]
            abs_pressures.extend(ph.tolist())
            if self._residual:
                if baseline is None:
                    baseline_ids = list(sc["node_ids"])
                    baseline = self._intact_baseline(baseline_ids)
                assert list(sc["node_ids"]) == baseline_ids, "node order differs between scenarios"
                ph = ph - np.asarray(baseline, dtype=np.float32)
            elevations.extend(nf[:, 0].tolist())
            demands.extend(nf[:, 1].tolist())
            pressures.extend(ph.tolist())
            lengths.extend(np.log10(np.maximum(ef[:, 0], 1e-6)).tolist())
            diameters.extend(ef[:, 1].tolist())
            # ef[:, 3] duplicates ef[:, 2] (raw C_hw) — index 2 stays raw for
            # physics_loss_term, index 3 is the copy fed to the GAT and needs
            # its own scale so it doesn't dwarf the 0/1 failure-status feature.
            roughness.extend(ef[:, 3].tolist())
            log_resistances.extend(ef[:, 5].tolist())

        stats = {
            "elev_mean": float(np.mean(elevations)) if elevations else 0.0,
            "elev_std": float(np.std(elevations)) if elevations else 1.0,
            "demand_max": float(np.max(demands)) if demands else 1.0,
            "pressure_min": float(np.min(pressures)) if pressures else 0.0,
            "pressure_range": float(np.max(pressures) - np.min(pressures)) if pressures else 1.0,
            "log_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "log_length_std": float(np.std(lengths)) if lengths else 1.0,
            "diam_max": float(np.max(diameters)) if diameters else 1.0,
            "roughness_max": float(np.max(roughness)) if roughness else 1.0,
            "log_res_mean": float(np.mean(log_resistances)) if log_resistances else 0.0,
            "log_res_std": float(np.std(log_resistances)) if log_resistances else 1.0,
        }
        stats["abs_pressure_min"] = float(np.min(abs_pressures)) if abs_pressures else 0.0
        stats["abs_pressure_range"] = float(np.max(abs_pressures) - np.min(abs_pressures)) if abs_pressures else 1.0
        if self._residual:
            stats["residual"] = True
            stats["baseline"] = baseline
            stats["baseline_node_ids"] = baseline_ids
        # Avoid zero division
        for key in ["elev_std", "pressure_range", "log_length_std", "demand_max", "diam_max",
                    "roughness_max", "log_res_std"]:
            if stats[key] == 0:
                stats[key] = 1.0

        ns_path = os.path.join(GENERATED_DIR, f"{self.network_name}_norm_stats.json")
        with open(ns_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    def _normalize(self, sc: dict) -> Data:
        ns = self.norm_stats or {}
        nf = sc["node_features"].copy()
        ef = sc["edge_features"].copy()
        ph = sc["pressure_heads_m"].copy()

        # Normalize node features
        nf[:, 0] = (nf[:, 0] - ns.get("elev_mean", 0)) / max(ns.get("elev_std", 1), 1e-8)
        d_max = max(ns.get("demand_max", 1), 1e-8)
        nf[:, 1] = nf[:, 1] / d_max

        # Normalize pressure (target)
        p_min = ns.get("pressure_min", 0)
        p_range = max(ns.get("pressure_range", 1), 1e-8)
        if ns.get("residual"):
            baseline_vec = np.asarray(ns["baseline"], dtype=np.float32)
            assert list(sc["node_ids"]) == list(ns["baseline_node_ids"]), "node order differs from the baseline"
            ph = ph - baseline_vec
        else:
            baseline_vec = np.zeros_like(ph)
        ph_norm = (ph - p_min) / p_range

        # Normalize edge features
        log_lens = np.log10(np.maximum(ef[:, 0], 1e-6))
        ll_mean = ns.get("log_length_mean", 0)
        ll_std = max(ns.get("log_length_std", 1), 1e-8)
        ef[:, 0] = (log_lens - ll_mean) / ll_std
        dm_max = max(ns.get("diam_max", 1), 1e-8)
        ef[:, 1] = ef[:, 1] / dm_max
        # ef[:, 2] is left raw — physics_loss_term needs real Hazen-Williams C.
        # ef[:, 3] is the duplicate C fed to the GAT's forward pass; ef[:, 5]
        # (log resistance) is also raw. Both previously entered the model at
        # ~100-140 / large magnitude next to the 0/1 failure-status bit at
        # ef[:, 4], which drowns that bit out in gradient scale — normalize
        # them so the model can actually learn to condition on which pipe failed.
        c_max = max(ns.get("roughness_max", 1), 1e-8)
        ef[:, 3] = ef[:, 3] / c_max
        lr_mean = ns.get("log_res_mean", 0)
        lr_std = max(ns.get("log_res_std", 1), 1e-8)
        ef[:, 5] = (ef[:, 5] - lr_mean) / lr_std

        x = torch.tensor(nf, dtype=torch.float32)
        edge_index = torch.tensor(sc["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(ef, dtype=torch.float32)
        y = torch.tensor(ph_norm, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.node_ids = sc["node_ids"]
        data.network_name = sc["network_name"]
        data.scenario_id = sc["scenario_id"]
        data.failed_pipe_id = sc["failed_pipe_id"]
        data.pressure_raw = torch.tensor(sc["pressure_heads_m"], dtype=torch.float32)
        data.baseline = torch.tensor(baseline_vec, dtype=torch.float32)   # zeros unless residual mode
        data.pressure_min = float(p_min)
        data.pressure_range = float(p_range)
        return data

    def len(self) -> int:
        return len(self._scenario_indices)

    def get(self, idx: int) -> Data:
        scenario_idx = self._scenario_indices[idx]
        sc = self._load_scenario(scenario_idx)
        return self._normalize(sc)
