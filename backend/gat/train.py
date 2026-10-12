"""Train the PressureGAT model on WDN scenarios."""
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import GENERATED_DIR, MODELS_DIR, RESULTS_DIR, SEED_TRAIN
from backend.gat.dataset import WDNScenarioDataset
from backend.gat.model import PressureGAT

def _compute_impact_sample_weights(train_ds: WDNScenarioDataset, oversample_factor: float = 2.0) -> np.ndarray:
    """Weight each training scenario by how much its failure actually
    depressed pressure relative to the train-set average.

    80% of Net3 single-pipe failures barely move the network-wide pressure
    field at all (this network is heavily looped/redundant) — with uniform
    sampling, the model has little incentive to spend capacity learning to
    condition on the failed-pipe signal, since predicting the "typical"
    field already gets most examples right. Oversampling the high-impact
    minority forces more gradient signal through the cases where the
    failure identity actually matters.
    """
    means = np.array([float(train_ds.get(i).pressure_raw.mean()) for i in range(len(train_ds))])
    global_mean = means.mean()
    # Only reward scenarios that pulled pressure *down* — that's what a real
    # failure does; scenarios sitting above the mean aren't under-weighted,
    # just not boosted.
    impact = np.maximum(global_mean - means, 0.0)
    impact_std = impact.std()
    impact_norm = impact / impact_std if impact_std > 1e-6 else impact
    return 1.0 + oversample_factor * impact_norm


def train_model(
    network_name: str = "Net3",
    max_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    lambda_phys: float = 0.01,
    oversample_high_impact: bool = True,
    residual: bool = False,
) -> PressureGAT:
    """Train GAT on pre-generated scenarios. Returns trained model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {network_name} datasets...")
    train_ds = WDNScenarioDataset(network_name, split="train", residual=residual)
    norm_stats = train_ds.norm_stats
    val_ds = WDNScenarioDataset(network_name, split="val", norm_stats=norm_stats)
    test_ds = WDNScenarioDataset(network_name, split="test", norm_stats=norm_stats)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if oversample_high_impact:
        from torch.utils.data import WeightedRandomSampler
        weights = _compute_impact_sample_weights(train_ds)
        print(f"  Oversampling high-impact scenarios: weight range [{weights.min():.2f}, {weights.max():.2f}]")
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(SEED_TRAIN)
    np.random.seed(SEED_TRAIN)
    random.seed(SEED_TRAIN)

    node_in = int(train_ds.get(0).x.shape[1])      # 8 for legacy data, 10 with the load features
    print(f"  Node input features: {node_in}")
    model = PressureGAT(node_in=node_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    log_path = os.path.join(RESULTS_DIR, f"training_{network_name}_log.csv")
    with open(log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                mse = mse_loss(out, batch.y)
                demands = batch.x[:, 1]
                phys = model.physics_loss_term(
                    out, batch.edge_index, batch.edge_attr, demands, lambda_phys
                )
                loss = mse + phys
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr)
                    val_losses.append(float(mse_loss(out, batch.y).item()))

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses) if val_losses else float("inf")
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])
            lf.flush()

            if epoch % 10 == 0 or epoch <= 5:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.4f} at epoch {best_epoch})")
                    break

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{network_name}_gat.pt")
    torch.save({"model_state": model.state_dict(), "norm_stats": norm_stats, "node_in": node_in}, model_path)
    print(f"  Model saved: {model_path}")

    # Test evaluation (CPU: matches how the API serves the model, and keeps the
    # reported inference time comparable to production).
    model.cpu()
    model.eval()
    all_true, all_pred, all_base = [], [], []
    inference_times = []

    with torch.no_grad():
        for batch in test_loader:
            t0 = time.perf_counter()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            dt = (time.perf_counter() - t0) * 1000.0
            inference_times.append(dt)
            # Denormalize
            p_min = float(batch.pressure_min[0]) if hasattr(batch, 'pressure_min') else 0.0
            p_range_val = float(batch.pressure_range[0]) if hasattr(batch, 'pressure_range') else 1.0
            pred_m = out.numpy() * p_range_val + p_min + batch.baseline.numpy()
            true_m = batch.pressure_raw.numpy()
            all_true.extend(true_m.tolist())
            all_pred.extend(pred_m.tolist())
            all_base.extend(batch.baseline.numpy().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    mae = float(np.mean(np.abs(all_true - all_pred)))
    rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
    # Relative error (avoid div by zero)
    nonzero = np.abs(all_true) > 0.1
    mre = float(np.mean(np.abs((all_true[nonzero] - all_pred[nonzero]) / all_true[nonzero]) * 100)) if nonzero.sum() > 0 else 0.0
    mean_inf_time = float(np.mean(inference_times)) if inference_times else 0.0
    ss_res = float(np.sum((all_true - all_pred) ** 2))
    ss_tot = float(np.sum((all_true - all_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Naive reference: always predict the training-set mean pressure. A model
    # that does not clearly beat this has learned nothing about the failure.
    train_mean = float(np.mean([float(train_ds.get(i).pressure_raw.mean()) for i in range(len(train_ds))]))
    naive_mae = float(np.mean(np.abs(all_true - train_mean)))
    # "Assume nothing failed": predict the intact network's pressure (residual mode only;
    # the baseline is stored with the checkpoint). A model must beat this to be useful.
    intact_mae = float(np.mean(np.abs(all_true - np.array(all_base)))) if norm_stats.get("residual") else float("nan")

    print(f"\n=== Test Results ({network_name}) ===")
    print(f"  MAE: {mae:.4f} m")
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  R2: {r2:.4f}")
    print(f"  Naive (predict train mean) MAE: {naive_mae:.4f} m")
    if norm_stats.get("residual"):
        print(f"  Intact-network baseline MAE: {intact_mae:.4f} m")
    print(f"  Mean Relative Error: {mre:.2f}%")
    print(f"  Mean Inference Time: {mean_inf_time:.1f} ms")

    # Save test CSV
    test_csv_path = os.path.join(RESULTS_DIR, f"gat_test_{network_name}.csv")
    rows = [{"true_head_m": float(t), "pred_head_m": float(p),
             "abs_error_m": abs(float(t)-float(p)),
             "relative_error_pct": abs(float(t)-float(p))/max(abs(float(t)),0.1)*100}
            for t, p in zip(all_true, all_pred)]
    with open(test_csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Save summary — upsert by network so training Net3 then L-Town doesn't
    # silently discard the Net3 row (this file used to be opened in "w" mode
    # unconditionally, so only the most recently trained network's summary
    # ever survived).
    summary_path = os.path.join(RESULTS_DIR, "experiment_2_summary.csv")
    fieldnames = ["network", "MAE_m", "RMSE_m", "mean_relative_error_pct", "inference_ms", "R2", "naive_MAE_m", "intact_baseline_MAE_m", "residual"]
    existing_rows = {}
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                existing_rows[row["network"]] = row
    existing_rows[network_name] = {
        "network": network_name, "MAE_m": mae, "RMSE_m": rmse,
        "mean_relative_error_pct": mre, "inference_ms": mean_inf_time,
        "R2": r2, "naive_MAE_m": naive_mae,
        "intact_baseline_MAE_m": intact_mae, "residual": bool(norm_stats.get("residual")),
    }
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in existing_rows.values():
            writer.writerow(row)

    return model


if __name__ == "__main__":
    import argparse
    np.random.seed(SEED_TRAIN)
    random.seed(SEED_TRAIN)
    torch.manual_seed(SEED_TRAIN)

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="Net3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual", action="store_true",
                        help="predict the change from the intact network's pressure instead of absolute pressure")
    args = parser.parse_args()
    train_model(args.network, max_epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, lr=args.lr, residual=args.residual)
