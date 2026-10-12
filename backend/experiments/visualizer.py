"""Generate paper-quality figures from experiment CSVs."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.config import FIGURES_DIR, RESULTS_DIR

sns.set_palette("colorblind")
DPI = 300
FIGSIZE = (10, 6)


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    size = os.path.getsize(path)
    print(f"  Saved {name} ({size//1024}KB)")


def fig1_comparison_bar():
    """Grouped bar: CDS by method × node type."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_1.csv"))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    node_types = ["CDS_hospital", "CDS_school", "CDS_fire", "CDS_residential", "CDS_overall"]
    labels = ["Hospital", "School", "Fire Station", "Residential", "Overall"]
    methods = ["none", "dinic", "harp"]
    method_labels = ["No Recovery", "Dinic's", "HARP"]
    palette = sns.color_palette("colorblind", n_colors=3)

    x = np.arange(len(node_types))
    width = 0.25
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        sub = df[df["method"] == method]
        means = [sub[col].mean() * 100 for col in node_types]
        stds = [sub[col].std() * 100 for col in node_types]
        bars = ax.bar(x + i * width, means, width, label=label, color=palette[i], alpha=0.85)
        ax.errorbar(x + i * width, means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xlabel("Node Type", fontsize=12)
    ax.set_ylabel("Critical Demand Satisfaction (%)", fontsize=12)
    ax.set_title("Algorithm Comparison: Critical Demand Satisfaction by Node Type", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_1_comparison_bar.png")


def fig2_pvc_boxplot():
    """Box plot: PVC distribution by method."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_1.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ["none", "dinic", "harp"]
    labels = ["No Recovery", "Dinic's", "HARP"]
    data = [df[df["method"] == m]["PVC"].values for m in methods]
    palette = sns.color_palette("colorblind", n_colors=3)
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Recovery Method", fontsize=12)
    ax.set_ylabel("Pressure Violation Count (nodes < 10 PSI)", fontsize=12)
    ax.set_title("Pressure Violation Count Distribution by Method", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_2_pvc_boxplot.png")


def fig3_gat_scatter():
    """Scatter: GAT predicted vs. WNTR true pressure head."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_2.csv"))
    df = df.dropna(subset=["true_head_m", "pred_head_m"])
    # Only include nodes with meaningful pressure
    df = df[df["true_head_m"].abs() > 1.0]

    fig, ax = plt.subplots(figsize=(8, 8))
    palette = sns.color_palette("colorblind", n_colors=2)
    net_colors = {}
    for i, net in enumerate(df["network"].unique()):
        net_colors[net] = palette[i % len(palette)]

    for net, group in df.groupby("network"):
        # Subsample for readability
        sample = group.sample(min(500, len(group)), random_state=42)
        ax.scatter(sample["true_head_m"], sample["pred_head_m"],
                   alpha=0.4, s=8, color=net_colors[net], label=net)

    lims = [min(df["true_head_m"].min(), df["pred_head_m"].min()),
            max(df["true_head_m"].max(), df["pred_head_m"].max())]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True Pressure Head (m)", fontsize=12)
    ax.set_ylabel("GAT Predicted Pressure Head (m)", fontsize=12)
    ax.set_title("GAT Pressure Surrogate: Predicted vs. True", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, "fig_3_gat_scatter.png")


def fig4_gat_error_hist():
    """Histogram: relative error distribution."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_2.csv"))
    df = df.dropna(subset=["relative_error_pct"])
    df = df[df["relative_error_pct"].between(0, 200)]  # clip outliers for display

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["relative_error_pct"], bins=40, color=sns.color_palette("colorblind")[0], alpha=0.8, edgecolor="white")
    ax.axvline(5, color="red", linestyle="--", lw=2, label="5% target threshold")
    ax.axvline(df["relative_error_pct"].median(), color="orange", linestyle="--", lw=1.5,
               label=f"Median = {df['relative_error_pct'].median():.1f}%")
    ax.set_xlabel("Relative Prediction Error (%)", fontsize=12)
    ax.set_ylabel("Frequency (node-scenario pairs)", fontsize=12)
    ax.set_title("GAT Pressure Error Distribution", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, "fig_4_gat_error_hist.png")


def fig5_subgraph_tradeoff():
    """Dual-axis: CDS_critical + CT_ms vs k."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_3.csv"))
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()

    palette = sns.color_palette("colorblind", n_colors=2)
    k_order = [1, 2, 3, 4, "full"]
    df["k_str"] = df["k"].astype(str)

    means_cds, stds_cds, means_ct, stds_ct = [], [], [], []
    for k in k_order:
        sub = df[df["k_str"] == str(k)]
        means_cds.append(sub["CDS_critical"].mean() * 100)
        stds_cds.append(sub["CDS_critical"].std() * 100)
        means_ct.append(sub["CT_ms"].mean())
        stds_ct.append(sub["CT_ms"].std())

    x = np.arange(len(k_order))
    l1 = ax1.plot(x, means_cds, "o-", color=palette[0], lw=2, markersize=8, label="CDS Critical (%)")
    ax1.fill_between(x, np.array(means_cds) - np.array(stds_cds),
                     np.array(means_cds) + np.array(stds_cds), alpha=0.2, color=palette[0])
    l2 = ax2.plot(x, means_ct, "s--", color=palette[1], lw=2, markersize=8, label="CT (ms)")
    ax2.fill_between(x, np.array(means_ct) - np.array(stds_ct),
                     np.array(means_ct) + np.array(stds_ct), alpha=0.2, color=palette[1])

    ax1.set_xlabel("k (hop radius)", fontsize=12)
    ax1.set_ylabel("CDS Critical (%)", fontsize=12, color=palette[0])
    ax2.set_ylabel("Computation Time (ms)", fontsize=12, color=palette[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_order)
    ax1.set_title("Subgraph k: Efficiency vs. Quality Tradeoff", fontsize=13)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11)
    ax1.grid(alpha=0.3)
    _save(fig, "fig_5_subgraph_tradeoff.png")


def fig6_priority_bars():
    """Grouped bars: demand satisfaction by node type, Dinic's vs HARP."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_4.csv"))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    node_types = ["hospital", "fire_station", "school", "residential"]
    labels = ["Hospital", "Fire Station", "School", "Residential"]
    methods = ["dinic", "harp"]
    method_labels = ["Dinic's", "HARP"]
    palette = sns.color_palette("colorblind", n_colors=2)

    x = np.arange(len(node_types))
    width = 0.35
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        sub = df[df["method"] == method]
        means = []
        for nt in node_types:
            vals = sub[sub["node_type"] == nt]["demand_satisfaction"]
            means.append(vals.mean() * 100 if len(vals) > 0 else 0.0)
        ax.bar(x + i * width, means, width, label=label, color=palette[i], alpha=0.85)

    ax.set_xlabel("Node Type", fontsize=12)
    ax.set_ylabel("Demand Satisfaction (%)", fontsize=12)
    ax.set_title("Priority-Aware Recovery: Demand Satisfaction (5 Worst Failures)", fontsize=13)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_6_priority_bars.png")


def fig7_scalability():
    """Bar chart: CT_ms for Dinic's vs HARP on L-Town (log scale)."""
    path = os.path.join(RESULTS_DIR, "experiment_5.csv")
    if not os.path.exists(path):
        print("  experiment_5.csv not found, skipping fig7 (no placeholder figure is written)")
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = ["dinic", "harp_k2"]
    labels = ["Dinic's (full graph)", "HARP (k=2)"]
    palette = sns.color_palette("colorblind", n_colors=2)
    x = df["scenario_id"].unique()
    width = 0.35

    for i, (method, label) in enumerate(zip(methods, labels)):
        sub = df[df["method"] == method].set_index("scenario_id")
        ct = [sub.at[sid, "CT_ms"] if sid in sub.index else 0.0 for sid in sorted(x)]
        ax.bar(np.arange(len(x)) + i * width, ct, width, label=label, color=palette[i], alpha=0.85)

    ax.set_xlabel("Scenario ID", fontsize=12)
    ax.set_ylabel("Computation Time (ms, log scale)", fontsize=12)
    ax.set_title("Scalability: L-Town (782 nodes, 905 pipes)", fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(x)) + width / 2)
    ax.set_xticklabels([str(s) for s in sorted(x)])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_7_scalability.png")


def fig8_ablation_heatmap():
    """Heatmap: metric × variant for ablation study."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_6.csv"))
    variants = ["HARP_full", "HARP_no_hydraulic", "HARP_no_priority", "HARP_no_subgraph"]
    metrics_cols = ["CDS_hospital", "CDS_overall", "PVC", "CT_ms"]
    metric_labels = ["CDS Hospital (%)", "CDS Overall (%)", "PVC (count)", "CT (ms)"]

    pivot_data = {}
    for variant in variants:
        sub = df[df["variant"] == variant]
        pivot_data[variant] = {
            "CDS_hospital": sub["CDS_hospital"].mean() * 100,
            "CDS_overall": sub["CDS_overall"].mean() * 100,
            "PVC": sub["PVC"].mean(),
            "CT_ms": sub["CT_ms"].mean(),
        }

    pivot_df = pd.DataFrame(pivot_data, index=metrics_cols).T

    # Normalize 0–1 per metric
    norm_df = pivot_df.copy()
    for col in metrics_cols:
        mn, mx = pivot_df[col].min(), pivot_df[col].max()
        if col == "PVC" or col == "CT_ms":
            # Lower is better → invert
            norm_df[col] = 1 - (pivot_df[col] - mn) / max(mx - mn, 1e-9)
        else:
            norm_df[col] = (pivot_df[col] - mn) / max(mx - mn, 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    hm = sns.heatmap(
        norm_df[metrics_cols],
        ax=ax,
        annot=pivot_df[metrics_cols].round(2),
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        xticklabels=metric_labels,
    )
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("HARP Variant", fontsize=12)
    ax.set_title("Ablation Study: Contribution of Each HARP Component", fontsize=13)
    plt.tight_layout()
    _save(fig, "fig_8_ablation_heatmap.png")


def generate_all() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating all figures...")

    funcs = [
        ("fig1", fig1_comparison_bar, "experiment_1.csv"),
        ("fig2", fig2_pvc_boxplot, "experiment_1.csv"),
        ("fig3", fig3_gat_scatter, "experiment_2.csv"),
        ("fig4", fig4_gat_error_hist, "experiment_2.csv"),
        ("fig5", fig5_subgraph_tradeoff, "experiment_3.csv"),
        ("fig6", fig6_priority_bars, "experiment_4.csv"),
        ("fig7", fig7_scalability, None),  # optional dep
        ("fig8", fig8_ablation_heatmap, "experiment_6.csv"),
    ]

    for name, fn, dep in funcs:
        if dep and not os.path.exists(os.path.join(RESULTS_DIR, dep)):
            print(f"  Skipping {name}: {dep} not found")
            continue
        try:
            fn()
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Verify outputs
    figs = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    print(f"\nFigures in {FIGURES_DIR}: {len(figs)}")
    for f in sorted(figs):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f} ({size//1024}KB) {'✓' if size > 0 else '✗ EMPTY!'}")


if __name__ == "__main__":
    generate_all()
