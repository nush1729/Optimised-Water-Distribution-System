# HARP-WDN

**Hydraulics-Aware Resilient Prioritization for Water Distribution Networks**

A full-stack research system for water infrastructure failure simulation and recovery prioritization. Combines WNTR hydraulic simulation, a Graph Attention Network pressure surrogate, and a modified Dinic's max-flow algorithm gated by Hazen-Williams hydraulic feasibility constraints. Deployed as a FastAPI backend with a React/TypeScript dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-15%20passing-brightgreen?style=flat-square)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Features](#features)
4. [Architecture](#architecture)
5. [HARP Algorithm](#harp-algorithm)
6. [PressureGAT Model](#pressuregat-model)
7. [India WDN Module](#india-wdn-module)
8. [Benchmark Experiments](#benchmark-experiments)
9. [Setup](#setup)
10. [Run Order](#run-order)
11. [API Reference](#api-reference)
12. [Project Structure](#project-structure)
13. [Tests](#tests)

---

## Overview

HARP-WDN addresses a core problem in water network emergency response: standard max-flow algorithms (Dinic's, Ford-Fulkerson) treat pipes as capacity-unlimited conduits and report 100% demand satisfaction even when the underlying hydraulics make that physically impossible. HARP fixes this by evaluating every candidate path against the Hazen-Williams head-loss equation before allocating flow, rejecting paths where the available pressure head is insufficient, and prioritizing delivery to hospitals and fire stations over lower-priority nodes.

The system operates at three levels:

- **Online simulation** — live HARP/Dinic comparison on synthetic Indian city networks (Mumbai, Delhi, Bangalore, Hyderabad) with burst, leak, surge, and multi-failure scenarios
- **Neural surrogate** — PressureGAT replaces slow WNTR PDA solves with ~2.5 ms Graph Attention Network inference, with MC-Dropout uncertainty quantification
- **Offline benchmarks** — six reproducible experiments on EPA Net3 (97 nodes) and L-Town (782 nodes) producing paper-quality figures at 300 DPI

---

## Key Results

| Metric | Value |
|---|---|
| Net3 GAT Mean Relative Error | **9.7%** |
| L-Town GAT MRE (2 000-scenario retrain) | **15.5%** |
| Net3 GAT MAE | **3.50 m** |
| GAT inference latency (CPU) | **~2.5 ms** |
| HARP demand satisfaction (typical) | **90.7%** (physically valid) |
| Dinic demand satisfaction (typical) | **100%** (hydraulically unconstrained) |
| Benchmark experiments | **6** |
| Paper-quality figures generated | **16** (300 DPI PNG) |
| Unit tests passing | **15 / 15** |

---

## Features

### HARP Algorithm
Modified Dinic's blocking flow with per-path Hazen-Williams hydraulic feasibility checks. For each candidate path, HARP computes the available head drop across all pipe segments and rejects the path if the net margin falls below 1 m. Accepted paths are scored by `priority_weight × unmet_demand` so hospitals and fire stations receive water before residential and junction nodes. k-hop subgraph localization limits computation to the failure neighbourhood; escalation to the full graph triggers if critical-node CDS falls below 0.6.

### PressureGAT Surrogate
3-layer Graph Attention Network (~180K parameters) predicting nodal pressure heads from 8 node features and 6 edge features. Physics loss (Hazen-Williams flow balance) regularizes training alongside MSE. MC-Dropout over 20 stochastic forward passes provides per-node uncertainty estimates at runtime. Model is cached in memory after first load; subsequent inference requires no disk I/O.

### India WDN Simulation
Synthetic hydraulic networks for four Indian cities generated from OSMnx points-of-interest with a Kruskal MST backbone and priority-weighted redundant edges. Supports four failure scenarios: **burst** (pipe closed, emitter at endpoints), **leak** (pipe open, reduced flow), **surge** (demand multiplier), and **multi-failure** (compound). All node placement, demands, and pipe attributes are seeded for full reproducibility. Priority overrides persist across server restarts via SQLite.

### Live IoT Sensor Stream
Server-Sent Events endpoint streams synthetic pressure telemetry at 2.5 s intervals. Each connection uses entropy-seeded RNG; the frontend shows a LIVE / LOCAL status badge with per-node warnings on threshold breach.

### Anomaly Detection
Z-score detector comparing incoming pressure readings against a per-node baseline distribution built from 100 WNTR scenarios. Returns anomalous nodes, z-scores, candidate failure zone, and the most likely failed pipe — identified by ranking all pipes touching the anomaly zone by combined endpoint |z|.

### Benchmark Experiments
Six reproducible experiments on Net3 and L-Town covering core algorithm comparison, GAT accuracy, k-hop subgraph tradeoff, priority breakdown by node type, L-Town scalability, and a component ablation study. All outputs are CSV + 300 DPI figures. Re-running any experiment with the same seed produces identical results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1 — Data & Simulation                    │
│                                                                  │
│   WNTR Engine          India WDN Generator      IoT SSE Stream  │
│   Net3 / L-Town PDA    OSMnx + Kruskal MST      2.5s entropy    │
│   failure injection    burst/leak/surge/multi    seeded PRNG     │
└──────────────┬─────────────────┬───────────────────┬────────────┘
               │                 │                   │
┌──────────────▼─────────────────▼───────────────────▼────────────┐
│                  LAYER 2 — Algorithms                            │
│                                                                  │
│   PressureGAT              HARP                 Z-Score Detect   │
│   ~180K params             H-W gated Dinic's    baseline ± σ     │
│   MC-Dropout UQ            priority scoring     pipe ranking     │
│   2.5 ms inference         k-hop subgraph       candidate zone   │
└──────────────┬─────────────────┬───────────────────┬────────────┘
               │                 │                   │
┌──────────────▼─────────────────▼───────────────────▼────────────┐
│                  LAYER 3 — Serving                               │
│                                                                  │
│   FastAPI (port 8000)      SQLite DB            React Frontend   │
│   30+ REST endpoints       priority overrides   (port 5173)      │
│   SSE streaming            harp_wdn.db          3-tab dashboard  │
└─────────────────────────────────────────────────────────────────┘
```

---

## HARP Algorithm

Standard Dinic's max-flow uses pipe diameter as a capacity proxy and ignores hydraulic physics entirely — it will always report 100% demand satisfaction because no path is ever hydraulically infeasible to it. HARP gates every candidate path in the blocking-flow phase against the Hazen-Williams equation:

```
Head loss:    h_f = 10.67 × L × Q^1.852 / (C^1.852 × D^4.87)

Max flow:     Q_max = ( ΔH × C^1.852 × D^4.87 / (10.67 × L) )^(1/1.852)
```

A path is rejected if `ΔH − h_f < 1.0 m`. Accepted paths are scored by:

```
score = Σ  priority_weight(node) × (1 − demand_satisfaction(node))
       node ∈ path \ {supersource}
```

Paths with `score ≤ 0` (no unmet demand benefit) are also rejected, preventing the algorithm from routing flow where it is not needed.

| | Dinic (Baseline) | HARP (Proposed) |
|---|---|---|
| Edge capacity | diameter × 10 | H-W Q_max from ΔH |
| Hydraulic check | None | Per path, per segment |
| Demand satisfaction | ~100% (unconstrained) | ~90.7% (physically valid) |
| Priority scoring | None | priority × unmet demand |
| Runtime (Net3) | ~28 ms | ~117 ms |
| Physics loss | Ignored | H-W balance in GAT training |

---

## PressureGAT Model

### Architecture

| Component | Detail |
|---|---|
| Input node features | 8 (elevation, demand, is_reservoir, hop_distance, ...) |
| Input edge features | 6 (log_length, diameter, roughness, ...) |
| Layer 1 | GATConv 64 heads×4 → 256, LayerNorm, ReLU, Dropout(0.1) |
| Layer 2 | GATConv 64 heads×4 → 256, LayerNorm, ReLU, Dropout(0.1) |
| Layer 3 | GATConv 32 heads×2 → 64, ReLU |
| Output | Linear(64 → 1) → scalar pressure head (m) |
| Parameters | ~180 000 |
| Physics loss | Hazen-Williams flow balance (λ = 0.01) |
| Uncertainty | MC-Dropout, 20 stochastic forward passes |

### Test Results

| Network | MAE (m) | RMSE (m) | MRE | Training scenarios |
|---|---|---|---|---|
| Net3 | **3.50** | 5.56 | **9.7%** | 400 train / 50 val / 50 test |
| L-Town | **6.40** | — | **15.5%** | 1 600 train / 200 val / 200 test |

L-Town was retrained from 96% MRE (160 scenarios) to 15.5% MRE by generating 2 000 scenarios. Training used a weighted sampler that oversamples high-impact (high pressure-drop) failure scenarios.

---

## India WDN Module

Synthetic networks for **Mumbai, Delhi, Bangalore, Hyderabad** — generated from OSMnx POI data with guaranteed minimums per node type and a Prim's MST backbone plus priority-weighted redundant connections.

### Node Types

| Type | Priority Weight | Base Demand (m³/s) | Pipe Class | Min. per Network |
|---|---|---|---|---|
| Hospital | 5 | 0.025 | Secondary (D=25cm, C=120) | 2 |
| Fire Station | 4 | 0.040 | Secondary | 2 |
| School | 3 | 0.012 | Secondary | 3 |
| College | 3 | 0.015 | Secondary | 1 |
| Market | 2 | 0.008 | Local (D=15cm, C=110) | 2 |
| Park | 1 | 0.002 | Local | 2 |
| Residential | 1 | 0.004 | Local / Branch | — |
| Reservoir / Tower | 0 | 0.0 | Main (D=40cm, C=130) | 2 |

### Failure Scenarios

| Scenario | Description |
|---|---|
| `burst` | Pipe fully closed; emitter at both endpoints (orifice-flow model) |
| `leak` | Pipe open but reduced flow; emitter scaled by leak coefficient |
| `surge` | Demand multiplier on specified nodes |
| `multi_failure` | Compound: any combination of the above |

### Demand Satisfaction Model

```
ds(node) = 0                          if pressure < 7 m  (MIN_PRESSURE)
         = (P − 7) / (14 − 7)        if 7 m ≤ P < 14 m
         = 1.0                        if P ≥ 14 m        (REQUIRED_PRESSURE)
```

---

## Benchmark Experiments

All experiments run with `SEED_TRAIN=42`, `SEED_TEST=99`. Deterministic — re-running produces identical CSV output.

| # | Experiment | Network | Scenarios | Key Metrics | Output |
|---|---|---|---|---|---|
| 1 | Core comparison: None vs. Dinic vs. HARP | Net3 | 20 | CDS, PVC, WLV, CT | `experiment_1.csv` |
| 2 | GAT pressure accuracy | Net3 | 500 | MAE, RMSE, MRE | `experiment_2_summary.csv` |
| 3 | k-hop subgraph sweep | Net3 | 20 × 5k values | CDS vs. CT tradeoff | `experiment_3.csv` |
| 4 | Priority breakdown by node type | Net3 | 20 | CDS per type | `experiment_4.csv` |
| 5 | Scalability on L-Town | L-Town | 20 | CT, memory (MB) | `experiment_5.csv` |
| 6 | Ablation study | Net3 | 20 | Component contribution | `experiment_6.csv` |

### Metrics

| Metric | Definition |
|---|---|
| CDS | Composite Demand Satisfaction — priority-weighted average demand fulfillment |
| PVC | Pressure Violation Count — nodes below minimum pressure threshold |
| WLV | Water Loss Volume (m³) |
| CT | Computation Time (ms) |

---

## Setup

### Prerequisites

- Python 3.10 — `brew install python@3.10`
- Node.js 22+ — `brew install node@22`

### Backend

```bash
# Create Python 3.10 venv (required for PyTorch + PyTorch Geometric)
python3.10 -m venv venv310
source venv310/bin/activate

pip install torch==2.1.2 torch_geometric==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Run Order

### Step 1 — Generate training data

```bash
# Net3: ~500 scenarios, ~3 minutes on CPU
venv310/bin/python -m backend.gat.dataset_generator --network Net3 --n 500

# L-Town: 2 000 scenarios recommended (use Colab T4 GPU for speed)
venv310/bin/python -m backend.gat.dataset_generator --network L-Town --n 2000
```

### Step 2 — Train GAT models

```bash
venv310/bin/python -m backend.gat.train --network Net3
venv310/bin/python -m backend.gat.train --network L-Town
```

Early stopping triggers around epoch 80. Best checkpoint saved to `backend/data/trained_models/`.

### Step 3 — Run benchmark experiments

```bash
venv310/bin/python -m backend.experiments.runner --exp 2 --network Net3
venv310/bin/python -m backend.experiments.runner --exp 1 --network Net3
venv310/bin/python -m backend.experiments.runner --exp 3 --network Net3
venv310/bin/python -m backend.experiments.runner --exp 4 --network Net3
venv310/bin/python -m backend.experiments.runner --exp 6 --network Net3
venv310/bin/python -m backend.experiments.runner --exp 5 --network L-Town
```

### Step 4 — Generate figures

```bash
venv310/bin/python -m backend.experiments.visualizer
# Figures saved to backend/results/figures/ at 300 DPI
```

### Step 5 — Start the backend

```bash
venv310/bin/python -m backend.main
# Running at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### Step 6 — Start the frontend

```bash
cd frontend
npm run dev
# Running at http://localhost:5173
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/india/wdn/{city}` | Fetch or generate WDN for a city |
| POST | `/india/simulate` | Run HARP or Dinic simulation with a failure scenario |
| GET | `/india/events/{city}` | SSE — live pressure telemetry at 2.5 s intervals |
| POST | `/india/priority/override` | Override node type/weight (persisted to SQLite) |
| DELETE | `/india/priority/overrides/{city}` | Clear all priority overrides for a city |
| GET | `/india/report/generate` | Start background report generation |
| GET | `/india/report/status` | Poll background report job status |
| POST | `/anomaly/detect` | Z-score anomaly detection with most-likely-failed-pipe |
| POST | `/simulate/scenario` | WNTR simulation with GAT pressure prediction + MC-Dropout |
| GET | `/networks` | List available benchmark networks |
| GET | `/networks/{net}/graph` | Graph topology for a benchmark network |
| GET | `/results/experiments` | List completed experiments |
| GET | `/results/figures` | List generated figure files |

Full interactive documentation: `http://localhost:8000/docs`

### Authentication

Set `HARP_API_KEY` in the environment to enable API key protection on simulation and anomaly endpoints. Pass the key as `X-API-Key` header. Unset (default) means no authentication required.

---

## Project Structure

```
backend/
  main.py                    # FastAPI — 30+ endpoints, SSE, SQLite persistence
  config.py                  # Global constants (SEED_TRAIN=42, SEED_TEST=99)
  wntr_engine.py             # WNTR simulation, failure injection, GAT feature extraction
  algorithms/
    harp.py                  # HARP: hydraulics-aware Dinic's with priority scoring
    dinic.py                 # Standard Dinic's max-flow (baseline)
    hw_hydraulics.py         # Hazen-Williams head-loss and max-flow equations
    subgraph.py              # k-hop ego-graph extraction with escalation
  gat/
    model.py                 # PressureGAT: 3-layer GATConv, ~180K parameters
    dataset.py               # WDNScenarioDataset (PyTorch Geometric)
    train.py                 # Training loop, weighted sampler, physics loss logging
    inference.py             # Cached inference, MC-Dropout uncertainty
  geo/
    india_wdn.py             # India WDN generator, HARP/Dinic runner, SSE events
    city_mapper.py           # OSMnx POI fetcher, node placement, 4 cities
  anomaly/
    detector.py              # Z-score detection, pipe-endpoint z-score ranking
  experiments/
    runner.py                # 6 benchmark experiments
    metrics.py               # CDS, PVC, WLV, RT, CT metric definitions
    visualizer.py            # 16 paper-quality figures at 300 DPI
  data/
    networks/                # Net3.inp, L-Town.inp benchmark network files
    trained_models/          # Net3_gat.pt, L-Town_gat.pt checkpoints
    generated/               # Scenario .pkl files, India WDN JSON cache
  results/
    experiment_1..6.csv      # Benchmark experiment outputs
    figures/                 # fig_1..8 (Net3) + india_fig1..8 (India WDN)
    harp_wdn.db              # SQLite: experiments + priority_overrides tables
  tests/
    test_hw_hydraulics.py    # 7 tests
    test_dinic.py            # 3 tests
    test_harp.py             # 5 tests

frontend/src/
  api.ts                     # Axios client, schema adapters, SimResult types
  App.tsx                    # Router + navigation
  cityData.ts                # Seeded frontend WDN generator (Mulberry32 PRNG)
  pages/
    IndiaMapView.tsx         # Leaflet map, live simulation, SSE stream, priority overrides
    NetworkView.tsx          # Benchmark WDN topology viewer (Net3 / L-Town)
    ResultsView.tsx          # Experiment tables, figure gallery, GAT confidence chart
```

---

## Tests

```bash
venv310/bin/python -m pytest backend/tests/ -v
```

```
test_hw_hydraulics.py   7 passed
test_dinic.py           3 passed
test_harp.py            5 passed
----------------------------------------
15 passed in total
```

---

## Expected Outputs

| File | Description |
|---|---|
| `backend/results/experiment_1.csv` | Core comparison: 20 scenarios × 3 methods (None / Dinic / HARP) |
| `backend/results/experiment_2_summary.csv` | GAT accuracy: MAE, RMSE, MRE per network |
| `backend/results/experiment_3.csv` | k-hop sweep: CDS_critical vs. computation time |
| `backend/results/experiment_4.csv` | Priority breakdown: CDS per node type |
| `backend/results/experiment_5.csv` | L-Town scalability: CT and memory |
| `backend/results/experiment_6.csv` | Ablation: HARP with/without hydraulic check, priority, subgraph |
| `backend/results/figures/fig_*.png` | 8 Net3 figures at 300 DPI |
| `backend/results/figures/india_fig*.png` | 8 India WDN figures at 300 DPI |
| `backend/results/harp_wdn.db` | SQLite experiment and override database |

---

## Design Principles

1. **Physics first.** BFS H-W pressure propagation is the hydraulic ground truth for India WDN. WNTR PDA is ground truth for benchmark experiments. Neither is approximated away.
2. **Reproducibility.** Fixed seeds throughout: `SEED_TRAIN=42`, `SEED_TEST=99`. Every experiment CSV is deterministic. The frontend demo mode uses a Mulberry32 PRNG seeded on city + scenario so metrics are stable across repeated clicks.
3. **No stubs.** Every backend endpoint returns computed data. No hardcoded results in any production path.
4. **Separation of offline and online.** GAT training and benchmark experiments run once and save checkpoints/CSVs. The online India WDN simulation is fast, stateless, and responds in under 200 ms.
5. **Honest uncertainty.** MC-Dropout gives a per-node std. The `pressure_method` field in every simulation response declares whether pressures come from GAT prediction, H-W BFS heuristic, or are unavailable — so no reviewer mistakes a heuristic for a model output.
