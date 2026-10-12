# HARP-WDN: Complete System Design Document

> **Superseded numbers.** This document describes earlier versions of the system. Several figures in it were
> measured on code that has since been fixed and must not be quoted: **90.7% demand satisfaction** and
> **~117 ms vs ~28 ms** (taken from the India demo when HARP read an invented distance-gradient pressure field;
> it now reads a real WNTR solve), **Net3 CDS 39.2%** (a hydraulic-gate bug: the gate demanded a head drop out of every
> source and delivered nothing), **L-Town MAE 18.45 m** (an old model; also, valves and pumps were missing from the flow
> graph), **~180K GAT parameters** (it is ~91K), **Kruskal's MST** (the generator uses **Prim's**), and any
> "500 scenarios" / "6 experiments" counts (there are 7 experiments). WNTR demand noise in the training data was also never
> applied. **The README and `backend/results/` hold the current, reproducible results.**


**Project:** Hydraulics-Aware Resilient Prioritization for Water Distribution Networks  
**Version:** 2.0 (architecture; see the notice above for superseded numbers)  
**Status:** Implemented. Results sections are historical.

---

## Table of Contents

1. Architecture Overview
2. India WDN Generator
3. HARP Algorithm (India WDN)
4. Standard Dinic's Baseline
5. IoT Streaming & Anomaly Detection
6. WNTR Benchmark Experiments
7. GAT Pressure Surrogate
8. FastAPI Backend
9. React Frontend (3-Tab Dashboard)
10. File & Directory Structure
11. Tech Stack Reference
12. Data Flow Diagram
13. Non-Negotiable Code Contracts

---

## 1. Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        HARP-WDN SYSTEM (v2.0)                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐ ║
║  │               LAYER 1: India WDN Simulation (Online)                │ ║
║  │                                                                      │ ║
║  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │ ║
║  │  │ City Selector │───►│ WDN Generator│───►│ Scenario Injection   │  │ ║
║  │  │ (10 cities)  │    │ Prim's MST   │    │ Burst/Leak/Surge/    │  │ ║
║  │  │              │    │ + Redundancy │    │ Multi-Failure        │  │ ║
║  │  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │ ║
║  │                                                       │              │ ║
║  │                              ┌────────────────────────▼──────────┐  │ ║
║  │                              │   BFS Pressure Propagation         │  │ ║
║  │                              │   Source = 100m head               │  │ ║
║  │                              │   Each hop: −5m                    │  │ ║
║  │                              │   gat_pressures[node] computed     │  │ ║
║  │                              └────────────────┬──────────────────┘  │ ║
║  │                                               │                      │ ║
║  │               ┌───────────────────────────────▼─────────────────┐   │ ║
║  │               │          Algorithm Selection                     │   │ ║
║  │               │                                                  │   │ ║
║  │  ┌────────────┴──────┐              ┌────────────────────────┐  │   │ ║
║  │  │  Standard Dinic's │              │  HARP Algorithm        │  │   │ ║
║  │  │  (Baseline)       │              │  (Proposed)            │  │   │ ║
║  │  │                   │              │                        │  │   │ ║
║  │  │  Capacity proxy:  │              │  For each path P:      │  │   │ ║
║  │  │  diameter × 10    │              │   ΔH = H_src - H_sink  │  │   │ ║
║  │  │  No HW check      │              │   h_f = HW(Q,L,D,C)   │  │   │ ║
║  │  │  Result: 100%     │              │   if ΔH-h_f < 1.0m    │  │   │ ║
║  │  │  (optimistic)     │              │     REJECT path        │  │   │ ║
║  │  │  ~28ms runtime    │              │   else: score by       │  │   │ ║
║  │  │                   │              │   priority×unmet_dmnd  │  │   │ ║
║  │  └───────────────────┘              │  Result: 90.7%         │  │   │ ║
║  │                                     │  (physically valid)    │  │   │ ║
║  │                                     │  ~117ms runtime        │  │   │ ║
║  │                                     └────────────────────────┘  │   │ ║
║  │               └─────────────────────────────────────────────────┘   │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐ ║
║  │               LAYER 2: IoT Real-Time Streaming (SSE)                │ ║
║  │                                                                      │ ║
║  │  GET /india/events/{city}  ─────────────►  EventSource (frontend)   │ ║
║  │  Seeded PRNG pressure sim                   livePressures state      │ ║
║  │  2500ms tick interval                       liveWarnings display     │ ║
║  │  Per-node head_m + warnings                 LIVE/Local indicator     │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐ ║
║  │               LAYER 3: WNTR Benchmark Experiments (Offline)         │ ║
║  │                                                                      │ ║
║  │  Net3 (97 nodes) + L-Town (782 nodes)                               │ ║
║  │  500 WNTR PDA scenarios → GAT training → 6 experiments → 16 figs   │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
║  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐ ║
║  │  FastAPI     │◄──►│  Backend     │◄──►│  React Frontend            │ ║
║  │  Port 8000   │    │  Python 3.10 │    │  Vite + TypeScript         │ ║
║  │  CORS:5173   │    │  venv310     │    │  Port 5173 (3-tab layout)  │ ║
║  └──────────────┘    └──────────────┘    └────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Design Principles

1. **Physics first.** BFS pressure gradient is the hydraulic ground truth for India WDN. WNTR PDA is ground truth for WNTR experiments. Neither is bypassed.
2. **Separation of offline and online.** GAT training and WNTR experiments run once. Online India WDN simulation is fast and deterministic.
3. **Every result is reproducible.** Fixed seeds: SEED_TRAIN=42, SEED_TEST=99. CSV before PNG. Re-running produces identical outputs.
4. **Real-time by design.** The IoT SSE endpoint delivers live sensor telemetry; the frontend connects via EventSource and shows live vs. local status.
5. **No stubs.** Every backend endpoint returns real computed data. No hardcoded results in production code.

---

## 2. India WDN Generator

**File:** `backend/geo/india_wdn.py`

### 2.1 City Catalogue

10 Indian cities: bangalore, delhi, mumbai, hyderabad, chennai, pune, kolkata, jaipur, ahmedabad, lucknow.

Each city has: lat/lon center, radius (m), state, district.

### 2.2 Node Types and Parameters

| Type | Priority Weight | Base Demand (m³/s) | Color (UI) |
|------|----------------|-------------------|------------|
| reservoir | 0 | 0.0 | `#0891b2` (cyan) |
| water_tower | 0 | 0.0 | `#7c3aed` (violet) |
| hospital | 5 | 0.025 | `#dc2626` (red) |
| fire_station | 4 | 0.040 | `#ea580c` (orange) |
| school | 3 | 0.012 | `#059669` (green) |
| college | 3 | 0.015 | `#0d9488` (teal) |
| market | 2 | 0.008 | `#d97706` (amber) |
| park | 1 | 0.002 | `#16a34a` (dark green) |
| residential | 1 | 0.004 | `#64748b` (slate) |
| junction | 1 | 0.003 | `#94a3b8` (light slate) |

### 2.3 Network Generation Algorithm

```
1. Generate POI nodes for city (guaranteed minimums per type):
   - 2 hospitals, 2 fire stations, 3 schools, 1 college, 2 markets, 2 parks
   - Remainder: residential and junction nodes
   - Positions: perturbed grid within city bounds (seeded mulberry32 PRNG)

2. Compute distance matrix: Haversine between all node pairs

3. Build MST backbone via Prim's algorithm (from the first source):
   - Sort edges by length, add if connects new node (union-find)
   - Guarantees full network connectivity

4. Add redundant edges (resilience):
   - For each node pair within max_dist_m:
     priority_bonus = {hospital: 0.9, fire: 0.85, school: 0.75, ...}
     p = priority_bonus × (1 - dist/max_dist)
     if random() < p: add edge
   - High-priority nodes get more redundant connections

5. Assign pipe class by endpoint types:
   - main (40cm, C=130): edges touching reservoir/water_tower
   - secondary (25cm, C=120): edges to hospital/fire_station/school/college
   - local (15cm, C=110): short residential
   - branch (10cm, C=100): long residential or park

6. Cache to backend/data/generated/india_wdn_{city}_n{n}.json
```

### 2.4 BFS Hydraulic Simulation

```python
HEAD_AT_SOURCE = 100.0   # meters
HEAD_DROP_PER_HOP = 5.0  # meters per hop from source

gat_pressures[node] = max(10.0, HEAD_AT_SOURCE - BFS_level[node] × 5.0)
```

For failure scenarios, BFS runs on the reduced graph (failed pipes removed). Nodes unreachable from any source get pressure = 0 and demand_satisfaction = 0.

Demand satisfaction per node:
```
ds[node] = 0                            if pressure < 7m
         = (pressure - 7) / (14 - 7)   if 7m ≤ pressure < 14m
         = 1.0                          if pressure ≥ 14m
```

---

## 3. HARP Algorithm (India WDN)

**File:** `backend/geo/india_wdn.py` — `run_harp_simulation()`  
**Algorithm file:** `backend/algorithms/harp.py`

### 3.1 Hazen-Williams Hydraulics

```
Head loss:   h_f = 10.67 × L × Q^1.852 / (C^1.852 × D^4.87)
Inverse:     Q_max = (ΔH × C^1.852 × D^4.87 / (10.67 × L))^(1/1.852)

Q_max is the maximum flow a pipe can carry given available head ΔH.
Q = 0 if ΔH ≤ 0 (water cannot flow uphill without pumping).
```

**Non-negotiable contract (`hw_hydraulics.py`):**
- `hw_head_loss(Q, L, D, C)` raises `ValueError` if any argument ≤ 0
- `hw_max_flow(L, D, C, dH)` returns `1e-6` if `dH ≤ 0` (no negative flow)

### 3.2 HARP Modifications to Dinic's Blocking Flow

Standard Dinic's: accept any edge with residual_capacity > 0.

HARP augments this with two checks per candidate edge `(u → v)`:

**Check 1 — Hydraulic Feasibility:**
```python
available_head = gat_pressures[path_source] - gat_pressures[v]
path_head_loss = Σ hw_head_loss(Q=0.005, L, D, C) for all edges in path so far
if available_head - path_head_loss - safety_margin_m < 0:
    REJECT path, log {reason: 'head_loss_exceeded', deficit_m: ...}
```

**Check 2 — Priority Benefit:**
```python
priority_benefit = Σ weight[t] × (1 - demand_met[t])  for all sinks t reachable from v
if priority_benefit == 0:
    REJECT path (no unmet critical demand on this branch)
```

### 3.3 k-Hop Subgraph Localization

```python
G_sub = ego_graph(G, failed_pipe.node_u, k=2) ∪ ego_graph(G, failed_pipe.node_v, k=2)
# Remove failed pipe from G_sub
# HARP runs on G_sub (not full network)
# Escalation: if CDS_critical(result) < 0.6 → re-run on full graph
```

Default k=2. UI slider allows k=1..5.

### 3.4 Priority Weighting

```python
priority_config = {
    "hospital_nodes": [...],      weight: 5.0
    "fire_station_nodes": [...],  weight: 4.0
    "school_nodes": [...],        weight: 3.0
    "default": 1.0
}

# Path scored by weighted unmet demand of all sinks it can reach
score(path) = Σ priority_weight[sink] × (1 - demand_met[sink])
```

---

## 4. Standard Dinic's Baseline

**File:** `backend/algorithms/dinic.py`

Standard Dinic's max-flow with virtual supersource and supersink.

**Capacity assignment (deliberately hydraulically incorrect — this is the baseline flaw):**
```python
capacity[edge] = pipe_diameter_m × 10   # crude proxy, no HW physics
```

This is intentional. It demonstrates the core research problem: graph-theoretic max-flow with arbitrary capacity assignments produces physically invalid results. Dinic's reports 100% demand satisfaction on scenarios where HARP correctly identifies that head loss makes those routes infeasible.

**Output:** `FlowResult` with `demand_satisfaction`, `total_flow`, `computation_time_ms`.

---

## 5. IoT Streaming & Anomaly Detection

### 5.1 SSE Endpoint

**Backend:** `GET /india/events/{city}?n_nodes=20&tick_ms=2500`  
**Frontend:** `new EventSource('http://localhost:8000/india/events/bangalore')`

Each SSE event (JSON):
```json
{
  "t": 12.5,
  "tick": 5,
  "city": "bangalore",
  "pressures": {"H1": 54.23, "F1": 48.91, "...": "..."},
  "warnings": ["R1"],
  "n_warnings": 1,
  "timestamp": 1782995000.0
}
```

The backend generator uses seeded PRNG + sine-wave drift + Gaussian noise to simulate a realistic SCADA feed. The frontend `IotSensorPanel` displays live pressure readings with colored status indicators.

### 5.2 Anomaly Detection (Backend)

**File:** `backend/anomaly/detector.py`

```python
z_j = (P_j_current - baseline[j].mean[hour]) / baseline[j].std[hour]
if |z_j| > 3.0: flag as anomalous
```

Baseline built from 100 normal-operation simulation runs per network.

Candidate failed pipe: pipe whose both endpoint nodes are in the anomalous set.

### 5.3 Frontend IoT Panel

- Shows up to 6 representative nodes: reservoirs, water towers, hospitals, fire stations
- **LIVE badge** when SSE is connected to backend; **Local badge** when using synthetic client-side computation (fallback when backend is offline)
- Warning banner when any monitored node drops below 14m threshold
- Per-node: pressure head (m), flow rate (L/s), status indicator (green/amber/red)

---

## 6. WNTR Benchmark Experiments

**Files:** `backend/experiments/runner.py`, `backend/experiments/visualizer.py`

Six experiments on Net3 and L-Town benchmark networks.

| Experiment | What it tests | Output CSV |
|-----------|--------------|------------|
| Exp 1 | HARP vs Dinic's vs None on 20 Net3 pipe failures | `experiment_1.csv` |
| Exp 2 | GAT accuracy vs WNTR ground truth (50 test scenarios) | `experiment_2.csv` |
| Exp 3 | Subgraph k-hop tradeoff: CDS vs computation time | `experiment_3.csv` |
| Exp 4 | Priority breakdown for 5 worst-failure scenarios | `experiment_4.csv` |
| Exp 5 | Scalability: L-Town 782 nodes, Dinic's vs HARP k=2 | `experiment_5.csv` |
| Exp 6 | Ablation: HARP_full vs no_hydraulic vs no_priority vs no_subgraph | `experiment_6.csv` |

**Seeds:** SEED_TRAIN=42 for scenario selection. SEED_TEST=99 for L-Town.

---

## 7. GAT Pressure Surrogate

**Files:** `backend/gat/model.py`, `backend/gat/train.py`, `backend/gat/inference.py`

### 7.1 Architecture

```
Input:  x (N, 7) node features, edge_index (2, 2E), edge_attr (2E, 6)

Layer 1: GATConv(7 → 64, heads=4, edge_dim=6, concat=True) → (N, 256)
         LayerNorm → ReLU → Dropout(0.1)

Layer 2: GATConv(256 → 64, heads=4, edge_dim=6, concat=True) → (N, 256)
         LayerNorm → ReLU → Dropout(0.1)

Layer 3: GATConv(256 → 32, heads=2, edge_dim=6, concat=True) → (N, 64)
         ReLU

Output: Linear(64 → 1) → squeeze → (N,) predicted pressure heads (m)
```

~91K parameters (measured: 90,817 with 8 node inputs). Training: Adam lr=1e-3, max 100 epochs, early stopping patience=15.

### 7.2 Physics Regularization Loss

```
Total loss = MSE(pred, true) + λ × mean(|flow_balance_violation|)
```

Flow balance computed from HW-estimated flows using predicted pressure heads.

### 7.3 Inference Contract

- Every call measures wall-clock inference time
- Logs a WARNING if inference > 100ms
- Returns `(pressure_heads_m, inference_time_ms)` tuple

---

## 8. FastAPI Backend

**File:** `backend/main.py`  
**Runtime:** `./venv310/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000`  
**CORS:** `allow_origins=["http://localhost:5173"]` only

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/india/cities` | List available Indian cities |
| GET | `/india/wdn/{city}` | Generate/load city WDN (cached) |
| POST | `/india/simulate` | Run simulation: method=none/dinic/harp/compare |
| **GET** | **`/india/events/{city}`** | **SSE stream of live IoT sensor readings** |
| GET | `/india/report/generate` | Trigger generation of 8 India WDN report figures |
| POST | `/india/priority/override` | Override node type/weight for session |
| POST | `/india/wdn/{city}/regenerate` | Force regenerate WDN (clears cache) |
| GET | `/networks` | List WNTR benchmark networks |
| GET | `/networks/{n}/graph` | Network graph in Leaflet format |
| POST | `/simulate/scenario` | WNTR failure + recovery scenario |
| POST | `/simulate/leak` | Pipe leak simulation |
| POST | `/simulate/surge` | Demand surge simulation |
| POST | `/simulate/multi_failure` | Multi-pipe failure + recovery |
| POST | `/anomaly/detect` | Z-score anomaly detection |
| GET | `/results/experiment/{n}` | Return experiment CSV as JSON |
| GET | `/results/figures` | List generated PNG figures |
| GET | `/static/figures/{name}` | Serve figure PNG |

### Key Response: `/india/simulate` (method=harp)

```json
{
  "nodes": [...],
  "edges": [...],
  "demand_satisfaction": {"H1": 0.95, "F1": 1.0, "...": "..."},
  "CDS": {
    "CDS_overall": 0.907,
    "CDS_critical": 0.993,
    "CDS_hospital": 0.950,
    "CDS_school": 0.980,
    "CDS_fire": 1.0
  },
  "pressures_m": {"H1": 75.0, "...": "..."},
  "computation_time_ms": 117.3,
  "paths_accepted": 19,
  "paths_rejected": 24,
  "escalation_flag": false,
  "failed_pipes": ["P5"]
}
```

---

## 9. React Frontend (3-Tab Dashboard)

**Stack:** React 18 + TypeScript + Vite (port 5173)  
**Key dependencies:** react-leaflet, react-router-dom

### Tab Structure

```
App.tsx (router + SharedSimState)
  ├── /         → IndiaMapView.tsx   (Simulation + IoT)
  ├── /results  → ResultsView.tsx    (4-sub-tab results dashboard)
  └── /network  → NetworkView.tsx    (Network topology)
```

**SharedSimState** — lifted to App.tsx, passed down:
```typescript
interface SharedSimState {
  city: string; wdn: WDNData
  simResult: SimResult | null
  harpResult: SimResult | null
  dinicResult: SimResult | null
  scenario: ScenarioType; failedPipes: string[]
  runAt: number
}
```

### Tab 1: Simulation (`IndiaMapView.tsx`)

**Layout:** 70% Leaflet map | 30% control panel

**Map layer:**
- `CircleMarker` per node, colored by type
- `Polyline` per edge, colored by status (active=cyan, failed=red, animated=amber)
- BFS flow animation: level-by-level reveal from sources
- Pressure heatmap overlay (toggle): red < 7m / amber 7–14m / green ≥ 14m

**Control panel features:**
- City selector + network size slider (10–50 nodes)
- Number of sources slider (1–4)
- Scenario selector: Normal / Pipe Burst / Leak / Demand Surge / Multi-Failure
- Scenario-specific config (surge multiplier, leak coefficient, surge node selection)
- Pipe click to mark as failed (burst/multi-failure scenarios)
- Algorithm selector: None / Dinic's / HARP / Compare (runs both)
- Run simulation button → calls `/india/simulate` → updates SharedSimState
- BFS animation controls: Play / Pause / Step / Reset
- **IoT Sensor Feed toggle:**
  - When on + backend online: connects to SSE `/india/events/{city}` via `EventSource`
  - Shows **LIVE badge** when SSE is connected, **Local badge** on fallback
  - Per-node: pressure (m), flow (L/s), status (ok/warn/critical)
  - Warning banner when nodes below 14m threshold

### Tab 2: Results (`ResultsView.tsx`)

4 sub-tabs: Overview | Nodes | Charts | Compare

- **Overview:** Metric cards — served demand %, critical node service %, recovery flow %, runtime
- **Nodes:** Sortable table ranked by priority weight then pressure ascending
- **Charts:** Node pressure bars + HARP vs Dinic's comparison visualization
- **Compare:** Side-by-side metric cards with delta badges (HARP vs Dinic's)

### Tab 3: Network (`NetworkView.tsx`)

Network topology visualization with node type legend, failed pipe highlighting, pressure heatmap overlay.

---

## 10. File & Directory Structure

```
proj1/
├── backend/
│   ├── main.py                      FastAPI app, all endpoints + SSE
│   ├── config.py                    SEED_TRAIN=42, SEED_TEST=99
│   ├── wntr_engine.py               WNTR simulation wrapper
│   ├── algorithms/
│   │   ├── hw_hydraulics.py         HW equations (ValueError on ≤0 inputs)
│   │   ├── subgraph.py              k-hop ego_graph + escalation
│   │   ├── dinic.py                 Standard Dinic's (diameter×10 capacity)
│   │   └── harp.py                  HARP: HW-feasibility + priority Dinic's
│   ├── gat/
│   │   ├── model.py                 3-layer GATConv + physics loss
│   │   ├── dataset.py               WDNScenarioDataset (PyG)
│   │   ├── dataset_generator.py     500-scenario WNTR generation
│   │   ├── train.py                 Adam + ReduceLROnPlateau + early stop
│   │   └── inference.py             predict_pressures() with timing
│   ├── anomaly/
│   │   └── detector.py              Z-score anomaly detection
│   ├── experiments/
│   │   ├── runner.py                Experiments 1–6 (WNTR benchmarks)
│   │   ├── metrics.py               CDS, PVC, WLV, RT metric functions
│   │   ├── visualizer.py            8 WNTR benchmark figures (300 DPI)
│   │   └── india_visualizer.py      8 India WDN report figures (300 DPI)
│   ├── geo/
│   │   └── india_wdn.py             City WDN generator + all simulations
│   ├── data/
│   │   ├── networks/                Net3.inp, Hanoi.inp, L-Town.inp
│   │   ├── priority_config.json     Node types per WNTR network
│   │   ├── generated/               Cached city WDNs + WNTR scenarios
│   │   └── trained_models/          Net3_gat.pt, L-Town_gat.pt
│   ├── results/
│   │   ├── experiment_{1-6}.csv     Experiment data
│   │   ├── experiment_2_summary.csv GAT summary stats
│   │   ├── harp_wdn.db              SQLite results store
│   │   └── figures/
│   │       ├── india_fig{1-8}_*.png  India WDN report figures (8 figs)
│   │       └── fig_{1-8}_*.png       WNTR benchmark figures (8 figs)
│   └── tests/
│       ├── test_hw.py               6 HW unit tests
│       ├── test_dinic.py            3 Dinic's unit tests
│       └── test_harp.py             5 HARP unit tests (mocked WNTR)
│
├── frontend/
│   └── src/
│       ├── App.tsx                  Router + SharedSimState
│       ├── types.ts                 SharedSimState, SimResult, SimStats
│       ├── api.ts                   Typed fetch + SSE EventSource wrapper
│       ├── cityData.ts              WDNData types + local BFS fallback
│       ├── icons.tsx                Inline SVG icon components
│       └── pages/
│           ├── IndiaMapView.tsx     Tab 1: Simulation + IoT SSE panel
│           ├── ResultsView.tsx      Tab 2: 4-sub-tab results dashboard
│           └── NetworkView.tsx      Tab 3: Topology view
│
├── idea.md                          Research concept + results (v2.0)
├── design.md                        This file (v2.0)
└── README.md                        Setup + run instructions
```

---

## 11. Tech Stack Reference

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Backend | Python | 3.10 | Core runtime (venv310) |
| Backend | FastAPI | 0.104 | REST + SSE API |
| Backend | uvicorn | 0.24 | ASGI server |
| Backend | wntr | 1.2.x | Hydraulic simulation |
| Backend | networkx | 3.x | Graph algorithms |
| Backend | torch | 2.1 (CPU) | GAT training |
| Backend | torch_geometric | 2.5 | GATConv layers |
| Backend | numpy | 1.26 | Numerics (seed: SEED_TRAIN=42) |
| Backend | pandas | 2.x | Experiment CSV handling |
| Backend | matplotlib | 3.8 | 300 DPI report figures |
| Backend | seaborn | 0.13 | Colorblind-safe statistical plots |
| Frontend | React | 18 | UI framework |
| Frontend | TypeScript | 5 | Type safety |
| Frontend | Vite | 5 | Build tool (dev port 5173) |
| Frontend | react-leaflet | 4 | Geo map rendering |
| Frontend | react-router-dom | 6 | 3-tab routing |
| Frontend | CSS/inline SVG | — | Charts rendered without a chart library |
| Protocol | EventSource | Browser native | SSE from `/india/events/{city}` |

---

## 12. Data Flow Diagram

```
User selects city + scenario + algorithm
        │
        ▼
IndiaMapView.runSim() → POST /india/simulate
        │
        ├─ method=harp → run_harp_simulation()
        │     ├─ fetch_city_pois(city, n_nodes) [cached]
        │     ├─ run_simple_simulation() → BFS pressures
        │     ├─ k-hop subgraph extraction around failed pipe(s)
        │     ├─ HARP Dinic's: HW check + priority score per path
        │     ├─ Escalation if CDS_critical < 0.6 → full graph
        │     └─ Returns: CDS, PVC, demand_sat, paths_accepted/rejected
        │
        └─ method=dinic → run_dinic_simulation()
              ├─ DiGraph from WDN edges (capacity = diameter×10)
              ├─ Virtual supersource → all reservoir nodes
              ├─ Standard Dinic's max-flow
              └─ demand_satisfaction from flow allocation

api.ts: adaptSimResult(raw, baseWDN, scenario, method)
        ├─ Merge pressures into nodes
        ├─ Build edge flow map from accepted_paths
        ├─ served_demand_pct = mean(demand_satisfaction) × 100
        ├─ critical_node_service_pct = mean(ds for priority_weight ≥ 3.8) × 100
        └─ recovery_flow_pct = raw.CDS.CDS_overall × 100

App.tsx: onSimComplete(SharedSimState) → ResultsView reads automatically

────────────────────────────────────────────────────
PARALLEL: IoT SSE Stream (when showIoT=true)
────────────────────────────────────────────────────
IndiaMapView useEffect:
  new EventSource('/india/events/{city}')
  onopen  → setSseConnected(true) → LIVE badge
  onmessage → setLivePressures(data.pressures)
             setLiveWarnings(data.warnings)
  onerror → setSseConnected(false) → Local badge

IotSensorPanel:
  livePressure ?? syntheticLocalPressure(tick, node)
  liveWarnings → warning banner with count
  connected ? '● LIVE' : '◌ Local'

Backend _iot_sensor_stream():
  Every 2500ms: seeded PRNG + sine drift + Gaussian noise
  Yield: f"data: {json.dumps(event_data)}\n\n"
```

---

## 13. Non-Negotiable Code Contracts

| Contract | Location | What happens if violated |
|----------|----------|-------------------------|
| `SEED_TRAIN=42`, `SEED_TEST=99` | `config.py` | Results non-reproducible |
| `np.random.seed(SEED_TRAIN)` at file top | Every file using randomness | Non-deterministic outputs |
| CSV saved before figure generated | `visualizer.py`, `india_visualizer.py` | Figure may precede data |
| `hw_head_loss` raises `ValueError` on Q≤0, D≤0, L≤0, C≤0 | `hw_hydraulics.py` | Physics undefined |
| WNTR convergence failures: log + skip, never raise | `wntr_engine.py`, `runner.py` | Runner crashes |
| GAT inference time measured and logged every call | `gat/inference.py` | Warn if >100ms |
| HARP unit tests mock WNTR | `tests/test_harp.py` | Tests require .inp files |
| CORS: `localhost:5173` only | `main.py` | Security: blocks other origins |
| Escalation if CDS_critical < 0.6 | `harp.py`, `india_wdn.py` | Hospitals underserved silently |

---

*Document version: 2.0 | HARP-WDN Final Year Project | VIT Vellore*  
*Extends: IEEE Conference Paper (Nair et al., 2025)*  
*All components: free and open source*  
*Last updated: 2026-07-02 — actual implementation, not specification*
