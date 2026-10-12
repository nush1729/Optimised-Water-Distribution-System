# HARP-WDN

**Hydraulics-Aware Resilient Prioritization for Water Distribution Networks**

A full-stack research prototype for water-network failure simulation and recovery prioritization. It combines WNTR hydraulic simulation, a Graph Attention Network pressure surrogate, and a modified Dinic's max-flow algorithm gated by Hazen-Williams pressure feasibility. A FastAPI backend serves it; a React/TypeScript dashboard drives it.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square)
![CI](https://github.com/nush1729/Optimised-Water-Distribution-System/actions/workflows/ci.yml/badge.svg)

> **Status: research prototype, not a deployable control system.** HARP *recommends*; it never actuates valves. See [Limitations](#limitations) before drawing conclusions from any number below.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Features](#features)
4. [Architecture](#architecture)
5. [HARP Algorithm](#harp-algorithm)
6. [PressureGAT Model](#pressuregat-model)
7. [India WDN Module](#india-wdn-module)
8. [Sensors, Anomaly Detection and Auto-Respond](#sensors-anomaly-detection-and-auto-respond)
9. [Benchmark Experiments](#benchmark-experiments)
10. [Setup](#setup)
11. [Run Order](#run-order)
12. [Retraining on Colab](#retraining-on-colab)
13. [API Reference](#api-reference)
14. [Configuration](#configuration)
15. [Project Structure](#project-structure)
16. [Tests and CI](#tests-and-ci)
17. [Limitations](#limitations)

---

## Overview

When a pipe bursts, a classical max-flow routing (Dinic's) reports that nearly all demand can still be met, because it only sees pipe diameters. It cannot tell that a route is hydraulically impossible: too long, too narrow, too little pressure left at the end. HARP adds the physics. Every candidate route is checked against Hazen-Williams head loss and the pressure the destination would actually keep, and demand is served in priority order (hospitals and fire stations first).

The system has four parts:

- **WNTR simulation** of two benchmark networks (EPANET Net3, L-Town) and four synthetic Indian city networks.
- **HARP**, the routing algorithm, plus the standard Dinic's baseline.
- **PressureGAT**, a graph neural network that predicts nodal pressures much faster than a full hydraulic solve.
- **Sensing**: a z-score anomaly detector and an `auto_respond` endpoint that goes from sensor readings to a recovery recommendation.

Everything below was produced by the committed code and checkpoints (CSVs in `backend/results/`, GAT metrics in `gat_eval_*.json`). Experiments run on Net3 unless noted.

**PressureGAT (held-out test scenarios, residual target)**

| Network | Test scenarios | MAE | R² | "Nothing failed" baseline MAE | Gate agreement |
|---|---|---|---|---|---|
| Net3 | 265 | 2.01 m | 0.84 | 1.63 m | 98% |
| L-Town | 200 | 0.51 m | 0.97 | 0.11 m | 99.9% |

On Net3 the model beats "assume nothing failed" on the worst quartile of failures (3.13 m vs 3.30 m MAE) but not overall, and is closer to the truth in 27% of scenarios. On L-Town most failures barely move pressure, so the intact baseline wins. MC-dropout spread is **not** calibrated against real error (Spearman -0.07 on Net3, -0.41 on L-Town); the UI labels it as a spread, not a confidence.

**Single-pipe failures (experiment 1, 20 scenarios)**

| Method | Critical-demand satisfaction | Pressure violations / scenario | Water lost (m³) | Time |
|---|---|---|---|---|
| No recovery | 0.986 | 5.5 | 10.5 | - |
| Dinic's | 0.998 | n/a | 0.70 | 7 ms |
| HARP | 0.998 | 4.1 | 0.70 | 39 ms |

Net3 is robust to one broken pipe, so all methods serve hospitals, schools and fire stations fully. HARP's measurable gain here is fewer pressure violations than no recovery, not higher delivery.

**Stress scenarios (experiment 7, allocations solved with real hydraulics; 20 draws each)**

| Scenario | Method | Promised | Delivered | Critical delivered | Pressure violations |
|---|---|---|---|---|---|
| 3 failures | Dinic's | 0.99 | 0.85 | 0.99 | 6.35 |
| 3 failures | HARP | 0.82 | 0.83 | 0.99 | 0.65 |
| 6 cascading | Dinic's | 0.96 | 0.88 | 0.96 | 5.65 |
| 6 cascading | HARP | 0.87 | 0.88 | 0.96 | 2.85 |
| Combined | Dinic's | 0.99 | 0.91 | 1.00 | 4.35 |
| Combined | HARP | 0.90 | 0.90 | 1.00 | 1.10 |

Dinic's promises more than the network delivers; HARP's promises track what is delivered, and it keeps critical service equal to the baselines while cutting pressure violations by 50-90%. Supply-drop and demand-surge scenarios are served fully by every method.

**Other experiments.** Subgraph search always escalated to the full graph on Net3 (so the full graph is faster, 24 ms vs 38 ms), and the ablation variants (no priority, no hydraulic check, no subgraph) score identically on single failures. On L-Town (experiment 5) HARP with k=2 takes about 5.5 s per scenario against 0.36 s for Dinic's, so it is much slower on larger networks.


---

## Features

### HARP Algorithm
Modified Dinic's blocking flow on a graph that includes pipes, **valves and pumps**. Edge capacity is the Hazen-Williams flow at the larger of the predicted head difference and a design gradient, capped by a design velocity. Each complete source-to-sink path is accepted only if the sink keeps at least `MIN_PRESSURE + 1 m` after head loss at the flow the path actually carries; between minimum and full service pressure the sink receives a proportional share (pressure-dependent demand). Paths are explored in priority order. k-hop localization limits work to the failure neighbourhood and escalates to the full graph when a hospital or fire station falls below 60% served.

### PressureGAT Surrogate
3-layer Graph Attention Network predicting nodal pressure from node and edge features, trained on WNTR scenarios with a Hazen-Williams physics regulariser. Optionally predicts the *change from the intact network's pressure* (residual target) and takes per-node demand ratio and a network load factor as inputs. MC-Dropout gives per-node uncertainty.

### India WDN Simulation
Synthetic networks for four Indian cities from OSMnx points of interest, with a Prim's MST backbone plus priority-weighted redundant edges. Scenarios: burst, leak, demand surge, multi-failure. HARP reads a real WNTR steady-state solve of the failed network.

### Sensors, Anomaly Detection, Auto-Respond
Pressure readings arrive over HTTP (or MQTT), a z-score detector against a per-hour baseline flags anomalies and ranks the likely failed pipe, and `/anomaly/auto_respond` returns HARP's recovery plan for operator approval.

### Run History
Every simulation is stored in SQLite and listed in the dashboard's **History** tab (`GET /runs`).

### Benchmark Experiments
Seven reproducible experiments, including a stress-scenario experiment (multi-failure, cascading failure, supply loss, surge) that applies each recommended plan to the network and solves it with real hydraulics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1 — Data and Simulation                   │
│                                                                  │
│  WNTR engine           India WDN generator      Sensor ingest    │
│  Net3 / L-Town (PDA)   OSMnx + Prim's MST       HTTP / MQTT      │
│  failures, leaks,      burst/leak/surge/multi   latest readings  │
│  surges (real leaks)                            per network      │
└──────────────┬─────────────────┬───────────────────┬────────────┘
               │                 │                   │
┌──────────────▼─────────────────▼───────────────────▼────────────┐
│                  LAYER 2 — Algorithms                            │
│                                                                  │
│  PressureGAT            HARP                    Z-score detector │
│  residual target        gated Dinic's +         per-hour baseline│
│  MC-Dropout             priority + partial      pipe ranking     │
│                         service + k-hop                          │
└──────────────┬─────────────────┬───────────────────┬────────────┘
               │                 │                   │
┌──────────────▼─────────────────▼───────────────────▼────────────┐
│                  LAYER 3 — Serving                               │
│                                                                  │
│  FastAPI (8000)         SQLite                  React dashboard  │
│  REST + SSE             runs, overrides         Simulation       │
│  compute + admin keys                           Results, Network │
│                                                 History          │
└─────────────────────────────────────────────────────────────────┘
```

---

## HARP Algorithm

```
Head loss:    h_f = 10.67 · L · Q^1.852 / (C^1.852 · D^4.87)
Max flow:     Q_max = ( ΔH · C^1.852 · D^4.87 / (10.67 · L) )^(1/1.852)
```

**Graph.** Pipes, valves and pumps are edges. Valves and pumps are one-way and modelled as short pipes. (An earlier version used pipes only; on L-Town the reservoirs feed the network through pressure-reducing valves, so two of three sources were disconnected and HARP could reach only 86 of 701 demand nodes.)

**Capacity.** `Q_max` evaluated at `max(predicted ΔH, 0.01 · L)` (a 10 m/km design gradient), capped at 2 m/s.

**Feasibility gate (per complete path).** With `h_supply` the highest head along the path (pumps lift heads above the source) and `Q` the flow the path would carry:

```
residual_pressure = h_supply − Σ h_f(Q) − elevation(sink)
feasible          = min(residual_pressure, predicted_sink_pressure) ≥ 7.03 m + 1 m margin
service fraction  = clamp((p − 7.03) / (14.06 − 7.03), 0, 1)
```

Below full pressure a sink gets only its service fraction of demand. The previous gate demanded a 1 m head drop out of every source, which cannot hold in a looped, near-equilibrium network, and delivered nothing.

**Priority.** Neighbours are explored highest-priority first and a path is accepted only if it still serves unmet demand: `score = Σ weight(node) · (1 − served(node))`. Weights live in one place, `backend/config.py::NODE_TYPE_WEIGHTS` (hospital 5, fire station 4, school and college 3, market 2, residential and junction 1, park 0.5, sources 0).

**Dinic's loop.** BFS builds the level graph, DFS pushes blocking flow, residual edges are updated, and sinks that cannot be served are retired so farther sinks are still reached. At most |V| phases.

**Escalation.** If any hospital or fire station ends below 60% served in the k-hop subgraph, HARP re-runs on the full graph.

| | Dinic's (baseline) | HARP |
|---|---|---|
| Edge capacity | diameter × 10 | Hazen-Williams, design gradient, velocity cap |
| Hydraulic check | none | terminal pressure + head loss at path flow |
| Partial service | no | pressure-dependent |
| Priority | none | weight × unmet demand |
| Valves / pumps | n/a | included |

---

## PressureGAT Model

| Component | Detail |
|---|---|
| Node inputs | 8 (elevation or source head, demand, is-source, 4 priority flags, proximity to failure); **10** with demand ratio and load factor |
| Edge inputs | 6 (log length, diameter, roughness ×2, open/closed, log resistance) |
| Layers | GATConv 64×4 → LayerNorm → ReLU → dropout; GATConv 64×4 (same); GATConv 32×2 → ReLU; Linear → pressure |
| Parameters | ~91 000 (91 329 with the 10 node inputs; 90 817 with 8) |
| Loss | MSE + Hazen-Williams flow-balance regulariser |
| Target | absolute pressure, or the change from the intact network (`--residual`) |
| Uncertainty | MC-Dropout (20 passes) |

Training data: one random single-pipe failure per scenario with log-normal demand noise (σ = 0.2, clipped to 0.3–3.0), simulated for 24 h; the label is the hour-12 pressure. Generation is parallel and seed-deterministic.

> **Data-generator fix.** WNTR's `Junction.base_demand` is read-only. Earlier code assigned to it inside `try/except: pass`, so the demand noise was *never applied*, while the features claimed it was. The same bug silently disabled demand surges and the anomaly baseline's noise. Demand is now scaled on its demand timeseries (`scale_node_demand`) and errors are no longer swallowed. Datasets generated before this fix are invalid and must be regenerated.

Evaluation (`python -m backend.gat.evaluate`) reports accuracy against a naive mean predictor *and* against the intact-network baseline, error on the high-impact quartile, how often the model makes the same servable / not-servable call as the true pressures, MC-Dropout calibration, and HARP end-to-end with GAT vs true pressures.

---

## India WDN Module

Synthetic networks for **Mumbai, Delhi, Bangalore, Hyderabad** from OSMnx POI data, with guaranteed minimums per node type, a **Prim's MST** backbone from the first source and priority-weighted redundant connections.

| Type | Priority | Base demand (m³/s) | Pipe class | Min. per network |
|---|---|---|---|---|
| Hospital | 5 | 0.025 | Secondary (D = 25 cm, C = 120) | 2 |
| Fire station | 4 | 0.040 | Secondary | 2 |
| School | 3 | 0.012 | Secondary | 3 |
| College | 3 | 0.015 | Secondary | 1 |
| Market | 2 | 0.008 | Secondary | 2 |
| Park | 0.5 | 0.002 | Local (D = 15 cm, C = 110) | 2 |
| Residential | 1 | 0.004 | Local / branch | — |
| Reservoir / tower | 0 (source) | 0 | Main (D = 40 cm, C = 130) | 2 |

| Scenario | Model |
|---|---|
| `burst` | pipe closed; a leak (orifice flow) at both endpoints until isolated |
| `leak` | pipe open, capacity throttled, ongoing leak at its endpoints |
| `surge` | demand multiplier on chosen nodes |
| `multi_failure` | any combination |

Leaks use WNTR's leak model (`add_leak`). WNTRSimulator ignores `emitter_coefficient`, which earlier code relied on, so leaks and bursts previously drew no water.

---

## Sensors, Anomaly Detection and Auto-Respond

```
sensor / SCADA gateway ──► POST /sensors/readings ──┐
MQTT  harp/<network>/pressure ──────────────────────┤
                                                    ▼
                        latest readings per network
                                                    │
        POST /anomaly/auto_respond  (or readings in the request body)
                                                    ▼
   z-score vs per-hour baseline ─► rank pipes by endpoint |z| ─► suspected pipe
                                                    ▼
        HARP + GAT  ─►  recovery plan: isolate / route flow / still-unserved
                          (requires_operator_approval: true)
```

```bash
# 1. a gateway pushes readings (node id -> pressure in metres)
curl -X POST localhost:8000/sensors/readings -H 'content-type: application/json' \
  -d '{"network":"Net3","readings":{"10":41.2,"15":38.0,"20":29.5}}'

# 2. detect, locate, recommend
curl -X POST localhost:8000/anomaly/auto_respond -H 'content-type: application/json' \
  -d '{"network":"Net3"}'
```

The detector's baseline is built from 100 noisy WNTR runs per network on first use (cached in `backend/data/generated/`). **Sensor input is simulated in the dashboard**; only the HTTP and MQTT paths accept real data. MQTT is optional (`pip install paho-mqtt`, set `HARP_MQTT_URL`).

---

## Benchmark Experiments

All experiments run with `SEED_TRAIN=42`, `SEED_TEST=99` and are deterministic.

| # | Experiment | Network | Scenarios | Key metrics | Output |
|---|---|---|---|---|---|
| 1 | Core comparison: none vs Dinic's vs HARP | Net3 | 20 | CDS, PVC, WLV, CT | `experiment_1.csv` |
| 2 | GAT pressure accuracy | Net3 | held-out test split | MAE, RMSE, MRE | `experiment_2_summary.csv` |
| 3 | k-hop subgraph sweep | Net3 | 20 × 5 | CDS vs CT | `experiment_3.csv` |
| 4 | Priority breakdown by node type | Net3 | 20 | CDS per type | `experiment_4.csv` |
| 5 | Scalability | L-Town | 10 | CT, memory | `experiment_5.csv` |
| 6 | Ablation | Net3 | 20 | component contribution | `experiment_6.csv` |
| 7 | **Stress scenarios** | Net3 | 7 types × 20 | promised vs delivered CDS, PVC | `experiment_7.csv` |

Experiment 7 applies each method's allocation to the network and solves it with real hydraulics: **CDS_promised** is what the method claims, **CDS_delivered** is what customers would actually get. HARP uses true post-stress pressures there, to isolate the algorithm from GAT error.

| Metric | Definition |
|---|---|
| CDS | Critical Demand Satisfaction: priority-weighted share of demand served |
| PVC | Pressure Violation Count: demand nodes below minimum service pressure |
| WLV | Water Loss Volume (m³) |
| CT | Computation time (ms) |

---

## Setup

Python 3.11+ and Node 22+.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # use requirements-deploy.txt for CPU-only torch
cd frontend && npm ci && cd ..
```

## Run Order

```bash
# 1. Training data (parallel, seed-deterministic). Heavy: prefer Colab, see below.
python -m backend.gat.dataset_generator --network Net3   --n 3000 --workers 4
python -m backend.gat.dataset_generator --network L-Town --n 2000 --workers 4

# 2. Train (GPU used automatically). --residual predicts the change from intact pressure.
python -m backend.gat.train --network Net3 --epochs 300 --patience 40 --lr 2e-4 --residual
python -m backend.gat.train --network L-Town --epochs 150 --patience 20 --batch-size 16

# 3. Evaluate on held-out scenarios
python -m backend.gat.evaluate --network Net3

# 4. Benchmark experiments
for e in 1 2 3 4 6 7; do python -m backend.experiments.runner --exp $e --network Net3; done
python -m backend.experiments.runner --exp 5 --network L-Town

# 5. Run
python -m backend.main                    # API on :8000, docs at /docs
cd frontend && npm run dev                # dashboard on :5173
```

Output locations can be redirected with `HARP_GENERATED_DIR`, `HARP_MODELS_DIR`, `HARP_RESULTS_DIR`.

## Retraining on Colab

`notebooks/HARP_WDN_Colab_Training.ipynb` generates data (parallel, cached on Drive under a `DATA_VERSION` key), trains both networks, evaluates against explicit targets, and saves checkpoints to Drive. Data generation is CPU-bound (WNTR): pick a runtime with several vCPUs; training needs only a T4. Nothing downloads to your machine unless `DOWNLOAD_MODELS = True`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/config/priorities` | Authoritative node-type priority weights |
| GET | `/networks`, `/networks/{n}/info`, `/networks/{n}/graph`, `/networks/{n}/pipes` | Benchmark networks |
| GET | `/india/cities`, `/geo/cities`, `/geo/city/{name}` | City lists and OSM-derived city graphs |
| POST | `/simulate/scenario` | Single-pipe failure; method `none` / `dinic` / `harp`; returns a recovery plan and MC-Dropout uncertainty for `harp` |
| POST | `/simulate/leak`, `/simulate/surge`, `/simulate/multi_failure` | Other WNTR scenarios |
| POST | `/anomaly/detect` | Simulate a failure and detect it |
| POST | `/anomaly/auto_respond` | Readings → detect → locate → HARP recommendation |
| POST | `/sensors/readings` | Ingest pressure readings |
| GET | `/sensors/latest/{network}` | Latest ingested readings |
| GET | `/runs`, `/runs/{id}` | Run history |
| GET | `/india/wdn/{city}` | Fetch or generate a city network |
| POST | `/india/simulate` | HARP / Dinic's / baseline on a city network |
| GET | `/india/events/{city}` | SSE: **simulated** pressure telemetry, 2.5 s ticks |
| POST | `/india/priority/override` | Override a node's type or weight *(admin)* |
| DELETE | `/india/priority/overrides/{city}` | Clear overrides *(admin)* |
| POST | `/india/wdn/{city}/regenerate` | Regenerate a network *(admin)* |
| GET | `/india/report/generate`, `/india/report/status` | Background figure generation *(admin)* |
| GET | `/results/experiments`, `/results/experiment/{id}`, `/results/figures` | Experiment outputs |

Interactive docs: `http://localhost:8000/docs`.

## Configuration

| Variable | Effect |
|---|---|
| `HARP_API_KEY` | Compute key: required (`X-API-Key`) for simulate, anomaly, sensor and report endpoints |
| `HARP_ADMIN_KEY` | Admin key: required for overrides, regeneration, report generation (also satisfies compute). With only `HARP_API_KEY` set, it guards both. Neither set = open (dev) |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins for CORS. Unset allows only `http://localhost:5173` |
| `RATE_LIMIT_COMPUTE` / `RATE_LIMIT_GENERAL` | Requests per minute per IP (default 10 / 60) |
| `TRUST_PROXY=1` | Rate-limit on `X-Forwarded-For` (set behind Render/Vercel, otherwise everyone shares one bucket) |
| `HARP_DB_PATH` | SQLite file (run history, overrides). Default `backend/results/harp_wdn.db` (git-ignored) |
| `HARP_MQTT_URL` | Enable the MQTT sensor listener (`mqtt://user:pass@host:1883`) |
| `HARP_GENERATED_DIR`, `HARP_MODELS_DIR`, `HARP_RESULTS_DIR` | Redirect data, checkpoints, results |
| `VITE_API_BASE_URL` | Frontend: backend origin when served separately |

---

## Project Structure

```
backend/
  main.py                     FastAPI: REST, SSE, auth roles, rate limiting
  config.py                   constants, NODE_TYPE_WEIGHTS, load_priority_config
  wntr_engine.py              simulation, failure/leak/surge injection, graph features
  runs.py  sensors.py  plan.py   run history, sensor ingest, recovery plan
  algorithms/                 harp.py, dinic.py, hw_hydraulics.py, subgraph.py
  gat/                        model, dataset, dataset_generator, train, inference, evaluate
  geo/                        india_wdn.py, city_mapper.py
  anomaly/detector.py         z-score detector
  experiments/                runner.py (7 experiments), scenarios.py, metrics.py, visualizer.py
  data/                       networks/, trained_models/, priority_config.json
  results/                    experiment CSVs, figures
  tests/                      unit and API tests
frontend/src/                 App.tsx, api.ts, cityData.ts, pages/ (Simulation, Results, Network, History)
notebooks/                    Colab training notebook
.github/workflows/ci.yml      backend tests, frontend typecheck and build
```

## Tests and CI

```bash
python -m pytest backend/tests -q
```

The backend suite currently has 84 tests, all passing.

CI (GitHub Actions) runs the backend suite on Python 3.11 with CPU torch, and the frontend typecheck and build, on every push to `main`.

---

## Limitations

- **Research prototype.** HARP recommends; a human must approve valve actions. It has not been validated on a real utility network.
- **MC-dropout uncertainty is uncalibrated** (see Key Results); do not read it as a confidence interval.
- **HARP speed.** About 15x slower than Dinic's on L-Town; subgraph search does not save time on Net3.
- **GAT accuracy.** On Net3 the surrogate is not clearly better than simply assuming nothing failed (see the table above). The hydraulic gate is also inactive on well-pressurised networks: single-pipe failures on Net3 and L-Town barely change delivery, so every method scores near 100%. Use experiment 7's stress scenarios to see differences.
- **Sensors are simulated** in the dashboard and in `/india/events`; only `/sensors/readings` and MQTT take real data.
- **Priority node ids** for Net3 and L-Town are hand-picked in `backend/data/priority_config.json` (a test checks they exist). Hanoi is a bundled benchmark file with no trained model and is not exposed in the UI.
- **Mixed hydraulic models.** The India demo's "no recovery" baseline is a BFS heuristic, while its HARP and Dinic's evaluations use WNTR.
- **Demo mode.** If the backend is unreachable the frontend shows clearly labelled illustrative numbers, not results.
