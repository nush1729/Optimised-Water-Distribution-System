import axios from "axios";
import type { WDNData, WDNNode, WDNEdge } from "./cityData";
import { SOURCE_TYPES, TYPE_WEIGHT } from "./cityData";

// In production the frontend is served by the same FastAPI process, so use
// relative URLs (empty string). In dev, proxy to the local backend.
// Set VITE_API_BASE_URL to override (e.g. a separate backend deployment).
const BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");
const api = axios.create({ baseURL: BASE });

// ── Health check ───────────────────────────────────────────────────────────
export const checkHealth = () =>
  api.get("/health", { timeout: 2000 }).then(() => { void syncPriorityWeights(); return true }).catch(() => false);

// The backend owns the priority weights (backend/config.py NODE_TYPE_WEIGHTS).
// Pull them once the backend is reachable; the constants in cityData.ts are only
// the offline fallback. Sources keep their display-only values.
let _weightsSynced = false
export async function syncPriorityWeights(): Promise<void> {
  if (_weightsSynced) return
  try {
    const { weights } = (await api.get("/config/priorities", { timeout: 2000 })).data as { weights: Record<string, number> }
    for (const [type, w] of Object.entries(weights)) {
      if (!SOURCE_TYPES.includes(type)) TYPE_WEIGHT[type] = w
    }
    _weightsSynced = true
  } catch (e) {
    console.warn("Could not load priority weights from backend; using built-in defaults", e)
  }
}

// ── Classic WDN networks ───────────────────────────────────────────────────
export const getNetworks   = () => api.get("/networks").then((r) => r.data);
export const getNetworkInfo  = (net: string) => api.get(`/networks/${net}/info`).then((r) => r.data);
export const getNetworkGraph = (net: string) => api.get(`/networks/${net}/graph`).then((r) => r.data);
export const getPipes        = (net: string) => api.get(`/networks/${net}/pipes`).then((r) => r.data);
export const runScenario     = (payload: object) => api.post("/simulate/scenario", payload).then((r) => r.data);
export const listExperiments = () => api.get("/results/experiments").then((r) => r.data);
export const getExperiment   = (id: number) => api.get(`/results/experiment/${id}`).then((r) => r.data);
export const listFigures     = () => api.get("/results/figures").then((r) => r.data);
export const detectAnomaly   = (payload: object) => api.post("/anomaly/detect", payload).then((r) => r.data);
export const autoRespond     = (payload: object) => api.post("/anomaly/auto_respond", payload).then((r) => r.data);

// ── Run history (backend/runs.py) ──────────────────────────────────────────
export interface RunRecord {
  id: number; ts: number; kind: string; network: string
  method: string | null; scenario: string | null; failed_pipes: string[]
  cds_overall: number | null; cds_critical: number | null; pvc: number | null
  runtime_ms: number | null; escalated: boolean | null
}
export const listRuns = (limit = 100, kind?: string): Promise<RunRecord[]> =>
  api.get("/runs", { params: { limit, kind } }).then((r) => r.data.runs);

// ── Schema adapters ────────────────────────────────────────────────────────
// Backend uses different field names than the frontend WDNData types.

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped backend JSON
function adaptNode(raw: any): WDNNode {
  return {
    id:              raw.id,
    lat:             raw.lat,
    lon:             raw.lon,
    type:            raw.type ?? "junction",
    demand:          raw.demand ?? 0,
    elevation:       raw.elevation ?? 0,
    priority_weight: raw.priority_weight ?? TYPE_WEIGHT[raw.type] ?? 1.0,
    pressure:        raw.pressure,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped backend JSON
function adaptEdge(raw: any): WDNEdge {
  return {
    id:       raw.id,
    from:     raw.from ?? raw.source,
    to:       raw.to   ?? raw.target,
    diameter: raw.diameter ?? raw.diameter_m ?? 0.15,
    length:   raw.length   ?? raw.length_m   ?? 100,
    C_hw:     raw.C_hw ?? raw.roughness ?? 120,
    flow:     raw.flow,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped backend JSON
function adaptWDNData(raw: any): WDNData {
  const nodes: WDNNode[] = (raw.nodes ?? []).map(adaptNode)
  const edges: WDNEdge[] = (raw.edges ?? []).map(adaptEdge)
  const sources =
    raw.sources?.length
      ? raw.sources
      : nodes.filter((_, i) => raw.nodes?.[i]?.is_source).map((n) => n.id)
  return { city: raw.city, n_nodes: nodes.length, nodes, edges, sources }
}

// ── Sim result adapter ─────────────────────────────────────────────────────
export interface SimStats {
  served_demand_pct:        number
  critical_node_service_pct: number
  recovery_flow_pct?:       number
  pressure_deficit_nodes?:  number
  runtime_ms?:              number
  escalated?:               boolean
}
export interface SimResult {
  method: string; scenario: string
  nodes: WDNNode[]; edges: WDNEdge[]
  stats: SimStats; failed_pipes?: string[]
  harp?: SimResult; dinic?: SimResult
  pressure_uncertainty_m?: Record<string, number>
}

function adaptSimResult(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped backend JSON
  raw: any,
  baseWDN: WDNData,
  scenario: string,
  method: string,
): SimResult {
  // Merge pressure data from simulation into nodes
  const pressuresMap: Record<string, number> = raw.pressures_m ?? {}
  const demandMap:    Record<string, number> = raw.demand_satisfaction ?? {}

  const nodes: WDNNode[] = baseWDN.nodes.map((n) => ({
    ...n,
    pressure: pressuresMap[n.id] ?? n.pressure,
  }))

  // Build edge flow from accepted_paths.
  // Backend serialises paths as plain node-id arrays [[n1,n2,...], ...].
  // We derive the traversed pipe IDs by looking up each consecutive node pair
  // in the edge list, then accumulate a flow count per pipe.
  const nodePairToEdge: Record<string, string> = {}
  for (const e of baseWDN.edges) {
    nodePairToEdge[`${e.from}|${e.to}`] = e.id
    nodePairToEdge[`${e.to}|${e.from}`] = e.id
  }
  const flowMap: Record<string, number> = {}
  for (const path of raw.accepted_paths ?? []) {
    // Support both old object form {pipes:[...], flow:n} and new array form [n1,n2,...]
    const nodeSeq: string[] = Array.isArray(path) ? path : (path.nodes ?? [])
    for (let i = 0; i < nodeSeq.length - 1; i++) {
      const eid = nodePairToEdge[`${nodeSeq[i]}|${nodeSeq[i + 1]}`]
      if (eid) flowMap[eid] = (flowMap[eid] ?? 0) + 1
    }
  }
  const edges: WDNEdge[] = baseWDN.edges.map((e) => ({
    ...e,
    flow: flowMap[e.id] ?? e.flow,
  }))

  // Compute stats from demand_satisfaction
  const allVals    = Object.values(demandMap) as number[]
  const demandPct  = allVals.length
    ? (allVals.reduce((s, v) => s + v, 0) / allVals.length) * 100
    : 0

  // Critical = hospitals and fire stations (sources are excluded: they supply
  // water, they are not served). Prefer the backend's own CDS_critical so the
  // UI and the API report the same number.
  const backendCritical = raw.CDS?.CDS_critical
  const criticalIds = baseWDN.nodes
    .filter((n) => n.type === "hospital" || n.type === "fire_station")
    .map((n) => n.id)
  const criticalVals = criticalIds.map((id) => demandMap[id] ?? 1)
  const criticalPct  = typeof backendCritical === "number"
    ? backendCritical * 100
    : criticalVals.length
      ? (criticalVals.reduce((s, v) => s + v, 0) / criticalVals.length) * 100
      : 0

  const nodesWithPressure = nodes.filter((n) => n.pressure != null)
  const pressureDeficit = raw.pressure_method === "unavailable"
    ? undefined
    : nodesWithPressure.filter((n) => n.pressure! < 7).length

  const stats: SimStats = {
    served_demand_pct:         Math.round(demandPct  * 10) / 10,
    critical_node_service_pct: Math.round(criticalPct * 10) / 10,
    recovery_flow_pct:         method !== 'none' && raw.CDS?.CDS_overall != null
      ? Math.round(raw.CDS.CDS_overall * 1000) / 10
      : undefined,
    pressure_deficit_nodes:    pressureDeficit,
    runtime_ms:                raw.computation_time_ms,
    escalated:                 raw.escalation_flag ?? false,
  }

  return {
    method,
    scenario,
    nodes,
    edges,
    stats,
    failed_pipes: raw.failed_pipes ?? [],
    pressure_uncertainty_m: raw.pressure_uncertainty_m,
  }
}

// ── India WDN API ──────────────────────────────────────────────────────────
export const getIndiaCities = () =>
  api.get("/india/cities").then((r) => {
    const d = r.data
    // Backend returns { cities: [{city, name, ...}] }
    const arr = Array.isArray(d) ? d : (d.cities ?? d)
    return Array.isArray(arr) ? arr.map((c: string | { city: string }) => (typeof c === "string" ? c : c.city)) : []
  })

export const getIndiaWDN = (city: string, n_nodes = 20, regen = false): Promise<WDNData> =>
  api
    .get(`/india/wdn/${city}`, { params: { n_nodes, regen } })
    .then((r) => adaptWDNData(r.data))

export type IndiaSimMethod  = "none" | "dinic" | "harp" | "compare";
export type ScenarioType    = "normal" | "burst" | "leak" | "surge" | "multi_failure";

export interface IndiaSimRequest {
  city: string; failed_pipes: string[]; method: IndiaSimMethod
  k?: number; n_nodes?: number; scenario?: ScenarioType
  surge_nodes?: string[]; surge_multiplier?: number
  leak_pipe?: string; leak_coefficient?: number
  custom_priorities?: Record<string, string>; custom_sources?: string[]
}

export const runIndiaSimulation = (
  body: IndiaSimRequest,
  baseWDN: WDNData,
): Promise<SimResult | { harp?: SimResult; dinic?: SimResult }> => {
  if (body.method === "compare") {
    return Promise.all([
      api.post("/india/simulate", { ...body, method: "harp" }).then((r) =>
        adaptSimResult(r.data, baseWDN, body.scenario ?? "normal", "harp"),
      ),
      api.post("/india/simulate", { ...body, method: "dinic" }).then((r) =>
        adaptSimResult(r.data, baseWDN, body.scenario ?? "normal", "dinic"),
      ),
    ]).then(([harp, dinic]) => ({ harp, dinic }))
    // No per-request .catch: a failure (e.g. a 429 rate limit) must reach the
    // caller and be shown, not silently turn into "no result" and leave stale
    // numbers on screen.
  }
  return api.post("/india/simulate", body).then((r) =>
    adaptSimResult(r.data, baseWDN, body.scenario ?? "normal", body.method),
  )
}

export interface PriorityOverrideRequest {
  city: string; node_id: string; new_type: string; new_weight: number
}
export const overrideNodePriority = (body: PriorityOverrideRequest) =>
  api.post("/india/priority/override", body).then((r) => r.data)


/** Best-effort message from an axios-style error without using `any`. */
export function errorDetail(e: unknown, fallback: string): string {
  const err = e as { response?: { data?: { detail?: unknown } }; message?: string } | undefined
  const d = err?.response?.data?.detail
  if (typeof d === 'string') return d
  return err?.message ?? fallback
}
