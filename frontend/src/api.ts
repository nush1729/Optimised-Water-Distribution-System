import axios from "axios";
import type { WDNData, WDNNode, WDNEdge } from "./cityData";

// VITE_API_BASE_URL is set at build time (see .env.production / Render's
// environment settings) so the deployed frontend talks to the deployed
// backend instead of localhost. Falls back to localhost for `npm run dev`.
const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const api = axios.create({ baseURL: BASE });

// ── Health check ───────────────────────────────────────────────────────────
export const checkHealth = () =>
  api.get("/health", { timeout: 2000 }).then(() => true).catch(() => false);

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

// ── Schema adapters ────────────────────────────────────────────────────────
// Backend uses different field names than the frontend WDNData types.
const TYPE_WEIGHT: Record<string, number> = {
  reservoir: 5.0, water_tower: 4.5, hospital: 4.0, fire_station: 3.8,
  school: 3.2, college: 2.8, market: 2.5, park: 1.5, residential: 2.0, junction: 1.0,
}

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

  // Critical = hospitals (4.0), fire stations (3.8), reservoirs/towers
  const criticalIds = baseWDN.nodes
    .filter((n) => n.priority_weight >= 3.8)
    .map((n) => n.id)
  const criticalVals = criticalIds.map((id) => demandMap[id] ?? 1)
  const criticalPct  = criticalVals.length
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
    return Array.isArray(arr) ? arr.map((c: any) => (typeof c === "string" ? c : c.city)) : []
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
      ).catch(() => null),
      api.post("/india/simulate", { ...body, method: "dinic" }).then((r) =>
        adaptSimResult(r.data, baseWDN, body.scenario ?? "normal", "dinic"),
      ).catch(() => null),
    ]).then(([harp, dinic]) => ({ harp: harp ?? undefined, dinic: dinic ?? undefined }))
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
