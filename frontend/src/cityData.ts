/**
 * Self-sufficient city data layer.
 * Generates deterministic, seeded WDN node/pipe networks for 4 Indian cities.
 * Used as primary data source; API enhances with real hydraulic simulation.
 */

// ── Types (mirror backend-independent interfaces) ──────────────────────────
export interface WDNNode {
  id: string; lat: number; lon: number; type: string
  demand: number; elevation: number; priority_weight: number; pressure?: number
}
export interface WDNEdge {
  id: string; from: string; to: string
  diameter: number; length: number; C_hw: number; flow?: number
}
export interface WDNData {
  city: string; n_nodes: number; nodes: WDNNode[]; edges: WDNEdge[]; sources: string[]
}

// ── City catalogue ─────────────────────────────────────────────────────────
export const CITY_CONFIGS: Record<string, {
  displayName: string
  center: [number, number]
  zoom: number
  radiusDeg: number
  baseElevation: number
}> = {
  bangalore: { displayName: 'Bangalore',  center: [12.9716, 77.5946], zoom: 14, radiusDeg: 0.020, baseElevation: 920 },
  delhi:     { displayName: 'Delhi',      center: [28.6139, 77.2090], zoom: 13, radiusDeg: 0.025, baseElevation: 216 },
  mumbai:    { displayName: 'Mumbai',     center: [19.0760, 72.8777], zoom: 14, radiusDeg: 0.018, baseElevation: 14  },
  hyderabad: { displayName: 'Hyderabad',  center: [17.3850, 78.4867], zoom: 14, radiusDeg: 0.020, baseElevation: 542 },
}

// ── Node type config ───────────────────────────────────────────────────────
const TYPE_WEIGHT: Record<string, number> = {
  reservoir: 5.0, water_tower: 4.5, hospital: 4.0, fire_station: 3.8,
  school: 3.2, college: 2.8, market: 2.5, park: 1.5, residential: 2.0, junction: 1.0,
}
const TYPE_DEMAND: Record<string, number> = {
  reservoir: 0.0, water_tower: 0.0, hospital: 0.008, fire_station: 0.005,
  school: 0.004, college: 0.006, market: 0.007, park: 0.001, residential: 0.004, junction: 0.001,
}
const TYPE_HW_C: Record<string, number> = {
  reservoir: 140, water_tower: 135, hospital: 130, fire_station: 130,
  school: 120, college: 120, market: 120, park: 110, residential: 110, junction: 120,
}

// Ring radius (0=center, 1=outer) and angle bias (radians) per type
const TYPE_RING: Record<string, { rMin: number; rMax: number; angleBias: number }> = {
  reservoir:    { rMin: 0.70, rMax: 0.95, angleBias: 0 },
  water_tower:  { rMin: 0.55, rMax: 0.75, angleBias: Math.PI / 2 },
  hospital:     { rMin: 0.30, rMax: 0.65, angleBias: Math.PI / 6 },
  fire_station: { rMin: 0.40, rMax: 0.70, angleBias: Math.PI },
  school:       { rMin: 0.25, rMax: 0.60, angleBias: Math.PI / 3 },
  college:      { rMin: 0.35, rMax: 0.55, angleBias: Math.PI * 1.5 },
  market:       { rMin: 0.10, rMax: 0.35, angleBias: Math.PI / 4 },
  park:         { rMin: 0.30, rMax: 0.55, angleBias: Math.PI * 0.8 },
  residential:  { rMin: 0.55, rMax: 0.90, angleBias: Math.PI * 1.2 },
  junction:     { rMin: 0.05, rMax: 0.30, angleBias: Math.PI * 1.7 },
}

// Full 26-node template (sorted by priority descending for top-N slicing)
const NODE_TEMPLATE: { type: string; count: number }[] = [
  { type: 'reservoir',    count: 2 },
  { type: 'water_tower',  count: 2 },
  { type: 'hospital',     count: 3 },
  { type: 'fire_station', count: 2 },
  { type: 'school',       count: 4 },
  { type: 'college',      count: 1 },
  { type: 'market',       count: 3 },
  { type: 'park',         count: 2 },
  { type: 'residential',  count: 4 },
  { type: 'junction',     count: 3 },
]

// ── Seeded PRNG ────────────────────────────────────────────────────────────
function mulberry32(seed: number) {
  let s = seed >>> 0
  return () => {
    s |= 0; s = s + 0x6D2B79F5 | 0
    let t = Math.imul(s ^ s >>> 15, 1 | s)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}
function strSeed(str: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193)
  return h >>> 0
}

// ── Haversine distance (metres) ───────────────────────────────────────────
function haversineDist(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

// ── Node generator ─────────────────────────────────────────────────────────
function generateNodes(city: string, nNodes: number): WDNNode[] {
  const cfg = CITY_CONFIGS[city]
  if (!cfg) return []
  const [clat, clon] = cfg.center
  const r = cfg.radiusDeg
  const elev = cfg.baseElevation
  const rng = mulberry32(strSeed(city))

  // Build full list ordered by priority (high → low)
  const typeSeq: string[] = []
  for (const { type, count } of NODE_TEMPLATE) {
    for (let i = 0; i < count; i++) typeSeq.push(type)
  }

  const nodes: WDNNode[] = []
  const typeCounters: Record<string, number> = {}

  // Simpler short IDs
  const TYPE_ID: Record<string, string> = {
    reservoir: 'S', water_tower: 'T', hospital: 'H', fire_station: 'F',
    school: 'SC', college: 'CL', market: 'M', park: 'P',
    residential: 'R', junction: 'J',
  }

  const capped = typeSeq.slice(0, Math.min(typeSeq.length, nNodes))

  // Track angle spread per type so nodes of same type don't overlap
  const typeAngleOffset: Record<string, number> = {}

  for (const type of capped) {
    const cnt = typeCounters[type] ?? 0
    typeCounters[type] = cnt + 1

    const ring = TYPE_RING[type]
    const totalOfType = NODE_TEMPLATE.find(t => t.type === type)?.count ?? 1

    // Spread angles evenly within the ring, add per-type bias + noise
    if (typeAngleOffset[type] === undefined) {
      typeAngleOffset[type] = ring.angleBias + (rng() - 0.5) * 0.5
    }
    const angleStep = (2 * Math.PI) / totalOfType
    const angle = typeAngleOffset[type] + cnt * angleStep + (rng() - 0.5) * angleStep * 0.4

    const radius = ring.rMin + rng() * (ring.rMax - ring.rMin)

    // Convert polar (city-relative) to lat/lon
    const cosLat = Math.cos(clat * Math.PI / 180)
    const lat = clat + Math.sin(angle) * radius * r
    const lon = clon + Math.cos(angle) * radius * r / cosLat

    const id = `${TYPE_ID[type]}${cnt}`
    const isSource = type === 'reservoir' || type === 'water_tower'
    const elevNoise = (rng() - 0.5) * 25
    const elevBoost = type === 'water_tower' ? 15 + rng() * 10 : 0

    nodes.push({
      id,
      lat: Math.round(lat * 1e6) / 1e6,
      lon: Math.round(lon * 1e6) / 1e6,
      type,
      demand: isSource ? 0 : TYPE_DEMAND[type] + (rng() - 0.5) * TYPE_DEMAND[type] * 0.3,
      elevation: Math.round((elev + elevNoise + elevBoost) * 10) / 10,
      priority_weight: TYPE_WEIGHT[type],
    })
  }

  return nodes
}

// ── Edge generator (MST + extras) ─────────────────────────────────────────
function generateEdges(nodes: WDNNode[], city: string): WDNEdge[] {
  const rng = mulberry32(strSeed(city + '_edges'))
  const n = nodes.length

  // All pairs with distances
  type Pair = { i: number; j: number; distDeg: number; distM: number }
  const pairs: Pair[] = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dlat = nodes[i].lat - nodes[j].lat
      const dlon = nodes[i].lon - nodes[j].lon
      const distDeg = Math.sqrt(dlat * dlat + dlon * dlon)
      const distM = haversineDist(nodes[i].lat, nodes[i].lon, nodes[j].lat, nodes[j].lon)
      pairs.push({ i, j, distDeg, distM })
    }
  }
  pairs.sort((a, b) => a.distDeg - b.distDeg)

  // Union-Find for MST
  const parent = Array.from({ length: n }, (_, i) => i)
  const rank = new Array(n).fill(0)
  function find(x: number): number {
    if (parent[x] !== x) parent[x] = find(parent[x])
    return parent[x]
  }
  function union(x: number, y: number): boolean {
    const px = find(x), py = find(y)
    if (px === py) return false
    if (rank[px] < rank[py]) parent[px] = py
    else if (rank[px] > rank[py]) parent[py] = px
    else { parent[py] = px; rank[px]++ }
    return true
  }

  const edgeSet = new Set<string>()
  const edges: WDNEdge[] = []
  let edgeIdx = 0

  function addEdge(i: number, j: number, distM: number) {
    const key = `${Math.min(i, j)}-${Math.max(i, j)}`
    if (edgeSet.has(key)) return
    edgeSet.add(key)
    const isMain = nodes[i].type === 'reservoir' || nodes[j].type === 'reservoir' ||
      nodes[i].type === 'water_tower' || nodes[j].type === 'water_tower'
    const diameter = isMain ? 0.25 : distM < 500 ? 0.10 : distM < 1200 ? 0.15 : 0.20
    const typeI = nodes[i].type, typeJ = nodes[j].type
    const c = Math.round((TYPE_HW_C[typeI] + TYPE_HW_C[typeJ]) / 2)
    edges.push({
      id: `P${edgeIdx++}`,
      from: nodes[i].id,
      to: nodes[j].id,
      diameter,
      length: Math.round(distM),
      C_hw: c,
    })
  }

  // MST edges (guarantees connectivity)
  for (const { i, j, distM } of pairs) {
    if (union(i, j)) addEdge(i, j, distM)
  }

  // Extra redundancy edges (each source/tank connects to 2 more nearby nodes)
  for (let i = 0; i < n; i++) {
    if (nodes[i].type === 'reservoir' || nodes[i].type === 'water_tower') {
      let added = 0
      for (const { i: pi, j: pj, distM } of pairs) {
        if (added >= 2) break
        if (pi !== i && pj !== i) continue
        const other = pi === i ? pj : pi
        const key = `${Math.min(i, other)}-${Math.max(i, other)}`
        if (!edgeSet.has(key)) { addEdge(i, other, distM); added++ }
      }
    }
  }

  // Random short extras (~30% of possible short-range pairs not yet connected)
  let extras = 0
  const maxExtras = Math.floor(n * 0.5)
  for (const { i, j, distDeg, distM } of pairs) {
    if (extras >= maxExtras) break
    if (distDeg > 0.012) break  // only short-range extras
    const key = `${Math.min(i, j)}-${Math.max(i, j)}`
    if (!edgeSet.has(key) && rng() < 0.35) {
      addEdge(i, j, distM); extras++
    }
  }

  return edges
}

// ── Public API ─────────────────────────────────────────────────────────────
export function generateCityWDN(city: string, nNodes: number, nSources = 2): WDNData {
  const n = Math.max(15, Math.min(nNodes, 60))
  const nodes = generateNodes(city, n)
  const edges = generateEdges(nodes, city)
  const allSources = nodes
    .filter(nd => nd.type === 'reservoir' || nd.type === 'water_tower')
    .map(nd => nd.id)
  // Respect user-specified source count (min 1, max available)
  const sources = allSources.slice(0, Math.max(1, Math.min(nSources, allSources.length)))
  return { city, n_nodes: nodes.length, nodes, edges, sources }
}

export const CITY_LIST = Object.keys(CITY_CONFIGS)
