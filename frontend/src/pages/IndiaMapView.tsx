import { useState, useEffect, useCallback, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import {
  CITY_CONFIGS, CITY_LIST, generateCityWDN,
  type WDNNode, type WDNEdge, type WDNData,
} from '../cityData'
import {
  checkHealth, getIndiaWDN, runIndiaSimulation, overrideNodePriority,
  type IndiaSimRequest, type ScenarioType, type IndiaSimMethod, type SimResult,
} from '../api'
import {
  IconCheck, IconPipeBreak, IconDroplets, IconTrendingUp, IconAlertTriangle,
  IconSettings, IconLayers, IconBarChart, IconRefresh, IconPlay,
  IconZap, IconEdit, IconDroplet, IconX, IconGlobe,
  IconWave, IconSliders, IconThermometer, IconActivity,
} from '../icons'
import type { SharedSimState } from '../types'

// ── Constants ─────────────────────────────────────────────────────────────────
const NODE_COLORS: Record<string, string> = {
  reservoir: '#0891b2', water_tower: '#7c3aed', hospital: '#dc2626',
  fire_station: '#ea580c', school: '#059669', college: '#0d9488',
  market: '#d97706', park: '#16a34a', residential: '#64748b', junction: '#94a3b8',
}
const NODE_LABELS: Record<string, string> = {
  reservoir: 'Reservoir', water_tower: 'Water Tower', hospital: 'Hospital',
  fire_station: 'Fire Station', school: 'School', college: 'College',
  market: 'Market', park: 'Park', residential: 'Residential', junction: 'Junction',
}
const NODE_RADII: Record<string, number> = {
  reservoir: 12, water_tower: 11, hospital: 10, fire_station: 10,
  school: 9, college: 9, market: 8, park: 8, residential: 7, junction: 6,
}
type ScenarioInfo = { value: ScenarioType; label: string; desc: string; key: string }
const SCENARIOS: ScenarioInfo[] = [
  { value: 'normal',        label: 'Normal Operation',   desc: 'Baseline flow, no failures',   key: 'normal' },
  { value: 'burst',         label: 'Pipe Burst',         desc: 'One pipe fully closed',        key: 'burst'  },
  { value: 'leak',          label: 'Pipe Leak',          desc: 'Emitter fault on a pipe',      key: 'leak'   },
  { value: 'surge',         label: 'Demand Surge',       desc: 'Demand spike at select nodes', key: 'surge'  },
  { value: 'multi_failure', label: 'Multi-Pipe Failure', desc: 'Multiple pipes closed',        key: 'multi'  },
]

function pressureColor(p: number) {
  if (p < 7)  return '#dc2626'
  if (p < 14) return '#d97706'
  return '#059669'
}
function ScenarioIcon({ type, size = 14 }: { type: ScenarioType; size?: number }) {
  if (type === 'normal')        return <IconCheck size={size} />
  if (type === 'burst')         return <IconPipeBreak size={size} />
  if (type === 'leak')          return <IconDroplets size={size} />
  if (type === 'surge')         return <IconTrendingUp size={size} />
  if (type === 'multi_failure') return <IconAlertTriangle size={size} />
  return null
}

// ── BFS animation ─────────────────────────────────────────────────────────────
interface AnimFrame { nodeIds: string[]; edgeIds: string[] }

function computeAnimFrames(
  nodes: WDNNode[], edges: WDNEdge[], sources: string[], failedPipeIds: string[],
): AnimFrame[] {
  const adj = new Map<string, { nodeId: string; edgeId: string }[]>()
  for (const n of nodes) adj.set(n.id, [])
  for (const e of edges) {
    if (failedPipeIds.includes(e.id)) continue
    adj.get(e.from)?.push({ nodeId: e.to, edgeId: e.id })
    adj.get(e.to)?.push({ nodeId: e.from, edgeId: e.id })
  }
  const nodeDepth = new Map<string, number>()
  const edgeDepth = new Map<string, number>()
  const queue = [...sources]
  for (const s of sources) nodeDepth.set(s, 0)
  while (queue.length) {
    const cur = queue.shift()!
    const d = nodeDepth.get(cur)!
    for (const { nodeId, edgeId } of adj.get(cur) ?? []) {
      if (!edgeDepth.has(edgeId)) edgeDepth.set(edgeId, d)
      if (!nodeDepth.has(nodeId)) { nodeDepth.set(nodeId, d + 1); queue.push(nodeId) }
    }
  }
  const maxD = Math.max(0, ...[...nodeDepth.values()])
  return Array.from({ length: maxD + 1 }, (_, d) => ({
    nodeIds: [...nodeDepth.entries()].filter(([, v]) => v === d).map(([k]) => k),
    edgeIds: [...edgeDepth.entries()].filter(([, v]) => v === d).map(([k]) => k),
  }))
}

// ── IoT Sensor Panel ──────────────────────────────────────────────────────────
const STATUS_COLORS = { ok: '#059669', warn: '#d97706', critical: '#dc2626' } as const
type SensorStatus = 'ok' | 'warn' | 'critical'

interface IotSensorPanelProps {
  nodes: WDNNode[]
  failedPipes: string[]
  tick: number
  simResult: SimResult | null
  livePressures: Record<string, number> | null
  liveWarnings: string[]
  connected: boolean
}
function IotSensorPanel({ nodes, failedPipes, tick, simResult, livePressures, liveWarnings, connected }: IotSensorPanelProps) {
  // Pick representative nodes: sources + high-priority infrastructure
  const sampleNodes = nodes
    .filter(n => ['reservoir', 'water_tower', 'hospital', 'fire_station'].includes(n.type))
    .slice(0, 6)

  const now = new Date()
  const ts = now.toLocaleTimeString('en-IN', { hour12: false })

  return (
    <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <div style={{
          width: 7, height: 7, borderRadius: '50%',
          background: connected ? '#059669' : '#d97706',
          boxShadow: connected ? '0 0 6px #059669' : '0 0 6px #d97706',
          animation: 'pipPulse 2s ease-in-out infinite',
        }} />
        <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6, color: 'var(--cyan)' }}>
          IoT Sensor Feed
        </span>
        <span style={{ fontSize: 9, color: connected ? '#059669' : '#d97706', fontWeight: 600, marginLeft: 2 }}>
          {connected ? '● LIVE' : '◌ Local'}
        </span>
        <span style={{ fontSize: 9, color: 'var(--text-3)', marginLeft: 'auto', fontFamily: 'var(--mono)' }}>{ts}</span>
      </div>
      {liveWarnings.length > 0 && (
        <div style={{ padding: '4px 8px', background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.3)', borderRadius: 4, marginBottom: 6 }}>
          <span style={{ fontSize: 10, color: '#dc2626', fontWeight: 700 }}>
            ⚠ {liveWarnings.length} node{liveWarnings.length > 1 ? 's' : ''} below min pressure
          </span>
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {sampleNodes.map((node, i) => {
          // Use live SSE pressure if available, else fall back to local computation
          const livePressure = livePressures?.[node.id]
          const basePressure = simResult?.nodes.find(n => n.id === node.id)?.pressure
            ?? (14 + node.priority_weight * 1.5)
          const noise = Math.sin((tick + i) * 0.7 + i * 1.3) * 0.8 + Math.sin(tick * 0.3 + i) * 0.4
          const pressure = livePressure != null ? livePressure : Math.max(0, basePressure + noise)
          const flowBase = node.type === 'reservoir' ? 0 : 0.003 + i * 0.0005
          const flow = Math.max(0, flowBase + Math.sin((tick + i) * 0.5) * 0.0008)
          const isWarned = liveWarnings.includes(node.id)
          const status: SensorStatus = (failedPipes.length > 2 && i % 3 === 0) || isWarned ? 'critical'
            : pressure < 8 ? 'warn' : 'ok'
          const isSource = node.type === 'reservoir' || node.type === 'water_tower'
          return (
            <div key={node.id} style={{
              padding: '6px 8px', borderRadius: 'var(--r)',
              background: 'var(--surface-2)', border: `1px solid ${status === 'critical' ? 'rgba(220,38,38,0.3)' : status === 'warn' ? 'rgba(217,119,6,0.3)' : 'var(--border)'}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: STATUS_COLORS[status], flexShrink: 0 }} />
                <span style={{ fontSize: 10, fontWeight: 700, flex: 1, color: 'var(--text)' }}>{node.id}</span>
                <span style={{ fontSize: 9, color: 'var(--text-3)' }}>{NODE_LABELS[node.type]}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                <div style={{ fontSize: 10 }}>
                  <span style={{ color: 'var(--text-3)' }}>P: </span>
                  <span style={{ fontFamily: 'var(--mono)', color: pressure < 8 ? 'var(--red)' : 'var(--cyan)', fontWeight: 700 }}>{pressure.toFixed(2)} m</span>
                </div>
                <div style={{ fontSize: 10 }}>
                  <span style={{ color: 'var(--text-3)' }}>Q: </span>
                  <span style={{ fontFamily: 'var(--mono)', color: 'var(--teal)' }}>{isSource ? '—' : (flow * 1000).toFixed(2) + 'L/s'}</span>
                </div>
              </div>
              {status !== 'ok' && (
                <div style={{ fontSize: 9, color: STATUS_COLORS[status], marginTop: 3, fontWeight: 600 }}>
                  {status === 'critical' ? '⚠ ANOMALY DETECTED' : '! Below threshold'}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <div style={{ marginTop: 6, fontSize: 9, color: 'var(--text-3)', textAlign: 'center' }}>
        {connected ? 'Live SSE stream from backend · 2.5 s tick' : 'Simulated SCADA feed · 2.5 s interval'}
      </div>
    </div>
  )
}

// Map fly-to helper component
function MapFlyTo({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap()
  useEffect(() => { map.flyTo(center, zoom, { duration: 1.2 }) }, [center[0], center[1], zoom])
  return null
}

// ── Main ──────────────────────────────────────────────────────────────────────
interface Props { onSimComplete: (state: SharedSimState) => void }

export default function IndiaMapView({ onSimComplete }: Props) {
  const [selectedCity, setSelectedCity]     = useState('')
  const [nNodes, setNNodes]                 = useState(20)
  const [nSources, setNSources]             = useState(2)
  const [wdn, setWdn]                       = useState<WDNData | null>(null)
  const [loading, setLoading]               = useState(false)
  const [simLoading, setSimLoading]         = useState(false)
  const [error, setError]                   = useState('')
  const [backendOnline, setBackendOnline]   = useState<boolean | null>(null)

  const [scenario, setScenario]             = useState<ScenarioType>('normal')
  const [failedPipes, setFailedPipes]       = useState<string[]>([])
  const [surgeNodes, setSurgeNodes]         = useState<string[]>([])
  const [surgeMultiplier, setSurgeMultiplier] = useState(2.5)
  const [leakPipe, setLeakPipe]             = useState('')
  const [leakCoeff, setLeakCoeff]           = useState(0.001)

  const [simResult, setSimResult]           = useState<SimResult | null>(null)
  const [harpResult, setHarpResult]         = useState<SimResult | null>(null)
  const [dinicResult, setDinicResult]       = useState<SimResult | null>(null)
  const [showHeatmap, setShowHeatmap]       = useState(false)
  const [showFlow, setShowFlow]             = useState(true)

  const [priorityMode, setPriorityMode]     = useState<'auto' | 'manual'>('auto')
  const [selectedNode, setSelectedNode]     = useState<WDNNode | null>(null)
  const [overrideType, setOverrideType]     = useState('')
  const [overrideWeight, setOverrideWeight] = useState(1.0)
  const [tab, setTab]                       = useState<'control' | 'legend' | 'stats'>('control')

  // Multi-failure mode
  const [multiFailMode, setMultiFailMode]   = useState(false)
  const [activeScenarios, setActiveScenarios] = useState(new Set<ScenarioType>(['normal']))

  // IoT sensor simulation
  const [showIoT, setShowIoT]               = useState(false)
  const [iotTick, setIoTTick]               = useState(0)
  const iotTimerRef                         = useRef<ReturnType<typeof setInterval> | null>(null)

  // SSE live data from backend IoT stream
  const [livePressures, setLivePressures]   = useState<Record<string, number> | null>(null)
  const [liveWarnings, setLiveWarnings]     = useState<string[]>([])
  const [sseConnected, setSseConnected]     = useState(false)
  const sseRef                              = useRef<EventSource | null>(null)

  // Animation
  const [animFrames, setAnimFrames]         = useState<AnimFrame[]>([])
  const [animStep, setAnimStep]             = useState(-1)
  const [animMode, setAnimMode]             = useState<'off' | 'playing' | 'paused' | 'done'>('off')
  const [animSpeed, setAnimSpeed]           = useState(700)
  const [reachedNodes, setReachedNodes]     = useState(new Set<string>())
  const [reachedEdges, setReachedEdges]     = useState(new Set<string>())
  const animTimerRef                        = useRef<ReturnType<typeof setInterval> | null>(null)

  const [flyTarget, setFlyTarget]           = useState<{ center: [number, number]; zoom: number } | null>(null)

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark'

  useEffect(() => { checkHealth().then(setBackendOnline) }, [])

  // Recompute animation frames whenever WDN or failed pipes change
  useEffect(() => {
    if (wdn) {
      const frames = computeAnimFrames(wdn.nodes, wdn.edges, wdn.sources, failedPipes)
      setAnimFrames(frames)
    }
  }, [wdn, failedPipes])

  // Animation interval
  useEffect(() => {
    if (animMode !== 'playing') {
      if (animTimerRef.current) clearInterval(animTimerRef.current)
      return
    }
    animTimerRef.current = setInterval(() => {
      setAnimStep(prev => {
        const next = prev + 1
        if (next >= animFrames.length) {
          setAnimMode('done')
          if (animTimerRef.current) clearInterval(animTimerRef.current)
          return prev
        }
        const frame = animFrames[next]
        setReachedNodes(s => { const n = new Set(s); frame.nodeIds.forEach(id => n.add(id)); return n })
        setReachedEdges(s => { const n = new Set(s); frame.edgeIds.forEach(id => n.add(id)); return n })
        return next
      })
    }, animSpeed)
    return () => { if (animTimerRef.current) clearInterval(animTimerRef.current) }
  }, [animMode, animSpeed, animFrames])

  function resetAnim() {
    if (animTimerRef.current) clearInterval(animTimerRef.current)
    setAnimMode('off'); setAnimStep(-1)
    setReachedNodes(new Set()); setReachedEdges(new Set())
  }

  function startAnim() {
    if (!animFrames.length) return
    if (animTimerRef.current) clearInterval(animTimerRef.current)
    const frame0 = animFrames[0]
    setReachedNodes(new Set(frame0.nodeIds))
    setReachedEdges(new Set(frame0.edgeIds))
    setAnimStep(0)
    setAnimMode(animFrames.length > 1 ? 'playing' : 'done')
  }

  function stepAnim() {
    const next = animStep + 1
    if (next >= animFrames.length) { setAnimMode('done'); return }
    const frame = animFrames[next]
    setReachedNodes(s => { const n = new Set(s); frame.nodeIds.forEach(id => n.add(id)); return n })
    setReachedEdges(s => { const n = new Set(s); frame.edgeIds.forEach(id => n.add(id)); return n })
    setAnimStep(next)
    if (next === animFrames.length - 1) setAnimMode('done')
  }

  // IoT timer (local tick for fallback synthetic pressure)
  useEffect(() => {
    if (showIoT) {
      iotTimerRef.current = setInterval(() => setIoTTick(t => t + 1), 2500)
    } else {
      if (iotTimerRef.current) clearInterval(iotTimerRef.current)
    }
    return () => { if (iotTimerRef.current) clearInterval(iotTimerRef.current) }
  }, [showIoT])

  // SSE connection — connects to backend live stream when IoT panel is open and backend is online
  useEffect(() => {
    if (!showIoT || !backendOnline || !selectedCity) {
      if (sseRef.current) { sseRef.current.close(); sseRef.current = null }
      setSseConnected(false)
      return
    }
    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
    const es = new EventSource(`${apiBase}/india/events/${selectedCity}?n_nodes=${nNodes}`)
    sseRef.current = es
    es.onopen = () => setSseConnected(true)
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        setLivePressures(data.pressures ?? null)
        setLiveWarnings(data.warnings ?? [])
      } catch { /* ignore malformed */ }
    }
    es.onerror = () => {
      setSseConnected(false)
      es.close()
      sseRef.current = null
    }
    return () => { es.close(); sseRef.current = null; setSseConnected(false) }
  }, [showIoT, backendOnline, selectedCity, nNodes])

  const loadWDN = useCallback(async (city: string, n: number, forceApi = false, ns?: number) => {
    if (!city) return
    setLoading(true); setError('')
    setSimResult(null); setHarpResult(null); setDinicResult(null)
    setFailedPipes([]); setLeakPipe(''); setSurgeNodes([])
    resetAnim()

    const srcCount = ns !== undefined ? ns : nSources
    const local = generateCityWDN(city, n, srcCount)
    setWdn(local)
    const cfg = CITY_CONFIGS[city]
    setFlyTarget({ center: cfg.center, zoom: cfg.zoom })

    if (backendOnline || forceApi) {
      try {
        const api = await getIndiaWDN(city, n, forceApi)
        if (api.nodes.length > 0) setWdn(api)
      } catch { /* local data already shown */ }
    }
    setLoading(false)
  }, [backendOnline, nSources])

  useEffect(() => {
    if (selectedCity) loadWDN(selectedCity, nNodes)
  }, [selectedCity, nNodes, nSources, loadWDN])

  // Derive effective scenario and failed pipes from multi-failure state
  const effectiveScenario: ScenarioType = multiFailMode
    ? (activeScenarios.has('multi_failure') ? 'multi_failure'
      : activeScenarios.has('burst') ? 'burst'
      : activeScenarios.has('leak') ? 'leak'
      : activeScenarios.has('surge') ? 'surge' : 'normal')
    : scenario
  const effectiveFailed = failedPipes

  const runSim = async (method: IndiaSimMethod) => {
    if (!wdn) return
    setSimLoading(true); setError('')
    resetAnim()

    // Penalty per active failure type in multi mode
    const nFailTypes = multiFailMode ? Math.max(0, activeScenarios.size - (activeScenarios.has('normal') ? 1 : 0)) : (scenario !== 'normal' ? 1 : 0)
    const failPenalty = nFailTypes * 12

    const buildDemo = (m: string, pct: number): SimResult => {
      // Seeded PRNG so demo metrics are stable across repeated clicks
      const seedStr = (wdn?.city ?? '') + m + effectiveFailed.join(',') + effectiveScenario
      let s = 0x811c9dc5
      for (let i = 0; i < seedStr.length; i++) s = Math.imul(s ^ seedStr.charCodeAt(i), 0x01000193)
      const rng = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
      return {
        method: m, scenario: effectiveScenario,
        nodes: wdn.nodes.map(nd => ({
          ...nd,
          pressure: effectiveFailed.length > 0 || nFailTypes > 0
            ? 8 + nd.priority_weight * 1.5 + rng() * 5 - nFailTypes * 1.5
            : 14 + nd.priority_weight * 1.8 + rng() * 6,
        })),
        edges: wdn.edges.map(e => ({
          ...e,
          flow: effectiveFailed.includes(e.id) ? 0 : 0.0015 + rng() * 0.006,
        })),
        stats: {
          served_demand_pct: Math.max(10, pct),
          critical_node_service_pct: Math.min(Math.max(10, pct + 6), 100),
          recovery_flow_pct: m !== 'none' ? Math.min(100, Math.max(0, pct - 10)) : undefined,
          runtime_ms: m === 'harp' ? 82 + rng() * 20
            : m === 'dinic' ? 31 + rng() * 10 : 7 + rng() * 3,
          escalated: nFailTypes > 1,
        },
        failed_pipes: effectiveFailed,
      }
    }

    let sr: SimResult | null = null
    let hr: SimResult | null = harpResult
    let dr: SimResult | null = dinicResult

    if (!backendOnline) {
      await new Promise(r => setTimeout(r, 850))
      setError('Backend offline: the figures below are illustrative demo numbers, not real simulation results.')
      const isNormal = multiFailMode ? activeScenarios.has('normal') && activeScenarios.size === 1 : scenario === 'normal'
      const harpBase  = Math.max(10, (isNormal ? 97 : 85) - failPenalty)
      const dinicBase = Math.max(10, (isNormal ? 97 : 72) - failPenalty)
      const noneBase  = Math.max(10, (isNormal ? 97 : 44) - failPenalty)
      if (method === 'compare') {
        hr = buildDemo('harp', harpBase)
        dr = buildDemo('dinic', dinicBase)
        sr = hr
      } else {
        const basePct = method === 'harp' ? harpBase : method === 'dinic' ? dinicBase : noneBase
        sr = buildDemo(method, basePct)
        if (method === 'harp')  hr = sr
        if (method === 'dinic') dr = sr
      }
    } else {
      try {
        const body: IndiaSimRequest = {
          city: selectedCity, failed_pipes: failedPipes, method, n_nodes: nNodes,
          scenario, surge_nodes: surgeNodes, surge_multiplier: surgeMultiplier,
          leak_pipe: leakPipe || undefined, leak_coefficient: leakCoeff,
        }
        const result = await runIndiaSimulation(body, wdn)
        if (method === 'compare') {
          const cr = result as { harp?: SimResult; dinic?: SimResult }
          hr = cr.harp ?? null; dr = cr.dinic ?? null; sr = hr
        } else {
          sr = result as SimResult
          if (method === 'harp')  hr = sr
          if (method === 'dinic') dr = sr
        }
      } catch (e: any) {
        setError(e?.response?.data?.detail ?? 'Simulation failed')
        setSimLoading(false); return
      }
    }

    setSimResult(sr); setHarpResult(hr); setDinicResult(dr)
    setShowFlow(true); setTab('stats')

    onSimComplete({
      city: selectedCity, wdn,
      simResult: sr, harpResult: hr, dinicResult: dr,
      scenario: effectiveScenario, failedPipes: effectiveFailed, runAt: Date.now(),
    })

    // Rebuild animation frames using simulation flow data
    if (sr) {
      const frames = computeAnimFrames(sr.nodes, sr.edges, wdn.sources, failedPipes)
      setAnimFrames(frames)
    }
    setSimLoading(false)
  }

  const applyOverride = async () => {
    if (!selectedNode || !overrideType) return
    try {
      if (backendOnline) {
        await overrideNodePriority({ city: selectedCity, node_id: selectedNode.id, new_type: overrideType, new_weight: overrideWeight })
      }
      setWdn(prev => prev ? {
        ...prev,
        nodes: prev.nodes.map(n => n.id === selectedNode.id
          ? { ...n, type: overrideType, priority_weight: overrideWeight } : n),
      } : prev)
      setSelectedNode(null)
    } catch (e: any) { setError(e?.response?.data?.detail ?? 'Override failed') }
  }

  const togglePipe = (pid: string) =>
    setFailedPipes(prev => prev.includes(pid) ? prev.filter(p => p !== pid) : [...prev, pid])
  const toggleSurgeNode = (nid: string) =>
    setSurgeNodes(prev => prev.includes(nid) ? prev.filter(n => n !== nid) : [...prev, nid])

  const displayNodes: WDNNode[] = simResult?.nodes || wdn?.nodes || []
  const displayEdges: WDNEdge[] = simResult?.edges || wdn?.edges || []
  const nodeMap = new Map(displayNodes.map(n => [n.id, n]))

  function nodeFill(node: WDNNode): string {
    if (animMode !== 'off' && !reachedNodes.has(node.id)) return '#475569'
    if (showHeatmap && node.pressure != null) return pressureColor(node.pressure)
    return NODE_COLORS[node.type] || '#94a3b8'
  }
  function nodeOpacity(node: WDNNode): number {
    return animMode !== 'off' && !reachedNodes.has(node.id) ? 0.25 : 0.92
  }
  function edgeColor(edge: WDNEdge): string {
    if (failedPipes.includes(edge.id)) return '#dc2626'
    if (leakPipe === edge.id) return '#d97706'
    if (animMode !== 'off') return reachedEdges.has(edge.id) ? '#0891b2' : '#64748b'
    if (showFlow && edge.flow != null) return edge.flow > 0.001 ? '#0891b2' : '#94a3b8'
    return '#94a3b8'
  }
  function edgeOpacity(edge: WDNEdge): number {
    if (animMode !== 'off' && !reachedEdges.has(edge.id) && !failedPipes.includes(edge.id)) return 0.15
    return failedPipes.includes(edge.id) ? 1 : 0.8
  }
  function edgeWeight(edge: WDNEdge): number {
    if (failedPipes.includes(edge.id)) return 3.5
    if (animMode !== 'off') return reachedEdges.has(edge.id) ? Math.max(2, edge.diameter * 18) : 1
    if (showFlow && edge.flow != null) return Math.max(1.5, Math.min(5, edge.flow * 200 + 1.5))
    return Math.max(1.5, edge.diameter * 15)
  }

  const animPct = animFrames.length > 0 ? Math.round(((animStep + 1) / animFrames.length) * 100) : 0
  const stats = simResult?.stats

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>

      {/* ══ Control Panel ══ */}
      <div className="ctrl-panel">
        <div className="ctrl-header">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <IconWave size={16} />
              <div>
                <div style={{ fontSize: 13, fontWeight: 700 }}>Water Distribution</div>
                <div style={{ fontSize: 10, opacity: 0.75, marginTop: 1 }}>Network Simulation</div>
              </div>
            </div>
            {backendOnline !== null && (
              <span style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase',
                padding: '2px 7px', borderRadius: 20,
                background: backendOnline ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)',
                color: backendOnline ? '#34d399' : '#f87171',
                border: `1px solid ${backendOnline ? '#34d39980' : '#f8717180'}`,
              }}>
                {backendOnline ? 'Live' : 'Demo'}
              </span>
            )}
          </div>
        </div>

        <div className="panel-tabs">
          {(['control', 'legend', 'stats'] as const).map(t => (
            <button key={t} className={`panel-tab${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {t === 'control' ? <IconSettings size={12} /> : t === 'legend' ? <IconLayers size={12} /> : <IconBarChart size={12} />}
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        <div className="panel-body">

          {/* CONTROL TAB */}
          {tab === 'control' && <>
            <div className="form-group">
              <label className="form-label">City</label>
              <select className="form-control" value={selectedCity}
                onChange={e => { setSelectedCity(e.target.value); resetAnim() }}>
                <option value="">— Select a city —</option>
                {CITY_LIST.map(c => (
                  <option key={c} value={c}>{CITY_CONFIGS[c].displayName}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">
                <span>Node Count</span>
                <strong style={{ color: 'var(--cyan)', fontFamily: 'var(--mono)' }}>{nNodes}</strong>
              </label>
              <input type="range" min={15} max={50} step={1} value={nNodes}
                onChange={e => setNNodes(Number(e.target.value))} className="range-input" />
              <div className="range-labels"><span>15 min</span><span>50 max</span></div>
            </div>

            <div className="form-group">
              <label className="form-label">
                <span>Water Sources</span>
                <strong style={{ color: 'var(--amber)', fontFamily: 'var(--mono)' }}>{nSources}</strong>
              </label>
              <input type="range" min={1} max={4} step={1} value={nSources}
                onChange={e => setNSources(Number(e.target.value))} className="range-input" />
              <div className="range-labels"><span>1 min</span><span>4 max</span></div>
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 3 }}>
                Reservoirs + water towers acting as sources
              </div>
            </div>

            {selectedCity && (
              <button className="btn btn-ghost btn-sm btn-block"
                onClick={() => loadWDN(selectedCity, nNodes, true)} disabled={loading}>
                {loading ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Generating…</>
                  : <><IconRefresh size={13} /> Regenerate Network</>}
              </button>
            )}

            <div className="divider" />

            {/* ── Water Flow Animation ── */}
            {wdn && (
              <div>
                <div className="section-label" style={{ marginBottom: 8 }}>Water Flow Animation</div>
                {animMode === 'off' ? (
                  <button className="btn btn-teal btn-block" onClick={startAnim}>
                    <IconPlay size={13} /> Play Step-by-Step Flow
                  </button>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>
                          {animMode === 'done' ? 'Complete' : `Step ${animStep + 1} / ${animFrames.length}`}
                        </span>
                        <span style={{ fontSize: 11, color: 'var(--cyan)', fontWeight: 700 }}>
                          {reachedNodes.size}/{displayNodes.length}
                        </span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div className="progress-fill good" style={{ width: `${animPct}%`, transition: 'width 0.25s ease' }} />
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 3 }}>
                        nodes reached
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 5 }}>
                      {animMode === 'playing'
                        ? <button className="btn btn-ghost btn-sm" style={{ flex: 1 }} onClick={() => setAnimMode('paused')}>⏸ Pause</button>
                        : animMode === 'done'
                          ? <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={startAnim}><IconRefresh size={12} /> Replay</button>
                          : <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={() => setAnimMode('playing')}><IconPlay size={12} /> Resume</button>
                      }
                      {animMode !== 'done' && (
                        <button className="btn btn-ghost btn-sm" onClick={stepAnim} disabled={animMode === 'playing'}>Next</button>
                      )}
                      <button className="btn btn-ghost btn-sm" onClick={resetAnim}>Reset</button>
                    </div>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">
                        Speed
                        <span style={{ color: 'var(--cyan)', fontFamily: 'var(--mono)', fontSize: 10, marginLeft: 4 }}>
                          {animSpeed <= 350 ? 'Fast' : animSpeed <= 700 ? 'Medium' : 'Slow'}
                        </span>
                      </label>
                      <input type="range" min={200} max={1400} step={100} value={animSpeed}
                        onChange={e => setAnimSpeed(Number(e.target.value))} className="range-input" />
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="divider" />

            {/* Failure Scenario */}
            <div className="form-group">
              <label className="form-label" style={{ justifyContent: 'space-between' }}>
                <span>Failure Scenario</span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: 10, fontWeight: 600, color: multiFailMode ? 'var(--amber)' : 'var(--text-3)' }}>
                  <input type="checkbox" checked={multiFailMode}
                    onChange={e => { setMultiFailMode(e.target.checked); setActiveScenarios(new Set(['normal'])); setFailedPipes([]); setLeakPipe(''); setSurgeNodes([]) }}
                    style={{ width: 12, height: 12, accentColor: 'var(--amber)', cursor: 'pointer' }} />
                  Multi
                </label>
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {SCENARIOS.map(s => {
                  const isActive = multiFailMode ? activeScenarios.has(s.value) : scenario === s.value
                  return (
                    <div key={s.value}
                      className={`scenario-card${isActive ? ` sc-${s.key} sc-active` : ''}`}
                      onClick={() => {
                        if (multiFailMode) {
                          setActiveScenarios(prev => {
                            const next = new Set(prev)
                            if (s.value === 'normal') return new Set(['normal'])
                            next.delete('normal')
                            if (next.has(s.value)) { next.delete(s.value); if (next.size === 0) next.add('normal') }
                            else next.add(s.value)
                            return next
                          })
                        } else {
                          setScenario(s.value); setFailedPipes([]); setLeakPipe(''); setSurgeNodes([])
                        }
                      }}>
                      <div className={`sc-icon${isActive ? ` sc-icon-${s.key}` : ''}`}>
                        <ScenarioIcon type={s.value} size={14} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="sc-name">{s.label}</div>
                        <div className="sc-desc">{s.desc}</div>
                      </div>
                      {multiFailMode
                        ? <div style={{ width: 14, height: 14, borderRadius: 3, border: `2px solid ${isActive ? 'var(--cyan)' : 'var(--border)'}`, background: isActive ? 'var(--cyan)' : 'transparent', flexShrink: 0 }} />
                        : <div className="sc-dot"><div className="sc-dot-inner" /></div>}
                    </div>
                  )
                })}
              </div>
              {multiFailMode && activeScenarios.size > 1 && (
                <div style={{ marginTop: 5, fontSize: 10, padding: '4px 8px', background: 'rgba(251,191,36,0.1)', borderRadius: 'var(--r)', color: 'var(--amber)', border: '1px solid rgba(251,191,36,0.25)' }}>
                  {activeScenarios.size} failure types active simultaneously
                </div>
              )}
            </div>

            {(effectiveScenario === 'burst' || effectiveScenario === 'multi_failure' || (multiFailMode && (activeScenarios.has('burst') || activeScenarios.has('multi_failure')))) && wdn && (
              <div className="form-group">
                <label className="form-label">
                  {scenario === 'burst' ? 'Select Failed Pipe' : `Select Pipes (${failedPipes.length})`}
                </label>
                <select className="form-control"
                  onChange={e => {
                    if (!e.target.value) return
                    if (scenario === 'burst') setFailedPipes([e.target.value])
                    else togglePipe(e.target.value)
                    e.target.value = ''
                  }}>
                  <option value="">— Select pipe —</option>
                  {wdn.edges.map(e => <option key={e.id} value={e.id}>{e.id}  ({e.from} → {e.to})</option>)}
                </select>
                {failedPipes.length > 0 && (
                  <div className="chip-list" style={{ marginTop: 5 }}>
                    {failedPipes.map(p => (
                      <span key={p} className="chip">
                        {p} <span className="chip-x" onClick={() => togglePipe(p)}><IconX size={10} /></span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {(effectiveScenario === 'leak' || (multiFailMode && activeScenarios.has('leak'))) && wdn && (
              <div className="form-group">
                <label className="form-label">Leak Pipe</label>
                <select className="form-control" value={leakPipe} onChange={e => setLeakPipe(e.target.value)}>
                  <option value="">— Select pipe —</option>
                  {wdn.edges.map(e => <option key={e.id} value={e.id}>{e.id}</option>)}
                </select>
                <label className="form-label" style={{ marginTop: 6 }}>
                  Emitter Coefficient
                  <strong style={{ marginLeft: 4, color: 'var(--cyan)', fontFamily: 'var(--mono)', fontSize: 11 }}>{leakCoeff.toFixed(4)}</strong>
                </label>
                <input type="range" min={0.0001} max={0.01} step={0.0001}
                  value={leakCoeff} onChange={e => setLeakCoeff(Number(e.target.value))} className="range-input" />
              </div>
            )}

            {(effectiveScenario === 'surge' || (multiFailMode && activeScenarios.has('surge'))) && wdn && (
              <div className="form-group">
                <label className="form-label">Surge Nodes ({surgeNodes.length})</label>
                <div className="chip-list" style={{ minHeight: 34 }}>
                  {surgeNodes.length === 0
                    ? <span style={{ fontSize: 11, color: 'var(--text-3)', padding: '4px 0' }}>Click nodes on the map</span>
                    : surgeNodes.map(n => (
                      <span key={n} className="chip">
                        {n} <span className="chip-x" onClick={() => toggleSurgeNode(n)}><IconX size={10} /></span>
                      </span>
                    ))}
                </div>
                <label className="form-label" style={{ marginTop: 6 }}>
                  Multiplier
                  <strong style={{ marginLeft: 4, color: 'var(--amber)', fontFamily: 'var(--mono)', fontSize: 11 }}>×{surgeMultiplier.toFixed(1)}</strong>
                </label>
                <input type="range" min={1.5} max={5} step={0.5}
                  value={surgeMultiplier} onChange={e => setSurgeMultiplier(Number(e.target.value))} className="range-input" />
              </div>
            )}

            <div className="divider" />

            <div className="form-group">
              <label className="form-label">Priority Assignment</label>
              <div style={{ display: 'flex', gap: 6 }}>
                {(['auto', 'manual'] as const).map(m => (
                  <button key={m} onClick={() => setPriorityMode(m)}
                    className={`btn btn-sm ${priorityMode === m ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}>
                    {m === 'auto' ? <IconZap size={12} /> : <IconEdit size={12} />}
                    {m === 'auto' ? 'Auto' : 'Manual'}
                  </button>
                ))}
              </div>
              {priorityMode === 'manual' && (
                <div style={{ fontSize: 11, marginTop: 5, padding: '6px 8px', background: 'var(--cyan-dim)', borderRadius: 'var(--r)', color: 'var(--cyan)' }}>
                  Click any node on the map to override its type and weight.
                </div>
              )}
            </div>

            {selectedNode && priorityMode === 'manual' && (
              <div className="surface-card">
                <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 8, display: 'flex', gap: 6, alignItems: 'center' }}>
                  <IconEdit size={12} /> Override: <code style={{ fontSize: 11, color: 'var(--cyan)' }}>{selectedNode.id}</code>
                </div>
                <div className="form-group" style={{ marginBottom: 6 }}>
                  <label className="form-label">Type</label>
                  <select className="form-control" value={overrideType} onChange={e => setOverrideType(e.target.value)}>
                    <option value="">— Select —</option>
                    {Object.entries(NODE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                  </select>
                </div>
                <div className="form-group" style={{ marginBottom: 8 }}>
                  <label className="form-label">
                    Weight
                    <strong style={{ marginLeft: 4, color: 'var(--cyan)', fontFamily: 'var(--mono)', fontSize: 11 }}>{overrideWeight.toFixed(1)}</strong>
                  </label>
                  <input type="range" min={0.5} max={5} step={0.1} value={overrideWeight}
                    onChange={e => setOverrideWeight(Number(e.target.value))} className="range-input" />
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={applyOverride}><IconCheck size={12} /> Apply</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setSelectedNode(null)}>Cancel</button>
                </div>
              </div>
            )}

            {error && (
              <div className="status status-error" style={{ fontSize: 12, display: 'flex', gap: 5, alignItems: 'center' }}>
                <IconAlertTriangle size={13} /> {error}
              </div>
            )}
          </>}

          {/* LEGEND TAB */}
          {tab === 'legend' && <>
            <div className="section-label">Node Types</div>
            <div className="legend">
              {Object.entries(NODE_LABELS).map(([type, label]) => (
                <div key={type} className="legend-row">
                  <span className="legend-dot" style={{ background: NODE_COLORS[type] }} />
                  <span style={{ flex: 1 }}>{label}</span>
                  <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
                    {displayNodes.filter(n => n.type === type).length}
                  </span>
                </div>
              ))}
            </div>
            <div className="divider" />
            <div className="section-label">Pipe State</div>
            <div className="legend">
              <div className="legend-row"><span className="legend-line" style={{ background: '#0891b2' }} /><span>Active flow</span></div>
              <div className="legend-row"><span className="legend-line" style={{ background: '#94a3b8' }} /><span>No flow / idle</span></div>
              <div className="legend-row"><span className="legend-line" style={{ background: '#dc2626' }} /><span>Burst / failed</span></div>
              <div className="legend-row"><span className="legend-line" style={{ background: '#d97706' }} /><span>Leak location</span></div>
            </div>
            <div className="divider" />
            <div className="section-label">Pressure</div>
            <div className="legend">
              <div className="legend-row"><span className="legend-dot" style={{ background: '#059669' }} /><span>High (&gt;14 m)</span></div>
              <div className="legend-row"><span className="legend-dot" style={{ background: '#d97706' }} /><span>Medium (7–14 m)</span></div>
              <div className="legend-row"><span className="legend-dot" style={{ background: '#dc2626' }} /><span>Low (&lt;7 m)</span></div>
            </div>
            <div className="divider" />
            <div className="section-label">Display Toggles</div>
            <label className="toggle-row">
              <input type="checkbox" checked={showHeatmap} onChange={e => setShowHeatmap(e.target.checked)} className="toggle-checkbox" />
              <IconThermometer size={13} /> Pressure Heatmap
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={showFlow} onChange={e => setShowFlow(e.target.checked)} className="toggle-checkbox" />
              <IconDroplet size={13} /> Flow Colours
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={showIoT} onChange={e => setShowIoT(e.target.checked)} className="toggle-checkbox" />
              <IconActivity size={13} /> IoT Sensor Feed
            </label>
            {showIoT && wdn && <IotSensorPanel nodes={displayNodes} failedPipes={effectiveFailed} tick={iotTick} simResult={simResult} livePressures={livePressures} liveWarnings={liveWarnings} connected={sseConnected} />}
          </>}

          {/* STATS TAB */}
          {tab === 'stats' && <>
            {!stats && !harpResult && !dinicResult ? (
              <div className="empty-state" style={{ padding: '30px 0' }}>
                <div className="empty-icon" style={{ width: 44, height: 44 }}><IconBarChart size={22} /></div>
                <div className="empty-title" style={{ fontSize: 14 }}>No results yet</div>
                <div className="empty-desc" style={{ fontSize: 12 }}>Run a simulation below to see statistics.</div>
              </div>
            ) : <>
              {stats && (
                <>
                  <div className="section-label">
                    {simResult?.method?.toUpperCase()} · {simResult?.scenario?.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  {([
                    { label: 'Demand Served',         value: stats.served_demand_pct },
                    { label: 'Critical Nodes Served', value: stats.critical_node_service_pct },
                    { label: 'Flow Recovered',        value: stats.recovery_flow_pct },
                  ] as const).map(m => m.value != null && (
                    <div key={m.label} className="kpi" style={{ marginTop: 8 }}>
                      <div className="kpi-label">{m.label}</div>
                      <div className={`kpi-value ${m.value >= 80 ? 'good' : m.value >= 50 ? 'warn' : 'bad'}`}>
                        {m.value.toFixed(1)}%
                      </div>
                      <div className="progress">
                        <div className={`progress-fill ${m.value >= 80 ? 'good' : m.value >= 50 ? 'warn' : 'bad'}`}
                          style={{ width: `${Math.min(m.value, 100)}%` }} />
                      </div>
                    </div>
                  ))}
                  {stats.runtime_ms != null && (
                    <div className="kpi" style={{ marginTop: 8 }}>
                      <div className="kpi-label">Runtime</div>
                      <div className="kpi-value info" style={{ fontFamily: 'var(--mono)' }}>{stats.runtime_ms.toFixed(1)} ms</div>
                    </div>
                  )}
                  {stats.escalated && (
                    <div className="status status-warn" style={{ fontSize: 11, display: 'flex', gap: 5, alignItems: 'center', marginTop: 8 }}>
                      <IconZap size={12} /> Escalated to full graph
                    </div>
                  )}
                </>
              )}

              {harpResult?.stats && dinicResult?.stats && (
                <>
                  <div className="divider" style={{ marginTop: 12 }} />
                  <div className="section-label" style={{ marginBottom: 8 }}>HARP vs Dinic's</div>
                  <table className="data-table" style={{ fontSize: 11 }}>
                    <thead>
                      <tr><th>Metric</th><th style={{ color: 'var(--cyan)' }}>HARP</th><th>Dinic's</th></tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Demand %</td>
                        <td style={{ color: 'var(--cyan)', fontWeight: 700 }}>{harpResult.stats.served_demand_pct.toFixed(1)}%</td>
                        <td>{dinicResult.stats.served_demand_pct.toFixed(1)}%</td>
                      </tr>
                      <tr>
                        <td>Critical %</td>
                        <td style={{ color: 'var(--cyan)', fontWeight: 700 }}>{harpResult.stats.critical_node_service_pct.toFixed(1)}%</td>
                        <td>{dinicResult.stats.critical_node_service_pct.toFixed(1)}%</td>
                      </tr>
                      {harpResult.stats.runtime_ms != null && (
                        <tr>
                          <td>Runtime</td>
                          <td style={{ fontFamily: 'var(--mono)' }}>{harpResult.stats.runtime_ms.toFixed(1)} ms</td>
                          <td style={{ fontFamily: 'var(--mono)' }}>{dinicResult.stats.runtime_ms?.toFixed(1)} ms</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                  {harpResult.stats.served_demand_pct > dinicResult.stats.served_demand_pct && (
                    <div className="status status-success" style={{ fontSize: 11, display: 'flex', gap: 5, alignItems: 'center', marginTop: 8 }}>
                      <IconCheck size={12} /> HARP +{(harpResult.stats.served_demand_pct - dinicResult.stats.served_demand_pct).toFixed(1)}% over Dinic's
                    </div>
                  )}
                </>
              )}
            </>}
          </>}
        </div>

        {/* Run Buttons */}
        {wdn && (
          <div className="panel-footer">
            {effectiveScenario === 'normal' && !(multiFailMode && activeScenarios.size > 1) ? (
              <button className="btn btn-primary btn-block" onClick={() => runSim('none')} disabled={simLoading}>
                {simLoading
                  ? <><span className="spinner" style={{ width: 13, height: 13 }} /> Simulating…</>
                  : <><IconPlay size={13} /> Run Baseline Simulation</>}
              </button>
            ) : (
              <>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => runSim('harp')} disabled={simLoading}>
                    {simLoading ? <span className="spinner" style={{ width: 12, height: 12 }} /> : <IconZap size={12} />} HARP
                  </button>
                  <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => runSim('dinic')} disabled={simLoading}>
                    Dinic's
                  </button>
                </div>
                <button className="btn btn-teal btn-block" onClick={() => runSim('compare')} disabled={simLoading}>
                  <IconSliders size={13} /> Compare Both
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {/* ══ Map ══ */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {!selectedCity && (
          <div className="map-empty-state">
            <div className="map-empty-card">
              <div className="map-empty-icon"><IconGlobe size={36} /></div>
              <div className="map-empty-title">Select a City to Begin</div>
              <div className="map-empty-desc">
                Choose a city to load its water distribution network — hospitals, schools,
                fire stations, reservoirs — then animate water flow step by step and
                simulate failure scenarios.
              </div>
            </div>
          </div>
        )}

        {(loading || simLoading) && (
          <div className="map-loading-overlay"
            style={simLoading ? { background: 'rgba(8,145,178,0.07)', backdropFilter: 'blur(2px)' } : undefined}>
            <div style={{ textAlign: 'center' }}>
              <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3, margin: '0 auto 10px', borderTopColor: 'var(--cyan)' }} />
              <div style={{ fontSize: 13, color: simLoading ? 'var(--cyan)' : 'var(--text-2)', fontWeight: 600 }}>
                {simLoading ? 'Running simulation…' : `Generating ${CITY_CONFIGS[selectedCity]?.displayName}…`}
              </div>
            </div>
          </div>
        )}

        <MapContainer center={[20.5937, 78.9629]} zoom={5}
          style={{ width: '100%', height: '100%' }} zoomControl>
          {/* CARTO's free basemap tiles now require an API key (they return a placeholder tile),
              so use OpenStreetMap's standard tiles, which need none. */}
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            className={isDark ? 'osm-dark-tiles' : undefined}
          />
          {flyTarget && <MapFlyTo center={flyTarget.center} zoom={flyTarget.zoom} />}

          {/* Pipes */}
          {displayEdges.map(edge => {
            const a = nodeMap.get(edge.from), b = nodeMap.get(edge.to)
            if (!a || !b) return null
            const isFailed  = failedPipes.includes(edge.id)
            const isLeak    = leakPipe === edge.id
            const isFlowing = animMode !== 'off' && reachedEdges.has(edge.id)
            return (
              <Polyline key={edge.id}
                positions={[[a.lat, a.lon], [b.lat, b.lon]]}
                pathOptions={{
                  color:     edgeColor(edge),
                  weight:    edgeWeight(edge),
                  opacity:   edgeOpacity(edge),
                  dashArray: isFailed ? '8 4' : isLeak ? '5 3' : undefined,
                  className: isFlowing && !isFailed ? 'flow-active' : '',
                }}
                eventHandlers={{
                  click: () => {
                    if (effectiveScenario === 'burst')         setFailedPipes([edge.id])
                    else if (effectiveScenario === 'multi_failure' || (multiFailMode && activeScenarios.has('burst'))) togglePipe(edge.id)
                    else if (effectiveScenario === 'leak' || (multiFailMode && activeScenarios.has('leak'))) setLeakPipe(edge.id)
                  },
                }}>
                <Tooltip sticky>
                  <div style={{ fontSize: 12, lineHeight: 1.7, minWidth: 155 }}>
                    <strong>{edge.id}</strong><br />
                    {edge.from} → {edge.to}<br />
                    Ø {(edge.diameter * 1000).toFixed(0)} mm · {edge.length.toFixed(0)} m<br />
                    HW-C = {edge.C_hw}
                    {edge.flow != null && <><br />Flow: <strong style={{ color: '#0891b2' }}>{edge.flow.toFixed(5)} m³/s</strong></>}
                    {isFailed && <><br /><span style={{ color: '#dc2626', fontWeight: 700 }}>Burst / Failed</span></>}
                    {isLeak   && <><br /><span style={{ color: '#d97706', fontWeight: 700 }}>Leak Location</span></>}
                    {!isFailed && !isLeak && (scenario !== 'normal') && (
                      <><br /><span style={{ fontSize: 10, color: '#94a3b8' }}>Click to select</span></>
                    )}
                  </div>
                </Tooltip>
              </Polyline>
            )
          })}

          {/* Nodes */}
          {displayNodes.map(node => {
            const isSource      = wdn?.sources?.includes(node.id)
            const isSurge       = surgeNodes.includes(node.id)
            const isNewlyHit    = animMode !== 'off' && reachedNodes.has(node.id)
              && animFrames[animStep]?.nodeIds?.includes(node.id)
            return (
              <CircleMarker key={node.id}
                center={[node.lat, node.lon]}
                radius={(NODE_RADII[node.type] || 7) + (isSurge ? 4 : 0) + (isSource ? 2 : 0)}
                pathOptions={{
                  fillColor:   nodeFill(node),
                  color:       isSource ? '#fbbf24' : isSurge ? '#ea580c' : isNewlyHit ? '#fff' : 'rgba(255,255,255,0.7)',
                  weight:      isSource ? 3 : isNewlyHit ? 2.5 : 1.5,
                  fillOpacity: nodeOpacity(node),
                  className:   isNewlyHit ? 'node-flash' : '',
                }}
                eventHandlers={{
                  click: () => {
                    if (effectiveScenario === 'surge' || (multiFailMode && activeScenarios.has('surge'))) toggleSurgeNode(node.id)
                    else if (priorityMode === 'manual') {
                      setSelectedNode(node); setOverrideType(node.type)
                      setOverrideWeight(node.priority_weight); setTab('control')
                    }
                  },
                }}>
                <Tooltip>
                  <div style={{ fontSize: 12, lineHeight: 1.7, minWidth: 155 }}>
                    <strong>{NODE_LABELS[node.type] || node.type}</strong><br />
                    <code style={{ fontSize: 11 }}>{node.id}</code><br />
                    Demand: {node.demand.toFixed(4)} m³/s<br />
                    Elevation: {node.elevation.toFixed(1)} m<br />
                    Priority: {node.priority_weight.toFixed(1)}
                    {node.pressure != null && (
                      <><br />Pressure: <strong style={{ color: pressureColor(node.pressure) }}>{node.pressure.toFixed(2)} m</strong></>
                    )}
                    {isSource && <><br /><span style={{ color: '#0891b2', fontWeight: 700 }}>Water Source</span></>}
                    {isSurge  && <><br /><span style={{ color: '#ea580c', fontWeight: 700 }}>Surge Node</span></>}
                    {animMode !== 'off' && reachedNodes.has(node.id) && (
                      <><br /><span style={{ color: '#059669', fontWeight: 600 }}>Water reached ✓</span></>
                    )}
                    {priorityMode === 'manual' && !isSurge && scenario === 'normal' && (
                      <><br /><span style={{ fontSize: 10, color: '#94a3b8' }}>Click to override</span></>
                    )}
                  </div>
                </Tooltip>
              </CircleMarker>
            )
          })}
        </MapContainer>

        {/* Top-right info card */}
        {wdn && (
          <div style={{ position: 'absolute', top: 12, right: 12, zIndex: 1000, display: 'flex', flexDirection: 'column', gap: 6, pointerEvents: 'none' }}>
            <div className="map-info-card" style={{ pointerEvents: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                <IconDroplet size={13} />
                <strong style={{ fontSize: 13 }}>{CITY_CONFIGS[selectedCity]?.displayName}</strong>
              </div>
              <div style={{ color: 'var(--text-2)', fontSize: 11 }}>{displayNodes.length} nodes · {displayEdges.length} pipes</div>
              <div style={{ color: 'var(--text-2)', fontSize: 11 }}>{wdn.sources?.length || 0} water source{wdn.sources?.length !== 1 ? 's' : ''}</div>
              {animMode !== 'off' && (
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border)' }}>
                  <div className="progress" style={{ height: 4, marginBottom: 3 }}>
                    <div className="progress-fill good" style={{ width: `${animPct}%` }} />
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--cyan)', fontWeight: 600 }}>
                    {animMode === 'done' ? 'Flow complete' : `Flowing — ${reachedNodes.size}/${displayNodes.length} nodes`}
                  </div>
                </div>
              )}
              {simResult && animMode === 'off' && (
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border)', color: 'var(--green)', fontSize: 11, fontWeight: 600, display: 'flex', gap: 4, alignItems: 'center' }}>
                  <IconCheck size={11} /> {simResult.method?.toUpperCase()} complete
                </div>
              )}
            </div>

            {(effectiveScenario !== 'normal' || (multiFailMode && !activeScenarios.has('normal'))) && (
              <div className={`scenario-badge sc-${effectiveScenario === 'multi_failure' ? 'multi' : effectiveScenario}`} style={{ pointerEvents: 'auto' }}>
                <ScenarioIcon type={effectiveScenario} size={12} />
                {multiFailMode && activeScenarios.size > 1
                  ? `${activeScenarios.size} failures active`
                  : SCENARIOS.find(s => s.value === effectiveScenario)?.label}
                {failedPipes.length > 0 && <> · {failedPipes.length} pipe{failedPipes.length > 1 ? 's' : ''}</>}
                {leakPipe && <> · {leakPipe}</>}
                {surgeNodes.length > 0 && <> · {surgeNodes.length} nodes</>}
              </div>
            )}
          </div>
        )}

        {/* Bottom hint */}
        {wdn && effectiveScenario !== 'normal' && failedPipes.length === 0 && !leakPipe && surgeNodes.length === 0 && (
          <div className="map-hint-bar">
            <IconDroplet size={13} />
            {(effectiveScenario === 'burst' || effectiveScenario === 'multi_failure') && 'Click a pipe on the map, or use the dropdown in the panel'}
            {effectiveScenario === 'leak'  && 'Click a pipe to mark the leak location'}
            {effectiveScenario === 'surge' && 'Click demand nodes to add them to the surge list'}
          </div>
        )}
      </div>
    </div>
  )
}
