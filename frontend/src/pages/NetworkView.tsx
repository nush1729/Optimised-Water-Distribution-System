import { useEffect, useRef, useState, useCallback } from 'react'
import { CITY_CONFIGS, CITY_LIST, SOURCE_TYPES, TYPE_WEIGHT, generateCityWDN, type WDNNode, type WDNEdge, type WDNData } from '../cityData'
import { checkHealth, getIndiaWDN } from '../api'
import { IconDroplet, IconZap, IconNetwork } from '../icons'

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
import type { SharedSimState } from '../types'

const NODE_RADII: Record<string, number> = {
  reservoir: 14, water_tower: 12, hospital: 11, fire_station: 11,
  school: 9, college: 9, market: 8, park: 8, residential: 7, junction: 6,
}
// Built from the (backend-synced) weights so the legend can never drift from them.
const typePriorityLabel = (type: string) =>
  SOURCE_TYPES.includes(type) ? 'source' : `P=${(TYPE_WEIGHT[type] ?? 1).toFixed(1)}`

// Project lat/lon to normalised SVG coords (flip Y since lat goes up)
function projectNodes(nodes: WDNNode[], W: number, H: number, pad = 60) {
  if (nodes.length === 0) return new Map<string, { x: number; y: number }>()
  const lats = nodes.map(n => n.lat), lons = nodes.map(n => n.lon)
  const minLat = Math.min(...lats), maxLat = Math.max(...lats)
  const minLon = Math.min(...lons), maxLon = Math.max(...lons)
  const rangeW = maxLon - minLon || 0.001
  const rangeH = maxLat - minLat || 0.001
  const scaleX = (W - pad * 2) / rangeW
  const scaleY = (H - pad * 2) / rangeH
  const scale  = Math.min(scaleX, scaleY)
  const offX   = pad + ((W - pad * 2) - rangeW * scale) / 2
  const offY   = pad + ((H - pad * 2) - rangeH * scale) / 2
  const result = new Map<string, { x: number; y: number }>()
  for (const n of nodes) {
    result.set(n.id, {
      x: offX + (n.lon - minLon) * scale,
      y: offY + (maxLat - n.lat) * scale,  // flip Y
    })
  }
  return result
}

interface Props { sharedSim: SharedSimState | null }

export default function NetworkView({ sharedSim }: Props) {
  const [selectedCity, setSelectedCity]   = useState(sharedSim?.city || '')
  const [nNodes, setNNodes]               = useState(sharedSim?.wdn?.n_nodes || 20)
  const [wdn, setWdn]                     = useState<WDNData | null>(sharedSim?.wdn || null)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [apiLoading, setApiLoading]       = useState(false)
  const [coords, setCoords]               = useState(new Map<string, { x: number; y: number }>())
  const [hoveredNode, setHoveredNode]     = useState<WDNNode | null>(null)
  const [hoveredEdge, setHoveredEdge]     = useState<WDNEdge | null>(null)
  const [transform, setTransform]         = useState({ x: 0, y: 0, scale: 1 })
  const [dragging, setDragging]           = useState(false)
  const [lastMouse, setLastMouse]         = useState({ x: 0, y: 0 })
  const [colorMode, setColorMode]         = useState<'type' | 'priority' | 'pressure'>('type')
  const [showLabels, setShowLabels]       = useState(true)
  const [filterType, setFilterType]       = useState<string>('all')

  const containerRef = useRef<HTMLDivElement>(null)
  const [svgSize, setSvgSize]             = useState({ w: 800, h: 600 })

  // Measure SVG container
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSvgSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Recompute coords when WDN or SVG size changes
  useEffect(() => {
    if (wdn) setCoords(projectNodes(wdn.nodes, svgSize.w, svgSize.h))
  }, [wdn, svgSize])

  useEffect(() => { checkHealth().then(setBackendOnline) }, [])

  // Load city — show local data instantly, then enhance with real API data
  const loadCity = useCallback(async (city: string, n: number) => {
    if (!city) return
    const local = generateCityWDN(city, n)
    setWdn(local)
    setTransform({ x: 0, y: 0, scale: 1 })
    if (backendOnline) {
      setApiLoading(true)
      try {
        const api = await getIndiaWDN(city, n)
        if (api.nodes.length > 0) setWdn(api)
      } catch { /* local data already shown */ }
      finally { setApiLoading(false) }
    }
  }, [backendOnline])

  useEffect(() => {
    if (selectedCity) loadCity(selectedCity, nNodes)
  }, [selectedCity, nNodes, loadCity])

  // If we have a shared sim result, preload that city
  useEffect(() => {
    if (sharedSim?.city && !selectedCity) {
      setSelectedCity(sharedSim.city)
      setNNodes(sharedSim.wdn.n_nodes)
    }
    // Only preload when the shared result changes; re-running on selectedCity would fight the city picker.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sharedSim])

  // Pan / zoom
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.15 : 0.87
    setTransform(t => ({ ...t, scale: Math.max(0.3, Math.min(10, t.scale * factor)) }))
  }, [])
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    setDragging(true); setLastMouse({ x: e.clientX, y: e.clientY })
  }, [])
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging) return
    setTransform(t => ({ ...t, x: t.x + e.clientX - lastMouse.x, y: t.y + e.clientY - lastMouse.y }))
    setLastMouse({ x: e.clientX, y: e.clientY })
  }, [dragging, lastMouse])
  const onMouseUp = useCallback(() => setDragging(false), [])

  function fitView() {
    setTransform({ x: 0, y: 0, scale: 1 })
  }

  // Node colour based on selected mode
  function nodeColor(node: WDNNode): string {
    if (colorMode === 'priority') {
      const p = node.priority_weight
      if (p >= 4.5) return '#0891b2'
      if (p >= 3.5) return '#7c3aed'
      if (p >= 2.5) return '#d97706'
      return '#94a3b8'
    }
    if (colorMode === 'pressure') {
      const p = sharedSim?.simResult?.nodes.find(n => n.id === node.id)?.pressure
      if (p == null) return '#64748b'
      if (p < 7)  return '#dc2626'
      if (p < 14) return '#d97706'
      return '#059669'
    }
    return NODE_COLORS[node.type] || '#94a3b8'
  }

  const displayNodes = wdn ? (filterType === 'all' ? wdn.nodes : wdn.nodes.filter(n => n.type === filterType)) : []
  const filteredIds  = new Set(displayNodes.map(n => n.id))
  const displayEdges = wdn ? wdn.edges.filter(e => filteredIds.has(e.from) && filteredIds.has(e.to)) : []

  const simNodes = sharedSim?.simResult?.nodes ?? []
  const simNodeMap = new Map(simNodes.map(n => [n.id, n]))

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', background: 'var(--bg)' }}>

      {/* Sidebar */}
      <div style={{ width: 250, flexShrink: 0, background: 'var(--surface)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '13px 15px 11px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
            <div style={{ width: 30, height: 30, borderRadius: 'var(--r)', background: 'var(--cyan-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)' }}>
              <IconNetwork size={15} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>Network Topology</div>
              <div style={{ fontSize: 10, color: 'var(--text-3)' }}>Zoomable graph view</div>
            </div>
            {backendOnline !== null && (
              <span style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase',
                padding: '2px 7px', borderRadius: 20,
                background: backendOnline ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)',
                color: backendOnline ? '#34d399' : '#f87171',
                border: `1px solid ${backendOnline ? '#34d39980' : '#f8717180'}`,
              }}>
                {apiLoading ? 'Loading…' : backendOnline ? 'Live' : 'Local'}
              </span>
            )}
          </div>
        </div>

        <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, borderBottom: '1px solid var(--border)', overflowY: 'auto', flex: 1 }}>
          {/* City */}
          <div className="form-group">
            <label className="form-label">City</label>
            <select className="form-control" value={selectedCity} onChange={e => setSelectedCity(e.target.value)}>
              <option value="">— Select city —</option>
              {CITY_LIST.map(c => <option key={c} value={c}>{CITY_CONFIGS[c].displayName}</option>)}
            </select>
          </div>

          {/* Node count */}
          <div className="form-group">
            <label className="form-label">
              Node Count
              <strong style={{ color: 'var(--cyan)', fontFamily: 'var(--mono)' }}>{nNodes}</strong>
            </label>
            <input type="range" min={15} max={50} step={1} value={nNodes}
              onChange={e => setNNodes(Number(e.target.value))} className="range-input" />
            <div className="range-labels"><span>15</span><span>50</span></div>
          </div>

          {/* Color mode */}
          <div className="form-group">
            <label className="form-label">Colour By</label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {([
                { id: 'type',     label: 'Node Type' },
                { id: 'priority', label: 'Priority Weight' },
                { id: 'pressure', label: 'Pressure (if simulated)', disabled: simNodes.length === 0 },
              ] as const).map(m => (
                <label key={m.id} style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 12, cursor: (m as { disabled?: boolean }).disabled ? 'not-allowed' : 'pointer', opacity: (m as { disabled?: boolean }).disabled ? 0.45 : 1 }}>
                  <input type="radio" name="colorMode" value={m.id}
                    checked={colorMode === m.id} disabled={(m as { disabled?: boolean }).disabled}
                    onChange={() => setColorMode(m.id as 'type' | 'priority' | 'pressure')}
                    style={{ accentColor: 'var(--cyan)' }} />
                  {m.label}
                </label>
              ))}
            </div>
          </div>

          {/* Label toggle */}
          <label className="toggle-row">
            <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} className="toggle-checkbox" />
            <IconDroplet size={13} /> Show Labels
          </label>

          {/* Filter by type */}
          <div className="form-group">
            <label className="form-label">Filter by Type</label>
            <select className="form-control" value={filterType} onChange={e => setFilterType(e.target.value)}>
              <option value="all">All types</option>
              {Object.entries(NODE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </div>

          <div className="divider" />

          {/* Legend */}
          <div className="section-label">Legend</div>
          {colorMode === 'type' && (
            <div className="legend">
              {Object.entries(NODE_LABELS).map(([type, label]) => (
                <div key={type} className="legend-row">
                  <span className="legend-dot" style={{ background: NODE_COLORS[type] }} />
                  <span style={{ flex: 1, fontSize: 11 }}>{label}</span>
                  <span style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
                    {typePriorityLabel(type)}
                  </span>
                </div>
              ))}
            </div>
          )}
          {colorMode === 'priority' && (
            <div className="legend">
              {[
                { label: 'Critical (≥4.5)',  color: '#0891b2' },
                { label: 'High (3.5–4.5)',   color: '#7c3aed' },
                { label: 'Medium (2.5–3.5)', color: '#d97706' },
                { label: 'Low (<2.5)',        color: '#94a3b8' },
              ].map(r => (
                <div key={r.label} className="legend-row">
                  <span className="legend-dot" style={{ background: r.color }} />
                  <span style={{ fontSize: 11 }}>{r.label}</span>
                </div>
              ))}
            </div>
          )}
          {colorMode === 'pressure' && (
            <div className="legend">
              {[
                { label: 'High (≥14 m)',   color: '#059669' },
                { label: 'Medium (7–14)', color: '#d97706' },
                { label: 'Low (<7 m)',     color: '#dc2626' },
                { label: 'No data',        color: '#64748b' },
              ].map(r => (
                <div key={r.label} className="legend-row">
                  <span className="legend-dot" style={{ background: r.color }} />
                  <span style={{ fontSize: 11 }}>{r.label}</span>
                </div>
              ))}
            </div>
          )}

          {/* Hovered info */}
          {(hoveredNode || hoveredEdge) && (
            <>
              <div className="divider" />
              {hoveredNode && (
                <div className="surface-card">
                  <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6, color: nodeColor(hoveredNode) }}>
                    {NODE_LABELS[hoveredNode.type] || hoveredNode.type}
                  </div>
                  <table style={{ fontSize: 11, width: '100%' }}>
                    <tbody>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>ID</td><td style={{ fontFamily: 'var(--mono)' }}>{hoveredNode.id}</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>Priority</td><td>{hoveredNode.priority_weight.toFixed(1)}</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>Demand</td><td style={{ fontFamily: 'var(--mono)' }}>{hoveredNode.demand.toFixed(4)} m³/s</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>Elevation</td><td>{hoveredNode.elevation.toFixed(1)} m</td></tr>
                      {simNodeMap.get(hoveredNode.id)?.pressure != null && (
                        <tr><td style={{ color: 'var(--text-3)', paddingRight: 8 }}>Pressure</td><td style={{ fontFamily: 'var(--mono)', fontWeight: 700 }}>{simNodeMap.get(hoveredNode.id)!.pressure!.toFixed(2)} m</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
              {hoveredEdge && !hoveredNode && (
                <div className="surface-card">
                  <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6, color: 'var(--cyan)' }}>Pipe {hoveredEdge.id}</div>
                  <table style={{ fontSize: 11, width: '100%' }}>
                    <tbody>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>From → To</td><td style={{ fontFamily: 'var(--mono)' }}>{hoveredEdge.from} → {hoveredEdge.to}</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>Diameter</td><td>{(hoveredEdge.diameter * 1000).toFixed(0)} mm</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8, paddingBottom: 3 }}>Length</td><td>{hoveredEdge.length.toFixed(0)} m</td></tr>
                      <tr><td style={{ color: 'var(--text-3)', paddingRight: 8 }}>HW-C</td><td>{hoveredEdge.C_hw}</td></tr>
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>

        {/* Controls */}
        <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border)', display: 'flex', gap: 6 }}>
          <button className="btn btn-ghost btn-sm" style={{ flex: 1 }} onClick={fitView} disabled={!wdn}>Reset View</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setTransform(t => ({ ...t, scale: Math.min(10, t.scale * 1.3) }))} disabled={!wdn}>+</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setTransform(t => ({ ...t, scale: Math.max(0.3, t.scale * 0.77) }))} disabled={!wdn}>−</button>
        </div>
      </div>

      {/* SVG canvas */}
      <div ref={containerRef}
        style={{ flex: 1, position: 'relative', overflow: 'hidden', cursor: dragging ? 'grabbing' : 'grab', background: 'var(--bg)' }}
        onWheel={onWheel} onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>

        {!selectedCity && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 12, pointerEvents: 'none' }}>
            <div style={{ width: 60, height: 60, borderRadius: 'var(--r-xl)', background: 'var(--cyan-dim)', border: '1px solid var(--cyan-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)' }}>
              <IconNetwork size={28} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>Select a City</div>
            <div style={{ fontSize: 13, color: 'var(--text-3)' }}>Explore the network topology as a graph</div>
          </div>
        )}

        {wdn && (
          <svg width={svgSize.w} height={svgSize.h} style={{ display: 'block', position: 'absolute', inset: 0 }}>
            <defs>
              <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#0891b2" opacity="0.6" />
              </marker>
            </defs>
            <g transform={`translate(${transform.x},${transform.y}) scale(${transform.scale})`}>
              {/* Edges */}
              {displayEdges.map(edge => {
                const a = coords.get(edge.from), b = coords.get(edge.to)
                if (!a || !b) return null
                const isHovered = hoveredEdge?.id === edge.id
                const isSource = wdn.sources.includes(edge.from) || wdn.sources.includes(edge.to)
                const w = Math.max(1, edge.diameter * 25)
                const simEdge = sharedSim?.simResult?.edges?.find(e => e.id === edge.id)
                const hasFlow = (simEdge?.flow ?? 0) > 0.001
                return (
                  <line key={edge.id}
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={isHovered ? '#0891b2' : hasFlow ? '#0891b2' : isSource ? '#7c3aed44' : 'var(--border)'}
                    strokeWidth={isHovered ? w + 1.5 : w}
                    strokeOpacity={isHovered ? 1 : hasFlow ? 0.65 : 0.5}
                    strokeDasharray={hasFlow ? undefined : undefined}
                    style={{ cursor: 'pointer', transition: 'stroke 0.15s, stroke-opacity 0.15s' }}
                    onMouseEnter={() => setHoveredEdge(edge)}
                    onMouseLeave={() => setHoveredEdge(null)}
                  />
                )
              })}

              {/* Nodes */}
              {displayNodes.map(node => {
                const pos = coords.get(node.id)
                if (!pos) return null
                const r = (NODE_RADII[node.type] || 7) / transform.scale
                const isHovered = hoveredNode?.id === node.id
                const isSource  = wdn.sources.includes(node.id)
                const fill      = nodeColor(node)
                const simNode   = simNodeMap.get(node.id)
                return (
                  <g key={node.id} style={{ cursor: 'pointer' }}
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}>
                    {/* Outer glow ring for sources */}
                    {isSource && (
                      <circle cx={pos.x} cy={pos.y} r={r + 4 / transform.scale}
                        fill="none" stroke="#fbbf24" strokeWidth={2 / transform.scale} strokeOpacity={0.5}
                        strokeDasharray={`${3 / transform.scale} ${3 / transform.scale}`} />
                    )}
                    <circle cx={pos.x} cy={pos.y} r={r}
                      fill={fill}
                      stroke={isHovered ? '#fff' : isSource ? '#fbbf24' : 'rgba(255,255,255,0.55)'}
                      strokeWidth={(isHovered ? 2.5 : 1.5) / transform.scale}
                      fillOpacity={0.92}
                      style={{ transition: 'fill 0.2s, r 0.1s' }} />
                    {/* Pressure indicator ring */}
                    {simNode?.pressure != null && colorMode !== 'pressure' && (
                      <circle cx={pos.x} cy={pos.y} r={r + 2 / transform.scale}
                        fill="none"
                        stroke={simNode.pressure < 7 ? '#dc2626' : simNode.pressure < 14 ? '#d97706' : '#059669'}
                        strokeWidth={1.5 / transform.scale} strokeOpacity={0.7} />
                    )}
                    {/* Label */}
                    {showLabels && transform.scale > 0.7 && (
                      <text x={pos.x} y={pos.y + r + 10 / transform.scale}
                        textAnchor="middle" fontSize={9 / transform.scale}
                        fill="var(--text-2)" style={{ pointerEvents: 'none', userSelect: 'none' }}>
                        {node.id}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          </svg>
        )}

        {/* Zoom badge */}
        {wdn && (
          <div style={{
            position: 'absolute', bottom: 12, right: 12,
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--r)', padding: '4px 12px',
            fontSize: 11, color: 'var(--text-2)', display: 'flex', gap: 10,
          }}>
            <span>{Math.round(transform.scale * 100)}%</span>
            <span style={{ color: 'var(--border)' }}>|</span>
            <span>{displayNodes.length} nodes · {displayEdges.length} pipes</span>
            {sharedSim?.simResult && <span style={{ color: 'var(--cyan)' }}>· sim overlay active</span>}
          </div>
        )}

        {/* Sim overlay banner */}
        {sharedSim?.simResult && colorMode === 'pressure' && (
          <div style={{
            position: 'absolute', top: 12, right: 12,
            background: 'var(--glass)', border: '1px solid var(--glass-border)',
            backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
            borderRadius: 'var(--r)', padding: '7px 12px',
            fontSize: 11, color: 'var(--text-2)', display: 'flex', gap: 6, alignItems: 'center',
          }}>
            <IconZap size={12} />
            Showing pressure from last {sharedSim.simResult.method?.toUpperCase()} simulation
          </div>
        )}
      </div>
    </div>
  )
}
