import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  IconBarChart, IconActivity, IconCheck, IconAlertTriangle,
  IconTrendingUp, IconDroplet, IconZap, IconPlay, IconThermometer,
} from '../icons'
import type { SharedSimState } from '../types'

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

// Colour helpers
function pressureColor(p: number) {
  if (p < 7)  return '#dc2626'
  if (p < 14) return '#d97706'
  return '#059669'
}

interface Props { sharedSim: SharedSimState | null }

export default function ResultsView({ sharedSim }: Props) {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'overview' | 'nodes' | 'compare' | 'analytics'>('overview')

  if (!sharedSim) {
    return (
      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
        <div style={{ textAlign: 'center', maxWidth: 400, padding: 40 }}>
          <div style={{ width: 72, height: 72, borderRadius: '50%', background: 'var(--cyan-dim)', border: '1px solid var(--cyan-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)', margin: '0 auto 20px' }}>
            <IconBarChart size={32} />
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 10, color: 'var(--text)' }}>No Simulation Run Yet</div>
          <div style={{ fontSize: 14, color: 'var(--text-2)', lineHeight: 1.7, marginBottom: 24 }}>
            Go to the Simulation tab, select a city, pick a scenario, and run HARP or Dinic's algorithm.
            Results will appear here automatically.
          </div>
          <button className="btn btn-primary" onClick={() => navigate('/')}>
            <IconPlay size={14} /> Go to Simulation
          </button>
        </div>
      </div>
    )
  }

  const { city, wdn, simResult, harpResult, dinicResult, scenario, failedPipes, runAt } = sharedSim
  const stats  = simResult?.stats
  const hStats = harpResult?.stats
  const dStats = dinicResult?.stats
  const hasCompare = !!(hStats && dStats)
  const runDate = new Date(runAt).toLocaleTimeString()

  const nodes = simResult?.nodes ?? wdn.nodes
  const edges = simResult?.edges ?? wdn.edges

  // Node stats
  const withPressure = nodes.filter(n => n.pressure != null)
  const lowPressure  = withPressure.filter(n => n.pressure! < 7)
  const highPressure = withPressure.filter(n => n.pressure! >= 14)
  const activeEdges  = edges.filter(e => (e.flow ?? 0) > 0.001 && !failedPipes.includes(e.id))
  const failedEdges  = edges.filter(e => failedPipes.includes(e.id))

  // Sort nodes: highest priority first, then by pressure asc (most at-risk first within tier)
  const sortedNodes = [...nodes].sort((a, b) => {
    const dp = b.priority_weight - a.priority_weight
    if (dp !== 0) return dp
    return (a.pressure ?? 99) - (b.pressure ?? 99)
  })

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', background: 'var(--bg)' }}>

      {/* ── Sidebar ── */}
      <div style={{ width: 240, flexShrink: 0, background: 'var(--surface)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '13px 15px 11px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <div style={{ width: 32, height: 32, borderRadius: 'var(--r)', background: 'var(--cyan-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)', border: '1px solid var(--cyan-glow)' }}>
              <IconActivity size={16} />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700 }}>Simulation Results</div>
              <div style={{ fontSize: 10, color: 'var(--text-3)' }}>Run at {runDate}</div>
            </div>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <span className="pill pill-cyan" style={{ fontSize: 10 }}>{city.charAt(0).toUpperCase() + city.slice(1)}</span>
            <span className="pill pill-gray" style={{ fontSize: 10 }}>{scenario.replace(/_/g, ' ')}</span>
            {simResult?.method && <span className="pill pill-green" style={{ fontSize: 10 }}>{simResult.method.toUpperCase()}</span>}
          </div>
        </div>

        {/* KPIs */}
        <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {stats && ([
            { label: 'Demand Served',  value: stats.served_demand_pct,           suffix: '%' },
            { label: 'Critical Nodes', value: stats.critical_node_service_pct,    suffix: '%' },
            { label: 'Runtime',        value: stats.runtime_ms,                   suffix: ' ms', mono: true },
          ] as const).map(m => m.value != null && (
            <div key={m.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{m.label}</span>
              <span style={{
                fontSize: 13, fontWeight: 700,
                fontFamily: (m as any).mono ? 'var(--mono)' : undefined,
                color: (m as any).mono ? 'var(--cyan)'
                  : m.value >= 80 ? 'var(--green)' : m.value >= 50 ? 'var(--amber)' : 'var(--red)',
              }}>
                {m.value.toFixed(1)}{m.suffix}
              </span>
            </div>
          ))}
        </div>

        {/* Network summary */}
        <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 8 }}>Network</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {[
              { label: 'Nodes',   value: nodes.length },
              { label: 'Pipes',   value: edges.length },
              { label: 'Active',  value: activeEdges.length },
              { label: 'Failed',  value: failedEdges.length },
            ].map(r => (
              <div key={r.label} className="kpi" style={{ padding: '6px 8px' }}>
                <div className="kpi-label" style={{ fontSize: 9 }}>{r.label}</div>
                <div className="kpi-value info" style={{ fontSize: 16 }}>{r.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Tab nav */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
          {([
            { id: 'overview',  label: 'Overview' },
            { id: 'nodes',     label: 'Nodes' },
            { id: 'analytics', label: 'Charts' },
            { id: 'compare',   label: 'Compare' },
          ] as const).map(t => (
            <button key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                flex: 1, border: 'none', padding: '8px 2px', fontSize: 10, fontWeight: 700,
                letterSpacing: 0.3, textTransform: 'uppercase', cursor: 'pointer',
                fontFamily: 'var(--font)', transition: 'all var(--t-fast)',
                background: activeTab === t.id ? 'var(--cyan-dim)' : 'transparent',
                color: activeTab === t.id ? 'var(--cyan)' : 'var(--text-3)',
                borderBottom: activeTab === t.id ? '2px solid var(--cyan)' : '2px solid transparent',
              }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Run new */}
        <div style={{ padding: 10, marginTop: 'auto', borderTop: '1px solid var(--border)' }}>
          <button className="btn btn-primary btn-block btn-sm" onClick={() => navigate('/')}>
            <IconPlay size={12} /> New Simulation
          </button>
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>

        {/* OVERVIEW */}
        {activeTab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 900 }}>

            {/* Big KPI row */}
            {stats && (
              <div className="kpi-grid kpi-4">
                {[
                  { label: 'Demand Served',         value: stats.served_demand_pct,           suffix: '%', good: 80, warn: 50 },
                  { label: 'Critical Node Service', value: stats.critical_node_service_pct,    suffix: '%', good: 80, warn: 60 },
                  { label: 'Flow Recovered',        value: stats.recovery_flow_pct,            suffix: '%', good: 80, warn: 50 },
                  { label: 'Runtime',               value: stats.runtime_ms,                   suffix: ' ms', time: true },
                ].filter(m => m.value != null).map(m => (
                  <div key={m.label} className="kpi">
                    <div className="kpi-label">{m.label}</div>
                    <div className={`kpi-value ${(m as any).time ? 'info' : m.value! >= m.good! ? 'good' : m.value! >= m.warn! ? 'warn' : 'bad'}`}
                      style={{ fontFamily: (m as any).time ? 'var(--mono)' : undefined }}>
                      {m.value!.toFixed(1)}{m.suffix}
                    </div>
                    {!(m as any).time && (
                      <div className="progress">
                        <div className={`progress-fill ${m.value! >= m.good! ? 'good' : m.value! >= m.warn! ? 'warn' : 'bad'}`}
                          style={{ width: `${Math.min(m.value!, 100)}%` }} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Pressure distribution */}
            {withPressure.length > 0 && (
              <div className="card">
                <div className="card-hd"><IconThermometer size={13} /> Pressure Distribution</div>
                <div className="card-body">
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
                    {[
                      { label: 'High Pressure', count: highPressure.length, color: 'var(--green)', desc: '≥ 14 m' },
                      { label: 'Medium Pressure', count: withPressure.length - lowPressure.length - highPressure.length, color: 'var(--amber)', desc: '7 – 14 m' },
                      { label: 'Low Pressure', count: lowPressure.length, color: 'var(--red)', desc: '< 7 m' },
                    ].map(b => (
                      <div key={b.label} style={{ padding: '12px 14px', background: 'var(--surface-2)', borderRadius: 'var(--r)', textAlign: 'center', border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: 26, fontWeight: 800, color: b.color, fontVariantNumeric: 'tabular-nums' }}>{b.count}</div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginTop: 2 }}>{b.label}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-3)' }}>{b.desc}</div>
                      </div>
                    ))}
                  </div>
                  {/* Pressure bar chart per node type */}
                  {Object.entries(
                    nodes.reduce((acc: Record<string, number[]>, n) => {
                      if (n.pressure != null) {
                        if (!acc[n.type]) acc[n.type] = []
                        acc[n.type].push(n.pressure)
                      }
                      return acc
                    }, {}),
                  ).map(([type, pressures]) => {
                    const avg = pressures.reduce((s, v) => s + v, 0) / pressures.length
                    return (
                      <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                        <span style={{ width: 90, fontSize: 11, color: 'var(--text-2)', textAlign: 'right', flexShrink: 0 }}>
                          {NODE_LABELS[type] || type}
                        </span>
                        <div style={{ flex: 1, height: 8, background: 'var(--surface-2)', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{
                            height: '100%', borderRadius: 4, transition: 'width 0.6s ease',
                            width: `${Math.min((avg / 25) * 100, 100)}%`,
                            background: pressureColor(avg),
                          }} />
                        </div>
                        <span style={{ width: 52, fontSize: 11, color: pressureColor(avg), fontWeight: 700, fontFamily: 'var(--mono)', flexShrink: 0 }}>
                          {avg.toFixed(1)} m
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Flow summary */}
            {activeEdges.length > 0 && (
              <div className="card">
                <div className="card-hd"><IconDroplet size={13} /> Pipe Flow Summary</div>
                <div className="card-body">
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                    {[
                      { label: 'Active Pipes',  value: activeEdges.length,  color: 'var(--cyan)' },
                      { label: 'Failed Pipes',  value: failedEdges.length,  color: 'var(--red)'  },
                      {
                        label: 'Avg Flow',
                        value: (activeEdges.reduce((s, e) => s + (e.flow ?? 0), 0) / (activeEdges.length || 1) * 1000).toFixed(2) + ' L/s',
                        color: 'var(--teal)',
                      },
                    ].map(b => (
                      <div key={b.label} style={{ padding: '12px 14px', background: 'var(--surface-2)', borderRadius: 'var(--r)', border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: 22, fontWeight: 800, color: b.color, fontFamily: 'var(--mono)' }}>{b.value}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 4 }}>{b.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Escalation notice */}
            {stats?.escalated && (
              <div className="status status-warn" style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 13 }}>
                <IconZap size={15} />
                <div>
                  <strong>Escalated to full graph.</strong> The initial subgraph CDS fell below 0.6, so HARP re-ran on the full network with <code>escalation_flag=True</code>.
                </div>
              </div>
            )}
          </div>
        )}

        {/* NODES */}
        {activeTab === 'nodes' && (
          <div style={{ maxWidth: 960 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, fontSize: 15 }}>Node Status</div>
              <span className="pill pill-cyan">{nodes.length} nodes</span>
              {lowPressure.length > 0 && (
                <span className="pill pill-red" style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <IconAlertTriangle size={10} /> {lowPressure.length} low pressure
                </span>
              )}
              {simResult?.pressure_uncertainty_m && (
                <span className="pill pill-gray" style={{ fontSize: 10 }}>GAT confidence shown</span>
              )}
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Priority</th>
                    <th>Demand m³/s</th>
                    <th>Elevation m</th>
                    <th>Pressure m</th>
                    {simResult?.pressure_uncertainty_m && <th>Confidence</th>}
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedNodes.map(node => {
                    const std = simResult?.pressure_uncertainty_m?.[node.id]
                    const confLabel = std == null ? null : std < 2 ? 'High' : std < 5 ? 'Med' : 'Low'
                    const confColor = std == null ? '' : std < 2 ? 'var(--green)' : std < 5 ? 'var(--amber)' : 'var(--red)'
                    return (
                      <tr key={node.id}>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--cyan)' }}>{node.id}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: NODE_COLORS[node.type] || '#94a3b8', display: 'inline-block', flexShrink: 0 }} />
                            {NODE_LABELS[node.type] || node.type}
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ width: Math.round(node.priority_weight / 5 * 40), height: 5, background: 'var(--cyan)', borderRadius: 3, minWidth: 2 }} />
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{node.priority_weight.toFixed(1)}</span>
                          </div>
                        </td>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{node.demand.toFixed(4)}</td>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{node.elevation.toFixed(1)}</td>
                        <td>
                          {node.pressure != null ? (
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color: pressureColor(node.pressure) }}>
                              {node.pressure.toFixed(2)}
                              {std != null && <span style={{ fontSize: 9, color: 'var(--text-3)', fontWeight: 400, marginLeft: 3 }}>±{std.toFixed(1)}</span>}
                            </span>
                          ) : <span style={{ color: 'var(--text-3)', fontSize: 11 }}>—</span>}
                        </td>
                        {simResult?.pressure_uncertainty_m && (
                          <td>
                            {confLabel != null ? (
                              <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                <div style={{ width: 28, height: 5, borderRadius: 3, background: 'var(--surface-2)', overflow: 'hidden' }}>
                                  <div style={{ height: '100%', borderRadius: 3, background: confColor, width: confLabel === 'High' ? '100%' : confLabel === 'Med' ? '55%' : '20%' }} />
                                </div>
                                <span style={{ fontSize: 10, color: confColor, fontWeight: 700 }}>{confLabel}</span>
                              </div>
                            ) : <span style={{ color: 'var(--text-3)', fontSize: 11 }}>—</span>}
                          </td>
                        )}
                        <td>
                          {node.pressure == null
                            ? <span className="pill pill-gray" style={{ fontSize: 9 }}>No data</span>
                            : node.pressure < 7
                              ? <span className="pill pill-red" style={{ fontSize: 9, display: 'inline-flex', gap: 3, alignItems: 'center' }}><IconAlertTriangle size={9} /> Low</span>
                              : node.pressure >= 14
                                ? <span className="pill pill-green" style={{ fontSize: 9, display: 'inline-flex', gap: 3, alignItems: 'center' }}><IconCheck size={9} /> High</span>
                                : <span className="pill pill-amber" style={{ fontSize: 9 }}>Med</span>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* COMPARE */}
        {activeTab === 'compare' && (
          <div style={{ maxWidth: 860 }}>
            {!hasCompare ? (
              <div className="empty-state">
                <div className="empty-icon"><IconTrendingUp size={28} /></div>
                <div className="empty-title">No Comparison Available</div>
                <div className="empty-desc">
                  Go to Simulation, pick a failure scenario, and click <strong>Compare Both</strong> to run HARP and Dinic's back-to-back.
                </div>
                <button className="btn btn-primary" style={{ marginTop: 4 }} onClick={() => navigate('/')}>
                  <IconPlay size={13} /> Go to Simulation
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                {/* Header comparison */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  {([
                    { label: 'HARP', color: 'var(--cyan)', stats: hStats!, result: harpResult },
                    { label: "Dinic's", color: 'var(--purple)', stats: dStats!, result: dinicResult },
                  ] as const).map(col => (
                    <div key={col.label} style={{ background: 'var(--surface)', border: `1px solid var(--border)`, borderRadius: 'var(--r-xl)', overflow: 'hidden' }}>
                      <div style={{ padding: '12px 16px', background: `color-mix(in srgb, ${col.color} 12%, var(--surface-2))`, borderBottom: '1px solid var(--border)', fontWeight: 800, fontSize: 14, color: col.color }}>
                        {col.label}
                      </div>
                      <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {[
                          { label: 'Demand Served',   value: col.stats.served_demand_pct,          suffix: '%' },
                          { label: 'Critical Nodes',  value: col.stats.critical_node_service_pct,   suffix: '%' },
                          { label: 'Flow Recovered',  value: col.stats.recovery_flow_pct,           suffix: '%' },
                          { label: 'Runtime',         value: col.stats.runtime_ms,                  suffix: ' ms', time: true },
                        ].filter(m => m.value != null).map(m => (
                          <div key={m.label} className="kpi" style={{ padding: 0 }}>
                            <div className="kpi-label">{m.label}</div>
                            <div className={`kpi-value ${(m as any).time ? 'info' : m.value! >= 80 ? 'good' : m.value! >= 50 ? 'warn' : 'bad'}`}
                              style={{ fontFamily: (m as any).time ? 'var(--mono)' : undefined, fontSize: 20 }}>
                              {m.value!.toFixed(1)}{m.suffix}
                            </div>
                            {!(m as any).time && (
                              <div className="progress">
                                <div className={`progress-fill ${m.value! >= 80 ? 'good' : m.value! >= 50 ? 'warn' : 'bad'}`}
                                  style={{ width: `${Math.min(m.value!, 100)}%` }} />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Delta summary */}
                <div className="card">
                  <div className="card-hd"><IconTrendingUp size={13} /> HARP Advantage over Dinic's</div>
                  <div className="card-body">
                    {[
                      { label: 'Demand Served',    hv: hStats!.served_demand_pct,         dv: dStats!.served_demand_pct,         suffix: '%' },
                      { label: 'Critical Nodes',   hv: hStats!.critical_node_service_pct,  dv: dStats!.critical_node_service_pct,  suffix: '%' },
                      { label: 'Runtime',          hv: hStats!.runtime_ms ?? 0,            dv: dStats!.runtime_ms ?? 0,            suffix: ' ms', lowerIsBetter: true },
                    ].map(row => {
                      const delta = row.lowerIsBetter ? row.dv - row.hv : row.hv - row.dv
                      const isGood = delta > 0
                      return (
                        <div key={row.label} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border-2)' }}>
                          <span style={{ width: 120, fontSize: 12, color: 'var(--text-2)' }}>{row.label}</span>
                          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--cyan)', fontWeight: 700, width: 70 }}>
                              {row.hv.toFixed(1)}{row.suffix}
                            </span>
                            <span style={{ fontSize: 10, color: 'var(--text-3)' }}>vs</span>
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, width: 70 }}>{row.dv.toFixed(1)}{row.suffix}</span>
                          </div>
                          <span style={{
                            fontSize: 12, fontWeight: 700, fontFamily: 'var(--mono)',
                            color: isGood ? 'var(--green)' : 'var(--red)',
                          }}>
                            {isGood ? '+' : ''}{delta.toFixed(1)}{row.suffix}
                          </span>
                        </div>
                      )
                    })}
                    <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--cyan-dim)', borderRadius: 'var(--r)', color: 'var(--cyan)', fontSize: 12, fontWeight: 600 }}>
                      {hStats!.served_demand_pct > dStats!.served_demand_pct
                        ? `HARP serves ${(hStats!.served_demand_pct - dStats!.served_demand_pct).toFixed(1)}% more demand — GAT pressure awareness gives a significant advantage over Dinic's baseline.`
                        : hStats!.served_demand_pct === dStats!.served_demand_pct
                          ? 'Both algorithms achieve the same demand satisfaction on this scenario.'
                          : `Dinic's outperforms HARP by ${(dStats!.served_demand_pct - hStats!.served_demand_pct).toFixed(1)}% — consider adjusting the GAT model parameters.`}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        {/* ANALYTICS */}
        {activeTab === 'analytics' && (
          <div style={{ maxWidth: 900, display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Node Pressure Chart */}
            {nodes.filter(n => n.pressure != null).length > 0 && (
              <div className="card">
                <div className="card-hd"><IconThermometer size={13} /> Node Pressure (m) — All Nodes</div>
                <div className="card-body">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 340, overflowY: 'auto' }}>
                    {[...nodes].filter(n => n.pressure != null)
                      .sort((a, b) => (b.pressure ?? 0) - (a.pressure ?? 0))
                      .map(node => {
                        const pct = Math.min(((node.pressure ?? 0) / 30) * 100, 100)
                        const col = (node.pressure ?? 0) < 7 ? '#dc2626' : (node.pressure ?? 0) < 14 ? '#d97706' : '#059669'
                        return (
                          <div key={node.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: NODE_COLORS[node.type] || '#94a3b8', flexShrink: 0 }} />
                            <span style={{ width: 44, fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--mono)', flexShrink: 0 }}>{node.id}</span>
                            <div style={{ flex: 1, height: 10, background: 'var(--surface-2)', borderRadius: 5, overflow: 'hidden' }}>
                              <div style={{ height: '100%', borderRadius: 5, background: col, width: `${pct}%`, transition: 'width 0.5s ease' }} />
                            </div>
                            <span style={{ width: 50, fontSize: 11, fontWeight: 700, fontFamily: 'var(--mono)', color: col, textAlign: 'right', flexShrink: 0 }}>
                              {(node.pressure ?? 0).toFixed(1)} m
                            </span>
                          </div>
                        )
                      })}
                  </div>
                  <div style={{ display: 'flex', gap: 14, marginTop: 12, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
                    {[['#059669', '≥ 14 m High'], ['#d97706', '7–14 m Med'], ['#dc2626', '< 7 m Low']].map(([c, l]) => (
                      <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: 'var(--text-2)' }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />{l}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Flow Rate Chart */}
            {edges.filter(e => (e.flow ?? 0) > 0).length > 0 && (
              <div className="card">
                <div className="card-hd"><IconDroplet size={13} /> Pipe Flow Rates (L/s)</div>
                <div className="card-body">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5, maxHeight: 280, overflowY: 'auto' }}>
                    {[...edges].filter(e => (e.flow ?? 0) > 0)
                      .sort((a, b) => (b.flow ?? 0) - (a.flow ?? 0))
                      .slice(0, 20)
                      .map(edge => {
                        const flowLs = (edge.flow ?? 0) * 1000
                        const maxFlow = Math.max(...edges.map(e => (e.flow ?? 0))) * 1000
                        const pct = maxFlow > 0 ? (flowLs / maxFlow) * 100 : 0
                        const isFailed = failedPipes.includes(edge.id)
                        return (
                          <div key={edge.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ width: 36, fontSize: 10, color: isFailed ? 'var(--red)' : 'var(--text-2)', fontFamily: 'var(--mono)', flexShrink: 0 }}>{edge.id}</span>
                            <div style={{ flex: 1, height: 8, background: 'var(--surface-2)', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{ height: '100%', borderRadius: 4, background: isFailed ? '#dc2626' : 'var(--cyan)', width: `${pct}%`, transition: 'width 0.5s ease' }} />
                            </div>
                            <span style={{ width: 54, fontSize: 10, fontFamily: 'var(--mono)', color: isFailed ? 'var(--red)' : 'var(--teal)', textAlign: 'right', flexShrink: 0 }}>
                              {isFailed ? 'FAILED' : flowLs.toFixed(2) + ' L/s'}
                            </span>
                          </div>
                        )
                      })}
                  </div>
                </div>
              </div>
            )}

            {/* HARP vs Dinic's Visual Bar Chart */}
            {hasCompare && hStats && dStats && (
              <div className="card">
                <div className="card-hd"><IconTrendingUp size={13} /> HARP vs Dinic's — Visual Comparison</div>
                <div className="card-body">
                  {[
                    { label: 'Demand Served (%)',        h: hStats.served_demand_pct,        d: dStats.served_demand_pct,        max: 100 },
                    { label: 'Critical Node Service (%)', h: hStats.critical_node_service_pct, d: dStats.critical_node_service_pct, max: 100 },
                    { label: 'Flow Recovered (%)',        h: hStats.recovery_flow_pct ?? 0,   d: dStats.recovery_flow_pct ?? 0,   max: 100 },
                  ].map(row => {
                    const hPct = (row.h / row.max) * 100
                    const dPct = (row.d / row.max) * 100
                    const hWins = row.h >= row.d
                    return (
                      <div key={row.label} style={{ marginBottom: 18 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-2)', fontWeight: 600 }}>{row.label}</span>
                          <span style={{ fontSize: 11, fontWeight: 700, color: hWins ? 'var(--green)' : 'var(--amber)' }}>
                            {hWins ? `HARP +${(row.h - row.d).toFixed(1)}%` : `Dinic's +${(row.d - row.h).toFixed(1)}%`}
                          </span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ width: 52, fontSize: 10, color: 'var(--cyan)', fontWeight: 700, flexShrink: 0 }}>HARP</span>
                            <div style={{ flex: 1, height: 14, background: 'var(--surface-2)', borderRadius: 7, overflow: 'hidden', position: 'relative' }}>
                              <div style={{ height: '100%', borderRadius: 7, background: 'linear-gradient(90deg, #0891b2, #06b6d4)', width: `${hPct}%`, transition: 'width 0.6s ease' }} />
                            </div>
                            <span style={{ width: 42, fontSize: 11, fontFamily: 'var(--mono)', fontWeight: 700, color: 'var(--cyan)', textAlign: 'right', flexShrink: 0 }}>{row.h.toFixed(1)}%</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ width: 52, fontSize: 10, color: 'var(--purple)', fontWeight: 700, flexShrink: 0 }}>Dinic's</span>
                            <div style={{ flex: 1, height: 14, background: 'var(--surface-2)', borderRadius: 7, overflow: 'hidden' }}>
                              <div style={{ height: '100%', borderRadius: 7, background: 'linear-gradient(90deg, #7c3aed, #a855f7)', width: `${dPct}%`, transition: 'width 0.6s ease' }} />
                            </div>
                            <span style={{ width: 42, fontSize: 11, fontFamily: 'var(--mono)', color: '#a855f7', textAlign: 'right', flexShrink: 0 }}>{row.d.toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                  {/* Runtime comparison (lower is better) */}
                  {hStats.runtime_ms != null && dStats.runtime_ms != null && (
                    <div style={{ marginTop: 4, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 'var(--r)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Runtime (lower is better)</span>
                      <div style={{ display: 'flex', gap: 16 }}>
                        <span style={{ fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--cyan)', fontWeight: 700 }}>HARP: {hStats.runtime_ms.toFixed(1)} ms</span>
                        <span style={{ fontSize: 12, fontFamily: 'var(--mono)', color: '#a855f7' }}>Dinic's: {dStats.runtime_ms.toFixed(1)} ms</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* GAT Uncertainty Chart */}
            {simResult?.pressure_uncertainty_m && Object.keys(simResult.pressure_uncertainty_m).length > 0 && (
              <div className="card">
                <div className="card-hd"><IconThermometer size={13} /> GAT Prediction Confidence — Per Node</div>
                <div className="card-body">
                  <div style={{ display: 'flex', gap: 16, marginBottom: 10 }}>
                    {[['var(--green)', '< 2 m std', 'High confidence'], ['var(--amber)', '2–5 m std', 'Moderate'], ['var(--red)', '> 5 m std', 'Uncertain']].map(([c, range, label]) => (
                      <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: 'var(--text-2)' }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />
                        <span style={{ color: c, fontWeight: 700 }}>{label}</span>
                        <span style={{ color: 'var(--text-3)' }}>({range})</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5, maxHeight: 300, overflowY: 'auto' }}>
                    {Object.entries(simResult.pressure_uncertainty_m)
                      .sort(([, a], [, b]) => b - a)
                      .map(([nodeId, std]) => {
                        const maxStd = Math.max(...Object.values(simResult.pressure_uncertainty_m!))
                        const pct = maxStd > 0 ? (std / maxStd) * 100 : 0
                        const col = std < 2 ? 'var(--green)' : std < 5 ? 'var(--amber)' : 'var(--red)'
                        const nodeType = nodes.find(n => n.id === nodeId)?.type ?? 'junction'
                        return (
                          <div key={nodeId} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: NODE_COLORS[nodeType] || '#94a3b8', flexShrink: 0 }} />
                            <span style={{ width: 44, fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--mono)', flexShrink: 0 }}>{nodeId}</span>
                            <div style={{ flex: 1, height: 10, background: 'var(--surface-2)', borderRadius: 5, overflow: 'hidden' }}>
                              <div style={{ height: '100%', borderRadius: 5, background: col, width: `${pct}%`, transition: 'width 0.5s ease' }} />
                            </div>
                            <span style={{ width: 54, fontSize: 11, fontWeight: 700, fontFamily: 'var(--mono)', color: col, textAlign: 'right', flexShrink: 0 }}>
                              ±{std.toFixed(2)} m
                            </span>
                          </div>
                        )
                      })}
                  </div>
                  <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 'var(--r)', fontSize: 11, color: 'var(--text-2)' }}>
                    Uncertainty estimated via MC-Dropout (20 stochastic forward passes). High uncertainty nodes should be verified with field sensors before operational decisions.
                  </div>
                </div>
              </div>
            )}

            {/* Demand by node type chart */}
            {nodes.length > 0 && (
              <div className="card">
                <div className="card-hd"><IconBarChart size={13} /> Demand Distribution by Node Type</div>
                <div className="card-body">
                  {(() => {
                    const byType = nodes.reduce((acc: Record<string, number>, n) => {
                      acc[n.type] = (acc[n.type] ?? 0) + n.demand
                      return acc
                    }, {})
                    const maxDemand = Math.max(...Object.values(byType))
                    return Object.entries(byType)
                      .sort(([, a], [, b]) => b - a)
                      .map(([type, demand]) => (
                        <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: NODE_COLORS[type] || '#94a3b8', flexShrink: 0 }} />
                          <span style={{ width: 84, fontSize: 11, color: 'var(--text-2)', flexShrink: 0 }}>{NODE_LABELS[type] || type}</span>
                          <div style={{ flex: 1, height: 10, background: 'var(--surface-2)', borderRadius: 5, overflow: 'hidden' }}>
                            <div style={{ height: '100%', borderRadius: 5, background: NODE_COLORS[type] || '#94a3b8', width: `${(demand / maxDemand) * 100}%`, opacity: 0.8, transition: 'width 0.5s ease' }} />
                          </div>
                          <span style={{ width: 70, fontSize: 10, fontFamily: 'var(--mono)', color: 'var(--text-2)', textAlign: 'right', flexShrink: 0 }}>
                            {(demand * 1000).toFixed(2)} L/s
                          </span>
                        </div>
                      ))
                  })()}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
