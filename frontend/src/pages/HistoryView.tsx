import { useCallback, useEffect, useState } from 'react'
import { checkHealth, listRuns, type RunRecord } from '../api'

const pct = (v: number | null | undefined) => (v == null ? '-' : `${(v * 100).toFixed(1)}%`)

/** Every simulation the backend has served, newest first (backend/runs.py). */
export default function HistoryView() {
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [online, setOnline] = useState<boolean | null>(null)
  const [error, setError] = useState('')
  const [kind, setKind] = useState('')

  const load = useCallback(async () => {
    setError('')
    try {
      const up = await checkHealth()
      setOnline(up)
      if (!up) { setRuns([]); return }
      setRuns(await listRuns(100, kind || undefined))
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Could not load run history')
    }
  }, [kind])

  useEffect(() => { void load() }, [load])

  return (
    <div style={{ padding: 20, overflow: 'auto', height: '100%' }}>
      <div className="card">
        <div className="card-hd">
          Run history
          <span style={{ flex: 1 }} />
          <select value={kind} onChange={e => setKind(e.target.value)} style={{ fontSize: 11 }}>
            <option value="">All kinds</option>
            <option value="india">India simulation</option>
            <option value="scenario">Benchmark scenario</option>
            <option value="multi_failure">Multi-failure</option>
            <option value="auto_respond">Auto-respond</option>
          </select>
          <button className="btn" onClick={() => void load()}>Refresh</button>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          {online === false && (
            <div style={{ padding: 16 }}>Backend offline: run history is stored on the server, so there is nothing to show.</div>
          )}
          {error && <div style={{ padding: 16, color: 'var(--red, #c0392b)' }}>{error}</div>}
          {online && !error && runs.length === 0 && (
            <div style={{ padding: 16 }}>No runs recorded yet. Run a simulation and it will appear here.</div>
          )}
          {runs.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th><th>When</th><th>Kind</th><th>Network</th><th>Method</th><th>Scenario</th>
                  <th>Failed pipes</th><th>Demand served</th><th>Critical served</th><th>PVC</th><th>Runtime</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{new Date(r.ts * 1000).toLocaleString()}</td>
                    <td>{r.kind}</td>
                    <td>{r.network}</td>
                    <td>{r.method ?? '-'}{r.escalated ? ' (escalated)' : ''}</td>
                    <td>{r.scenario ?? '-'}</td>
                    <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {r.failed_pipes.join(', ') || '-'}
                    </td>
                    <td>{pct(r.cds_overall)}</td>
                    <td>{pct(r.cds_critical)}</td>
                    <td>{r.pvc ?? '-'}</td>
                    <td>{r.runtime_ms == null ? '-' : `${r.runtime_ms.toFixed(1)} ms`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
