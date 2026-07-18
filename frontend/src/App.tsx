import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom'
import './App.css'
import { IconDroplet, IconActivity, IconBarChart, IconNetwork, IconSun, IconMoon } from './icons'
import type { SharedSimState } from './types'
import SimulationView from './pages/IndiaMapView'
import ResultsView from './pages/ResultsView'
import NetworkView from './pages/NetworkView'

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [sharedSim, setSharedSim] = useState<SharedSimState | null>(null)

  const toggleTheme = () => {
    setTheme(t => {
      const next = t === 'light' ? 'dark' : 'light'
      document.documentElement.setAttribute('data-theme', next)
      return next
    })
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <nav className="top-nav">
          <NavLink to="/" className="nav-brand" style={{ textDecoration: 'none' }}>
            <div className="nav-brand-icon"><IconDroplet size={16} /></div>
            <div>
              <div className="nav-brand-name">HARP-WDN</div>
              <div className="nav-brand-sub">Water Network System</div>
            </div>
          </NavLink>

          <div className="nav-sep" />

          <div className="nav-links">
            <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              <IconActivity size={14} />
              Simulation
            </NavLink>
            <NavLink to="/results" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              <IconBarChart size={14} />
              Results
              {sharedSim?.simResult && <span className="nav-pip" />}
            </NavLink>
            <NavLink to="/network" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              <IconNetwork size={14} />
              Network
            </NavLink>
          </div>

          <div className="nav-spacer" />

          <div className="nav-actions">
            <span className="nav-badge">Hydraulic Resilience Platform</span>
            <button className="theme-toggle" onClick={toggleTheme}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
              {theme === 'light' ? <IconMoon size={15} /> : <IconSun size={15} />}
            </button>
          </div>
        </nav>

        <div className="page">
          <Routes>
            <Route path="/" element={<SimulationView onSimComplete={setSharedSim} />} />
            <Route path="/results" element={<ResultsView sharedSim={sharedSim} />} />
            <Route path="/network" element={<NetworkView sharedSim={sharedSim} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
