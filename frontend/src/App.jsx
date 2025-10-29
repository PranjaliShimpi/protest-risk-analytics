import React, { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const pct = (value, digits = 1) => `${(Number(value || 0) * 100).toFixed(digits)}%`
const formatNumber = (value) => Number(value || 0).toLocaleString()
const formatDays = (value) => {
  const num = Number(value)
  if (!Number.isFinite(num)) return '—'
  return `${Math.round(num)} days`
}

export default function App() {
  const [agencyId, setAgencyId] = useState('')
  const [naics, setNaics] = useState('')
  const [valueBand, setValueBand] = useState('')

  const [aggRows, setAggRows] = useState([])
  const [aggTotal, setAggTotal] = useState(0)
  const [aggLoading, setAggLoading] = useState(false)

  const [riskOpp, setRiskOpp] = useState('')
  const [risk, setRisk] = useState(null)
  const [riskLoading, setRiskLoading] = useState(false)

  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')
  const refreshedAt = useMemo(() => new Date(), [])

  const summaryMedianDays = useMemo(() => {
    if (!aggRows.length || !aggTotal) return null
    const weighted = aggRows.reduce(
      (acc, row) =>
        acc +
        Number(row.median_resolution_days || 0) * Number(row.opportunity_count || 0),
      0
    )
    if (!Number.isFinite(weighted)) return null
    return weighted / aggTotal
  }, [aggRows, aggTotal])

  const hasFilters = useMemo(
    () => Boolean(agencyId || naics || valueBand),
    [agencyId, naics, valueBand]
  )

  const fetchAggregates = async (params = {}) => {
    setAggLoading(true)
    setError('')
    try {
      const res = await axios.get(`${API}/protestAgg`, { params })
      const { records = [], total = 0 } = res.data || {}
      setAggRows(records)
      setAggTotal(total)
    } catch (e) {
      setAggRows([])
      setAggTotal(0)
      setError(e?.response?.data?.detail || e.message || 'Unable to load aggregates')
    } finally {
      setAggLoading(false)
    }
  }

  const handleAggSubmit = (evt) => {
    evt.preventDefault()
    fetchAggregates({
      agencyId: agencyId || undefined,
      naics: naics || undefined,
      valueBand: valueBand || undefined,
    })
  }

  const handleAggReset = () => {
    setAgencyId('')
    setNaics('')
    setValueBand('')
    fetchAggregates({})
  }

  const handleRiskScore = async (evt) => {
    evt.preventDefault()
    if (!riskOpp) {
      setError('Enter an opportunity ID to score risk.')
      return
    }
    setRisk(null)
    setRiskLoading(true)
    setError('')
    try {
      const res = await axios.get(`${API}/protestRisk`, {
        params: { oppId: Number(riskOpp) },
      })
      setRisk(res.data)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Unable to score opportunity')
    } finally {
      setRiskLoading(false)
    }
  }

  useEffect(() => {
    fetchAggregates({})
    axios
      .get(`${API}/modelMetrics`)
      .then((r) => setMetrics(r.data))
      .catch(() => setMetrics(null))
  }, [])

  const sortedDrivers = useMemo(() => {
    if (!risk?.drivers) return []
    return Object.entries(risk.drivers)
      .map(([feature, weight]) => ({ feature, weight }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 5)
  }, [risk])

  const sortedSustainDrivers = useMemo(() => {
    if (!risk?.sustain_drivers) return []
    return Object.entries(risk.sustain_drivers)
      .map(([feature, weight]) => ({ feature, weight }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 5)
  }, [risk])

  const protestMetrics = metrics?.protest
  const sustainMetrics = metrics?.sustain

  return (
    <div className="app-shell">
      <header className="hero-card">
        <div className="hero-headline">
          <span className="eyebrow">Protest Risk Analytics</span>
          <h1>
            Understand protest exposure across agencies, opportunities, and capture strategies.
          </h1>
          <p>
            Explore historical activity, simulate risk for individual opportunities, and review
            model calibration before decisions go live.
          </p>
          <div className="cta-row">
            <a className="btn btn-primary" href={`${API}/exportAggregates`} target="_blank" rel="noreferrer">
              Export CSV
            </a>
            <button type="button" className="btn btn-ghost" onClick={handleAggReset}>
              Reset filters
            </button>
          </div>
        </div>
        {(protestMetrics || sustainMetrics) && (
          <div className="metric-stack" title="Back-tested metrics generated from the latest training run.">
            {protestMetrics && (
              <section>
                <header>Protest model</header>
                <dl>
                  <div>
                    <dt>AUC</dt>
                    <dd>{protestMetrics.auc.toFixed(2)}</dd>
                  </div>
                  <div>
                    <dt>Calibration error</dt>
                    <dd>{protestMetrics.calibration_error.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Brier score</dt>
                    <dd>{protestMetrics.brier.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Train / Test</dt>
                    <dd>
                      {formatNumber(protestMetrics.n_train)} / {formatNumber(protestMetrics.n_test)}
                    </dd>
                  </div>
                </dl>
              </section>
            )}
            {sustainMetrics && (
              <section>
                <header>Sustain model</header>
                <dl>
                  <div>
                    <dt>AUC</dt>
                    <dd>{sustainMetrics.auc.toFixed(2)}</dd>
                  </div>
                  <div>
                    <dt>Calibration error</dt>
                    <dd>{sustainMetrics.calibration_error.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Brier score</dt>
                    <dd>{sustainMetrics.brier.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Train / Test</dt>
                    <dd>
                      {formatNumber(sustainMetrics.n_train)} / {formatNumber(sustainMetrics.n_test)}
                    </dd>
                  </div>
                </dl>
              </section>
            )}
          </div>
        )}
      </header>

      <main className="content-grid">
        <section className="card layout-span-2">
          <header className="section-header">
            <div>
              <h2>Agency Filters</h2>
              <p>Select filters to recalculate protest exposure at the portfolio level.</p>
            </div>
            <span className={`badge ${hasFilters ? 'badge-active' : ''}`}>
              {hasFilters ? 'Filters applied' : 'All agencies'}
            </span>
          </header>
          <form className="filter-grid" onSubmit={handleAggSubmit}>
            <label className="field">
              <span>Agency ID</span>
              <input
                value={agencyId}
                onChange={(e) => setAgencyId(e.target.value)}
                placeholder="e.g., DHS"
              />
            </label>
            <label className="field">
              <span>NAICS (prefix)</span>
              <input
                value={naics}
                onChange={(e) => setNaics(e.target.value)}
                placeholder="e.g., 5415"
              />
            </label>
            <label className="field">
              <span>Value band</span>
              <select value={valueBand} onChange={(e) => setValueBand(e.target.value)}>
                <option value="">Any</option>
                <option value="LT1M">LT1M</option>
                <option value="1-10M">1-10M</option>
                <option value="10-50M">10-50M</option>
                <option value="50M+">50M+</option>
              </select>
            </label>
            <div className="filter-actions">
              <button className="btn btn-primary" type="submit" disabled={aggLoading}>
                {aggLoading ? 'Loading…' : 'Update aggregates'}
              </button>
              <button className="btn btn-ghost" type="button" onClick={handleAggReset}>
                Clear
              </button>
            </div>
          </form>
          {error && (
            <div className="error-banner">
              <strong>Heads up:</strong> {error}
            </div>
          )}
          <div className="summary-cards">
            <article>
              <h3>Total opportunities</h3>
              <p>{formatNumber(aggTotal)}</p>
            </article>
            <article>
              <h3>Unique segments</h3>
              <p>{formatNumber(aggRows.length)}</p>
            </article>
            <article>
              <h3>Median resolution</h3>
              <p>{summaryMedianDays !== null ? formatDays(summaryMedianDays) : '—'}</p>
            </article>
          </div>
          <small
            className="data-refresh"
            title={`Synthetic dataset regenerated on ${refreshedAt.toLocaleString()}.`}
          >
            Data refreshed {refreshedAt.toLocaleDateString()} via latest pipeline run.
          </small>
        </section>

        <section className="card layout-span-2">
          <header className="section-header">
            <div>
              <h2>Agency × NAICS drilldown</h2>
              <p>Compare protest dynamics across mission areas and contract sizes.</p>
            </div>
          </header>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Agency</th>
                  <th>NAICS</th>
                  <th>Value band</th>
                  <th className="num">Opps</th>
                  <th className="num">Protest rate</th>
                  <th className="num">Sustain rate</th>
                  <th className="num">Median days</th>
                </tr>
              </thead>
              <tbody>
                {aggLoading ? (
                  <tr>
                    <td colSpan={7} className="empty">
                      Fetching latest figures…
                    </td>
                  </tr>
                ) : aggRows.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="empty">
                      No matching records. Adjust filters to broaden the search.
                    </td>
                  </tr>
                ) : (
                  aggRows.map((row, idx) => (
                    <tr key={`${row.agency_id}-${row.naics}-${idx}`}>
                      <td>{row.agency_id || '—'}</td>
                      <td>{row.naics || '—'}</td>
                      <td>{row.value_band || '—'}</td>
                      <td className="num">{formatNumber(row.opportunity_count)}</td>
                      <td className="num">{pct(row.protest_rate)}</td>
                      <td className="num">{pct(row.sustain_rate)}</td>
                      <td className="num">{formatDays(row.median_resolution_days)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card layout-span-1 sticky-card">
          <header className="section-header">
            <div>
              <h2>Opportunity risk meter</h2>
              <p>Score a specific solicitation to surface primary drivers.</p>
            </div>
          </header>
          <form className="risk-form" onSubmit={handleRiskScore}>
            <label className="field">
              <span>Opportunity ID</span>
              <input
                value={riskOpp}
                onChange={(e) => setRiskOpp(e.target.value)}
                placeholder="e.g., 1000"
                inputMode="numeric"
              />
            </label>
            <button className="btn btn-primary" type="submit" disabled={riskLoading}>
              {riskLoading ? 'Scoring…' : 'Score opportunity'}
            </button>
          </form>

          {risk && (
            <div className="risk-card">
              <div className="risk-prob-grid">
                <div className="probability-tile protest">
                  <div className="probability-header">
                    <span className="label">Protest probability</span>
                    <span className="pill">Filings</span>
                  </div>
                  <div className="probability-value">{pct(risk.probability, 1)}</div>
                  <div className="risk-meter">
                    <span className="fill protest" style={{ width: pct(risk.probability, 2) }} />
                  </div>
                  <p className="probability-subcopy">Likelihood this award receives a formal protest.</p>
                </div>
                <div className="probability-tile sustain">
                  <div className="probability-header">
                    <span className="label">Sustain probability</span>
                    <span className="pill">Outcome</span>
                  </div>
                  <div className="probability-value">{pct(risk.sustain_probability, 1)}</div>
                  <div className="risk-meter">
                    <span className="fill sustain" style={{ width: pct(risk.sustain_probability, 2) }} />
                  </div>
                  <p className="probability-subcopy">Chance a filed protest is ultimately sustained.</p>
                </div>
              </div>

              <div className="risk-meta">
                <div>
                  <span className="fact-label">Opportunity</span>
                  <span className="fact-value">#{risk.oppId}</span>
                </div>
                <div>
                  <span className="fact-label">Generated</span>
                  <span className="fact-value">{new Date().toLocaleTimeString()}</span>
                </div>
              </div>

              <div className="driver-columns">
                <div className="driver-group">
                  <h4 title="Top contributors to the protest probability.">Protest drivers</h4>
                  <ul className="driver-list">
                    {sortedDrivers.map(({ feature, weight }) => (
                      <li key={`p-${feature}`} title={`Impact on log-odds: ${weight.toFixed(3)}`}>
                        <span>{feature}</span>
                        <span>{pct(Math.abs(weight), 1)}</span>
                      </li>
                    ))}
                    {sortedDrivers.length === 0 && <li className="empty">Insufficient signal.</li>}
                  </ul>
                </div>
                <div className="driver-group">
                  <h4 title="Top contributors to the sustain probability (conditional on protest).">
                    Sustain drivers
                  </h4>
                  <ul className="driver-list">
                    {sortedSustainDrivers.map(({ feature, weight }) => (
                      <li key={`s-${feature}`} title={`Impact on sustain log-odds: ${weight.toFixed(3)}`}>
                        <span>{feature}</span>
                        <span>{pct(Math.abs(weight), 1)}</span>
                      </li>
                    ))}
                    {sortedSustainDrivers.length === 0 && <li className="empty">Insufficient signal.</li>}
                  </ul>
                </div>
              </div>

              <p className="footnote" title="Drivers normalised by absolute coefficient weight in the logistic regression.">
                Drivers show relative contribution (absolute weight) to logistic regression scores.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
