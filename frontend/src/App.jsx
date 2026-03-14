import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import 'bootstrap/dist/css/bootstrap.min.css';
import './index.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
);

const API_URL = 'http://127.0.0.1:8000/';

// ─── Colour palette for charts ───────────────────────────────────────────────
const CHART_COLORS = [
  '#00e5ff', '#7c4dff', '#00e676', '#ff6d00', '#ff4081',
  '#ffea00', '#40c4ff', '#69f0ae', '#ff6e40', '#e040fb',
];

// ─── Signal badge ─────────────────────────────────────────────────────────────
function SignalBadge({ value }) {
  if (value === null || value === undefined) {
    return <span className="badge bg-secondary">Analyzing…</span>;
  }
  const num = parseFloat(value);
  if (num > 0.55) return <span className="badge signal-buy">BUY {num.toFixed(2)}</span>;
  if (num < 0.45) return <span className="badge signal-sell">SELL {num.toFixed(2)}</span>;
  return <span className="badge signal-hold">HOLD {num.toFixed(2)}</span>;
}

// ─── Stock drill-down modal ───────────────────────────────────────────────────
function StockModal({ stock, historicalCharts, onClose }) {
  const backdropRef = useRef(null);

  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  if (!stock) return null;

  const ticker = stock.symbol || stock.tradingsymbol || '';
  const hist = historicalCharts[ticker] || null;

  const lineData = hist
    ? {
        labels: hist.dates,
        datasets: [
          {
            label: `${ticker} Close Price`,
            data: hist.closes,
            borderColor: '#00e5ff',
            backgroundColor: 'rgba(0,229,255,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
          },
        ],
      }
    : null;

  const lineOptions = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c9d1d9' } },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    },
  };

  return (
    <div
      className="tp-modal-backdrop"
      ref={backdropRef}
      onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
      role="dialog"
      aria-modal="true"
      aria-label={`${ticker} drill-down`}
    >
      <div className="tp-modal">
        <button className="tp-modal-close" onClick={onClose} aria-label="Close">✕</button>
        <h4 className="tp-modal-title">
          {ticker}
          <span className="ms-3">
            <SignalBadge value={stock.ai_signal_float} />
          </span>
        </h4>
        <div className="row mb-3 text-muted small">
          <div className="col-6">
            <span className="me-3">Qty: <strong className="text-light">{stock.quantity}</strong></span>
            <span>Avg: <strong className="text-light">₹{parseFloat(stock.avg_price || 0).toFixed(2)}</strong></span>
          </div>
          <div className="col-6 text-end">
            Market Value: <strong className="text-light">₹{parseFloat(stock.market_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}</strong>
          </div>
        </div>
        {lineData ? (
          <Line data={lineData} options={lineOptions} />
        ) : (
          <div className="text-center text-muted py-5">No historical data available for {ticker}.</div>
        )}
      </div>
    </div>
  );
}

// ─── Portfolio table ──────────────────────────────────────────────────────────
function PortfolioTable({ portfolio, onRowClick }) {
  if (!portfolio.length) {
    return <p className="text-muted text-center py-4">No portfolio data available.</p>;
  }
  return (
    <div className="table-responsive">
      <table className="table table-dark table-hover tp-table mb-0">
        <thead>
          <tr>
            <th>Symbol</th>
            <th className="text-end">Quantity</th>
            <th className="text-end">Avg Price</th>
            <th className="text-end">Market Value</th>
            <th className="text-center">AI Signal</th>
          </tr>
        </thead>
        <tbody>
          {portfolio.map((stock, idx) => {
            const ticker = stock.symbol || stock.tradingsymbol || `row-${idx}`;
            return (
              <tr
                key={ticker}
                onClick={() => onRowClick(stock)}
                className="tp-table-row"
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter') onRowClick(stock); }}
              >
                <td className="fw-semibold text-info">{ticker}</td>
                <td className="text-end">{stock.quantity}</td>
                <td className="text-end">₹{parseFloat(stock.avg_price || 0).toFixed(2)}</td>
                <td className="text-end">₹{parseFloat(stock.market_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}</td>
                <td className="text-center">
                  <SignalBadge value={stock.ai_signal_float} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Dashboard page ───────────────────────────────────────────────────────────
function DashboardPage({ data }) {
  const [selectedStock, setSelectedStock] = useState(null);

  const { portfolio, chartLabels, chartValues, sectorLabels, sectorValues, historicalCharts } = data;

  // Portfolio net-worth line chart
  const netWorthData = {
    labels: chartLabels,
    datasets: [
      {
        label: 'Portfolio Net Worth (₹)',
        data: chartValues,
        borderColor: '#00e5ff',
        backgroundColor: 'rgba(0,229,255,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const netWorthOptions = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#c9d1d9' } },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    },
  };

  // Sector allocation doughnut
  const sectorData = {
    labels: sectorLabels,
    datasets: [
      {
        data: sectorValues,
        backgroundColor: CHART_COLORS,
        borderColor: '#0d1117',
        borderWidth: 2,
      },
    ],
  };

  const sectorOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'right', labels: { color: '#c9d1d9', boxWidth: 12 } },
    },
  };

  return (
    <>
      {/* Top row — charts */}
      <div className="row g-3 mb-3">
        <div className="col-12 col-lg-7">
          <div className="tp-card h-100">
            <div className="tp-card-header">
              <span className="tp-card-title">Portfolio Net Worth</span>
            </div>
            <div className="tp-card-body">
              {chartLabels.length ? (
                <Line data={netWorthData} options={netWorthOptions} />
              ) : (
                <p className="text-muted text-center py-5">No snapshot data yet.</p>
              )}
            </div>
          </div>
        </div>
        <div className="col-12 col-lg-5">
          <div className="tp-card h-100">
            <div className="tp-card-header">
              <span className="tp-card-title">Sector Allocation</span>
            </div>
            <div className="tp-card-body d-flex align-items-center justify-content-center">
              {sectorLabels.length ? (
                <Doughnut data={sectorData} options={sectorOptions} />
              ) : (
                <p className="text-muted text-center py-5">No sector data yet.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Middle row — portfolio table + news sidebar */}
      <div className="row g-3">
        <div className="col-12 col-lg-8">
          <div className="tp-card">
            <div className="tp-card-header">
              <span className="tp-card-title">Holdings</span>
              <span className="tp-card-hint">Click a row for drill-down</span>
            </div>
            <div className="tp-card-body p-0">
              <PortfolioTable portfolio={portfolio} onRowClick={setSelectedStock} />
            </div>
          </div>
        </div>
        <div className="col-12 col-lg-4">
          <div className="tp-card h-100">
            <div className="tp-card-header">
              <span className="tp-card-title">News &amp; Sentiment</span>
              <span className="tp-badge-coming">Coming soon</span>
            </div>
            <div className="tp-card-body tp-news-placeholder">
              <div className="tp-news-icon">📰</div>
              <p className="tp-news-text">Real-time news feed &amp; sentiment scores will appear here once the news pipeline is wired up.</p>
              <div className="tp-news-items">
                {['Earnings season kicks off…', 'Fed rate decision next week', 'Global markets rally on data'].map((headline, i) => (
                  <div key={i} className="tp-news-item">
                    <span className="tp-news-bullet">▸</span>
                    <span className="text-muted">{headline}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Drill-down modal */}
      {selectedStock && (
        <StockModal
          stock={selectedStock}
          historicalCharts={historicalCharts}
          onClose={() => setSelectedStock(null)}
        />
      )}
    </>
  );
}

// ─── About page ───────────────────────────────────────────────────────────────
function AboutPage() {
  return (
    <div className="tp-card">
      <div className="tp-card-header">
        <span className="tp-card-title">About TradePulse</span>
      </div>
      <div className="tp-card-body">
        <p className="text-muted">
          TradePulse is a quantitative trading dashboard that combines live portfolio data from
          Angel One with machine-learning AI signals and sector analysis.
        </p>
        <ul className="text-muted mt-3">
          <li>Live portfolio ingestion via Angel One API</li>
          <li>XGBoost + LSTM meta-learner for directional AI signals</li>
          <li>Historical close-price drill-downs per holding</li>
          <li>Sector allocation at a glance</li>
          <li>News &amp; sentiment integration (coming soon)</li>
        </ul>
      </div>
    </div>
  );
}

// ─── Navigation bar ───────────────────────────────────────────────────────────
function NavBar() {
  const location = useLocation();
  return (
    <nav className="tp-navbar">
      <div className="tp-navbar-brand">
        <span className="tp-brand-icon">⚡</span>
        TradePulse
      </div>
      <div className="tp-navbar-links">
        <Link to="/" className={`tp-nav-link${location.pathname === '/' ? ' active' : ''}`}>Dashboard</Link>
        <Link to="/about" className={`tp-nav-link${location.pathname === '/about' ? ' active' : ''}`}>About</Link>
      </div>
    </nav>
  );
}

// ─── Root App ─────────────────────────────────────────────────────────────────
function AppInner() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  useEffect(() => {
    axios
      .get(API_URL)
      .then((res) => {
        const raw = res.data;
        setData({
          portfolio: raw.portfolio || [],
          chartLabels: JSON.parse(raw.chart_labels || '[]'),
          chartValues: JSON.parse(raw.chart_values || '[]'),
          sectorLabels: JSON.parse(raw.sector_labels || '[]'),
          sectorValues: JSON.parse(raw.sector_values || '[]'),
          historicalCharts: JSON.parse(raw.historical_charts_json || '{}'),
          errorMessage: raw.error_message || null,
        });
      })
      .catch((err) => {
        setError(err.message || 'Failed to fetch data from the API.');
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="tp-shell">
      <NavBar />
      <main className="tp-main container-fluid">
        {loading && (
          <div className="tp-loading">
            <div className="tp-spinner" />
            <span>Connecting to TradePulse API…</span>
          </div>
        )}
        {!loading && error && (
          <div className="alert tp-alert-error mt-4" role="alert">
            <strong>API Error:</strong> {error}
          </div>
        )}
        {!loading && data && data.errorMessage && (
          <div className="alert tp-alert-warning mt-4" role="alert">
            <strong>Notice:</strong> {data.errorMessage}
          </div>
        )}
        {!loading && data && (
          <Routes>
            <Route path="/" element={<DashboardPage data={data} />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        )}
      </main>
      <footer className="tp-footer">
        TradePulse © {new Date().getFullYear()} — Quantitative Trading Dashboard
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  );
}
