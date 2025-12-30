import React, { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Clock,
  RefreshCw,
  Search,
  Sparkles,
  TrendingUp,
  Zap
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const formatCurrency = (value) => {
  if (value === null || value === undefined) return '--';
  const abs = Math.abs(value);
  if (abs > 0 && abs < 1) return `$${value.toFixed(3)}`;
  if (abs >= 1 && abs < 10) return `$${value.toFixed(2)}`;
  if (abs >= 10 && abs < 100) return `$${value.toFixed(1)}`;
  if (abs >= 1000000) return `$${(value / 1000000).toFixed(2)}M`;
  if (abs >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
};

const formatNumber = (value) => {
  if (value === null || value === undefined) return '--';
  return value.toLocaleString();
};

const formatPercent = (value) => {
  if (value === null || value === undefined) return '--';
  return `${(value * 100).toFixed(1)}%`;
};

const formatCoverageLabel = (source, count) => {
  if (source === 'trade') return `trades ${formatNumber(count || 0)}`;
  if (source === 'insufficient') return `low sample ${formatNumber(count || 0)}`;
  return 'snapshot';
};

const resolveRetailStatus = (signals) => {
  if (!signals) return { level: 'low', label: 'low retail' };
  if (signals.coverage === 'insufficient') return { level: 'insufficient', label: 'insufficient data' };
  return { level: signals.level || 'low', label: `${signals.level || 'low'} retail` };
};

const formatTimestamp = (value) => {
  if (!value) return '--';
  return new Date(value).toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const STALE_SNAPSHOT_HOURS = 24;

const isTruthyFlag = (value) => {
  if (value === true) return true;
  if (value === false || value === null || value === undefined) return false;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    return normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'y';
  }
  return false;
};

const isMarketClosed = (market) => {
  if (!market) return false;
  if (isTruthyFlag(market.closed) || isTruthyFlag(market.archived)) return true;
  const status = typeof market.status === 'string' ? market.status.trim().toLowerCase() : '';
  return status === 'inactive' || status === 'closed';
};

const hoursSince = (value) => {
  if (!value) return null;
  const ts = new Date(value);
  if (Number.isNaN(ts.getTime())) return null;
  return (Date.now() - ts.getTime()) / 36e5;
};

const formatRollupDate = (value) => {
  if (!value) return '--';
  const parts = String(value).split('-');
  if (parts.length !== 3) return value;
  const monthIndex = Number(parts[1]) - 1;
  const day = parts[2];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const monthLabel = months[monthIndex];
  if (!monthLabel) return value;
  return `${monthLabel} ${day}`;
};

const formatHourlyLabel = (value) => {
  if (!value) return '--';
  return new Date(value).toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const TREND_RANGES = [
  { label: '24h', mode: 'hourly', hours: 24 },
  { label: '72h', mode: 'hourly', hours: 72 },
  { label: '7d', mode: 'daily', days: 7 },
  { label: '30d', mode: 'daily', days: 30 },
  { label: '90d', mode: 'daily', days: 90 },
  { label: '180d', mode: 'daily', days: 180 }
];

const getCategoryColor = (category) => {
  const colors = {
    politics: '#d1495b',
    crypto: '#f28f3b',
    sports: '#1b998b',
    business: '#2d7dd2',
    entertainment: '#7f5af0',
    science: '#4ea8de',
    world: '#7c9885',
    uncategorized: '#9aa0a6'
  };
  return colors[category] || '#9aa0a6';
};

const normalizeTags = (tags) => {
  if (!tags) return [];
  if (Array.isArray(tags)) return tags;
  if (typeof tags === 'string') {
    try {
      const parsed = JSON.parse(tags);
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }
  return [];
};

export default function App() {
  const [overview, setOverview] = useState(null);
  const [markets, setMarkets] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedMarketId, setSelectedMarketId] = useState(null);
  const [marketDetail, setMarketDetail] = useState(null);
  const [marketHistory, setMarketHistory] = useState([]);
  const [marketTrades, setMarketTrades] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [windowDays, setWindowDays] = useState(30);
  const [hideWhales, setHideWhales] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [trendCategory, setTrendCategory] = useState('all');
  const [trendDays, setTrendDays] = useState(180);
  const [trendHours, setTrendHours] = useState(24);
  const [trendGranularity, setTrendGranularity] = useState('daily');
  const [trendSeries, setTrendSeries] = useState([]);
  const [trendSummary, setTrendSummary] = useState(null);
  const [trendLoading, setTrendLoading] = useState(true);

  const fetchJson = async (url) => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return response.json();
  };

  const loadOverview = async (days = windowDays) => {
    const data = await fetchJson(`${API_BASE}/overview?days=${days}`);
    setOverview(data);
    setLastUpdated(new Date());
  };

  const loadMarkets = async (category, days = windowDays) => {
    const params = new URLSearchParams();
    if (category && category !== 'all') {
      params.set('category', category);
    }
    params.set('limit', '50');
    params.set('days', String(days));
    if (hideWhales) {
      params.set('hide_whales', 'true');
    }
    const data = await fetchJson(`${API_BASE}/markets?${params.toString()}`);
    setMarkets(data.markets || []);
    if (selectedMarketId && !(data.markets || []).some((m) => m.id === selectedMarketId)) {
      setSelectedMarketId(null);
      setMarketDetail(null);
      setMarketHistory([]);
      setMarketTrades(null);
    }
  };

  const loadMarketDetail = async (marketId, days = windowDays) => {
    const windowHours = days * 24;
    const detailPromise = fetchJson(`${API_BASE}/markets/${marketId}?hours=${windowHours}`);
    const historyPromise = fetchJson(`${API_BASE}/markets/${marketId}/history?hours=${windowHours}`);
    const tradesPromise = fetchJson(`${API_BASE}/markets/${marketId}/trades?hours=${windowHours}&limit=500`);

    const [detailData, historyData, tradesData] = await Promise.all([
      detailPromise,
      historyPromise,
      tradesPromise
    ]);

    setMarketDetail(detailData);
    setMarketHistory(historyData.series || []);
    setMarketTrades(tradesData);
  };

  const loadTrend = async (
    category = trendCategory,
    granularity = trendGranularity,
    days = trendDays,
    hours = trendHours
  ) => {
    setTrendLoading(true);
    try {
      const params = new URLSearchParams();
      params.set('category', category);
      let endpoint = '/analytics/retail-index';
      if (granularity === 'hourly') {
        endpoint = '/analytics/retail-index/hourly';
        params.set('hours', String(hours));
      } else {
        params.set('days', String(days));
      }
      const data = await fetchJson(`${API_BASE}${endpoint}?${params.toString()}`);
      setTrendSeries(data.series || []);
      setTrendSummary(data.summary || null);
    } finally {
      setTrendLoading(false);
    }
  };

  const refreshAll = async () => {
    setRefreshing(true);
    setError('');
    try {
      await Promise.all([loadOverview(), loadMarkets(selectedCategory), loadTrend()]);
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    const boot = async () => {
      setLoading(true);
      setError('');
      try {
        await Promise.all([loadOverview(), loadMarkets(selectedCategory), loadTrend()]);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    boot();
  }, []);

  useEffect(() => {
    if (!loading) {
      loadMarkets(selectedCategory).catch((err) => setError(err.message));
    }
  }, [selectedCategory]);

  useEffect(() => {
    if (!loading) {
      loadMarkets(selectedCategory).catch((err) => setError(err.message));
    }
  }, [hideWhales]);

  useEffect(() => {
    if (!loading) {
      loadTrend(trendCategory, trendGranularity, trendDays, trendHours)
        .catch((err) => setError(err.message));
    }
  }, [trendCategory, trendGranularity, trendDays, trendHours]);

  useEffect(() => {
    if (selectedMarketId) {
      loadMarketDetail(selectedMarketId).catch((err) => setError(err.message));
    }
  }, [selectedMarketId]);

  useEffect(() => {
    if (!loading) {
      refreshAll();
      if (selectedMarketId) {
        loadMarketDetail(selectedMarketId).catch((err) => setError(err.message));
      }
    }
  }, [windowDays]);

  const categories = useMemo(() => {
    if (!overview?.categories) return [];
    return overview.categories
      .slice()
      .sort((a, b) => (b.volume_24h || 0) - (a.volume_24h || 0));
  }, [overview]);

  const topRetailShare = useMemo(() => {
    if (!categories.length) return 0;
    const shares = categories
      .map((category) => (category.retail_volume_share ?? category.retail_trade_share))
      .filter((value) => typeof value === 'number');
    if (!shares.length) return 0;
    return Math.max(...shares);
  }, [categories]);

  const categoryCoverageHint = useMemo(() => {
    if (!overview) return '';
    const minTrades = overview.retail_min_trades ?? 25;
    return `Coverage: ${minTrades}+ trades for retail share`;
  }, [overview]);

  const filteredMarkets = useMemo(() => {
    if (!markets.length) return [];
    const term = searchTerm.trim().toLowerCase();
    return markets.filter((market) => {
      if (hideWhales && market.retail_signals?.whale_dominated) return false;
      if (!term) return true;
      return (
        market.title?.toLowerCase().includes(term) ||
        market.category?.toLowerCase().includes(term)
      );
    });
  }, [markets, searchTerm]);

  const historyChartData = useMemo(() => {
    return marketHistory.map((point) => ({
      time: new Date(point.timestamp).toLocaleDateString('en-US', {
        month: 'short',
        day: '2-digit'
      }),
      price: point.price,
      volume: point.volume_24h,
      liquidity: point.liquidity,
      avgTrade: point.avg_trade_size
    }));
  }, [marketHistory]);

  const hourlyData = useMemo(() => {
    if (!marketTrades?.retail_analysis?.hourly_patterns) return [];
    return marketTrades.retail_analysis.hourly_patterns.map((item) => ({
      hour: `${item.hour}:00`,
      totalTrades: item.trades,
      retailTrades: item.retail_trades
    }));
  }, [marketTrades]);

  const tradeSizeData = useMemo(() => {
    return marketTrades?.trade_size_distribution || [];
  }, [marketTrades]);

  const trendChartData = useMemo(() => {
    return trendSeries.map((point) => ({
      date: trendGranularity === 'hourly'
        ? formatHourlyLabel(point.timestamp)
        : formatRollupDate(point.date),
      retailScore: point.retail_score ?? null,
      flowIndex: point.flow_score !== null && point.flow_score !== undefined ? point.flow_score * 10 : null,
      attentionIndex: point.attention_score !== null && point.attention_score !== undefined ? point.attention_score * 10 : null,
      retailShare: point.retail_trade_share ?? null,
      whaleShare: point.whale_share ?? null,
      trades: point.total_trades ?? 0
    }));
  }, [trendSeries, trendGranularity]);

  const flowMetric = useMemo(() => {
    if (typeof trendSummary?.latest_flow_score === 'number') {
      return trendSummary.latest_flow_score.toFixed(2);
    }
    return '--';
  }, [trendSummary]);

  const attentionMetric = useMemo(() => {
    if (typeof trendSummary?.latest_attention_score === 'number') {
      return trendSummary.latest_attention_score.toFixed(2);
    }
    return '--';
  }, [trendSummary]);

  const detailCoverageNote = useMemo(() => {
    const signals = marketDetail?.retail_signals;
    if (!signals || signals.coverage !== 'insufficient') return '';
    const sample = signals.sample_trades ?? 0;
    const minTrades = signals.min_trades ?? 25;
    return `Insufficient trades in window (${sample}/${minTrades}). Showing snapshot-derived signals.`;
  }, [marketDetail]);

  const detailConfidenceLabel = useMemo(() => {
    const label = marketDetail?.retail_signals?.confidence_label;
    if (!label || label === 'insufficient') return '';
    return `Confidence: ${label}`;
  }, [marketDetail]);

  const detailIsClosed = useMemo(() => isMarketClosed(marketDetail?.market), [marketDetail]);

  const staleNote = useMemo(() => {
    const latestTimestamp = marketDetail?.latest?.timestamp;
    if (!latestTimestamp) return '';
    const hours = hoursSince(latestTimestamp);
    if (hours === null) return '';
    if (detailIsClosed) {
      return `Market marked closed. Data frozen since ${formatTimestamp(latestTimestamp)}.`;
    }
    if (hours >= STALE_SNAPSHOT_HOURS) {
      const rounded = Math.round(hours);
      return `No fresh snapshots in ~${rounded}h. Market may be closed or inactive; treat signals as stale.`;
    }
    return '';
  }, [marketDetail, detailIsClosed]);

  const qualityNote = useMemo(() => {
    const signals = marketDetail?.retail_signals;
    if (!signals || signals.quality_ok !== false) return '';
    if (signals.quality_reason === 'low_volume') return 'Low volume market; expect noisy signals.';
    if (signals.quality_reason === 'low_liquidity') return 'Low liquidity market; price impact is noisy.';
    if (signals.quality_reason === 'low_volume_and_liquidity') {
      return 'Low volume and liquidity; signals are unreliable.';
    }
    return 'Market quality below threshold.';
  }, [marketDetail]);

  const flowAttention = useMemo(() => {
    const flow = marketDetail?.retail_signals?.flow_score ?? 0;
    const attention = marketDetail?.retail_signals?.attention_score ?? 0;
    let label = 'Low signal';
    if (flow >= 0.6 && attention >= 0.6) label = 'Hot retail + hype';
    else if (flow >= 0.6) label = 'Steady retail flow';
    else if (attention >= 0.6) label = 'Hype burst';
    return { flow, attention, label };
  }, [marketDetail]);

  const whaleNote = useMemo(() => {
    const whaleShare = marketDetail?.retail_signals?.whale_share;
    if (whaleShare === null || whaleShare === undefined) return '';
    if (whaleShare < 0.5) return '';
    return `Whale dominated (${formatPercent(whaleShare)} of volume)`;
  }, [marketDetail]);

  const reliabilityNote = useMemo(() => {
    const signals = marketDetail?.retail_signals;
    if (!signals || signals.rank_reliable !== false) return '';
    return 'Retail ranking not reliable (insufficient trades, volume, or liquidity).';
  }, [marketDetail]);

  const selectedMarketRow = useMemo(() => {
    if (!selectedMarketId) return null;
    return markets.find((market) => market.id === selectedMarketId) || null;
  }, [markets, selectedMarketId]);

  if (loading) {
    return (
      <div className="app-shell">
        <div className="panel loading-panel">
          <div className="loading-spinner" />
          <p>Loading Polymarket retail signals…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="logo-chip">
            <Sparkles className="icon" />
          </div>
          <div>
            <p className="eyebrow">Polymarket only</p>
            <h1 className="title">Retail Pulse Dashboard</h1>
            <p className="subtitle">Track retail flow, attention, and meme-market cycles in real time.</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="btn-secondary" onClick={refreshAll} disabled={refreshing}>
            <RefreshCw className={`icon ${refreshing ? 'spin' : ''}`} />
            {refreshing ? 'Refreshing' : 'Refresh'}
          </button>
          <div className="window-toggle">
            <span className="toggle-label">Window</span>
            <div className="toggle-group">
              {[7, 30].map((days) => (
                <button
                  key={days}
                  className={`toggle-btn ${windowDays === days ? 'active' : ''}`}
                  onClick={() => setWindowDays(days)}
                  type="button"
                >
                  {days}d
                </button>
              ))}
            </div>
          </div>
          <div className="window-toggle">
            <span className="toggle-label">Whales</span>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${hideWhales ? 'active' : ''}`}
                onClick={() => setHideWhales(true)}
                type="button"
              >
                Hide
              </button>
              <button
                className={`toggle-btn ${!hideWhales ? 'active' : ''}`}
                onClick={() => setHideWhales(false)}
                type="button"
              >
                Show
              </button>
            </div>
          </div>
          <div className="timestamp">
            <Clock className="icon" />
            {lastUpdated ? lastUpdated.toLocaleTimeString() : '--'}
          </div>
        </div>
      </header>

      {error && (
        <div className="panel error-panel">
          <p>Data connection issue: {error}</p>
        </div>
      )}

      <section className="metrics-grid">
        <div className="panel metric-card">
          <div className="metric-header">
            <Activity className="icon" />
            <span>Markets tracked</span>
          </div>
          <div className="metric-value">{formatNumber(overview?.totals?.markets || 0)}</div>
          <p className="metric-footnote">Coverage from latest snapshots</p>
        </div>
        <div className="panel metric-card">
          <div className="metric-header">
            <BarChart3 className="icon" />
            <span>Trades in last {overview?.window_days ?? windowDays}d</span>
          </div>
          <div className="metric-value">{formatNumber(overview?.totals?.trades_24h || 0)}</div>
          <p className="metric-footnote">Historical trade activity</p>
        </div>
        <div className="panel metric-card">
          <div className="metric-header">
            <Zap className="icon" />
            <span>Top retail share (volume)</span>
          </div>
          <div className="metric-value">
            {formatPercent(topRetailShare)}
          </div>
          <p className="metric-footnote">Peak category retail share</p>
        </div>
        <div className="panel metric-card">
          <div className="metric-header">
            <TrendingUp className="icon" />
            <span>Latest collection</span>
          </div>
          <div className="metric-value">{formatTimestamp(overview?.last_collection)}</div>
          <p className="metric-footnote">Database refresh checkpoint</p>
        </div>
      </section>

      <section className="panel section">
        <div className="section-header">
          <div>
            <p className="eyebrow">Retail attention map</p>
            <h2 className="section-title">Where retail is leaning right now</h2>
          </div>
          <span className="hint">{categoryCoverageHint || 'Retail share + liquidity mix by category'}</span>
        </div>
        <div className="category-grid">
          {categories.map((category) => (
            <div key={category.category} className="category-card">
              <div className="category-top">
                <span className="category-dot" style={{ backgroundColor: getCategoryColor(category.category) }} />
                <span className="category-name">{category.category}</span>
                <span className="category-markets">{category.market_count} mkts</span>
                <span className={`coverage-pill ${category.retail_share_source || 'snapshot'}`}>
                  {formatCoverageLabel(category.retail_share_source, category.trade_count_24h)}
                </span>
              </div>
              <div className="category-metrics">
                <div>
                  <p>Retail share (count)</p>
                  <strong>{formatPercent(category.retail_trade_share)}</strong>
                </div>
                <div>
                  <p>Retail share (volume)</p>
                  <strong>{formatPercent(category.retail_volume_share)}</strong>
                </div>
                <div>
                  <p>Avg trade size</p>
                  <strong>{formatCurrency(category.avg_trade_size)}</strong>
                </div>
                <div>
                  <p>Volume 24h</p>
                  <strong>{formatCurrency(category.volume_24h)}</strong>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="market-section">
        <div className="panel market-list">
          <div className="section-header">
            <div>
              <p className="eyebrow">Market board</p>
              <h2 className="section-title">Retail flow watchlist</h2>
            </div>
            <div className="controls">
              <div className="search-field">
                <Search className="icon" />
                <input
                  type="text"
                  placeholder="Search markets"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                />
              </div>
              <div className="select-field">
                <select
                  value={selectedCategory}
                  onChange={(event) => setSelectedCategory(event.target.value)}
                >
                  <option value="all">All categories</option>
                  {categories.map((cat) => (
                    <option key={cat.category} value={cat.category}>
                      {cat.category}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="market-list-body">
            {filteredMarkets.map((market) => {
              const retailStatus = resolveRetailStatus(market.retail_signals);
              const marketClosed = isMarketClosed(market);
              const rankReliable = market.retail_signals?.rank_reliable;
              return (
              <button
                key={market.id}
                className={`market-row ${selectedMarketId === market.id ? 'active' : ''}`}
                onClick={() => setSelectedMarketId(market.id)}
              >
                <div>
                  <p className="market-title">{market.title}</p>
                  <div className="market-meta">
                    <span className="category-dot" style={{ backgroundColor: getCategoryColor(market.category) }} />
                    <span>{market.category}</span>
                    {marketClosed && <span className="status-pill closed">Closed</span>}
                    {rankReliable === false && <span className="status-pill caution">Rank caution</span>}
                    <span>•</span>
                    <span>{formatTimestamp(market.last_updated)}</span>
                    <span>•</span>
                    <span>
                      {market.last_trade_time
                        ? `Last trade ${formatTimestamp(market.last_trade_time)}`
                        : 'No recent trades'}
                    </span>
                  </div>
                </div>
                <div className="market-stats">
                  <div>
                    <p>Volume</p>
                    <strong>{formatCurrency(market.volume_24h)}</strong>
                  </div>
                  <div>
                    <p>Avg trade</p>
                    <strong>{formatCurrency(market.avg_trade_size_window)}</strong>
                  </div>
                  <div>
                    <p>Cat z</p>
                    <strong>
                      {typeof market.retail_score_z === 'number'
                        ? market.retail_score_z.toFixed(2)
                        : '--'}
                    </strong>
                  </div>
                  <div className={`signal-pill ${retailStatus.level}`}>
                    {retailStatus.label}
                  </div>
                </div>
                <ArrowUpRight className="icon" />
              </button>
            )})}
          </div>
        </div>

        <div className="panel market-detail">
          {!marketDetail && (
            <div className="empty-state">
              <p className="eyebrow">Pick a market</p>
              <h3 className="section-title">Select a market to unpack retail flow.</h3>
              <p className="hint">We will surface trade size bias, spikes, and lifecycle signals.</p>
            </div>
          )}

          {marketDetail && (
            <>
              <div className="detail-header">
                <div>
                  <p className="eyebrow">{marketDetail.market.category}</p>
                  <h2 className="section-title">{marketDetail.market.title}</h2>
                  <div className="detail-tags">
                    {normalizeTags(marketDetail.market.event_tags).slice(0, 3).map((tag) => (
                      <span key={tag.label || tag} className="tag">
                        {tag.label || tag}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="detail-status">
                  <p>Lifecycle</p>
                  <strong>{marketDetail.lifecycle.stage}</strong>
                  <span className="hint">Growth {marketDetail.lifecycle.growth_rate?.toFixed(2)}x</span>
                  <span className="hint">Confidence {marketDetail.lifecycle.confidence?.toFixed(2)}</span>
                  {typeof selectedMarketRow?.retail_score_z === 'number' && (
                    <span className="hint">Category z {selectedMarketRow.retail_score_z.toFixed(2)}</span>
                  )}
                  <span className={`coverage-pill ${marketDetail.retail_signals.coverage || 'snapshot'}`}>
                    {formatCoverageLabel(
                      marketDetail.retail_signals.coverage,
                      marketDetail.retail_signals.sample_trades
                    )}
                  </span>
                  {typeof marketDetail.retail_signals.rank_reliable === 'boolean' && (
                    <span className={`status-pill ${marketDetail.retail_signals.rank_reliable ? 'reliable' : 'caution'}`}>
                      {marketDetail.retail_signals.rank_reliable ? 'Rank reliable' : 'Rank caution'}
                    </span>
                  )}
                  {detailIsClosed && <span className="status-pill closed">Closed</span>}
                  {detailConfidenceLabel && <span className="hint">{detailConfidenceLabel}</span>}
                </div>
              </div>

              <div className="detail-metrics">
                <div>
                  <p>Last price</p>
                  <strong>{marketDetail.latest.price?.toFixed(3) ?? '--'}</strong>
                </div>
                <div>
                  <p>24h volume</p>
                  <strong>{formatCurrency(marketDetail.latest.volume_24h)}</strong>
                </div>
                <div>
                  <p>Liquidity</p>
                  <strong>{formatCurrency(marketDetail.latest.liquidity)}</strong>
                </div>
                <div>
                  <p>Avg trade</p>
                  <strong>{formatCurrency(marketDetail.latest.avg_trade_size_window)}</strong>
                </div>
                <div>
                  <p>Last trade</p>
                  <strong>{formatTimestamp(marketDetail.latest.last_trade_time)}</strong>
                </div>
                <div>
                  <p>Last snapshot</p>
                  <strong>{formatTimestamp(marketDetail.latest.timestamp)}</strong>
                </div>
              </div>

              {detailCoverageNote && (
                <div className="coverage-note">{detailCoverageNote}</div>
              )}
              {staleNote && (
                <div className="coverage-note warning">{staleNote}</div>
              )}
              {qualityNote && (
                <div className="coverage-note warning">{qualityNote}</div>
              )}
              {whaleNote && (
                <div className="coverage-note warning">{whaleNote}</div>
              )}
              {reliabilityNote && (
                <div className="coverage-note warning">{reliabilityNote}</div>
              )}

              <div className="signal-grid">
                <div className="signal-card">
                  <p>Retail bias</p>
                  <strong>{formatPercent(marketDetail.retail_signals.small_trade_share)}</strong>
                  <span className="hint">Trades below threshold</span>
                </div>
                <div className="signal-card">
                  <p>Retail volume share</p>
                  <strong>{formatPercent(marketDetail.retail_signals.retail_volume_share)}</strong>
                  <span className="hint">Retail share by volume</span>
                </div>
                <div className="signal-card">
                  <p>Off-hours share</p>
                  <strong>{formatPercent(marketDetail.retail_signals.evening_share)}</strong>
                  <span className="hint">Evening & late night</span>
                </div>
                <div className="signal-card">
                  <p>Weekend share</p>
                  <strong>{formatPercent(marketDetail.retail_signals.weekend_share)}</strong>
                  <span className="hint">Sat/Sun trade count</span>
                </div>
                <div className="signal-card">
                  <p>Burstiness</p>
                  <strong>{marketDetail.retail_signals.burstiness?.toFixed(2)}x</strong>
                  <span className="hint">Peak hourly surge</span>
                </div>
                <div className="signal-card">
                  <p>Whale share</p>
                  <strong>{formatPercent(marketDetail.retail_signals.whale_share)}</strong>
                  <span className="hint">Top 1% trade volume</span>
                </div>
                <div className="signal-card flow-attention-card">
                  <p>Flow vs attention</p>
                  <div className="flow-attention-grid">
                    <div className="flow-attention-dot" style={{
                      left: `${Math.min(Math.max(flowAttention.flow, 0), 1) * 100}%`,
                      top: `${(1 - Math.min(Math.max(flowAttention.attention, 0), 1)) * 100}%`
                    }} />
                    <span className="flow-label">Flow</span>
                    <span className="attention-label">Attention</span>
                  </div>
                  <span className="hint">{flowAttention.label}</span>
                </div>
              </div>

              <div className="chart-grid">
                <div className="chart-card">
                  <div className="chart-header">
                    <span>Price + volume arc</span>
                    <span className="hint">Last {windowDays} days</span>
                  </div>
                  <div className="chart-body">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={historyChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                        <XAxis dataKey="time" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip
                          formatter={(value, name) => {
                            if (name === 'volume') return [formatCurrency(value), 'Volume'];
                            if (name === 'price') return [value?.toFixed(3), 'Price'];
                            return [value, name];
                          }}
                        />
                        <Area yAxisId="left" type="monotone" dataKey="volume" fill="#f2b13455" stroke="#f2b134" />
                        <Line yAxisId="right" type="monotone" dataKey="price" stroke="#1b998b" strokeWidth={2} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-card">
                  <div className="chart-header">
                    <span>Trade size split</span>
                    <span className="hint">Last {windowDays} days trades</span>
                  </div>
                  <div className="chart-body">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={tradeSizeData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                        <XAxis dataKey="range" />
                        <YAxis />
                        <Tooltip
                          formatter={(value, name) => [
                            name === 'count' ? formatNumber(value) : formatPercent(value),
                            name === 'count' ? 'Trades' : 'Share'
                          ]}
                        />
                        <Bar dataKey="count" fill="#2d7dd2" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-card">
                  <div className="chart-header">
                    <span>Hourly retail rhythm</span>
                    <span className="hint">Last {windowDays} days trades</span>
                  </div>
                  <div className="chart-body">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={hourlyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                        <XAxis dataKey="hour" />
                        <YAxis />
                        <Tooltip
                          formatter={(value, name) => [
                            formatNumber(value),
                            name === 'retailTrades' ? 'Retail trades' : 'Total trades'
                          ]}
                        />
                        <Area type="monotone" dataKey="totalTrades" stroke="#e07a3f" fill="#e07a3f33" />
                        <Area type="monotone" dataKey="retailTrades" stroke="#d1495b" fill="#d1495b33" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="insight-row">
                <div>
                  <p className="eyebrow">Retail drivers</p>
                  <div className="driver-tags">
                    {marketDetail.retail_signals.drivers.length ? (
                      marketDetail.retail_signals.drivers.map((driver) => (
                        <span key={driver} className="tag">
                          {driver.replace(/_/g, ' ')}
                        </span>
                      ))
                    ) : (
                      <span className="hint">No dominant retail drivers yet.</span>
                    )}
                  </div>
                </div>
                <div className="retail-score">
                  <div>
                    <p>Retail score</p>
                    <strong className={`score ${marketDetail.retail_signals.level}`}>
                      {marketDetail.retail_signals.level}
                    </strong>
                  </div>
                  <div>
                    <p>Score</p>
                    <strong>
                      {marketDetail.retail_signals.score}/{marketDetail.retail_signals.score_max ?? 10}
                    </strong>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="trend-section">
        <div className="panel trend-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Long-term signal</p>
              <h2 className="section-title">Retail trend index</h2>
              <p className="hint">Daily rollups for flow vs attention across categories.</p>
            </div>
            <div className="controls">
              <div className="select-field">
                <select
                  value={trendCategory}
                  onChange={(event) => setTrendCategory(event.target.value)}
                >
                  <option value="all">All categories</option>
                  {categories.map((cat) => (
                    <option key={cat.category} value={cat.category}>
                      {cat.category}
                    </option>
                  ))}
                </select>
              </div>
              <div className="toggle-group">
                {TREND_RANGES.map((range) => {
                  const isActive = range.mode === 'hourly'
                    ? trendGranularity === 'hourly' && trendHours === range.hours
                    : trendGranularity === 'daily' && trendDays === range.days;
                  return (
                    <button
                      key={range.label}
                      className={`toggle-btn ${isActive ? 'active' : ''}`}
                      onClick={() => {
                        setTrendGranularity(range.mode);
                        if (range.mode === 'hourly') {
                          setTrendHours(range.hours);
                        } else {
                          setTrendDays(range.days);
                        }
                      }}
                      type="button"
                    >
                      {range.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="metrics-grid trend-metrics">
            <div className="panel metric-card">
              <div className="metric-header">
                <TrendingUp className="icon" />
                Retail score
              </div>
              <div className="metric-value">
                {trendSummary?.latest_retail_score?.toFixed(2) ?? '--'}
              </div>
              <p className="hint">
                Latest {trendGranularity === 'hourly' ? 'hourly' : 'daily'} index
              </p>
            </div>
            <div className="panel metric-card">
              <div className="metric-header">
                <Zap className="icon" />
                Flow vs attention
              </div>
              <div className="metric-value">
                {flowMetric} / {attentionMetric}
              </div>
              <p className="hint">Flow / attention scores</p>
            </div>
            <div className="panel metric-card">
              <div className="metric-header">
                <Activity className="icon" />
                Retail share
              </div>
              <div className="metric-value">
                {formatPercent(trendSummary?.latest_retail_trade_share)}
              </div>
              <p className="hint">
                {trendGranularity === 'hourly' ? 'Hourly' : 'Daily'} retail trade share
              </p>
            </div>
            <div className="panel metric-card">
              <div className="metric-header">
                <BarChart3 className="icon" />
                Whale share
              </div>
              <div className="metric-value">
                {formatPercent(trendSummary?.latest_whale_share)}
              </div>
              <p className="hint">Top 1% volume share</p>
            </div>
          </div>

          <div className="chart-grid trend-chart-grid">
            <div className="chart-card">
              <div className="chart-header">
                <span>Retail index ({trendGranularity === 'hourly' ? 'hourly' : 'daily'})</span>
                <span className="hint">
                  {trendGranularity === 'hourly'
                    ? `${trendHours}h view`
                    : `${trendDays} day view`}
                </span>
              </div>
              <div className="chart-body">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={trendChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip
                      formatter={(value, name) => {
                        if (name === 'retailScore') return [value?.toFixed(2), 'Retail score'];
                        if (name === 'flowIndex') return [value?.toFixed(2), 'Flow index'];
                        if (name === 'attentionIndex') return [value?.toFixed(2), 'Attention index'];
                        return [value, name];
                      }}
                    />
                    <Area type="monotone" dataKey="retailScore" fill="#d1495b33" stroke="#d1495b" />
                    <Line type="monotone" dataKey="flowIndex" stroke="#1b998b" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="attentionIndex" stroke="#f2b134" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-card">
              <div className="chart-header">
                <span>Participation + whales</span>
                <span className="hint">Trade count vs whale share</span>
              </div>
              <div className="chart-body">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={trendChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                    <XAxis dataKey="date" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip
                      formatter={(value, name) => {
                        if (name === 'trades') return [formatNumber(value), 'Trades'];
                        if (name === 'whaleShare') return [formatPercent(value), 'Whale share'];
                        return [value, name];
                      }}
                    />
                    <Bar yAxisId="left" dataKey="trades" fill="#2d7dd2" radius={[6, 6, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="whaleShare" stroke="#8f5d2e" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {trendLoading && (
            <div className="coverage-note">Loading rollup data…</div>
          )}
        </div>
      </section>
    </div>
  );
}
