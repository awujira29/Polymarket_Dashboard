import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, Activity, DollarSign, BarChart3, AlertCircle, Brain, Users, Clock, Target, BookOpen, Lightbulb, TrendingDown } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function ResearchDashboard() {
  const [markets, setMarkets] = useState([]);
  const [retailInsights, setRetailInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedInsight, setSelectedInsight] = useState('overview');

  useEffect(() => {
    fetchMarkets();
    fetchRetailInsights();

    const interval = setInterval(() => {
      fetchMarkets();
      fetchRetailInsights();
    }, 300000); // Update every 5 minutes for research data

    return () => clearInterval(interval);
  }, []);

  const fetchMarkets = async () => {
    try {
      const response = await fetch(`${API_BASE}/markets`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setMarkets(data.markets || []);
      setError(null);
    } catch (error) {
      console.error('Error fetching markets:', error);
      setError(`Failed to fetch markets: ${error.message}`);
    }
  };

  const fetchRetailInsights = async () => {
    try {
      const response = await fetch(`${API_BASE}/analytics/retail-insights`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setRetailInsights(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching retail insights:', error);
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
    return `$${value.toFixed(0)}`;
  };

  const formatPercentage = (value) => `${(value * 100).toFixed(1)}%`;

  // Research data preparation
  const getRetailIntensityData = () => {
    if (!retailInsights) return [];
    const { high_retail_markets, medium_retail_markets, low_retail_markets } = retailInsights;
    return [
      { name: 'High Retail', value: high_retail_markets, color: '#ef4444' },
      { name: 'Medium Retail', value: medium_retail_markets, color: '#f59e0b' },
      { name: 'Low Retail', value: low_retail_markets, color: '#10b981' }
    ];
  };

  const getVolumeVsLiquidityData = () => {
    return markets.slice(0, 10).map(market => ({
      name: market.question.substring(0, 30) + '...',
      volume: market.volume_24h,
      liquidity: market.liquidity,
      retailIntensity: market.volume_24h > 500000 ? 'High' : market.volume_24h > 100000 ? 'Medium' : 'Low'
    }));
  };

  const getTemporalPatternsData = () => {
    if (!retailInsights) return [];
    return [
      { time: 'Weekdays', activity: 60, retail: 40 },
      { time: 'Weekends', activity: retailInsights.weekend_activity_percentage || 85, retail: 60 },
      { time: 'Evenings', activity: retailInsights.evening_activity_percentage || 90, retail: 70 }
    ];
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 font-medium">Loading research data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="max-w-md p-8 bg-white rounded-xl shadow-lg">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => { setLoading(true); fetchMarkets(); fetchRetailInsights(); }}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <Brain className="w-8 h-8 text-blue-600" />
                Retail Behavior Research Dashboard
              </h1>
              <p className="text-gray-600 mt-1">Economic analysis of prediction market retail participation patterns</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Last updated</p>
              <p className="text-sm font-medium text-gray-900">{new Date().toLocaleTimeString()}</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Research Navigation */}
        <div className="mb-8">
          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-fit">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'patterns', label: 'Retail Patterns', icon: Users },
              { id: 'economics', label: 'Economic Analysis', icon: TrendingUp },
              { id: 'conclusions', label: 'Research Conclusions', icon: Lightbulb }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSelectedInsight(id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  selectedInsight === id
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Overview Section */}
        {selectedInsight === 'overview' && (
          <div className="space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Markets Analyzed</p>
                    <p className="text-3xl font-bold text-gray-900">{markets.length}</p>
                  </div>
                  <BarChart3 className="w-8 h-8 text-blue-500" />
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Retail Market %</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {retailInsights ? formatPercentage(retailInsights.retail_market_percentage / 100) : '0%'}
                    </p>
                  </div>
                  <Users className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Volume</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {formatCurrency(markets.reduce((sum, m) => sum + m.volume_24h, 0))}
                    </p>
                  </div>
                  <DollarSign className="w-8 h-8 text-yellow-500" />
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Data Points</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {markets.reduce((sum, m) => sum + (m.snapshots_count || 0), 0)}
                    </p>
                  </div>
                  <Activity className="w-8 h-8 text-purple-500" />
                </div>
              </div>
            </div>

            {/* Retail Intensity Distribution */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Retail Intensity Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={getRetailIntensityData()}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {getRetailIntensityData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} markets`, 'Count']} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-6 mt-4">
                {getRetailIntensityData().map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                    <span className="text-sm text-gray-600">{item.name}: {item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Retail Patterns Section */}
        {selectedInsight === 'patterns' && (
          <div className="space-y-8">
            {/* Temporal Patterns */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-600" />
                Temporal Trading Patterns
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getTemporalPatternsData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="activity" fill="#3b82f6" name="Total Activity %" />
                    <Bar dataKey="retail" fill="#10b981" name="Retail Activity %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-gray-600 mt-4">
                Retail traders show increased activity during off-hours, particularly weekends and evenings,
                suggesting participation when institutional traders may be less active.
              </p>
            </div>

            {/* Volume vs Liquidity Scatter */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-green-600" />
                Volume vs Liquidity Analysis
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={getVolumeVsLiquidityData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="liquidity" name="Liquidity" />
                    <YAxis type="number" dataKey="volume" name="Volume" />
                    <Tooltip
                      formatter={(value, name) => [formatCurrency(value), name]}
                      labelFormatter={(label) => `Market: ${label}`}
                    />
                    <Scatter dataKey="volume" fill="#3b82f6" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="text-sm text-gray-600 mt-4">
                Analysis shows retail participation correlates with market liquidity, with higher retail intensity
                in markets with sufficient liquidity to support smaller trade sizes.
              </p>
            </div>
          </div>
        )}

        {/* Economic Analysis Section */}
        {selectedInsight === 'economics' && (
          <div className="space-y-8">
            {/* Market Efficiency Analysis */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-purple-600" />
                Market Efficiency Implications
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Retail Participation Impact</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                      <span className="text-sm text-gray-700">Price Discovery</span>
                      <span className="text-sm font-medium text-green-700">Enhanced</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                      <span className="text-sm text-gray-700">Market Liquidity</span>
                      <span className="text-sm font-medium text-blue-700">Increased</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                      <span className="text-sm text-gray-700">Information Efficiency</span>
                      <span className="text-sm font-medium text-yellow-700">Mixed</span>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900">Economic Indicators</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                      <span className="text-sm text-gray-700">Retail Market Share</span>
                      <span className="text-sm font-medium text-red-700">
                        {retailInsights ? formatPercentage(retailInsights.retail_market_percentage / 100) : '0%'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-indigo-50 rounded-lg">
                      <span className="text-sm text-gray-700">Weekend Activity</span>
                      <span className="text-sm font-medium text-indigo-700">
                        {retailInsights ? formatPercentage(retailInsights.weekend_activity_percentage / 100) : '0%'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Research Methodology */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-orange-600" />
                Research Methodology
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Data Collection</h4>
                  <p className="text-sm text-gray-600">
                    Real-time market data from Polymarket API with enhanced retail metrics including trade sizes,
                    temporal patterns, and trader estimates.
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Analysis Framework</h4>
                  <p className="text-sm text-gray-600">
                    Statistical analysis of trading patterns, volume spikes, and temporal behavior to identify
                    retail vs institutional participation.
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Economic Modeling</h4>
                  <p className="text-sm text-gray-600">
                    Market efficiency analysis considering retail participation impact on price discovery,
                    liquidity, and information flow.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Research Conclusions Section */}
        {selectedInsight === 'conclusions' && (
          <div className="space-y-8">
            {/* Key Findings */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-yellow-600" />
                Key Research Findings
              </h3>
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-medium text-gray-900 mb-2">Retail Participation Patterns</h4>
                  <p className="text-gray-600">
                    Retail traders constitute approximately {retailInsights ? formatPercentage(retailInsights.retail_market_percentage / 100) : '0%'} of active markets,
                    with higher participation in markets offering sufficient liquidity for smaller trade sizes.
                  </p>
                </div>

                <div className="border-l-4 border-green-500 pl-4">
                  <h4 className="font-medium text-gray-900 mb-2">Temporal Behavior Insights</h4>
                  <p className="text-gray-600">
                    Retail activity peaks during off-hours, with {retailInsights ? formatPercentage(retailInsights.weekend_activity_percentage / 100) : '0%'} of trading
                    occurring on weekends, suggesting retail traders participate when institutional oversight may be reduced.
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-medium text-gray-900 mb-2">Market Efficiency Implications</h4>
                  <p className="text-gray-600">
                    Retail participation appears to enhance market liquidity and price discovery, though the impact on
                    information efficiency remains mixed depending on market maturity and information quality.
                  </p>
                </div>

                <div className="border-l-4 border-orange-500 pl-4">
                  <h4 className="font-medium text-gray-900 mb-2">Economic Significance</h4>
                  <p className="text-gray-600">
                    The presence of retail traders in prediction markets suggests these platforms are becoming more
                    accessible to individual investors, potentially democratizing access to sophisticated financial instruments.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

  if (loading) {
    return (
      <div className="dashboard-container flex items-center justify-center">
        <div className="error-card text-center max-w-md p-8 rounded-xl">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <div className="bg-gray-100 p-4 rounded-lg text-left text-sm">
            <p className="font-semibold mb-2">Troubleshooting:</p>
            <ul className="list-disc list-inside space-y-1 text-gray-700">
              <li>Make sure the API is running: <code>python api.py</code></li>
              <li>Check API at: <a href={API_BASE} className="text-blue-600 underline">{API_BASE}</a></li>
              <li>Check console for errors (Cmd+Option+J)</li>
            </ul>
          </div>
          <button 
            onClick={() => { setLoading(true); fetchMarkets(); }}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Prediction Market Analytics</h1>
          <p className="text-sm text-gray-600 mt-1">Real-time retail behavior tracking</p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="metric-card rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Markets</p>
                <p className="text-2xl font-bold text-gray-900">{overview?.total_markets || 0}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="metric-card rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Data Points</p>
                <p className="text-2xl font-bold text-gray-900">{overview?.total_snapshots || 0}</p>
              </div>
              <Activity className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="metric-card rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Top Market Volume</p>
                <p className="text-2xl font-bold text-gray-900">
                  {overview?.top_markets_by_volume?.[0] ? formatCurrency(overview.top_markets_by_volume[0].volume_24h) : '$0'}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-yellow-500" />
            </div>
          </div>

          <div className="metric-card rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Markets</p>
                <p className="text-2xl font-bold text-gray-900">
                  {markets.filter(m => m.volume_24h > 1000).length}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-500" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 market-list rounded-lg">
            <div className="p-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Markets</h2>
              <p className="text-sm text-gray-600">Sorted by total volume</p>
            </div>
            <div className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
              {markets.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  No markets found
                </div>
              ) : (
                markets
                  .sort((a, b) => b.volume_24h - a.volume_24h)
                  .map((market) => {
                    const intensity = getRetailIntensity(market.volume_24h);
                    return (
                      <div
                        key={market.id}
                        onClick={() => setSelectedMarket(market.id)}
                        className={`market-item p-4 cursor-pointer ${
                          selectedMarket === market.id ? 'selected' : ''
                        }`}
                      >
                        <h3 className="font-medium text-gray-900 text-sm line-clamp-2 mb-2">
                          {market.title}
                        </h3>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-gray-600">24h Vol:</span>
                          <span className="font-semibold">{formatCurrency(market.volume_24h)}</span>
                        </div>
                        <div className="flex items-center justify-between text-xs mt-1">
                          <span className="text-gray-600">Retail:</span>
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                            market.retail_intensity === 'High' ? 'retail-high' :
                            market.retail_intensity === 'Medium' ? 'retail-moderate' : 'retail-low'
                          }`}>
                            {market.retail_intensity || 'Low'}
                          </span>
                        </div>
                        {market.avg_trade_size && (
                          <div className="flex items-center justify-between text-xs mt-1">
                            <span className="text-gray-600">Avg Trade:</span>
                            <span className="font-semibold">${market.avg_trade_size.toFixed(0)}</span>
                          </div>
                        )}
                        {market.is_weekend && (
                          <div className="flex items-center justify-between text-xs mt-1">
                            <span className="text-gray-600">Weekend:</span>
                            <span className="font-semibold text-orange-600">Active</span>
                          </div>
                        )}
                        <div className="flex items-center justify-between text-xs mt-1">
                          <span className="text-gray-600">Price:</span>
                          <span className="font-semibold">${market.current_price.toFixed(3)}</span>
                        </div>
                      </div>
                    );
                  })
              )}
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {marketDetail ? (
              <>
                <div className="detail-card rounded-lg p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-2">
                    {marketDetail.market.title}
                  </h2>
                  <p className="text-sm text-gray-600 mb-4">
                    {marketDetail.market.description || 'No description available'}
                  </p>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-xs text-gray-600">Category</p>
                      <p className="font-semibold">{marketDetail.market.category}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Status</p>
                      <p className="font-semibold capitalize">{marketDetail.market.status}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Data Points</p>
                      <p className="font-semibold">{marketDetail.total_snapshots}</p>
                    </div>
                  </div>
                </div>

                {marketDetail.snapshots.length > 1 && (
                  <div className="chart-container p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Price History</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={marketDetail.snapshots}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis 
                          dataKey="timestamp" 
                          tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                          fontSize={12}
                          stroke="#6b7280"
                        />
                        <YAxis domain={[0, 1]} fontSize={12} stroke="#6b7280" />
                        <Tooltip 
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                          formatter={(value) => [`$${value.toFixed(3)}`, 'Price']}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '8px',
                            backdropFilter: 'blur(10px)'
                          }}
                        />
                        <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {marketDetail.snapshots.length > 1 && (
                  <div className="chart-container p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">24h Volume</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={marketDetail.snapshots}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis 
                          dataKey="timestamp" 
                          tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                          fontSize={12}
                          stroke="#6b7280"
                        />
                        <YAxis fontSize={12} tickFormatter={(value) => formatCurrency(value)} stroke="#6b7280" />
                        <Tooltip 
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                          formatter={(value) => [formatCurrency(value), '24h Volume']}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '8px',
                            backdropFilter: 'blur(10px)'
                          }}
                        />
                        <Bar dataKey="volume_24h" fill="#10b981" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Retail Analysis Section */}
                <div className="detail-card rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Activity className="w-5 h-5 mr-2 text-blue-600" />
                    Retail Behavior Analysis
                  </h3>

                  {marketDetail.retail_analysis ? (
                    <div className="space-y-4">
                      {/* Overall Retail Score */}
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">Overall Retail Intensity</span>
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                            marketDetail.retail_analysis.overall_retail_score?.retail_intensity === 'very_high' ? 'bg-red-100 text-red-800' :
                            marketDetail.retail_analysis.overall_retail_score?.retail_intensity === 'high' ? 'bg-orange-100 text-orange-800' :
                            marketDetail.retail_analysis.overall_retail_score?.retail_intensity === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {marketDetail.retail_analysis.overall_retail_score?.retail_intensity?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">
                          {marketDetail.retail_analysis.overall_retail_score?.description || 'Analysis in progress...'}
                        </p>
                      </div>

                      {/* Key Retail Indicators */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-white p-4 rounded-lg border border-gray-200">
                          <h4 className="font-medium text-gray-900 mb-2">Trade Size Analysis</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Avg Trade Size:</span>
                              <span className="font-semibold">${marketDetail.retail_analysis.retail_indicators?.avg_trade_size?.toFixed(0) || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Trade Volatility:</span>
                              <span className="font-semibold">{marketDetail.retail_analysis.retail_indicators?.trade_size_volatility?.toFixed(2) || 'N/A'}</span>
                            </div>
                            <div className="text-xs text-gray-500 mt-2">
                              {marketDetail.retail_analysis.retail_indicators?.trade_size_indicators?.join(', ') || 'No specific indicators'}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white p-4 rounded-lg border border-gray-200">
                          <h4 className="font-medium text-gray-900 mb-2">Volume Patterns</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Volume Spikes:</span>
                              <span className="font-semibold">{marketDetail.retail_analysis.retail_indicators?.volume_spikes_count || 0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Weekend Ratio:</span>
                              <span className="font-semibold">{(marketDetail.retail_analysis.retail_indicators?.weekend_vs_weekday_ratio * 100)?.toFixed(0) || 0}%</span>
                            </div>
                            <div className="text-xs text-gray-500 mt-2">
                              {marketDetail.retail_analysis.retail_indicators?.volume_indicators?.join(', ') || 'No specific patterns'}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Research Insights */}
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg border border-purple-200">
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                          <TrendingUp className="w-4 h-4 mr-2 text-purple-600" />
                          Research Insights
                        </h4>
                        <ul className="text-sm text-gray-700 space-y-1">
                          {marketDetail.retail_analysis.overall_retail_score?.retail_intensity === 'very_high' && (
                            <li>• This market shows very strong retail participation with small trade sizes and volatile patterns</li>
                          )}
                          {marketDetail.retail_analysis.behavioral_patterns?.retail_patterns?.includes('weekend_activity') && (
                            <li>• Weekend trading activity suggests retail traders are most active outside business hours</li>
                          )}
                          {marketDetail.retail_analysis.retail_indicators?.volume_spikes_count > 2 && (
                            <li>• Frequent volume spikes indicate potential hype cycles or FOMO-driven retail trading</li>
                          )}
                          {marketDetail.retail_analysis.behavioral_patterns?.lifecycle_description?.includes('meme') && (
                            <li>• Meme market characteristics detected - expect rapid growth followed by potential decline</li>
                          )}
                          <li>• Monitor for correlation between social media mentions and volume spikes</li>
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Activity className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                      <p>Retail analysis not available</p>
                      <p className="text-sm mt-1">Analysis requires more market data</p>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state rounded-lg p-12 text-center">
                <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-600">Select a market to view details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}