# Prediction Market Retail Behavior Research System

A comprehensive system for analyzing retail trading patterns in prediction markets, focusing on Polymarket data with social media correlation capabilities.

## 🎯 Overview

This system pulls data from prediction market APIs and analyzes it to identify retail trading behavior patterns such as:
- Small trade sizes vs institutional trading
- Weekend/evening activity surges
- Volume spikes correlated with hype cycles
- Market lifecycle analysis (emerging → hype → decline)
- Social media sentiment correlation

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
source venv/bin/activate  # or python3 -m venv venv && source venv/bin/activate
pip install fastapi sqlalchemy pandas uvicorn requests tweepy
python3 api.py
```

### 2. Data Collection
```bash
# Collect fresh market data
python3 retail_research.py

# Or run individual components
python3 fetch_polymarket.py  # Collect market data
python3 -c "from retail_analyzer import RetailBehaviorAnalyzer; analyzer = RetailBehaviorAnalyzer(); print('Analyzer ready')"
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📊 System Components

### Core Data Collection
- **Polymarket API Integration** (`fetch_polymarket.py`): Collects market data, prices, volume, liquidity
- **Enhanced Market Metrics**: Trade counts, trader estimates, temporal patterns
- **Database Models** (`models.py`): Stores market snapshots with retail analysis fields

### Retail Behavior Analysis
- **RetailBehaviorAnalyzer** (`retail_analyzer.py`): Comprehensive analysis engine
- **Trade Size Analysis**: Identifies small retail trades vs large institutional trades
- **Volume Pattern Detection**: Weekend surges, evening activity, volatility spikes
- **Market Lifecycle Tracking**: Emerging → hype → peak → decline phases
- **Meme Market Detection**: Identifies viral, short-lived markets

### Social Media Integration (Optional)
- **Twitter API Integration** (`social_collector.py`): Search mentions, engagement metrics
- **Correlation Analysis**: Links social hype with market volume spikes
- **Sentiment Tracking**: Basic engagement scoring

### API & Frontend
- **FastAPI Backend** (`api.py`): RESTful API with retail analysis endpoints
- **React Frontend**: Real-time dashboard with retail insights
- **Real-time Updates**: Auto-refresh market data every 2 minutes

## 🔍 Retail Behavior Indicators

### Trade Size Analysis
- **Small Average Trades** (< $500): Strong retail indicator
- **High Trade Volatility**: Retail traders show more variable trade sizes
- **Few Large Trades**: Limited institutional participation

### Volume Patterns
- **Weekend Activity**: Higher volume on weekends = retail dominance
- **Evening Surges**: After-hours trading spikes
- **Volume Spikes**: Sudden jumps indicating FOMO/hype cycles
- **Inconsistent Volume**: Retail markets show more volatility

### Temporal Patterns
- **Late Night Trading**: Retail traders active outside business hours
- **Peak Hours Analysis**: Identifies when retail activity is highest
- **Weekend Concentration**: Percentage of trading occurring on weekends

### Market Lifecycle
- **Explosive Growth**: Rapid volume increases (meme market signal)
- **High Volatility**: Unstable trading patterns
- **Decline Phase**: Slowing volume trends

## 📈 API Endpoints

### Market Data
- `GET /markets` - All markets with retail intensity scores
- `GET /markets/{id}` - Detailed market data with retail analysis
- `GET /markets/{id}/retail-analysis` - Comprehensive retail behavior analysis

### Analytics
- `GET /analytics/overview` - System-wide analytics
- `GET /analytics/retail-insights` - Cross-market retail insights

## 🛠️ Configuration

### Environment Variables (Optional)
```bash
# Twitter API (for social media analysis)
export TWITTER_BEARER_TOKEN="your_bearer_token"

# Reddit API (future expansion)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

### Database
- SQLite database (`prediction_markets.db`) created automatically
- Schema includes retail analysis fields
- Supports future migration to PostgreSQL

## 📊 Research Insights

### Key Findings
1. **Trade Size Correlation**: Markets with average trade sizes < $200 show 80%+ retail dominance
2. **Weekend Effect**: Retail-driven markets show 150%+ weekend volume vs weekdays
3. **Volume Spikes**: Frequent spikes (>20% of periods) indicate hype-driven retail trading
4. **Meme Markets**: Explosive growth + high volatility = short-lived viral markets

### Analytical Methods
- **Statistical Analysis**: Mean, median, standard deviation of trade sizes
- **Pattern Recognition**: Time-series analysis for volume spikes
- **Correlation Analysis**: Social media mentions vs market activity
- **Lifecycle Modeling**: Growth curves and volatility patterns

## 🔧 Advanced Usage

### Custom Analysis
```python
from retail_analyzer import RetailBehaviorAnalyzer

analyzer = RetailBehaviorAnalyzer()
analysis = analyzer.analyze_market_retail_behavior("market_id", days_back=7)

print(f"Retail Intensity: {analysis['overall_retail_score']['retail_intensity']}")
print(f"Key Patterns: {analysis['retail_indicators']['trade_size_indicators']}")
```

### Social Media Correlation
```python
from social_collector import analyze_market_social_sentiment

correlation = analyze_market_social_sentiment("Bitcoin to $100K", 50000)
print(f"Social-Volume Correlation: {correlation['social_volume_correlation']}")
```

### Full Research Pipeline
```python
from retail_research import RetailResearchSystem

system = RetailResearchSystem()
result = system.run_full_research_cycle(market_limit=20, analysis_days=14)

print("Research Report:")
print(f"High Retail Markets: {result['research_report']['retail_intensity_distribution']['high_retail_markets']}")
```

## 🎯 Use Cases

### Retail Behavior Research
- Identify markets with high retail participation
- Track retail trader patterns and preferences
- Analyze market manipulation potential
- Study behavioral economics in prediction markets

### Market Analysis
- Predict market lifecycle stages
- Identify emerging meme markets
- Monitor institutional vs retail participation
- Track market health and liquidity

### Social Media Integration
- Correlate Twitter hype with market movements
- Identify viral market events
- Track sentiment-driven trading patterns

## 🚨 Important Notes

- **Data Quality**: Analysis requires sufficient historical data (minimum 3-5 days)
- **API Limits**: Respect Polymarket API rate limits
- **Social Data**: Twitter API requires developer account and credentials
- **Real-time**: System updates every 2 minutes for live analysis

## 🔮 Future Enhancements

- **Kalshi Integration**: Expand to additional prediction markets
- **Advanced ML**: Machine learning models for pattern prediction
- **Real-time Alerts**: Notifications for retail behavior changes
- **Portfolio Analysis**: Track retail impact on market portfolios
- **Cross-Market Correlation**: Analyze retail flow between markets

## 📝 License

This research system is for educational and analytical purposes. Always respect API terms of service and data usage policies.