import os
from pathlib import Path
from dotenv import load_dotenv

# Polymarket API Configuration
POLYMARKET_API_BASE = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")

# Database Configuration
DEFAULT_DB_PATH = BASE_DIR / "prediction_markets.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Markets to track (we'll start with popular ones)
TRACKED_MARKETS = [
    # We'll populate this after we see what's available
]

# Categories to track for research focus
TRACKED_CATEGORIES = ["crypto", "sports", "politics"]

# Public trade feed backfill depth (pages of 500 trades)
TRADE_PAGES = 30
TRADE_LOOKBACK_DAYS_FAST = 7
TRADE_LOOKBACK_DAYS_FULL = 30

# Data collection interval (in seconds)
COLLECTION_INTERVAL = 900  # 15 minutes

# Noise filters / quality gates
MIN_MARKET_VOLUME_24H = 1000.0
MIN_MARKET_LIQUIDITY = 500.0
TRADE_TRIM_PCT = 0.05
TRADE_WINSOR_PCT = 0.05
BURSTINESS_SMOOTH_WINDOW = 3
ROLLUP_DAYS = 180

# Retail detection thresholds
RETAIL_THRESHOLDS = {
    "volume_spike_multiplier": 2.0,
    "high_volatility_threshold": 0.15,
    "retail_hours_utc": [0, 1, 2, 3, 4, 22, 23],  # Evening US time
}

# Retail trade classification settings
RETAIL_SIZE_PERCENTILES = {
    "crypto": 0.25,
    "sports": 0.3,
    "politics": 0.35,
    "default": 0.3
}
RETAIL_MIN_TRADES = 25
RETAIL_FALLBACK_THRESHOLD = 100.0
