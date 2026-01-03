from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func
from database import get_db_session, init_db
from models import Market, MarketSnapshot, Trade, DataCollectionRun, RetailRollup
from retail_analyzer import RetailBehaviorAnalyzer
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from math import sqrt
import re
import time
import logging
import threading
from config import (
    TRACKED_CATEGORIES,
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD,
    MIN_MARKET_VOLUME_24H,
    MIN_MARKET_LIQUIDITY,
    TRADE_TRIM_PCT,
    TRADE_WINSOR_PCT,
    BURSTINESS_SMOOTH_WINDOW
)

app = FastAPI(title="Prediction Market Retail Behavior API")
logger = logging.getLogger("api")

@app.on_event("startup")
def startup_db():
    init_db()

@app.on_event("startup")
def start_cache_warmer():
    t = threading.Thread(target=_cache_warmer_loop, daemon=True)
    t.start()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retail analyzer
retail_analyzer = RetailBehaviorAnalyzer()

MAX_MARKET_DAYS = 14
MAX_ANALYTICS_DAYS = 90
MAX_HISTORY_HOURS = 240
MAX_TRADES_LIMIT = 300
MAX_HOURLY_RETAIL_HOURS = 168

DEFAULT_MARKET_DAYS = 5
DEFAULT_MARKET_LIMIT = 25
DEFAULT_OVERVIEW_DAYS = 5

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_SECONDS = 300
_CACHE_WARM_INTERVAL = 180

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Prediction Market Retail Behavior API",
        "version": "3.0.0",
        "features": ["real_time_retail_monitoring", "categorized_markets", "trade_data_analysis"]
    }

def _safe_median(values: List[float]) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2

def _normalize_category(value: str | None) -> str:
    if not value:
        return "uncategorized"
    return str(value).strip().lower()

_CLOSED_STATUS_VALUES = {
    "closed",
    "inactive",
    "resolved",
    "settled",
    "void",
    "voided",
    "cancelled",
    "canceled",
    "ended"
}

def _truthy_flag(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False

def _parse_date(value):
    """Best-effort parse of date/datetime strings into a naive UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is None else value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return None

def _is_closed_market(market, latest_snapshot=None) -> bool:
    """Derive a closed flag using multiple cues to avoid stale 'active' states."""
    if not market:
        return False
    if _truthy_flag(getattr(market, "closed", None)) or _truthy_flag(getattr(market, "archived", None)):
        return True
    status = str(market.status or "").strip().lower()
    if status in _CLOSED_STATUS_VALUES:
        return True
    end_date = _parse_date(getattr(market, "end_date", None) or getattr(market, "end_date_iso", None))
    if end_date and datetime.utcnow() >= end_date:
        return True
    return False

def _status_label(market, closed_flag: bool) -> str:
    """Normalize status, defaulting to closed when the derived flag is set."""
    status = str(getattr(market, "status", "") or "").strip().lower()
    if closed_flag:
        if status in _CLOSED_STATUS_VALUES:
            return status
        return "closed"
    return status or "active"

def _clamp_int(value: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value) -> None:
    _CACHE[key] = (time.time(), value)

def _cache_warmer_loop():
    while True:
        try:
            get_overview(DEFAULT_OVERVIEW_DAYS)
            get_overview(MAX_MARKET_DAYS)
            list_markets(limit=DEFAULT_MARKET_LIMIT, days=DEFAULT_MARKET_DAYS)
        except Exception as exc:  # pragma: no cover
            logger.warning("cache warm failed: %s", exc)
        time.sleep(_CACHE_WARM_INTERVAL)

def _tracked_categories_set() -> set[str]:
    if not TRACKED_CATEGORIES:
        return set()
    return {str(category).strip().lower() for category in TRACKED_CATEGORIES if category}

def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0
    if percentile <= 0:
        return min(values)
    if percentile >= 1:
        return max(values)
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight

def _retail_percentile_for_category(category: str | None) -> float:
    if not category:
        return RETAIL_SIZE_PERCENTILES.get("default", 0.3)
    return RETAIL_SIZE_PERCENTILES.get(category, RETAIL_SIZE_PERCENTILES.get("default", 0.3))

def _retail_threshold(
    values: List[float],
    category: str | None = None,
    min_trades: int = RETAIL_MIN_TRADES,
    fallback: float = RETAIL_FALLBACK_THRESHOLD
) -> tuple[float, str]:
    percentile = _retail_percentile_for_category(category)
    if len(values) < min_trades:
        return fallback, "fixed"
    return _percentile(values, percentile), "percentile"

def _confidence_label(sample: int, min_trades: int = RETAIL_MIN_TRADES) -> str:
    if sample >= min_trades * 3:
        return "high"
    if sample >= min_trades * 2:
        return "medium"
    if sample >= min_trades:
        return "low"
    return "insufficient"

def _confidence_score(sample: int, min_trades: int = RETAIL_MIN_TRADES) -> float:
    if sample <= 0:
        return 0.0
    return min(1.0, sample / (min_trades * 3))

def _retail_score(
    small_trade_share: float,
    evening_share: float,
    weekend_share: float,
    burstiness: float
) -> tuple[float, str, List[str], float]:
    small_score = min(small_trade_share / 0.6, 1.0) if small_trade_share else 0.0
    evening_score = min(evening_share / 0.35, 1.0) if evening_share else 0.0
    weekend_score = min(weekend_share / 0.35, 1.0) if weekend_share else 0.0
    burst_score = min(max(burstiness - 1.0, 0.0) / 1.5, 1.0) if burstiness else 0.0

    weighted = (
        (0.45 * small_score) +
        (0.15 * evening_score) +
        (0.15 * weekend_score) +
        (0.25 * burst_score)
    )
    score = round(weighted * 10, 2)

    drivers = []
    if small_score >= 0.7:
        drivers.append("small_trade_bias")
    if evening_score >= 0.7:
        drivers.append("evening_activity")
    if weekend_score >= 0.7:
        drivers.append("weekend_activity")
    if burst_score >= 0.6:
        drivers.append("volume_spikes")

    level = "low"
    if score >= 7:
        level = "high"
    elif score >= 4:
        level = "medium"

    return score, level, drivers, 10.0

def _winsorize(values: List[float], lower_pct: float, upper_pct: float) -> List[float]:
    if not values:
        return []
    lower = _percentile(values, lower_pct)
    upper = _percentile(values, 1 - upper_pct)
    return [min(max(v, lower), upper) for v in values]

def _trimmed_mean(values: List[float], trim_pct: float) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    trim = int(len(sorted_values) * trim_pct)
    if trim * 2 >= len(sorted_values):
        return sum(sorted_values) / len(sorted_values)
    trimmed = sorted_values[trim:-trim]
    return sum(trimmed) / len(trimmed) if trimmed else 0

def _smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start:idx + 1]
        smoothed.append(sum(bucket) / len(bucket))
    return smoothed

def _market_quality_ok(volume_24h: float | None, liquidity: float | None) -> tuple[bool, str | None]:
    volume_ok = (volume_24h or 0) >= MIN_MARKET_VOLUME_24H
    liquidity_ok = (liquidity or 0) >= MIN_MARKET_LIQUIDITY
    if volume_ok and liquidity_ok:
        return True, None
    if not volume_ok and not liquidity_ok:
        return False, "low_volume_and_liquidity"
    if not volume_ok:
        return False, "low_volume"
    return False, "low_liquidity"

def _canonical_market_id(market: Market) -> str:
    if getattr(market, "condition_id", None):
        return str(market.condition_id)
    title = (market.title or "").strip().lower()
    if not title:
        return str(market.id)
    slug = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    return slug or str(market.id)

def _rank_reliable(
    coverage: str | None,
    quality_ok: bool | None,
    sample_trades: int | None,
    trade_count_window: int | None,
    volume_24h: float | None,
    liquidity: float | None,
    min_trades: int = RETAIL_MIN_TRADES,
    min_volume: float = MIN_MARKET_VOLUME_24H,
    min_liquidity: float = MIN_MARKET_LIQUIDITY
) -> bool:
    if coverage != "trade":
        return False
    if not quality_ok:
        return False
    if (sample_trades or 0) < min_trades:
        return False
    if (trade_count_window or 0) < min_trades:
        return False
    if (volume_24h or 0) < min_volume:
        return False
    if (liquidity or 0) < min_liquidity:
        return False
    return True

def _whale_share(values: List[float]) -> tuple[float, int]:
    if not values:
        return 0.0, 0
    sorted_values = sorted(values, reverse=True)
    top_k = max(1, int(len(sorted_values) * 0.01))
    total = sum(sorted_values)
    if total <= 0:
        return 0.0, top_k
    whale_volume = sum(sorted_values[:top_k])
    return whale_volume / total, top_k

def _trade_size_to_share(avg_trade_size: float) -> float:
    if not avg_trade_size:
        return 0
    if avg_trade_size < 100:
        return 0.7
    if avg_trade_size < 250:
        return 0.5
    if avg_trade_size < 500:
        return 0.35
    if avg_trade_size < 1000:
        return 0.2
    return 0.1

def _linear_trend(values: List[float]) -> float:
    if len(values) < 2:
        return 0
    n = len(values)
    x_sum = (n - 1) * n / 2
    y_sum = sum(values)
    xy_sum = sum(i * v for i, v in enumerate(values))
    x2_sum = sum(i * i for i in range(n))
    denom = (n * x2_sum) - (x_sum ** 2)
    if denom == 0:
        return 0
    return (n * xy_sum - x_sum * y_sum) / denom

def _to_utc_iso(value: datetime | None) -> str | None:
    if not value:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()

def _parse_trade_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
        except ValueError:
            return None
    return None

def _compute_retail_signals(trades: List[Trade], category: str | None = None) -> Dict:
    if not trades:
        return {
            "total_trades": 0,
            "total_volume": 0,
            "avg_trade_size": 0,
            "median_trade_size": 0,
            "small_trade_share": 0,
            "retail_volume_share": 0,
            "evening_share": 0,
            "weekend_share": 0,
            "burstiness": 0,
            "volatility": 0,
            "whale_share": None,
            "whale_trade_count": None,
            "whale_dominated": False,
            "flow_score": 0,
            "attention_score": 0,
            "score": 0,
            "level": "low",
            "drivers": [],
            "retail_threshold": None,
            "retail_threshold_method": None,
            "coverage": "insufficient",
            "sample_trades": 0,
            "min_trades": RETAIL_MIN_TRADES,
            "confidence_label": "insufficient",
            "confidence_score": 0.0,
            "score_max": 10.0
        }

    filtered_trades = [
        t for t in trades
        if t.value is not None and t.value > 0 and t.timestamp is not None
    ]
    values = [t.value for t in filtered_trades]
    total_trades = len(values)
    if total_trades == 0:
        return _compute_retail_signals([])
    total_volume = sum(values)
    winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
    avg_trade_size = _trimmed_mean(winsorized, TRADE_TRIM_PCT)
    median_trade_size = _safe_median(winsorized)

    threshold, method = _retail_threshold(winsorized, category)
    small_trade_values = [v for v in values if v <= threshold]
    small_trade_count = len(small_trade_values)
    small_trade_share = small_trade_count / total_trades
    retail_volume_share = sum(small_trade_values) / total_volume if total_volume > 0 else 0
    whale_share, whale_count = _whale_share(values)
    whale_dominated = whale_share >= 0.5

    evening_count = sum(
        1 for t in filtered_trades
        if t.timestamp and (t.timestamp.hour >= 18 or t.timestamp.hour <= 5)
    )
    weekend_count = sum(
        1 for t in filtered_trades
        if t.timestamp and t.timestamp.weekday() >= 5
    )
    evening_share = evening_count / total_trades
    weekend_share = weekend_count / total_trades

    hourly_volume = {}
    for trade in filtered_trades:
        if not trade.timestamp:
            continue
        bucket = trade.timestamp.replace(minute=0, second=0, microsecond=0)
        hourly_volume[bucket] = hourly_volume.get(bucket, 0) + trade.value

    hourly_values = [hourly_volume[key] for key in sorted(hourly_volume.keys())]
    hourly_values = _smooth_series(hourly_values, BURSTINESS_SMOOTH_WINDOW)
    if hourly_values:
        hourly_mean = sum(hourly_values) / len(hourly_values)
        hourly_std = sqrt(sum((v - hourly_mean) ** 2 for v in hourly_values) / len(hourly_values))
        burstiness = max(hourly_values) / hourly_mean if hourly_mean > 0 else 0
        volatility = hourly_std / hourly_mean if hourly_mean > 0 else 0
    else:
        burstiness = 0
        volatility = 0

    score, level, drivers, score_max = _retail_score(
        small_trade_share,
        evening_share,
        weekend_share,
        burstiness
    )
    flow_score = min(1.0, (small_trade_share / 0.6) * 0.7 + (retail_volume_share / 0.5) * 0.3)
    attention_score = min(1.0, max(burstiness - 1.0, 0.0) / 1.5)

    return {
        "total_trades": total_trades,
        "total_volume": total_volume,
        "avg_trade_size": avg_trade_size,
        "median_trade_size": median_trade_size,
        "small_trade_share": small_trade_share,
        "retail_volume_share": retail_volume_share,
        "evening_share": evening_share,
        "weekend_share": weekend_share,
        "burstiness": burstiness,
        "volatility": volatility,
        "whale_share": whale_share,
        "whale_trade_count": whale_count,
        "whale_dominated": whale_dominated,
        "flow_score": flow_score,
        "attention_score": attention_score,
        "score": score,
        "score_max": score_max,
        "level": level,
        "drivers": drivers,
        "retail_threshold": threshold,
        "retail_threshold_method": method,
        "coverage": "trade",
        "sample_trades": total_trades,
        "min_trades": RETAIL_MIN_TRADES,
        "confidence_label": _confidence_label(total_trades),
        "confidence_score": _confidence_score(total_trades)
    }

def _compute_snapshot_signals(snapshots: List[MarketSnapshot]) -> Dict:
    if not snapshots:
        return _compute_retail_signals([])

    volumes = [s.volume_24h or 0 for s in snapshots]
    trade_counts = [s.volume_num_trades or 0 for s in snapshots]
    avg_trade_sizes = [s.avg_trade_size for s in snapshots if s.avg_trade_size]

    total_trades = sum(trade_counts)
    total_volume = volumes[-1] if volumes else 0
    avg_trade_size = _trimmed_mean(avg_trade_sizes, TRADE_TRIM_PCT) if avg_trade_sizes else 0
    median_trade_size = _safe_median(avg_trade_sizes) if avg_trade_sizes else avg_trade_size

    smoothed_volumes = _smooth_series(volumes, BURSTINESS_SMOOTH_WINDOW)
    mean_volume = sum(smoothed_volumes) / len(smoothed_volumes) if smoothed_volumes else 0
    std_volume = sqrt(sum((v - mean_volume) ** 2 for v in smoothed_volumes) / len(smoothed_volumes)) if mean_volume else 0
    burstiness = max(smoothed_volumes) / mean_volume if mean_volume > 0 else 0
    volatility = std_volume / mean_volume if mean_volume > 0 else 0

    evening_share = sum(1 for s in snapshots if s.is_evening) / len(snapshots)
    weekend_share = sum(1 for s in snapshots if s.is_weekend) / len(snapshots)

    small_trade_share = _trade_size_to_share(avg_trade_size)

    score, level, drivers, score_max = _retail_score(
        small_trade_share,
        evening_share,
        weekend_share,
        burstiness
    )
    flow_score = min(1.0, small_trade_share / 0.6) if small_trade_share else 0.0
    attention_score = min(1.0, max(burstiness - 1.0, 0.0) / 1.5) if burstiness else 0.0

    return {
        "total_trades": total_trades,
        "total_volume": total_volume,
        "avg_trade_size": avg_trade_size,
        "median_trade_size": median_trade_size,
        "small_trade_share": small_trade_share,
        "retail_volume_share": None,
        "evening_share": evening_share,
        "weekend_share": weekend_share,
        "burstiness": burstiness,
        "volatility": volatility,
        "whale_share": None,
        "whale_trade_count": None,
        "whale_dominated": False,
        "flow_score": flow_score,
        "attention_score": attention_score,
        "score": score,
        "score_max": score_max,
        "level": level,
        "drivers": drivers,
        "retail_threshold": None,
        "retail_threshold_method": None,
        "coverage": "snapshot",
        "sample_trades": total_trades,
        "min_trades": RETAIL_MIN_TRADES,
        "confidence_label": "insufficient",
        "confidence_score": 0.0
    }

def _compute_lifecycle(snapshots: List[MarketSnapshot]) -> Dict:
    if len(snapshots) < 2:
        return {
            "stage": "insufficient_data",
            "trend": 0,
            "growth_rate": 0,
            "volatility": 0,
            "spike_count": 0,
            "confidence": 0
        }

    volumes = [s.volume_24h or 0 for s in snapshots]
    trend = _linear_trend(volumes)
    growth_rate = (volumes[-1] / volumes[0] - 1) if volumes[0] > 0 else 0
    mean_volume = sum(volumes) / len(volumes)
    std_volume = sqrt(sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)) if mean_volume > 0 else 0
    volatility = std_volume / mean_volume if mean_volume > 0 else 0

    spike_count = 0
    for idx in range(1, len(volumes)):
        if volumes[idx - 1] > 0 and (volumes[idx] / volumes[idx - 1]) >= 1.8:
            spike_count += 1

    if growth_rate > 1.5 and volatility > 1.0:
        stage = "meme_spike"
    elif growth_rate > 0.5:
        stage = "rising"
    elif trend < -0.05:
        stage = "cooling"
    else:
        stage = "steady"

    coverage = min(1.0, len(snapshots) / 30)
    stability = max(0.0, 1.0 - min(volatility, 2.0) / 2.0)
    confidence = round(coverage * stability, 2)

    return {
        "stage": stage,
        "trend": trend,
        "growth_rate": growth_rate,
        "volatility": volatility,
        "spike_count": spike_count,
        "confidence": confidence
    }

def _bucket_trade_sizes(trades: List[Trade]) -> List[Dict]:
    buckets = [
        ("0-10", 0, 10),
        ("10-50", 10, 50),
        ("50-100", 50, 100),
        ("100-500", 100, 500),
        ("500-1000", 500, 1000),
        ("1000+", 1000, float("inf"))
    ]
    counts = {name: 0 for name, _, _ in buckets}
    for trade in trades:
        value = trade.value or 0
        for name, low, high in buckets:
            if low <= value < high:
                counts[name] += 1
                break

    total = sum(counts.values()) or 1
    return [
        {"range": name, "count": count, "share": count / total}
        for name, count in counts.items()
    ]

@app.get("/overview")
def get_overview(days: int = DEFAULT_OVERVIEW_DAYS, response: Response | None = None):
    """High-level overview for the Polymarket-only retail dashboard."""
    days = _clamp_int(days, minimum=1, maximum=MAX_MARKET_DAYS)
    cache_key = f"overview:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        if response is not None:
            response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
        return cached

    start_time = time.time()
    db = get_db_session()

    try:
        latest_run = db.query(DataCollectionRun)\
            .order_by(desc(DataCollectionRun.timestamp))\
            .first()

        markets = db.query(Market).all()
        tracked_set = _tracked_categories_set()
        market_by_id = {m.id: m for m in markets}

        category_stats: Dict[str, Dict] = {}
        market_rollup = []

        for market in markets:
            category = _normalize_category(market.category)
            if tracked_set and category not in tracked_set:
                continue
            if category not in category_stats:
                category_stats[category] = {
                    "market_count": 0,
                    "volume_24h": 0,
                    "liquidity": 0,
                    "avg_trade_size_sum": 0,
                    "avg_trade_size_count": 0,
                    "snapshot_count": 0,
                    "trade_count_24h": 0,
                    "retail_trades_24h": 0,
                    "trade_volume_24h": 0,
                    "retail_volume_24h": 0,
                    "retail_share_source": "snapshot",
                    "quality_market_count": 0
                }

            latest = (db.query(MarketSnapshot)
                      .filter_by(market_id=market.id)
                      .order_by(desc(MarketSnapshot.timestamp))
                      .first())

            category_stats[category]["market_count"] += 1
            if latest:
                category_stats[category]["volume_24h"] += latest.volume_24h or 0
                category_stats[category]["liquidity"] += latest.liquidity or 0
                category_stats[category]["snapshot_count"] += 1
                if latest.avg_trade_size is not None:
                    category_stats[category]["avg_trade_size_sum"] += latest.avg_trade_size
                    category_stats[category]["avg_trade_size_count"] += 1

                quality_ok, _ = _market_quality_ok(latest.volume_24h, latest.liquidity)
                if quality_ok:
                    category_stats[category]["quality_market_count"] += 1
                    market_rollup.append({
                        "id": market.id,
                        "canonical_id": _canonical_market_id(market),
                        "title": market.title,
                        "category": category,
                        "price": latest.price,
                        "volume_24h": latest.volume_24h,
                        "liquidity": latest.liquidity,
                        "avg_trade_size": latest.avg_trade_size
                    })

        since = datetime.utcnow() - timedelta(days=days)
        trades_24h = db.query(Trade)\
            .filter(Trade.timestamp >= since)\
            .all()

        category_trade_values: Dict[str, List[float]] = {}
        for trade in trades_24h:
            market = market_by_id.get(trade.market_id)
            category = _normalize_category(market.category) if market else "uncategorized"
            if tracked_set and category not in tracked_set:
                continue
            category_trade_values.setdefault(category, []).append(trade.value or 0)

        for category, values in category_trade_values.items():
            if category not in category_stats:
                category_stats[category] = {
                    "market_count": 0,
                    "volume_24h": 0,
                    "liquidity": 0,
                    "avg_trade_size_sum": 0,
                    "avg_trade_size_count": 0,
                    "snapshot_count": 0,
                    "trade_count_24h": 0,
                    "retail_trades_24h": 0,
                    "trade_volume_24h": 0,
                    "retail_volume_24h": 0,
                    "retail_share_source": "snapshot",
                    "quality_market_count": 0
                }

            trade_count = len(values)
            trade_volume = sum(values)
            category_stats[category]["trade_count_24h"] = trade_count
            category_stats[category]["trade_volume_24h"] = trade_volume

            if trade_count >= RETAIL_MIN_TRADES:
                threshold, _ = _retail_threshold(values, category)
                category_stats[category]["retail_trades_24h"] = sum(1 for v in values if v <= threshold)
                category_stats[category]["retail_volume_24h"] = sum(v for v in values if v <= threshold)
                category_stats[category]["retail_share_source"] = "trade"
            elif trade_count > 0:
                category_stats[category]["retail_share_source"] = "insufficient"

        categories = []
        for category, stats in category_stats.items():
            snapshot_count = stats["snapshot_count"] or 1
            avg_trade_size_count = stats["avg_trade_size_count"] or 0
            trade_count = stats.get("trade_count_24h", 0)
            trade_volume = stats.get("trade_volume_24h", 0)

            avg_trade_size = (
                stats["avg_trade_size_sum"] / avg_trade_size_count
                if avg_trade_size_count > 0 else None
            )

            if trade_count >= RETAIL_MIN_TRADES and stats.get("retail_share_source") == "trade":
                retail_share = stats["retail_trades_24h"] / stats["trade_count_24h"]
                retail_volume_share = stats["retail_volume_24h"] / trade_volume if trade_volume > 0 else 0
                retail_share_source = "trade"
            elif avg_trade_size is not None:
                retail_share = _trade_size_to_share(avg_trade_size)
                retail_volume_share = None
                retail_share_source = stats.get("retail_share_source", "snapshot")
            else:
                retail_share = None
                retail_volume_share = None
                retail_share_source = "insufficient"

            categories.append({
                "category": category,
                "market_count": stats["market_count"],
                "volume_24h": stats["volume_24h"],
                "avg_trade_size": avg_trade_size,
                "liquidity": stats["liquidity"],
                "trade_count_24h": stats["trade_count_24h"],
                "retail_trade_share": retail_share,
                "retail_volume_share": retail_volume_share,
                "retail_share_source": retail_share_source,
                "quality_market_count": stats.get("quality_market_count", 0)
            })

        deduped_rollup = {}
        for entry in market_rollup:
            key = entry.get("canonical_id") or entry.get("id")
            existing = deduped_rollup.get(key)
            if not existing or (entry.get("volume_24h") or 0) > (existing.get("volume_24h") or 0):
                deduped_rollup[key] = entry

        market_rollup = list(deduped_rollup.values())
        market_rollup.sort(key=lambda x: x["volume_24h"] or 0, reverse=True)
        top_volume = market_rollup[:8]
        top_retail = sorted(
            [m for m in market_rollup if m.get("avg_trade_size")],
            key=lambda x: x["avg_trade_size"]
        )[:8]

        result = {
            "timestamp": _to_utc_iso(datetime.utcnow()),
            "last_collection": _to_utc_iso(latest_run.timestamp) if latest_run else None,
            "window_days": days,
            "retail_min_trades": RETAIL_MIN_TRADES,
            "retail_size_percentiles": RETAIL_SIZE_PERCENTILES,
            "totals": {
                "markets": len(markets),
                "trades_24h": len(trades_24h),
                "snapshots": db.query(MarketSnapshot).count()
            },
            "categories": categories,
            "top_volume_markets": top_volume,
            "top_retail_markets": top_retail
        }
        if response is not None:
            response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
        return result
    finally:
        db.close()
        duration = time.time() - start_time
        logger.info("overview duration=%.2fs days=%s", duration, days)
        if "result" in locals():
            _cache_set(cache_key, locals()["result"])

@app.get("/markets")
def list_markets(category: str | None = None, limit: int = DEFAULT_MARKET_LIMIT, days: int = DEFAULT_MARKET_DAYS, hide_whales: bool = False, response: Response | None = None):
    """List markets with latest stats and retail signals."""
    days = _clamp_int(days, minimum=1, maximum=MAX_MARKET_DAYS)
    limit = _clamp_int(limit, minimum=1, maximum=100)
    cache_key = f"markets:{category or 'all'}:{limit}:{days}:{int(hide_whales)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        if response is not None:
            response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
        return cached

    start_time = time.time()
    db = get_db_session()

    try:
        query = db.query(Market)
        if category:
            category_norm = _normalize_category(category)
            query = query.filter(func.lower(Market.category) == category_norm)
        else:
            tracked_set = _tracked_categories_set()
            if tracked_set:
                query = query.filter(func.lower(Market.category).in_(tracked_set))
        markets = query.all()
        since = datetime.utcnow() - timedelta(days=days)

        rows = []
        for market in markets:
            latest = (db.query(MarketSnapshot)
                      .filter_by(market_id=market.id)
                      .order_by(desc(MarketSnapshot.timestamp))
                      .first())
            closed_flag = _is_closed_market(market, latest)
            status_label = _status_label(market, closed_flag)
            if not latest:
                continue
            quality_ok, quality_reason = _market_quality_ok(latest.volume_24h, latest.liquidity)
            trade_count, total_value, last_trade_time = db.query(
                func.count(Trade.id),
                func.sum(Trade.value),
                func.max(Trade.timestamp)
            ).filter(
                Trade.market_id == market.id,
                Trade.timestamp >= since
            ).one()
            trade_count = trade_count or 0
            total_value = total_value or 0
            avg_trade_size_window = (
                total_value / trade_count
                if trade_count >= RETAIL_MIN_TRADES else None
            )

            rows.append({
                "id": market.id,
                "canonical_id": _canonical_market_id(market),
                "title": market.title,
                "category": _normalize_category(market.category),
                "description": market.description,
                "end_date": market.end_date_iso,
                "closed": closed_flag,
                "archived": _truthy_flag(market.archived),
                "status": status_label,
                "price": latest.price,
                "volume_24h": latest.volume_24h,
                "liquidity": latest.liquidity,
                "avg_trade_size": latest.avg_trade_size,
                "avg_trade_size_window": avg_trade_size_window,
                "trade_count_window": trade_count,
                "last_updated": _to_utc_iso(latest.timestamp),
                "last_trade_time": _to_utc_iso(last_trade_time),
                "quality_ok": quality_ok,
                "quality_reason": quality_reason
            })

        rows.sort(key=lambda x: x["volume_24h"] or 0, reverse=True)
        deduped = {}
        for row in rows:
            key = row.get("canonical_id") or row.get("id")
            existing = deduped.get(key)
            if not existing:
                deduped[key] = row
                continue
            if row.get("quality_ok") and not existing.get("quality_ok"):
                deduped[key] = row
                continue
            if (row.get("trade_count_window") or 0) > (existing.get("trade_count_window") or 0):
                deduped[key] = row
                continue
            if (row.get("volume_24h") or 0) > (existing.get("volume_24h") or 0):
                deduped[key] = row
        rows = list(deduped.values())
        rows.sort(key=lambda x: x["volume_24h"] or 0, reverse=True)

        for row in rows:
            trades = db.query(Trade)\
                .filter_by(market_id=row["id"])\
                .filter(Trade.timestamp >= since)\
                .order_by(desc(Trade.timestamp))\
                .limit(300)\
                .all()
            if len(trades) >= RETAIL_MIN_TRADES and row.get("quality_ok"):
                row["retail_signals"] = _compute_retail_signals(trades, row["category"])
            else:
                snapshots = db.query(MarketSnapshot)\
                    .filter_by(market_id=row["id"])\
                    .filter(MarketSnapshot.timestamp >= since)\
                    .order_by(MarketSnapshot.timestamp)\
                    .all()
                if not snapshots:
                    latest_snapshot = db.query(MarketSnapshot)\
                        .filter_by(market_id=row["id"])\
                        .order_by(desc(MarketSnapshot.timestamp))\
                        .first()
                    snapshots = [latest_snapshot] if latest_snapshot else []
                row["retail_signals"] = _compute_snapshot_signals(snapshots)
                row["retail_signals"]["coverage"] = "insufficient"
                row["retail_signals"]["sample_trades"] = len(trades)
            row["retail_signals"]["quality_ok"] = row.get("quality_ok", False)
            row["retail_signals"]["quality_reason"] = row.get("quality_reason")
            row["retail_signals"]["rank_reliable"] = _rank_reliable(
                row["retail_signals"].get("coverage"),
                row["retail_signals"].get("quality_ok"),
                row["retail_signals"].get("sample_trades"),
                row.get("trade_count_window"),
                row.get("volume_24h"),
                row.get("liquidity")
            )

        scores_by_category = {}
        for row in rows:
            signals = row.get("retail_signals") or {}
            if not signals.get("rank_reliable"):
                continue
            score = signals.get("score")
            if score is None:
                continue
            scores_by_category.setdefault(row["category"], []).append(score)

        stats_by_category = {}
        for cat, scores in scores_by_category.items():
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = variance ** 0.5
            stats_by_category[cat] = (mean, std)

        for row in rows:
            signals = row.get("retail_signals") or {}
            score = signals.get("score")
            mean_std = stats_by_category.get(row["category"])
            if (not signals.get("rank_reliable")) or score is None or not mean_std or mean_std[1] == 0:
                row["retail_score_z"] = None
            else:
                row["retail_score_z"] = (score - mean_std[0]) / mean_std[1]

        if hide_whales:
            rows = [
                row for row in rows
                if not (row.get("retail_signals") or {}).get("whale_dominated")
            ]

        rows = rows[:limit]

        result = {
            "total": len(rows),
            "markets": rows
        }
    finally:
        db.close()
        duration = time.time() - start_time
        logger.info("markets duration=%.2fs days=%s limit=%s cached=%s", duration, days, limit, cached is not None)
        if "result" in locals():
            _cache_set(cache_key, locals()["result"])
    if response is not None:
        response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
    return result

@app.get("/markets/{market_id}")
def get_market_detail(market_id: str, hours: int = 24):
    """Detailed market view with retail signals and lifecycle analysis."""
    hours = _clamp_int(hours, minimum=1, maximum=MAX_HISTORY_HOURS)
    db = get_db_session()

    try:
        market = db.query(Market).filter_by(id=market_id).first()
        if not market:
            return {"error": "Market not found"}

        latest = (db.query(MarketSnapshot)
                  .filter_by(market_id=market_id)
                  .order_by(desc(MarketSnapshot.timestamp))
                  .first())

        since = datetime.utcnow() - timedelta(hours=hours)
        trades = db.query(Trade)\
            .filter_by(market_id=market_id)\
            .filter(Trade.timestamp >= since)\
            .order_by(desc(Trade.timestamp))\
            .limit(MAX_TRADES_LIMIT)\
            .all()
        trade_count, total_value, last_trade_time = db.query(
            func.count(Trade.id),
            func.sum(Trade.value),
            func.max(Trade.timestamp)
        ).filter(
            Trade.market_id == market_id,
            Trade.timestamp >= since
        ).one()
        trade_count = trade_count or 0
        total_value = total_value or 0
        avg_trade_size_window = total_value / trade_count if trade_count >= RETAIL_MIN_TRADES else None

        quality_ok, quality_reason = _market_quality_ok(
            latest.volume_24h if latest else None,
            latest.liquidity if latest else None
        )
        closed_flag = _is_closed_market(market, latest)
        status_label = _status_label(market, closed_flag)

        snapshots = db.query(MarketSnapshot)\
            .filter_by(market_id=market_id)\
            .order_by(MarketSnapshot.timestamp)\
            .all()

        if len(trades) >= RETAIL_MIN_TRADES and quality_ok:
            retail_signals = _compute_retail_signals(trades, _normalize_category(market.category))
        else:
            recent_snapshots = [s for s in snapshots if s.timestamp >= since]
            retail_signals = _compute_snapshot_signals(recent_snapshots or snapshots[-24:])
            retail_signals["coverage"] = "insufficient"
            retail_signals["sample_trades"] = len(trades)
        retail_signals["quality_ok"] = quality_ok
        retail_signals["quality_reason"] = quality_reason
        lifecycle = _compute_lifecycle(snapshots[-60:])

        retail_signals["rank_reliable"] = _rank_reliable(
            retail_signals.get("coverage"),
            retail_signals.get("quality_ok"),
            retail_signals.get("sample_trades"),
            trade_count,
            latest.volume_24h if latest else None,
            latest.liquidity if latest else None
        )

        return {
            "market": {
                "id": market.id,
            "canonical_id": _canonical_market_id(market),
                "title": market.title,
                "category": _normalize_category(market.category),
                "subcategory": market.subcategory,
                "description": market.description,
                "end_date": market.end_date_iso,
                "closed": closed_flag,
                "archived": _truthy_flag(market.archived),
                "status": status_label,
                "event_title": market.event_title,
                "event_tags": market.event_tags,
                "outcomes": market.outcomes
            },
            "latest": {
                "price": latest.price if latest else None,
                "volume_24h": latest.volume_24h if latest else None,
                "liquidity": latest.liquidity if latest else None,
                "avg_trade_size": latest.avg_trade_size if latest else None,
                "avg_trade_size_window": avg_trade_size_window,
                "trade_count_window": trade_count,
                "last_trade_time": _to_utc_iso(last_trade_time),
                "timestamp": _to_utc_iso(latest.timestamp) if latest else None
            },
            "retail_signals": retail_signals,
            "lifecycle": lifecycle
        }
    finally:
        db.close()

@app.get("/markets/{market_id}/history")
def get_market_history(market_id: str, hours: int = 168):
    """Time series of market snapshots for charting."""
    hours = _clamp_int(hours, minimum=1, maximum=MAX_HISTORY_HOURS)
    db = get_db_session()

    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        snapshots = db.query(MarketSnapshot)\
            .filter_by(market_id=market_id)\
            .filter(MarketSnapshot.timestamp >= since)\
            .order_by(MarketSnapshot.timestamp)\
            .all()

        series = [
            {
                "timestamp": _to_utc_iso(snap.timestamp),
                "price": snap.price,
                "volume_24h": snap.volume_24h,
                "liquidity": snap.liquidity,
                "avg_trade_size": snap.avg_trade_size,
                "num_trades": snap.volume_num_trades
            }
            for snap in snapshots
        ]

        return {
            "market_id": market_id,
            "hours": hours,
            "series": series
        }
    finally:
        db.close()

@app.get("/categories")
def get_market_categories():
    """Get available market categories and their statistics"""
    db = get_db_session()

    try:
        # Get category statistics
        categories = db.query(
            Market.category,
            func.count(Market.id).label('market_count'),
            func.sum(MarketSnapshot.volume_24h).label('total_volume'),
            func.avg(MarketSnapshot.liquidity).label('avg_liquidity')
        ).join(MarketSnapshot, Market.id == MarketSnapshot.market_id)\
         .group_by(Market.category)\
         .all()

        result = []
        for cat in categories:
            result.append({
                "category": cat.category,
                "market_count": cat.market_count,
                "total_volume_24h": float(cat.total_volume or 0),
                "avg_liquidity": float(cat.avg_liquidity or 0)
            })

        return {"categories": result}

    finally:
        db.close()

@app.get("/markets/category/{category}")
def get_markets_by_category(category: str, limit: int = 20):
    """Get markets for a specific category with real-time data"""
    db = get_db_session()

    try:
        category_norm = _normalize_category(category)
        markets = db.query(Market).filter(func.lower(Market.category) == category_norm).all()

        result = []
        for market in markets:
            # Get latest snapshot
            latest = (db.query(MarketSnapshot)
                     .filter_by(market_id=market.id)
                     .order_by(desc(MarketSnapshot.timestamp))
                     .first())

            if latest:
                quality_ok, quality_reason = _market_quality_ok(latest.volume_24h, latest.liquidity)
                if not quality_ok:
                    continue
                # Get recent trades for retail metrics
                recent_trades = db.query(Trade)\
                    .filter_by(market_id=market.id)\
                    .order_by(desc(Trade.timestamp))\
                    .limit(100)\
                    .all()
                last_trade_time = db.query(func.max(Trade.timestamp))\
                    .filter_by(market_id=market.id)\
                    .scalar()

                # Calculate retail metrics
                retail_metrics = calculate_retail_metrics(recent_trades, _normalize_category(market.category))
                retail_metrics["quality_ok"] = quality_ok
                retail_metrics["quality_reason"] = quality_reason

                result.append({
                    "id": market.id,
                    "canonical_id": _canonical_market_id(market),
                    "title": market.title,
                    "category": market.category,
                    "event_title": market.event_title,
                    "current_price": latest.price,
                    "volume_24h": latest.volume_24h,
                    "liquidity": latest.liquidity,
                    "last_updated": _to_utc_iso(latest.timestamp),
                    "last_trade_time": _to_utc_iso(last_trade_time),
                    "retail_metrics": retail_metrics
                })

        # Sort by volume
        result.sort(key=lambda x: x['volume_24h'], reverse=True)

        return {
            "category": category,
            "markets": result[:limit],
            "total": len(result)
        }

    finally:
        db.close()

@app.get("/markets/{market_id}/trades")
def get_market_trades(market_id: str, hours: int = 24, limit: int = MAX_TRADES_LIMIT):
    """Get trade data for a market with retail analysis"""
    hours = _clamp_int(hours, minimum=1, maximum=MAX_HISTORY_HOURS)
    limit = _clamp_int(limit, minimum=1, maximum=MAX_TRADES_LIMIT)
    db = get_db_session()

    try:
        market = db.query(Market).filter_by(id=market_id).first()
        market_category = market.category if market else None
        latest = (db.query(MarketSnapshot)
                  .filter_by(market_id=market_id)
                  .order_by(desc(MarketSnapshot.timestamp))
                  .first())
        quality_ok, quality_reason = _market_quality_ok(
            latest.volume_24h if latest else None,
            latest.liquidity if latest else None
        )

        # Get trades within time window
        since = datetime.utcnow() - timedelta(hours=hours)

        trades = db.query(Trade)\
            .filter(Trade.market_id == market_id)\
            .filter(Trade.timestamp >= since)\
            .order_by(desc(Trade.timestamp))\
            .limit(limit)\
            .all()

        valid_trades = [
            trade for trade in trades
            if trade.value is not None and trade.value > 0 and trade.timestamp is not None
        ]
        values = [trade.value for trade in valid_trades]
        winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
        threshold, method = _retail_threshold(winsorized, market_category) if values else (None, None)
        coverage = "trade" if len(values) >= RETAIL_MIN_TRADES and quality_ok else "insufficient"

        # Convert to dict and calculate rolling metrics
        trade_data = []
        for trade in valid_trades:
            is_retail = trade.value <= threshold if threshold is not None else False
            trade_data.append({
                "timestamp": _to_utc_iso(trade.timestamp),
                "price": trade.price,
                "quantity": trade.quantity,
                "value": trade.value,
                "side": trade.side,
                "is_retail": is_retail,
                "size_category": trade.trade_size_category
            })

        # Calculate time-bucketed retail metrics
        retail_analysis = analyze_trade_patterns(trade_data)
        trade_size_distribution = _bucket_trade_sizes(valid_trades)

        return {
            "market_id": market_id,
            "trades": trade_data,
            "retail_analysis": retail_analysis,
            "trade_size_distribution": trade_size_distribution,
            "time_window_hours": hours,
            "retail_threshold": threshold,
            "retail_threshold_method": method,
            "retail_threshold_trades": len(values),
            "retail_coverage": coverage,
            "quality_ok": quality_ok,
            "quality_reason": quality_reason,
            "retail_min_trades": RETAIL_MIN_TRADES
        }

    finally:
        db.close()

@app.get("/retail/dashboard")
def get_retail_dashboard():
    """Get real-time retail behavior dashboard data"""
    db = get_db_session()

    try:
        # Get latest data collection run
        latest_run = db.query(DataCollectionRun)\
            .order_by(desc(DataCollectionRun.timestamp))\
            .first()

        # Get category summaries
        categories_data = {}
        categories = ['politics', 'crypto', 'sports', 'business', 'entertainment', 'science', 'world']

        for category in categories:
            markets = db.query(Market).filter(func.lower(Market.category) == category).all()
            category_trades = []

            for market in markets:
                latest = (db.query(MarketSnapshot)
                          .filter_by(market_id=market.id)
                          .order_by(desc(MarketSnapshot.timestamp))
                          .first())
                quality_ok, _ = _market_quality_ok(
                    latest.volume_24h if latest else None,
                    latest.liquidity if latest else None
                )
                if not quality_ok:
                    continue
                recent_trades = db.query(Trade)\
                    .filter_by(market_id=market.id)\
                    .filter(Trade.timestamp >= datetime.utcnow() - timedelta(hours=24))\
                    .all()
                category_trades.extend(recent_trades)

            if category_trades:
                retail_metrics = calculate_retail_metrics(category_trades, category)
                categories_data[category] = {
                    "market_count": len(markets),
                    "trade_count_24h": len(category_trades),
                    "retail_metrics": retail_metrics
                }

        # Get top retail markets
        top_retail_markets = []
        markets = db.query(Market).all()

        for market in markets:
            recent_trades = db.query(Trade)\
                .filter_by(market_id=market.id)\
                .filter(Trade.timestamp >= datetime.utcnow() - timedelta(hours=24))\
                .all()

            latest = (db.query(MarketSnapshot)
                      .filter_by(market_id=market.id)
                      .order_by(desc(MarketSnapshot.timestamp))
                      .first())
            quality_ok, _ = _market_quality_ok(
                latest.volume_24h if latest else None,
                latest.liquidity if latest else None
            )

            if recent_trades and quality_ok and len(recent_trades) >= RETAIL_MIN_TRADES:
                metrics = calculate_retail_metrics(recent_trades, _normalize_category(market.category))
                if metrics['retail_percentage'] > 50:  # High retail activity
                    top_retail_markets.append({
                        "id": market.id,
                        "title": market.title,
                        "category": market.category,
                        "retail_percentage": metrics['retail_percentage'],
                        "avg_trade_size": metrics['avg_trade_size'],
                        "trade_count": len(recent_trades)
                    })

        top_retail_markets.sort(key=lambda x: x['retail_percentage'], reverse=True)

        return {
            "timestamp": _to_utc_iso(datetime.utcnow()),
            "last_collection": _to_utc_iso(latest_run.timestamp) if latest_run else None,
            "categories": categories_data,
            "top_retail_markets": top_retail_markets[:10],
            "total_markets": db.query(Market).count(),
            "total_trades_24h": db.query(Trade)\
                .filter(Trade.timestamp >= datetime.utcnow() - timedelta(hours=24))\
                .count()
        }

    finally:
        db.close()

@app.get("/retail/time-series/{category}")
def get_retail_time_series(category: str, hours: int = 24):
    """Get time-series retail behavior data for a category"""
    hours = _clamp_int(hours, minimum=1, maximum=MAX_HISTORY_HOURS)
    db = get_db_session()

    try:
        # Get markets in category
        category_norm = _normalize_category(category)
        markets = db.query(Market).filter(func.lower(Market.category) == category_norm).all()
        market_ids = [m.id for m in markets]

        if not market_ids:
            return {"error": f"No markets found for category {category}"}

        # Get trades in time window
        since = datetime.utcnow() - timedelta(hours=hours)

        trades = db.query(Trade)\
            .filter(Trade.market_id.in_(market_ids))\
            .filter(Trade.timestamp >= since)\
            .order_by(Trade.timestamp)\
            .all()

        valid_trades = [
            trade for trade in trades
            if trade.value is not None and trade.value > 0 and trade.timestamp is not None
        ]
        values = [trade.value for trade in valid_trades]
        winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
        threshold, method = _retail_threshold(winsorized, category) if values else (None, None)

        # Bucket trades by hour
        hourly_data = {}
        for trade in valid_trades:
            hour_key = trade.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {
                    "timestamp": _to_utc_iso(hour_key),
                    "total_trades": 0,
                    "total_volume": 0,
                    "retail_trades": 0,
                    "retail_volume": 0,
                    "avg_trade_size": 0
                }

            hourly_data[hour_key]["total_trades"] += 1
            hourly_data[hour_key]["total_volume"] += trade.value

            if threshold is not None and trade.value <= threshold:
                hourly_data[hour_key]["retail_trades"] += 1
                hourly_data[hour_key]["retail_volume"] += trade.value

        # Calculate averages and percentages
        for hour_data in hourly_data.values():
            if hour_data["total_trades"] > 0:
                hour_data["avg_trade_size"] = hour_data["total_volume"] / hour_data["total_trades"]
                hour_data["retail_percentage"] = (hour_data["retail_trades"] / hour_data["total_trades"]) * 100

        # Sort by timestamp
        time_series = list(hourly_data.values())
        time_series.sort(key=lambda x: x['timestamp'])

        return {
            "category": category,
            "time_series": time_series,
            "summary": {
                "total_hours": len(time_series),
                "total_trades": sum(h["total_trades"] for h in time_series),
                "total_volume": sum(h["total_volume"] for h in time_series),
                "avg_retail_percentage": sum(h.get("retail_percentage", 0) for h in time_series) / len(time_series) if time_series else 0,
                "retail_threshold": threshold,
                "retail_threshold_method": method,
                "retail_threshold_trades": len(values),
                "retail_coverage": "trade" if len(values) >= RETAIL_MIN_TRADES else "insufficient",
                "retail_min_trades": RETAIL_MIN_TRADES
            }
        }

    finally:
        db.close()

def calculate_retail_metrics(trades, category: str | None = None):
    """Calculate retail behavior metrics from trades"""
    if not trades:
        return {
            "total_trades": 0,
            "total_volume": 0,
            "avg_trade_size": 0,
            "retail_percentage": 0,
            "retail_trade_count": 0,
            "retail_volume_share": 0,
            "whale_share": None,
            "whale_trade_count": None,
            "whale_dominated": False,
            "flow_score": 0,
            "attention_score": 0,
            "retail_threshold": None,
            "retail_threshold_method": None,
            "coverage": "insufficient",
            "sample_trades": 0,
            "min_trades": RETAIL_MIN_TRADES,
            "confidence_label": "insufficient",
            "confidence_score": 0.0
        }

    filtered_trades = [
        trade for trade in trades
        if trade.value is not None and trade.value > 0 and trade.timestamp is not None
    ]
    trade_values = [trade.value for trade in filtered_trades]
    winsorized = _winsorize(trade_values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
    threshold, method = _retail_threshold(winsorized, category)
    retail_trades = [v for v in trade_values if v <= threshold]
    total_volume = sum(trade_values)
    retail_volume_share = sum(retail_trades) / total_volume if total_volume > 0 else 0
    avg_trade_size = _trimmed_mean(winsorized, TRADE_TRIM_PCT) if trade_values else 0
    whale_share, whale_count = _whale_share(trade_values)
    whale_dominated = whale_share >= 0.5
    small_trade_share = len(retail_trades) / len(trade_values) if trade_values else 0
    flow_score = min(1.0, (small_trade_share / 0.6) * 0.7 + (retail_volume_share / 0.5) * 0.3) if trade_values else 0
    attention_score = 0.0

    return {
        "total_trades": len(trade_values),
        "total_volume": total_volume,
        "avg_trade_size": avg_trade_size,
        "retail_percentage": (len(retail_trades) / len(trade_values)) * 100 if trade_values else 0,
        "retail_trade_count": len(retail_trades),
        "retail_volume_share": retail_volume_share,
        "median_trade_size": _safe_median(winsorized),
        "whale_share": whale_share,
        "whale_trade_count": whale_count,
        "whale_dominated": whale_dominated,
        "flow_score": flow_score,
        "attention_score": attention_score,
        "retail_threshold": threshold,
        "retail_threshold_method": method,
        "coverage": "trade" if len(trade_values) >= RETAIL_MIN_TRADES else "insufficient",
        "sample_trades": len(trade_values),
        "min_trades": RETAIL_MIN_TRADES,
        "confidence_label": _confidence_label(len(trade_values)),
        "confidence_score": _confidence_score(len(trade_values))
    }

def analyze_trade_patterns(trade_data):
    """Analyze trading patterns for retail behavior insights"""
    if not trade_data:
        return {}

    total_trades = len(trade_data)
    total_value = sum(trade["value"] for trade in trade_data)
    avg_trade_size = total_value / total_trades if total_trades > 0 else 0

    # Group by hour
    hourly_patterns = {}
    for trade in trade_data:
        dt = _parse_trade_timestamp(trade.get('timestamp'))
        if not dt:
            continue
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        hour = dt.hour

        if hour not in hourly_patterns:
            hourly_patterns[hour] = {
                "hour": hour,
                "trades": 0,
                "retail_trades": 0,
                "volume": 0,
                "retail_volume": 0
            }

        hourly_patterns[hour]["trades"] += 1
        hourly_patterns[hour]["volume"] += trade["value"]

        if trade["is_retail"]:
            hourly_patterns[hour]["retail_trades"] += 1
            hourly_patterns[hour]["retail_volume"] += trade["value"]

    # Calculate retail activity by time of day
    peak_retail_hours = sorted(hourly_patterns.items(),
                              key=lambda x: x[1]["retail_trades"], reverse=True)[:3]

    return {
        "total_trades": total_trades,
        "total_volume": total_value,
        "avg_trade_size": avg_trade_size,
        "hourly_patterns": list(hourly_patterns.values()),
        "peak_retail_hours": [h[0] for h in peak_retail_hours],
        "retail_dominance_score": sum(h["retail_trades"] for h in hourly_patterns.values()) / sum(h["trades"] for h in hourly_patterns.values()) * 100
    }

@app.get("/markets/{market_id}/retail-analysis")
def get_market_retail_analysis(market_id: str, days: int = 7):
    """Get comprehensive retail behavior analysis for a market"""
    try:
        analysis = retail_analyzer.analyze_market_retail_behavior(market_id, days)

        if "error" in analysis:
            return {"error": analysis["error"]}

        return {
            "market_id": market_id,
            "analysis_period_days": days,
            "retail_analysis": analysis
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.get("/analytics/retail-insights")
def get_retail_insights():
    """Get retail trading insights across all markets"""
    db = get_db_session()

    try:
        markets = db.query(Market).all()
        insights = {
            "high_retail_markets": [],
            "weekend_active_markets": [],
            "volatile_markets": [],
            "summary": {}
        }

        total_volume = 0
        retail_dominated_count = 0

        for market in markets:
            latest = (db.query(MarketSnapshot)
                     .filter_by(market_id=market.id)
                     .order_by(desc(MarketSnapshot.timestamp))
                     .first())

            if latest:
                quality_ok, _ = _market_quality_ok(latest.volume_24h, latest.liquidity)
                if not quality_ok:
                    continue
                total_volume += latest.volume_24h or 0

                # Identify high retail markets (small trade sizes)
                if latest.avg_trade_size and latest.avg_trade_size < 500:
                    insights["high_retail_markets"].append({
                        "id": market.id,
                        "title": market.title,
                        "avg_trade_size": latest.avg_trade_size,
                        "volume_24h": latest.volume_24h
                    })
                    retail_dominated_count += 1

                # Weekend active markets
                if latest.is_weekend:
                    insights["weekend_active_markets"].append({
                        "id": market.id,
                        "title": market.title,
                        "volume_24h": latest.volume_24h
                    })

                # Volatile markets (high volume changes)
                # This would need historical comparison, simplified for now
                if latest.volume_24h and latest.volume_24h > 10000:
                    insights["volatile_markets"].append({
                        "id": market.id,
                        "title": market.title,
                        "volume_24h": latest.volume_24h
                    })

        insights["summary"] = {
            "total_markets": len(markets),
            "retail_dominated_markets": retail_dominated_count,
            "total_volume_24h": total_volume,
            "retail_market_percentage": (retail_dominated_count / len(markets) * 100) if markets else 0
        }

        return insights

    finally:
        db.close()

@app.get("/analytics/overview")
def get_analytics_overview():
    """Get overall analytics dashboard data"""
    db = get_db_session()
    
    try:
        # Total markets
        total_markets = db.query(Market).count()
        
        # Total snapshots
        total_snapshots = db.query(MarketSnapshot).count()
        
        # Get markets with highest volume
        top_markets = []
        markets = db.query(Market).all()
        
        for market in markets:
            latest = (db.query(MarketSnapshot)
                     .filter_by(market_id=market.id)
                     .order_by(desc(MarketSnapshot.timestamp))
                     .first())

            if latest:
                quality_ok, _ = _market_quality_ok(latest.volume_24h, latest.liquidity)
                if not quality_ok:
                    continue
                top_markets.append({
                    "id": market.id,
                    "title": market.title,
                    "volume_24h": latest.volume_24h,
                    "price": latest.price
                })
        
        # Sort by volume
        top_markets.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        return {
            "total_markets": total_markets,
            "total_snapshots": total_snapshots,
            "top_markets_by_volume": top_markets[:5],
            "timestamp": _to_utc_iso(datetime.utcnow())
        }
    
    finally:
        db.close()

@app.get("/analytics/retail-index")
def get_retail_index(days: int = 180, category: str = "all"):
    """Long-term retail index rollups."""
    days = _clamp_int(days, minimum=1, maximum=MAX_ANALYTICS_DAYS)
    db = get_db_session()

    try:
        category_norm = _normalize_category(category) if category else "all"
        if category_norm in ("all", "overall"):
            category_norm = "all"

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=max(days - 1, 0))

        query = db.query(RetailRollup)\
            .filter(RetailRollup.date >= start_date)\
            .filter(RetailRollup.date <= end_date)\
            .filter(RetailRollup.category == category_norm)\
            .order_by(RetailRollup.date)

        rows = query.all()
        series = [
            {
                "date": r.date.isoformat(),
                "retail_score": r.retail_score,
                "retail_level": r.retail_level,
                "flow_score": r.flow_score,
                "attention_score": r.attention_score,
                "retail_trade_share": r.retail_trade_share,
                "retail_volume_share": r.retail_volume_share,
                "avg_trade_size": r.avg_trade_size,
                "total_trades": r.total_trades,
                "total_volume": r.total_volume,
                "whale_share": r.whale_share,
                "whale_dominated": r.whale_dominated,
                "markets_covered": r.markets_covered,
                "quality_markets": r.quality_markets,
                "confidence_label": r.confidence_label,
                "confidence_score": r.confidence_score
            }
            for r in rows
        ]

        summary = {}
        if rows:
            latest = rows[-1]
            summary = {
                "latest_date": latest.date.isoformat(),
                "latest_retail_score": latest.retail_score,
                "latest_flow_score": latest.flow_score,
                "latest_attention_score": latest.attention_score,
                "latest_retail_trade_share": latest.retail_trade_share,
                "latest_retail_volume_share": latest.retail_volume_share,
                "latest_whale_share": latest.whale_share,
                "latest_total_trades": latest.total_trades,
                "latest_total_volume": latest.total_volume,
                "markets_covered": latest.markets_covered,
                "quality_markets": latest.quality_markets,
                "confidence_label": latest.confidence_label
            }

        return {
            "category": category_norm,
            "days": days,
            "series": series,
            "summary": summary
        }
    finally:
        db.close()

@app.get("/analytics/retail-index/hourly")
def get_retail_index_hourly(hours: int = 72, category: str = "all"):
    """Short-term retail index rollups (hourly)."""
    hours = _clamp_int(hours, minimum=1, maximum=MAX_HOURLY_RETAIL_HOURS)
    db = get_db_session()

    try:
        category_norm = _normalize_category(category) if category else "all"
        if category_norm in ("all", "overall"):
            category_norm = "all"

        hours = max(hours, 1)
        end_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start_hour = end_hour - timedelta(hours=hours - 1)

        trade_query = db.query(Trade).filter(Trade.timestamp >= start_hour)
        if category_norm != "all":
            market_ids = [
                row[0] for row in db.query(Market.id)
                .filter(func.lower(Market.category) == category_norm)
                .all()
            ]
            if not market_ids:
                return {
                    "category": category_norm,
                    "hours": hours,
                    "series": [],
                    "summary": {},
                    "granularity": "hourly"
                }
            trade_query = trade_query.filter(Trade.market_id.in_(market_ids))

        trades = trade_query.order_by(Trade.timestamp).all()
        valid_trades = [
            trade for trade in trades
            if trade.value is not None and trade.value > 0 and trade.timestamp is not None
        ]
        values = [trade.value for trade in valid_trades]
        winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
        threshold, method = _retail_threshold(
            winsorized,
            category_norm if category_norm != "all" else None
        ) if values else (None, None)

        buckets = {}
        for trade in valid_trades:
            hour_key = trade.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key < start_hour or hour_key > end_hour:
                continue
            bucket = buckets.setdefault(
                hour_key,
                {
                    "values": [],
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "retail_trades": 0,
                    "retail_volume": 0.0
                }
            )
            bucket["values"].append(trade.value)
            bucket["total_trades"] += 1
            bucket["total_volume"] += trade.value
            if threshold is not None and trade.value <= threshold:
                bucket["retail_trades"] += 1
                bucket["retail_volume"] += trade.value

        cursor = start_hour
        while cursor <= end_hour:
            buckets.setdefault(
                cursor,
                {
                    "values": [],
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "retail_trades": 0,
                    "retail_volume": 0.0
                }
            )
            cursor += timedelta(hours=1)

        volumes = [bucket["total_volume"] for bucket in buckets.values()]
        mean_volume = sum(volumes) / len(volumes) if volumes else 0

        series = []
        for hour_key in sorted(buckets.keys()):
            bucket = buckets[hour_key]
            total_trades = bucket["total_trades"]
            total_volume = bucket["total_volume"]
            retail_trade_share = (bucket["retail_trades"] / total_trades) if total_trades else None
            retail_volume_share = (bucket["retail_volume"] / total_volume) if total_volume else None
            avg_trade_size = (total_volume / total_trades) if total_trades else None
            whale_share, _ = _whale_share(bucket["values"]) if bucket["values"] else (None, 0)
            burstiness = (total_volume / mean_volume) if mean_volume > 0 else 0
            flow_score = min(1.0, (retail_trade_share or 0) / 0.6) if retail_trade_share is not None else None
            attention_score = (
                min(1.0, max(burstiness - 1.0, 0.0) / 1.5)
                if total_trades
                else None
            )
            retail_score = round((flow_score or 0) * 10, 2) if total_trades else None

            series.append({
                "timestamp": _to_utc_iso(hour_key),
                "retail_score": retail_score,
                "retail_level": None,
                "flow_score": flow_score,
                "attention_score": attention_score,
                "retail_trade_share": retail_trade_share,
                "retail_volume_share": retail_volume_share,
                "avg_trade_size": avg_trade_size,
                "total_trades": total_trades,
                "total_volume": total_volume,
                "whale_share": whale_share,
                "whale_dominated": whale_share >= 0.5 if whale_share is not None else False,
                "confidence_label": _confidence_label(total_trades),
                "confidence_score": _confidence_score(total_trades),
                "trade_threshold": threshold,
                "trade_threshold_method": method
            })

        summary = {}
        if series:
            latest = next((row for row in reversed(series) if row["total_trades"] > 0), series[-1])
            summary = {
                "latest_date": latest["timestamp"],
                "latest_retail_score": latest["retail_score"],
                "latest_flow_score": latest["flow_score"],
                "latest_attention_score": latest["attention_score"],
                "latest_retail_trade_share": latest["retail_trade_share"],
                "latest_retail_volume_share": latest["retail_volume_share"],
                "latest_whale_share": latest["whale_share"],
                "latest_total_trades": latest["total_trades"],
                "latest_total_volume": latest["total_volume"],
                "confidence_label": latest["confidence_label"]
            }

        return {
            "category": category_norm,
            "hours": hours,
            "series": series,
            "summary": summary,
            "granularity": "hourly"
        }
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
