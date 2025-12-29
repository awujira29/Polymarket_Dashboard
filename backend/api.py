from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func
from database import get_db_session, init_db
from models import Market, MarketSnapshot, Trade, DataCollectionRun
from retail_analyzer import RetailBehaviorAnalyzer
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from math import sqrt
from config import (
    TRACKED_CATEGORIES,
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD
)

app = FastAPI(title="Prediction Market Retail Behavior API")

@app.on_event("startup")
def startup_db():
    init_db()

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
            "score": 0,
            "level": "low",
            "drivers": [],
            "retail_threshold": None,
            "retail_threshold_method": None,
            "coverage": "none",
            "sample_trades": 0,
            "min_trades": RETAIL_MIN_TRADES
        }

    values = [t.value for t in trades]
    total_trades = len(values)
    total_volume = sum(values)
    avg_trade_size = total_volume / total_trades
    median_trade_size = _safe_median(values)

    threshold, method = _retail_threshold(values, category)
    small_trade_values = [v for v in values if v <= threshold]
    small_trade_count = len(small_trade_values)
    small_trade_share = small_trade_count / total_trades
    retail_volume_share = sum(small_trade_values) / total_volume if total_volume > 0 else 0

    evening_count = sum(
        1 for t in trades
        if t.timestamp and (t.timestamp.hour >= 18 or t.timestamp.hour <= 5)
    )
    weekend_count = sum(
        1 for t in trades
        if t.timestamp and t.timestamp.weekday() >= 5
    )
    evening_share = evening_count / total_trades
    weekend_share = weekend_count / total_trades

    hourly_volume = {}
    for trade in trades:
        if not trade.timestamp:
            continue
        bucket = trade.timestamp.replace(minute=0, second=0, microsecond=0)
        hourly_volume[bucket] = hourly_volume.get(bucket, 0) + trade.value

    hourly_values = list(hourly_volume.values())
    if hourly_values:
        hourly_mean = sum(hourly_values) / len(hourly_values)
        hourly_std = sqrt(sum((v - hourly_mean) ** 2 for v in hourly_values) / len(hourly_values))
        burstiness = max(hourly_values) / hourly_mean if hourly_mean > 0 else 0
        volatility = hourly_std / hourly_mean if hourly_mean > 0 else 0
    else:
        burstiness = 0
        volatility = 0

    score = 0
    drivers = []
    if small_trade_share > 0.6:
        score += 3
        drivers.append("small_trade_bias")
    elif small_trade_share > 0.4:
        score += 2
        drivers.append("retail_trade_mix")

    if evening_share > 0.35:
        score += 1
        drivers.append("evening_activity")

    if weekend_share > 0.35:
        score += 1
        drivers.append("weekend_activity")

    if burstiness > 2:
        score += 2
        drivers.append("volume_spikes")
    elif burstiness > 1.5:
        score += 1
        drivers.append("volume_swings")

    level = "low"
    if score >= 6:
        level = "high"
    elif score >= 3:
        level = "medium"

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
        "score": score,
        "level": level,
        "drivers": drivers,
        "retail_threshold": threshold,
        "retail_threshold_method": method,
        "coverage": "trade",
        "sample_trades": total_trades,
        "min_trades": RETAIL_MIN_TRADES
    }

def _compute_snapshot_signals(snapshots: List[MarketSnapshot]) -> Dict:
    if not snapshots:
        return _compute_retail_signals([])

    volumes = [s.volume_24h or 0 for s in snapshots]
    trade_counts = [s.volume_num_trades or 0 for s in snapshots]
    avg_trade_sizes = [s.avg_trade_size for s in snapshots if s.avg_trade_size]

    total_trades = sum(trade_counts)
    total_volume = volumes[-1] if volumes else 0
    avg_trade_size = sum(avg_trade_sizes) / len(avg_trade_sizes) if avg_trade_sizes else 0
    median_trade_size = avg_trade_size

    mean_volume = sum(volumes) / len(volumes) if volumes else 0
    std_volume = sqrt(sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)) if mean_volume else 0
    burstiness = max(volumes) / mean_volume if mean_volume > 0 else 0
    volatility = std_volume / mean_volume if mean_volume > 0 else 0

    evening_share = sum(1 for s in snapshots if s.is_evening) / len(snapshots)
    weekend_share = sum(1 for s in snapshots if s.is_weekend) / len(snapshots)

    small_trade_share = _trade_size_to_share(avg_trade_size)

    score = 0
    drivers = []
    if small_trade_share >= 0.5:
        score += 3
        drivers.append("small_trade_bias")
    elif small_trade_share >= 0.35:
        score += 2
        drivers.append("retail_trade_mix")

    if evening_share > 0.35:
        score += 1
        drivers.append("evening_activity")

    if weekend_share > 0.35:
        score += 1
        drivers.append("weekend_activity")

    if burstiness > 2:
        score += 2
        drivers.append("volume_spikes")
    elif burstiness > 1.5:
        score += 1
        drivers.append("volume_swings")

    level = "low"
    if score >= 6:
        level = "high"
    elif score >= 3:
        level = "medium"

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
        "score": score,
        "level": level,
        "drivers": drivers,
        "retail_threshold": None,
        "retail_threshold_method": None,
        "coverage": "snapshot",
        "sample_trades": total_trades,
        "min_trades": RETAIL_MIN_TRADES
    }

def _compute_lifecycle(snapshots: List[MarketSnapshot]) -> Dict:
    if len(snapshots) < 2:
        return {
            "stage": "insufficient_data",
            "trend": 0,
            "growth_rate": 0,
            "volatility": 0,
            "spike_count": 0
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

    return {
        "stage": stage,
        "trend": trend,
        "growth_rate": growth_rate,
        "volatility": volatility,
        "spike_count": spike_count
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
def get_overview(days: int = 30):
    """High-level overview for the Polymarket-only retail dashboard."""
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
                    "snapshot_count": 0,
                    "trade_count_24h": 0,
                    "retail_trades_24h": 0,
                    "trade_volume_24h": 0,
                    "retail_volume_24h": 0,
                    "retail_share_source": "snapshot"
                }

            latest = (db.query(MarketSnapshot)
                      .filter_by(market_id=market.id)
                      .order_by(desc(MarketSnapshot.timestamp))
                      .first())

            category_stats[category]["market_count"] += 1
            if latest:
                category_stats[category]["volume_24h"] += latest.volume_24h or 0
                category_stats[category]["liquidity"] += latest.liquidity or 0
                category_stats[category]["avg_trade_size_sum"] += latest.avg_trade_size or 0
                category_stats[category]["snapshot_count"] += 1

                market_rollup.append({
                    "id": market.id,
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
                    "snapshot_count": 0,
                    "trade_count_24h": 0,
                    "retail_trades_24h": 0,
                    "trade_volume_24h": 0,
                    "retail_volume_24h": 0,
                    "retail_share_source": "snapshot"
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

        categories = []
        for category, stats in category_stats.items():
            snapshot_count = stats["snapshot_count"] or 1
            trade_count = stats.get("trade_count_24h", 0)
            trade_volume = stats.get("trade_volume_24h", 0)
            if trade_count >= RETAIL_MIN_TRADES and stats.get("retail_share_source") == "trade":
                retail_share = stats["retail_trades_24h"] / stats["trade_count_24h"]
                retail_volume_share = stats["retail_volume_24h"] / trade_volume if trade_volume > 0 else 0
                retail_share_source = "trade"
            else:
                avg_trade_size = stats["avg_trade_size_sum"] / snapshot_count
                retail_share = _trade_size_to_share(avg_trade_size)
                retail_volume_share = None
                retail_share_source = "snapshot"
            categories.append({
                "category": category,
                "market_count": stats["market_count"],
                "volume_24h": stats["volume_24h"],
                "avg_trade_size": stats["avg_trade_size_sum"] / snapshot_count,
                "liquidity": stats["liquidity"],
                "trade_count_24h": stats["trade_count_24h"],
                "retail_trade_share": retail_share,
                "retail_volume_share": retail_volume_share,
                "retail_share_source": retail_share_source
            })

        market_rollup.sort(key=lambda x: x["volume_24h"] or 0, reverse=True)
        top_volume = market_rollup[:8]
        top_retail = sorted(
            [m for m in market_rollup if m.get("avg_trade_size")],
            key=lambda x: x["avg_trade_size"]
        )[:8]

        return {
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
    finally:
        db.close()

@app.get("/markets")
def list_markets(category: str | None = None, limit: int = 25, days: int = 30):
    """List markets with latest stats and retail signals."""
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
            if not latest:
                continue
            trade_count, total_value = db.query(
                func.count(Trade.id),
                func.sum(Trade.value)
            ).filter(
                Trade.market_id == market.id,
                Trade.timestamp >= since
            ).one()
            trade_count = trade_count or 0
            total_value = total_value or 0
            avg_trade_size_window = total_value / trade_count if trade_count > 0 else None

            rows.append({
                "id": market.id,
                "title": market.title,
                "category": _normalize_category(market.category),
                "description": market.description,
                "end_date": market.end_date_iso,
                "price": latest.price,
                "volume_24h": latest.volume_24h,
                "liquidity": latest.liquidity,
                "avg_trade_size": latest.avg_trade_size,
                "avg_trade_size_window": avg_trade_size_window,
                "trade_count_window": trade_count,
                "last_updated": _to_utc_iso(latest.timestamp)
            })

        rows.sort(key=lambda x: x["volume_24h"] or 0, reverse=True)
        rows = rows[:limit]

        for row in rows:
            trades = db.query(Trade)\
                .filter_by(market_id=row["id"])\
                .filter(Trade.timestamp >= since)\
                .order_by(desc(Trade.timestamp))\
                .limit(300)\
                .all()
            if len(trades) >= RETAIL_MIN_TRADES:
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

        return {
            "total": len(rows),
            "markets": rows
        }
    finally:
        db.close()

@app.get("/markets/{market_id}")
def get_market_detail(market_id: str, hours: int = 24):
    """Detailed market view with retail signals and lifecycle analysis."""
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
            .limit(500)\
            .all()
        trade_count, total_value = db.query(
            func.count(Trade.id),
            func.sum(Trade.value)
        ).filter(
            Trade.market_id == market_id,
            Trade.timestamp >= since
        ).one()
        trade_count = trade_count or 0
        total_value = total_value or 0
        avg_trade_size_window = total_value / trade_count if trade_count > 0 else None

        snapshots = db.query(MarketSnapshot)\
            .filter_by(market_id=market_id)\
            .order_by(MarketSnapshot.timestamp)\
            .all()

        if len(trades) >= RETAIL_MIN_TRADES:
            retail_signals = _compute_retail_signals(trades, _normalize_category(market.category))
        else:
            recent_snapshots = [s for s in snapshots if s.timestamp >= since]
            retail_signals = _compute_snapshot_signals(recent_snapshots or snapshots[-24:])
            retail_signals["coverage"] = "insufficient"
            retail_signals["sample_trades"] = len(trades)
        lifecycle = _compute_lifecycle(snapshots[-60:])

        return {
            "market": {
                "id": market.id,
                "title": market.title,
                "category": _normalize_category(market.category),
                "subcategory": market.subcategory,
                "description": market.description,
                "end_date": market.end_date_iso,
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
                # Get recent trades for retail metrics
                recent_trades = db.query(Trade)\
                    .filter_by(market_id=market.id)\
                    .order_by(desc(Trade.timestamp))\
                    .limit(100)\
                    .all()

                # Calculate retail metrics
                retail_metrics = calculate_retail_metrics(recent_trades, _normalize_category(market.category))

                result.append({
                    "id": market.id,
                    "title": market.title,
                    "category": market.category,
                    "event_title": market.event_title,
                    "current_price": latest.price,
                    "volume_24h": latest.volume_24h,
                    "liquidity": latest.liquidity,
                    "last_updated": _to_utc_iso(latest.timestamp),
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
def get_market_trades(market_id: str, hours: int = 24, limit: int = 1000):
    """Get trade data for a market with retail analysis"""
    db = get_db_session()

    try:
        market = db.query(Market).filter_by(id=market_id).first()
        market_category = market.category if market else None

        # Get trades within time window
        since = datetime.utcnow() - timedelta(hours=hours)

        trades = db.query(Trade)\
            .filter(Trade.market_id == market_id)\
            .filter(Trade.timestamp >= since)\
            .order_by(desc(Trade.timestamp))\
            .limit(limit)\
            .all()

        values = [trade.value for trade in trades]
        threshold, method = _retail_threshold(values, market_category) if values else (None, None)
        coverage = "trade" if len(values) >= RETAIL_MIN_TRADES else "insufficient"

        # Convert to dict and calculate rolling metrics
        trade_data = []
        for trade in trades:
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
        trade_size_distribution = _bucket_trade_sizes(trades)

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

            if recent_trades:
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

        values = [trade.value for trade in trades]
        threshold, method = _retail_threshold(values, category) if values else (None, None)

        # Bucket trades by hour
        hourly_data = {}
        for trade in trades:
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
            "retail_threshold": None,
            "retail_threshold_method": None,
            "coverage": "none",
            "sample_trades": 0,
            "min_trades": RETAIL_MIN_TRADES
        }

    trade_values = [trade.value for trade in trades]
    threshold, method = _retail_threshold(trade_values, category)
    retail_trades = [v for v in trade_values if v <= threshold]
    total_volume = sum(trade_values)
    retail_volume_share = sum(retail_trades) / total_volume if total_volume > 0 else 0

    return {
        "total_trades": len(trades),
        "total_volume": total_volume,
        "avg_trade_size": total_volume / len(trades),
        "retail_percentage": (len(retail_trades) / len(trades)) * 100,
        "retail_trade_count": len(retail_trades),
        "retail_volume_share": retail_volume_share,
        "median_trade_size": sorted(trade_values)[len(trade_values)//2],
        "retail_threshold": threshold,
        "retail_threshold_method": method,
        "coverage": "trade" if len(trades) >= RETAIL_MIN_TRADES else "insufficient",
        "sample_trades": len(trades),
        "min_trades": RETAIL_MIN_TRADES
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
        dt = datetime.fromisoformat(trade['timestamp'])
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
