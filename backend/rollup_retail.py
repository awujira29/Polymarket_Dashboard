#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime, timedelta, timezone, date

from sqlalchemy import func

from config import (
    TRACKED_CATEGORIES,
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD,
    TRADE_TRIM_PCT,
    TRADE_WINSOR_PCT,
    BURSTINESS_SMOOTH_WINDOW,
    MIN_MARKET_VOLUME_24H,
    MIN_MARKET_LIQUIDITY,
    ROLLUP_DAYS
)
from database import init_db, get_db_session
from models import Market, MarketSnapshot, Trade, RetailRollup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_category(value: str | None) -> str:
    if not value:
        return "uncategorized"
    return str(value).strip().lower()


def _percentile(values, percentile):
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


def _winsorize(values, lower_pct, upper_pct):
    if not values:
        return []
    lower = _percentile(values, lower_pct)
    upper = _percentile(values, 1 - upper_pct)
    return [min(max(v, lower), upper) for v in values]


def _trimmed_mean(values, trim_pct):
    if not values:
        return 0
    sorted_values = sorted(values)
    trim = int(len(sorted_values) * trim_pct)
    if trim * 2 >= len(sorted_values):
        return sum(sorted_values) / len(sorted_values)
    trimmed = sorted_values[trim:-trim]
    return sum(trimmed) / len(trimmed) if trimmed else 0


def _smooth_series(values, window):
    if window <= 1 or not values:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        bucket = values[start:idx + 1]
        smoothed.append(sum(bucket) / len(bucket))
    return smoothed


def _retail_percentile_for_category(category):
    if not category:
        return RETAIL_SIZE_PERCENTILES.get("default", 0.3)
    return RETAIL_SIZE_PERCENTILES.get(category, RETAIL_SIZE_PERCENTILES.get("default", 0.3))


def _retail_threshold(values, category=None, min_trades=RETAIL_MIN_TRADES, fallback=RETAIL_FALLBACK_THRESHOLD):
    percentile = _retail_percentile_for_category(category)
    if len(values) < min_trades:
        return fallback, "fixed"
    return _percentile(values, percentile), "percentile"


def _confidence_label(sample, min_trades=RETAIL_MIN_TRADES):
    if sample >= min_trades * 3:
        return "high"
    if sample >= min_trades * 2:
        return "medium"
    if sample >= min_trades:
        return "low"
    return "insufficient"


def _confidence_score(sample, min_trades=RETAIL_MIN_TRADES):
    if sample <= 0:
        return 0.0
    return min(1.0, sample / (min_trades * 3))


def _retail_score(small_trade_share, evening_share, weekend_share, burstiness):
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

    level = "low"
    if score >= 7:
        level = "high"
    elif score >= 4:
        level = "medium"

    return score, level


def _market_quality_ok(volume_24h, liquidity):
    volume_ok = (volume_24h or 0) >= MIN_MARKET_VOLUME_24H
    liquidity_ok = (liquidity or 0) >= MIN_MARKET_LIQUIDITY
    return volume_ok and liquidity_ok


def _whale_share(values):
    if not values:
        return 0.0, 0
    sorted_values = sorted(values, reverse=True)
    top_k = max(1, int(len(sorted_values) * 0.01))
    total = sum(sorted_values)
    if total <= 0:
        return 0.0, top_k
    whale_volume = sum(sorted_values[:top_k])
    return whale_volume / total, top_k


def _get_quality_markets(session, tracked_set):
    latest_subq = session.query(
        MarketSnapshot.market_id,
        func.max(MarketSnapshot.timestamp).label("ts")
    ).group_by(MarketSnapshot.market_id).subquery()

    latest_snapshots = session.query(MarketSnapshot).join(
        latest_subq,
        (MarketSnapshot.market_id == latest_subq.c.market_id) &
        (MarketSnapshot.timestamp == latest_subq.c.ts)
    ).all()

    quality_ids = set()
    for snap in latest_snapshots:
        if _market_quality_ok(snap.volume_24h, snap.liquidity):
            quality_ids.add(snap.market_id)

    market_rows = session.query(Market).all()
    category_map = {m.id: _normalize_category(m.category) for m in market_rows}
    quality_counts = {}
    for market_id in quality_ids:
        category = category_map.get(market_id, "uncategorized")
        if tracked_set and category not in tracked_set:
            continue
        quality_counts[category] = quality_counts.get(category, 0) + 1

    quality_counts["all"] = len(quality_ids)
    return quality_ids, category_map, quality_counts


def build_retail_rollups(days=ROLLUP_DAYS, categories=None):
    init_db()
    session = get_db_session()

    try:
        categories = categories or TRACKED_CATEGORIES or []
        categories = [_normalize_category(cat) for cat in categories]
        tracked_set = set(categories)
        categories = sorted(tracked_set)
        categories_with_all = categories + ["all"]

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(days - 1, 0))
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

        quality_ids, category_map, quality_counts = _get_quality_markets(session, tracked_set)

        trades = session.query(Trade)\
            .filter(Trade.timestamp >= start_dt)\
            .filter(Trade.timestamp < end_dt)\
            .all()

        buckets = {}
        for trade in trades:
            if trade.value is None or trade.value <= 0:
                continue
            if trade.timestamp is None:
                continue
            market_category = category_map.get(trade.market_id, "uncategorized")
            if tracked_set and market_category not in tracked_set:
                continue
            if trade.market_id not in quality_ids:
                continue

            day_key = trade.timestamp.date()
            for category in (market_category, "all"):
                bucket = buckets.setdefault(
                    (day_key, category),
                    {
                        "values": [],
                        "total_volume": 0.0,
                        "total_trades": 0,
                        "evening": 0,
                        "weekend": 0,
                        "hourly": {},
                        "markets": set()
                    }
                )
                bucket["values"].append(trade.value)
                bucket["total_volume"] += trade.value
                bucket["total_trades"] += 1
                bucket["markets"].add(trade.market_id)
                hour = trade.timestamp.hour
                bucket["hourly"][hour] = bucket["hourly"].get(hour, 0) + trade.value
                if hour >= 18 or hour <= 5:
                    bucket["evening"] += 1
                if trade.timestamp.weekday() >= 5:
                    bucket["weekend"] += 1

        session.query(RetailRollup)\
            .filter(RetailRollup.date >= start_date)\
            .filter(RetailRollup.date <= end_date)\
            .filter(RetailRollup.category.in_(categories_with_all))\
            .delete(synchronize_session=False)

        day_cursor = start_date
        while day_cursor <= end_date:
            for category in categories_with_all:
                bucket = buckets.get((day_cursor, category))
                if not bucket:
                    rollup = RetailRollup(
                        date=day_cursor,
                        category=category,
                        total_trades=0,
                        total_volume=0,
                        avg_trade_size=None,
                        retail_trade_share=None,
                        retail_volume_share=None,
                        small_trade_share=None,
                        evening_share=None,
                        weekend_share=None,
                        burstiness=None,
                        flow_score=None,
                        attention_score=None,
                        retail_score=None,
                        retail_level=None,
                        whale_share=None,
                        whale_dominated=False,
                        trade_threshold=None,
                        trade_threshold_method=None,
                        markets_covered=0,
                        quality_markets=quality_counts.get(category, 0),
                        confidence_label="insufficient",
                        confidence_score=0.0
                    )
                    session.add(rollup)
                    continue

                values = bucket["values"]
                total_trades = bucket["total_trades"]
                total_volume = bucket["total_volume"]
                winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
                avg_trade_size = _trimmed_mean(winsorized, TRADE_TRIM_PCT) if values else None
                threshold, method = _retail_threshold(winsorized, category if category != "all" else None)
                small_values = [v for v in values if v <= threshold]
                small_trade_share = len(small_values) / total_trades if total_trades else None
                retail_volume_share = sum(small_values) / total_volume if total_volume else None
                evening_share = bucket["evening"] / total_trades if total_trades else None
                weekend_share = bucket["weekend"] / total_trades if total_trades else None

                hourly_values = [bucket["hourly"].get(h, 0) for h in range(24)]
                smoothed = _smooth_series(hourly_values, BURSTINESS_SMOOTH_WINDOW)
                mean_volume = sum(smoothed) / len(smoothed) if smoothed else 0
                burstiness = max(smoothed) / mean_volume if mean_volume > 0 else 0

                whale_share, _ = _whale_share(values)
                whale_dominated = whale_share >= 0.5
                flow_score = min(1.0, (small_trade_share or 0) / 0.6) if small_trade_share else 0
                attention_score = min(1.0, max(burstiness - 1.0, 0.0) / 1.5) if burstiness else 0
                retail_score, retail_level = _retail_score(
                    small_trade_share or 0,
                    evening_share or 0,
                    weekend_share or 0,
                    burstiness
                )

                rollup = RetailRollup(
                    date=day_cursor,
                    category=category,
                    total_trades=total_trades,
                    total_volume=total_volume,
                    avg_trade_size=avg_trade_size,
                    retail_trade_share=small_trade_share,
                    retail_volume_share=retail_volume_share,
                    small_trade_share=small_trade_share,
                    evening_share=evening_share,
                    weekend_share=weekend_share,
                    burstiness=burstiness,
                    flow_score=flow_score,
                    attention_score=attention_score,
                    retail_score=retail_score,
                    retail_level=retail_level,
                    whale_share=whale_share,
                    whale_dominated=whale_dominated,
                    trade_threshold=threshold,
                    trade_threshold_method=method,
                    markets_covered=len(bucket["markets"]),
                    quality_markets=quality_counts.get(category, 0),
                    confidence_label=_confidence_label(total_trades),
                    confidence_score=_confidence_score(total_trades)
                )
                session.add(rollup)

            day_cursor += timedelta(days=1)

        session.commit()
        logger.info("Retail rollups updated for %s days.", days)

    finally:
        session.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Build daily retail rollups")
    parser.add_argument("--days", type=int, default=ROLLUP_DAYS, help="Days to recompute")
    parser.add_argument("--category", action="append", default=None, help="Category to include (repeatable)")
    return parser.parse_args()


def main():
    args = _parse_args()
    build_retail_rollups(days=args.days, categories=args.category)


if __name__ == "__main__":
    main()
