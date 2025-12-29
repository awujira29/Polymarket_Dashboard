#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime, timezone, timedelta

from fetch_polymarket import PolymarketCollector
from database import init_db, get_db_session
from models import Market, Trade
from config import (
    TRACKED_CATEGORIES,
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD,
    TRADE_LOOKBACK_DAYS_FULL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def _retail_percentile_for_category(category):
    if not category:
        return RETAIL_SIZE_PERCENTILES.get("default", 0.3)
    return RETAIL_SIZE_PERCENTILES.get(category, RETAIL_SIZE_PERCENTILES.get("default", 0.3))

def _retail_threshold(values, category=None, min_trades=RETAIL_MIN_TRADES, fallback=RETAIL_FALLBACK_THRESHOLD):
    percentile = _retail_percentile_for_category(category)
    if len(values) < min_trades:
        return fallback, "fixed"
    return _percentile(values, percentile), "percentile"


def _parse_args():
    parser = argparse.ArgumentParser(description="Backfill public Polymarket trades")
    parser.add_argument("--pages", type=int, default=20, help="Number of pages to fetch (limit=500 per page)")
    parser.add_argument("--limit", type=int, default=500, help="Trades per page")
    parser.add_argument("--days", type=int, default=TRADE_LOOKBACK_DAYS_FULL, help="Lookback window in days")
    parser.add_argument("--taker-only", action="store_true", help="Use takerOnly=true on public feed")
    parser.add_argument("--dry-run", action="store_true", help="Fetch trades without writing to DB")
    return parser.parse_args()


def main():
    args = _parse_args()
    init_db()

    collector = PolymarketCollector()
    session = get_db_session()

    try:
        markets = session.query(Market).filter(Market.category.in_(TRACKED_CATEGORIES)).all()
        missing_condition = [m for m in markets if not m.condition_id]

        if missing_condition or not markets:
            logger.info("Syncing missing condition_id values from Polymarket events...")
            refreshed = collector.get_active_markets(limit=200, categories=TRACKED_CATEGORIES)
            refreshed_by_id = {m["id"]: m for m in refreshed}
            updated = 0

            for market in session.query(Market).all():
                data = refreshed_by_id.get(market.id)
                if not data:
                    continue
                if data.get("condition_id") and not market.condition_id:
                    market.condition_id = data["condition_id"]
                if data.get("category") and market.category != data["category"]:
                    market.category = data["category"]
                updated += 1

            if updated:
                session.commit()

            markets = session.query(Market).filter(Market.category.in_(TRACKED_CATEGORIES)).all()
        condition_map = {
            m.condition_id: m.id for m in markets if m.condition_id
        }
        market_category = {m.id: m.category for m in markets}
        condition_ids = set(condition_map.keys())

        if not condition_ids:
            logger.warning("No markets with condition_id found for tracked categories.")
            return

        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=args.days)
        trades = collector._fetch_public_trades_pages(
            pages=args.pages,
            limit=args.limit,
            taker_only=args.taker_only,
            since=since
        )

        logger.info("Fetched %s trades from public feed", len(trades))

        parsed_trades = []
        for trade in trades:
            condition_id = trade.get("conditionId")
            if condition_id not in condition_ids:
                continue

            price = collector._safe_float(trade.get("price", 0), 0.0)
            quantity = collector._safe_float(trade.get("size", trade.get("quantity", 0)), 0.0)
            timestamp = collector._parse_trade_timestamp(trade.get("timestamp") or trade.get("created_at"))
            if not timestamp or timestamp < since:
                continue

            value = price * quantity
            market_id = condition_map[condition_id]
            parsed_trades.append({
                "market_id": market_id,
                "timestamp": timestamp,
                "price": price,
                "quantity": quantity,
                "value": value,
                "side": trade.get("side", "unknown"),
                "taker": trade.get("proxyWallet"),
                "maker": trade.get("maker")
            })

        trade_values = {}
        for trade in parsed_trades:
            trade_values.setdefault(trade["market_id"], []).append(trade["value"])
        threshold_by_market = {
            market_id: _retail_threshold(values, market_category.get(market_id))[0]
            for market_id, values in trade_values.items()
        }

        stored = 0
        skipped = 0
        for trade in parsed_trades:
            market_id = trade["market_id"]
            threshold = threshold_by_market.get(market_id, 100.0)

            existing = session.query(Trade).filter_by(
                market_id=market_id,
                timestamp=trade["timestamp"],
                value=trade["value"]
            ).first()

            if existing:
                skipped += 1
                continue

            if not args.dry_run:
                session.add(Trade(
                    market_id=market_id,
                    timestamp=trade["timestamp"],
                    price=trade["price"],
                    quantity=trade["quantity"],
                    side=trade["side"],
                    value=trade["value"],
                    taker=trade["taker"],
                    maker=trade["maker"],
                    trade_size_category=collector._classify_trade_size(trade["value"]),
                    is_retail_trade=trade["value"] <= threshold
                ))
            stored += 1

        if not args.dry_run:
            session.commit()

        logger.info("Stored %s trades (skipped %s duplicates)", stored, skipped)

    finally:
        session.close()


if __name__ == "__main__":
    main()
