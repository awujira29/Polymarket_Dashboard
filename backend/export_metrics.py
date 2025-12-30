#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from api import (
    _bucket_trade_sizes,
    _canonical_market_id,
    _compute_lifecycle,
    _compute_retail_signals,
    _compute_snapshot_signals,
    _market_quality_ok,
    _normalize_category,
    _retail_threshold,
    _to_utc_iso,
    _winsorize,
    analyze_trade_patterns,
    RETAIL_MIN_TRADES,
    TRADE_WINSOR_PCT
)
from database import engine, get_db_session
from models import Market, MarketSnapshot, Trade

ALLOWED_TABLES = {
    "markets",
    "market_snapshots",
    "trades",
    "retail_rollups",
    "data_collection_runs"
}

TIME_FILTERS = {
    "market_snapshots": "timestamp",
    "trades": "timestamp",
    "data_collection_runs": "timestamp"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export prediction market metrics to CSV."
    )
    parser.add_argument(
        "--outdir",
        default="exports",
        help="Directory to write CSV files."
    )
    parser.add_argument(
        "--tables",
        default="markets,market_snapshots,retail_rollups,trades,data_collection_runs",
        help="Comma-separated list of tables to export."
    )
    parser.add_argument(
        "--snapshots-days",
        type=int,
        default=None,
        help="Only export snapshots from the last N days."
    )
    parser.add_argument(
        "--trades-days",
        type=int,
        default=None,
        help="Only export trades from the last N days."
    )
    parser.add_argument(
        "--runs-days",
        type=int,
        default=None,
        help="Only export data_collection_runs from the last N days."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows per table."
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip derived metrics exports."
    )
    return parser.parse_args()


def _since_days(days: int | None) -> datetime | None:
    if days is None:
        return None
    return datetime.utcnow() - timedelta(days=days)


def _load_table(table: str, since: datetime | None, limit: int | None) -> pd.DataFrame:
    sql = f"SELECT * FROM {table}"
    params = {}
    clauses = []
    if since and table in TIME_FILTERS:
        clauses.append(f"{TIME_FILTERS[table]} >= :since")
        params["since"] = since
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = limit
    return pd.read_sql_query(text(sql), engine, params=params)

def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value

def _flatten_signals(signals: dict, prefix: str) -> dict:
    row = {}
    for key, value in (signals or {}).items():
        column = f"{prefix}{key}"
        if isinstance(value, (list, dict)):
            row[column] = json.dumps(value, ensure_ascii=True)
        else:
            row[column] = value
    return row

def _fetch_snapshots(session, market_id: str, since: datetime | None):
    query = session.query(MarketSnapshot).filter_by(market_id=market_id)
    if since:
        query = query.filter(MarketSnapshot.timestamp >= since)
    return query.order_by(MarketSnapshot.timestamp).all()

def _fetch_trades(session, market_id: str, since: datetime | None):
    query = session.query(Trade).filter_by(market_id=market_id)
    if since:
        query = query.filter(Trade.timestamp >= since)
    return query.order_by(Trade.timestamp).all()

def _trade_analysis_rows(trades: list[Trade], category: str | None):
    valid_trades = [
        trade for trade in trades
        if trade.value is not None and trade.value > 0 and trade.timestamp is not None
    ]
    values = [trade.value for trade in valid_trades]
    if values:
        winsorized = _winsorize(values, TRADE_WINSOR_PCT, TRADE_WINSOR_PCT)
        threshold, method = _retail_threshold(winsorized, category)
    else:
        threshold, method = None, None

    trade_data = []
    for trade in valid_trades:
        is_retail = trade.value <= threshold if threshold is not None else False
        trade_data.append({
            "timestamp": _to_utc_iso(trade.timestamp),
            "value": trade.value,
            "is_retail": is_retail
        })

    retail_analysis = analyze_trade_patterns(trade_data)
    trade_size_distribution = _bucket_trade_sizes(valid_trades)

    return {
        "retail_analysis": retail_analysis or {},
        "trade_size_distribution": trade_size_distribution or [],
        "retail_threshold": threshold,
        "retail_threshold_method": method,
        "retail_threshold_trades": len(values)
    }

def _export_derived(outdir: Path, snapshots_since: datetime | None, trades_since: datetime | None):
    session = get_db_session()
    try:
        markets = session.query(Market).all()
        market_rows = []
        trade_signal_rows = []
        snapshot_signal_rows = []
        trade_summary_rows = []
        hourly_pattern_rows = []
        trade_size_rows = []

        score_by_category = {}

        for market in markets:
            category = _normalize_category(market.category)
            snapshots = _fetch_snapshots(session, market.id, snapshots_since)
            trades = _fetch_trades(session, market.id, trades_since)
            latest_snapshot = snapshots[-1] if snapshots else None

            last_trade_time = None
            if trades:
                last_trade_time = max((t.timestamp for t in trades if t.timestamp), default=None)

            quality_ok, quality_reason = _market_quality_ok(
                latest_snapshot.volume_24h if latest_snapshot else None,
                latest_snapshot.liquidity if latest_snapshot else None
            )

            trade_signals = _compute_retail_signals(trades, category) if trades else _compute_retail_signals([])
            snapshot_signals = _compute_snapshot_signals(snapshots) if snapshots else _compute_snapshot_signals([])

            if len(trades) >= RETAIL_MIN_TRADES and quality_ok:
                effective_signals = trade_signals.copy()
                effective_signals["coverage"] = "trade"
                effective_signals["sample_trades"] = len(trades)
            else:
                effective_signals = snapshot_signals.copy()
                effective_signals["coverage"] = "insufficient"
                effective_signals["sample_trades"] = len(trades)

            effective_signals["quality_ok"] = quality_ok
            effective_signals["quality_reason"] = quality_reason

            if effective_signals.get("coverage") == "trade" and quality_ok:
                score = effective_signals.get("score")
                if score is not None:
                    score_by_category.setdefault(category, []).append(score)

            lifecycle = _compute_lifecycle(snapshots[-60:]) if snapshots else _compute_lifecycle([])

            trade_analysis = _trade_analysis_rows(trades, category)
            retail_analysis = trade_analysis["retail_analysis"]

            if retail_analysis:
                trade_summary_rows.append({
                    "market_id": market.id,
                    "category": category,
                    "total_trades": retail_analysis.get("total_trades"),
                    "total_volume": retail_analysis.get("total_volume"),
                    "avg_trade_size": retail_analysis.get("avg_trade_size"),
                    "peak_retail_hours": json.dumps(retail_analysis.get("peak_retail_hours", []), ensure_ascii=True),
                    "retail_dominance_score": retail_analysis.get("retail_dominance_score"),
                    "retail_threshold": trade_analysis.get("retail_threshold"),
                    "retail_threshold_method": trade_analysis.get("retail_threshold_method"),
                    "retail_threshold_trades": trade_analysis.get("retail_threshold_trades")
                })

                for pattern in retail_analysis.get("hourly_patterns", []):
                    hourly_pattern_rows.append({
                        "market_id": market.id,
                        "category": category,
                        "hour": pattern.get("hour"),
                        "trades": pattern.get("trades"),
                        "retail_trades": pattern.get("retail_trades"),
                        "volume": pattern.get("volume"),
                        "retail_volume": pattern.get("retail_volume")
                    })

            for bucket in trade_analysis.get("trade_size_distribution", []):
                trade_size_rows.append({
                    "market_id": market.id,
                    "category": category,
                    "range": bucket.get("range"),
                    "count": bucket.get("count"),
                    "share": bucket.get("share")
                })

            trade_signal_rows.append({
                "market_id": market.id,
                "category": category,
                **_flatten_signals(trade_signals, "trade_")
            })
            snapshot_signal_rows.append({
                "market_id": market.id,
                "category": category,
                **_flatten_signals(snapshot_signals, "snapshot_")
            })

            total_value = sum(t.value for t in trades if t.value) if trades else 0
            trade_count = len([t for t in trades if t.value is not None])
            avg_trade_size_window = (
                total_value / trade_count
                if trade_count >= RETAIL_MIN_TRADES else None
            )

            market_rows.append({
                "market_id": market.id,
                "canonical_id": _canonical_market_id(market),
                "title": market.title,
                "category": category,
                "subcategory": market.subcategory,
                "description": market.description,
                "end_date": market.end_date_iso,
                "status": market.status,
                "closed": market.closed,
                "archived": market.archived,
                "platform": market.platform,
                "created_at": _to_utc_iso(market.created_at),
                "event_title": _json_value(market.event_title),
                "event_tags": _json_value(market.event_tags),
                "outcomes": _json_value(market.outcomes),
                "latest_price": latest_snapshot.price if latest_snapshot else None,
                "latest_volume_24h": latest_snapshot.volume_24h if latest_snapshot else None,
                "latest_liquidity": latest_snapshot.liquidity if latest_snapshot else None,
                "latest_avg_trade_size": latest_snapshot.avg_trade_size if latest_snapshot else None,
                "latest_timestamp": _to_utc_iso(latest_snapshot.timestamp) if latest_snapshot else None,
                "last_trade_time": _to_utc_iso(last_trade_time),
                "trade_count_total": trade_count,
                "trade_volume_total": total_value,
                "avg_trade_size_window": avg_trade_size_window,
                "quality_ok": quality_ok,
                "quality_reason": quality_reason,
                "retail_score_z": None,
                **_flatten_signals(effective_signals, "signal_"),
                "lifecycle_stage": lifecycle.get("stage"),
                "lifecycle_trend": lifecycle.get("trend"),
                "lifecycle_growth_rate": lifecycle.get("growth_rate"),
                "lifecycle_volatility": lifecycle.get("volatility"),
                "lifecycle_spike_count": lifecycle.get("spike_count"),
                "lifecycle_confidence": lifecycle.get("confidence")
            })

        stats_by_category = {}
        for category, scores in score_by_category.items():
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = variance ** 0.5
            stats_by_category[category] = (mean, std)

        for row in market_rows:
            category = row.get("category")
            score = row.get("signal_score")
            mean_std = stats_by_category.get(category)
            if score is None or not mean_std or mean_std[1] == 0:
                row["retail_score_z"] = None
            else:
                row["retail_score_z"] = (score - mean_std[0]) / mean_std[1]

        pd.DataFrame(market_rows).to_csv(outdir / "market_enriched.csv", index=False)
        pd.DataFrame(trade_signal_rows).to_csv(outdir / "market_signals_trade.csv", index=False)
        pd.DataFrame(snapshot_signal_rows).to_csv(outdir / "market_signals_snapshot.csv", index=False)
        pd.DataFrame(trade_summary_rows).to_csv(outdir / "market_trade_summary.csv", index=False)
        pd.DataFrame(hourly_pattern_rows).to_csv(outdir / "market_trade_patterns_hourly.csv", index=False)
        pd.DataFrame(trade_size_rows).to_csv(outdir / "market_trade_size_distribution.csv", index=False)
    finally:
        session.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    invalid = [t for t in tables if t not in ALLOWED_TABLES]
    if invalid:
        raise SystemExit(f"Unknown table(s): {', '.join(invalid)}")

    since_map = {
        "market_snapshots": _since_days(args.snapshots_days),
        "trades": _since_days(args.trades_days),
        "data_collection_runs": _since_days(args.runs_days)
    }

    for table in tables:
        df = _load_table(table, since_map.get(table), args.limit)
        output_path = outdir / f"{table}.csv"
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(df)} rows -> {output_path}")

    if not args.skip_derived:
        _export_derived(outdir, since_map.get("market_snapshots"), since_map.get("trades"))


if __name__ == "__main__":
    main()
