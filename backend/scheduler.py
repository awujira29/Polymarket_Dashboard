import schedule
import time
import logging
from datetime import datetime, timedelta, timezone
from fetch_polymarket import PolymarketCollector
from database import get_db_session, init_db
from models import Market, MarketSnapshot, Trade, DataCollectionRun
import threading
from config import (
    TRACKED_CATEGORIES,
    TRADE_PAGES,
    TRADE_LOOKBACK_DAYS_FAST,
    TRADE_LOOKBACK_DAYS_FULL,
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD,
    ROLLUP_DAYS
)
from rollup_retail import build_retail_rollups

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

class ContinuousDataCollector:
    def __init__(self):
        self.collector = PolymarketCollector()
        self.is_running = False
        self.collection_thread = None

    def collect_comprehensive_data(
        self,
        categories=None,
        markets_per_category=15,
        include_trades=True,
        trade_pages=TRADE_PAGES,
        trade_lookback_days=None
    ):
        """Collect comprehensive market and trade data"""
        start_time = datetime.now(timezone.utc)

        try:
            logger.info("🚀 Starting comprehensive data collection...")

            # Collect data
            data = self.collector.collect_comprehensive_data(
                categories=categories,
                markets_per_category=markets_per_category,
                include_trades=include_trades,
                trade_pages=trade_pages if include_trades else 0,
                trade_lookback_days=trade_lookback_days
            )

            # Store in database
            session = get_db_session()
            try:
                run_record = DataCollectionRun(
                    categories_collected=list(data['categories'].keys()),
                    total_markets=data['total_markets'],
                    total_trades=data['total_trades'],
                    duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    status='completed'
                )
                session.add(run_record)

                # Store market data and snapshots
                for category, markets in data['categories'].items():
                    for market_data in markets:
                        is_closed = self.collector._bool_flag(market_data.get('closed', False))
                        is_archived = self.collector._bool_flag(market_data.get('archived', False))
                        is_active = market_data.get('active', True)
                        raw_status = str(market_data.get('status') or "").strip().lower()
                        if raw_status:
                            status_label = raw_status
                        elif is_closed:
                            status_label = 'closed'
                        elif is_active:
                            status_label = 'active'
                        else:
                            status_label = 'inactive'
                        end_date_iso = (
                            market_data.get('endDateIso')
                            or market_data.get('end_date')
                            or market_data.get('endDate')
                        )

                        # Check if market exists
                        market = session.query(Market).filter_by(id=market_data['id']).first()
                        if not market:
                            # Create new market
                            market = Market(
                                id=market_data['id'],
                                condition_id=market_data.get('condition_id') or market_data.get('conditionId'),
                                title=market_data['question'],
                                category=category,
                                event_title=market_data['event_title'],
                                event_tags=market_data['event_tags'],
                                outcomes=market_data['outcomes'],
                                status=status_label,
                                closed='true' if is_closed else 'false',
                                archived='true' if is_archived else 'false',
                                end_date_iso=end_date_iso
                            )
                            session.add(market)
                        else:
                            # Update existing market
                            market.category = category
                            market.event_tags = market_data['event_tags']
                            market.status = status_label
                            market.closed = 'true' if is_closed else 'false'
                            market.archived = 'true' if is_archived else 'false'
                            if end_date_iso:
                                market.end_date_iso = end_date_iso
                            if market_data.get('condition_id') and not market.condition_id:
                                market.condition_id = market_data.get('condition_id')

                        # Create snapshot
                        volume_24h = market_data.get('volume_24h', 0)
                        volume_num_trades = market_data.get('trade_count', 0)
                        avg_trade_size = market_data.get('avg_trade_size')
                        if not avg_trade_size and volume_num_trades:
                            avg_trade_size = volume_24h / max(volume_num_trades, 1)
                        if not volume_num_trades or volume_num_trades < RETAIL_MIN_TRADES:
                            avg_trade_size = None
                        now = datetime.now(timezone.utc).replace(tzinfo=None)
                        snapshot = MarketSnapshot(
                            market_id=market_data['id'],
                            price=market_data['price'],
                            volume_24h=volume_24h,
                            volume_num_trades=volume_num_trades,
                            avg_trade_size=avg_trade_size,
                            liquidity=market_data['liquidity'],
                            hour_of_day=now.hour,
                            is_weekend=now.weekday() >= 5,
                            is_evening=now.hour >= 18 or now.hour <= 5
                        )
                        session.add(snapshot)

                        # Store trades if available
                        if 'recent_trades' in market_data and market_data['recent_trades']:
                            trade_values = [t['value'] for t in market_data['recent_trades']]
                            trade_threshold, _ = _retail_threshold(trade_values, category)
                            for trade_data in market_data['recent_trades']:
                                timestamp = self.collector._parse_trade_timestamp(trade_data.get('timestamp'))
                                if not timestamp:
                                    continue
                                # Check if trade already exists (avoid duplicates)
                                existing_trade = session.query(Trade).filter_by(
                                    market_id=market_data['id'],
                                    timestamp=timestamp,
                                    value=trade_data['value']
                                ).first()

                                if not existing_trade:
                                    trade = Trade(
                                        market_id=market_data['id'],
                                        timestamp=timestamp,
                                        price=trade_data['price'],
                                        quantity=trade_data['quantity'],
                                        side=trade_data['side'],
                                        value=trade_data['value'],
                                        taker=trade_data.get('taker'),
                                        maker=trade_data.get('maker'),
                                        trade_size_category=self._classify_trade_size(trade_data['value']),
                                        is_retail_trade=trade_data['value'] <= trade_threshold
                                    )
                                    session.add(trade)

                session.commit()
                logger.info(f"✅ Data collection completed: {data['total_markets']} markets, {data['total_trades']} trades")

            except Exception as e:
                session.rollback()
                logger.error(f"Database error during collection: {e}")
                run_record.status = 'failed'
                session.add(run_record)
                session.commit()
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")

    def _classify_trade_size(self, value: float) -> str:
        """Classify trade size for analysis"""
        if value < 50:
            return 'small'
        elif value < 500:
            return 'medium'
        elif value < 5000:
            return 'large'
        else:
            return 'xlarge'

    def collect_high_frequency_data(self):
        """Collect data more frequently for active markets (every 5 minutes)"""
        logger.info("📊 Collecting high-frequency data for active markets...")
        # Focus on high-volume markets with more frequent updates
        self.collect_comprehensive_data(
            categories=TRACKED_CATEGORIES,
            markets_per_category=12,
            include_trades=True,
            trade_pages=min(TRADE_PAGES, 10),
            trade_lookback_days=TRADE_LOOKBACK_DAYS_FAST
        )

    def collect_comprehensive_update(self):
        """Collect comprehensive data across all categories (every hour)"""
        logger.info("🌍 Collecting comprehensive data across all categories...")
        self.collect_comprehensive_data(
            categories=TRACKED_CATEGORIES,
            markets_per_category=25,
            include_trades=True,
            trade_pages=TRADE_PAGES,
            trade_lookback_days=TRADE_LOOKBACK_DAYS_FULL
        )
        try:
            build_retail_rollups(days=ROLLUP_DAYS)
        except Exception as exc:
            logger.error("Rollup build failed: %s", exc)

    def start_continuous_collection(self):
        """Start the continuous data collection scheduler"""
        if self.is_running:
            logger.warning("Data collector is already running")
            return

        self.is_running = True
        logger.info("🚀 Starting continuous data collection...")

        # Schedule different collection frequencies
        schedule.every(5).minutes.do(self.collect_high_frequency_data)  # Active markets
        schedule.every(1).hours.do(self.collect_comprehensive_update)   # All markets

        # Start collection thread
        self.collection_thread = threading.Thread(target=self._run_scheduler)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        logger.info("✅ Continuous collection started")
        logger.info("  - High-frequency data: every 5 minutes")
        logger.info("  - Comprehensive data: every hour")

    def stop_continuous_collection(self):
        """Stop the continuous data collection"""
        self.is_running = False
        schedule.clear()
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("🛑 Continuous collection stopped")

    def _run_scheduler(self):
        """Run the scheduler loop"""
        # Run initial collection
        self.collect_comprehensive_update()

        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying

def collect_job():
    """Legacy job function for backward compatibility"""
    collector = ContinuousDataCollector()
    collector.collect_comprehensive_data()

def main():
    """Run the continuous data collector"""
    logger.info("🚀 Starting Prediction Market Continuous Data Collector")
    logger.info("Press Ctrl+C to stop")

    # Initialize database
    init_db()

    # Start continuous collection
    collector = ContinuousDataCollector()
    collector.start_continuous_collection()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Stopping continuous collector...")
        collector.stop_continuous_collection()
        logger.info("Goodbye!")

if __name__ == "__main__":
    main()
