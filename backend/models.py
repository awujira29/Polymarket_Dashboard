from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean, JSON, Date, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Market(Base):
    __tablename__ = "markets"

    id = Column(String, primary_key=True)
    condition_id = Column(String)
    title = Column(Text)
    category = Column(String, index=True)  # politics, crypto, sports, etc.
    subcategory = Column(String, nullable=True)  # More specific categorization
    description = Column(Text, nullable=True)
    end_date = Column(DateTime, nullable=True)
    end_date_iso = Column(String, nullable=True)
    closed = Column(String, nullable=True)  # "true"/"false" or boolean
    archived = Column(String, nullable=True)  # "true"/"false" or boolean
    status = Column(String)
    platform = Column(String, default="polymarket")  # polymarket
    created_at = Column(DateTime, default=datetime.utcnow)

    # Market metadata
    event_title = Column(Text, nullable=True)
    event_tags = Column(JSON, nullable=True)  # Store tags as JSON array
    outcomes = Column(JSON, nullable=True)  # Store possible outcomes

    # Social media tracking
    twitter_mentions_24h = Column(Integer, default=0)
    social_hype_score = Column(Float, default=0)

    # Relationships
    snapshots = relationship("MarketSnapshot", back_populates="market")
    trades = relationship("Trade", back_populates="market")

    def __repr__(self):
        return f"<Market(id={self.id}, title={self.title[:30]}..., category={self.category})>"


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Price data
    price = Column(Float)  # Current price (0-1 for binary markets)

    # Volume data
    volume_24h = Column(Float, default=0)
    volume_num_trades = Column(Integer, default=0)
    volume_all_time = Column(Float, default=0)

    # Liquidity data
    liquidity = Column(Float, default=0)

    # Advanced trading metrics for retail analysis
    open_interest = Column(Float, default=0)
    num_traders = Column(Integer, default=0)
    avg_trade_size = Column(Float, default=0)
    large_trade_count = Column(Integer, default=0)  # Trades > $1000

    # Time-based patterns for retail detection
    is_weekend = Column(Boolean, default=False)
    is_evening = Column(Boolean, default=False)  # After 6 PM ET
    hour_of_day = Column(Integer, default=0)

    # Derived retail behavior metrics
    retail_score = Column(Float, nullable=True)  # 0-10 scale
    volume_spike_flag = Column(String, nullable=True)  # 'spike', 'surge', etc.
    social_correlation = Column(Float, default=0)  # Correlation with social mentions

    # Market lifecycle indicators
    lifecycle_stage = Column(String, nullable=True)  # 'emerging', 'hype', 'peak', 'decline'
    meme_market_score = Column(Float, default=0)  # 0-1 scale

    # Relationships
    market = relationship("Market", back_populates="snapshots")

    def __repr__(self):
        return f"<Snapshot(market_id={self.market_id}, timestamp={self.timestamp}, price={self.price})>"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), index=True)
    timestamp = Column(DateTime, index=True)
    price = Column(Float)
    quantity = Column(Float)
    side = Column(String)  # 'buy' or 'sell'
    value = Column(Float)  # price * quantity
    taker = Column(String, nullable=True)  # Trader ID
    maker = Column(String, nullable=True)  # Counterparty ID

    # Trade classification
    trade_size_category = Column(String, default='medium')  # 'small', 'medium', 'large'
    is_retail_trade = Column(Boolean, default=True)  # Based on size thresholds

    # Relationships
    market = relationship("Market", back_populates="trades")

    def __repr__(self):
        return f"<Trade(market_id={self.market_id}, timestamp={self.timestamp}, value={self.value}, side={self.side})>"


class DataCollectionRun(Base):
    __tablename__ = "data_collection_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    categories_collected = Column(JSON)  # List of categories collected
    total_markets = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0)
    status = Column(String, default='completed')  # 'completed', 'failed', 'partial'

    def __repr__(self):
        return f"<DataCollectionRun(timestamp={self.timestamp}, total_markets={self.total_markets}, status={self.status})>"


class RetailRollup(Base):
    __tablename__ = "retail_rollups"
    __table_args__ = (
        UniqueConstraint("date", "category", name="uq_retail_rollup_date_category"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    category = Column(String, index=True)

    total_trades = Column(Integer, default=0)
    total_volume = Column(Float, default=0)
    avg_trade_size = Column(Float, nullable=True)
    retail_trade_share = Column(Float, nullable=True)
    retail_volume_share = Column(Float, nullable=True)
    small_trade_share = Column(Float, nullable=True)
    evening_share = Column(Float, nullable=True)
    weekend_share = Column(Float, nullable=True)
    burstiness = Column(Float, nullable=True)
    flow_score = Column(Float, nullable=True)
    attention_score = Column(Float, nullable=True)
    retail_score = Column(Float, nullable=True)
    retail_level = Column(String, nullable=True)
    whale_share = Column(Float, nullable=True)
    whale_dominated = Column(Boolean, default=False)
    trade_threshold = Column(Float, nullable=True)
    trade_threshold_method = Column(String, nullable=True)
    markets_covered = Column(Integer, default=0)
    quality_markets = Column(Integer, default=0)
    confidence_label = Column(String, default="insufficient")
    confidence_score = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RetailRollup(date={self.date}, category={self.category}, trades={self.total_trades})>"
