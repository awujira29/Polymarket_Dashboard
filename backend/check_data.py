from database import get_db_session
from models import Market, MarketSnapshot
from sqlalchemy import func

def check_collected_data():
    """Check what data we've collected"""
    db = get_db_session()
    
    try:
        # Count markets
        market_count = db.query(Market).count()
        print(f"\n📊 Total Markets: {market_count}")
        print("=" * 80)
        
        # Show markets
        markets = db.query(Market).all()
        for i, market in enumerate(markets, 1):
            print(f"\n{i}. {market.title}")
            print(f"   ID: {market.id}")
            print(f"   Category: {market.category}")
            print(f"   Status: {market.status}")
            
            # Count snapshots for this market
            snapshot_count = db.query(MarketSnapshot).filter_by(market_id=market.id).count()
            print(f"   Snapshots collected: {snapshot_count}")
            
            # Show latest snapshot
            latest = db.query(MarketSnapshot).filter_by(market_id=market.id).order_by(MarketSnapshot.timestamp.desc()).first()
            if latest:
                print(f"   Latest price: ${latest.price:.3f}")
                print(f"   24h Volume: ${latest.volume_24h:,.0f}")
                print(f"   Liquidity: ${latest.liquidity:,.0f}")
        
        # Total snapshots
        total_snapshots = db.query(MarketSnapshot).count()
        print(f"\n📈 Total Snapshots: {total_snapshots}")
        
        # Time range
        first_snapshot = db.query(func.min(MarketSnapshot.timestamp)).scalar()
        last_snapshot = db.query(func.max(MarketSnapshot.timestamp)).scalar()
        
        if first_snapshot and last_snapshot:
            print(f"   First: {first_snapshot}")
            print(f"   Latest: {last_snapshot}")
        
    finally:
        db.close()

if __name__ == "__main__":
    check_collected_data()