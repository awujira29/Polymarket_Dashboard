#!/usr/bin/env python3
"""
Comprehensive Retail Behavior Analysis Tool
Analyzes Polymarket data to extract insights on retail trading patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
import sys
import os

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Market, MarketSnapshot, Trade, DataCollectionRun
from database import get_db_session, init_db
from retail_analyzer import RetailBehaviorAnalyzer
from config import RETAIL_SIZE_PERCENTILES, RETAIL_MIN_TRADES, RETAIL_FALLBACK_THRESHOLD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRetailAnalyzer:
    """Advanced analysis tool for retail behavior insights"""

    def __init__(self):
        self.analyzer = RetailBehaviorAnalyzer()
        init_db()  # Ensure database is initialized

    def comprehensive_market_analysis(self, market_id: Optional[str] = None, category: Optional[str] = None,
                                    days_back: int = 7) -> Dict:
        """Comprehensive analysis of market(s) for retail behavior insights"""

        db_session = get_db_session()

        try:
            # Get markets to analyze
            query = db_session.query(Market)

            if market_id:
                query = query.filter(Market.id == market_id)
            elif category:
                query = query.filter(Market.category == category)

            markets = query.all()

            if not markets:
                return {"error": f"No markets found for criteria: market_id={market_id}, category={category}"}

            results = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "parameters": {
                    "market_id": market_id,
                    "category": category,
                    "days_back": days_back,
                    "markets_analyzed": len(markets)
                },
                "market_insights": [],
                "category_summary": {},
                "retail_behavior_patterns": {},
                "anomaly_detection": {},
                "predictive_insights": {}
            }

            # Analyze each market
            for market in markets:
                market_analysis = self.analyzer.analyze_market_retail_behavior(market.id, days_back)
                if "error" not in market_analysis:
                    market_analysis["market_title"] = market.title
                    market_analysis["category"] = market.category
                    results["market_insights"].append(market_analysis)

            # Generate category-level summaries
            if results["market_insights"]:
                results["category_summary"] = self._generate_category_summary(results["market_insights"])
                results["retail_behavior_patterns"] = self._identify_behavioral_patterns(results["market_insights"])
                results["anomaly_detection"] = self._detect_anomalies(results["market_insights"])
                results["predictive_insights"] = self._generate_predictive_insights(results["market_insights"])

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
        finally:
            db_session.close()

    def trade_flow_analysis(self, market_id: str, hours_back: int = 24) -> Dict:
        """Detailed analysis of individual trade flows"""

        db_session = get_db_session()

        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

            # Get trades
            trades = db_session.query(Trade).filter(
                Trade.market_id == market_id,
                Trade.timestamp >= cutoff_time
            ).order_by(Trade.timestamp).all()

            if not trades:
                return {"error": "No trade data available"}

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'price': t.price,
                'quantity': t.quantity,
                'side': t.side,
                'value': t.value,
                'is_retail_trade': t.is_retail_trade,
                'trade_size_category': t.trade_size_category
            } for t in trades])

            market = db_session.query(Market).filter(Market.id == market_id).first()
            category = market.category if market else None
            percentile = RETAIL_SIZE_PERCENTILES.get(category, RETAIL_SIZE_PERCENTILES.get("default", 0.3))

            if len(df) >= RETAIL_MIN_TRADES:
                retail_threshold = float(df['value'].quantile(percentile))
                retail_method = "percentile"
            else:
                retail_threshold = RETAIL_FALLBACK_THRESHOLD
                retail_method = "fixed"
            df['is_retail_trade'] = df['value'] <= retail_threshold

            analysis = {
                "market_id": market_id,
                "time_period_hours": hours_back,
                "total_trades": len(df),
                "trade_characteristics": {},
                "flow_patterns": {},
                "retail_vs_institutional": {},
                "temporal_distribution": {},
                "price_impact_analysis": {}
            }

            # Trade characteristics
            analysis["trade_characteristics"] = {
                "avg_trade_size": float(df['value'].mean()),
                "median_trade_size": float(df['value'].median()),
                "total_volume": float(df['value'].sum()),
                "buy_sell_ratio": float(len(df[df['side'] == 'buy']) / max(len(df[df['side'] == 'sell']), 1)),
                "retail_trade_percentage": float(len(df[df['is_retail_trade'] == True]) / len(df)) if len(df) > 0 else 0,
                "retail_threshold": retail_threshold,
                "retail_threshold_method": retail_method
            }

            # Flow patterns
            analysis["flow_patterns"] = self._analyze_trade_flows(df)

            # Retail vs Institutional
            analysis["retail_vs_institutional"] = self._compare_retail_institutional(df)

            # Temporal distribution
            analysis["temporal_distribution"] = self._analyze_temporal_trade_distribution(df)

            # Price impact
            analysis["price_impact_analysis"] = self._analyze_price_impact(df)

            return analysis

        except Exception as e:
            logger.error(f"Error in trade flow analysis: {e}")
            return {"error": str(e)}
        finally:
            db_session.close()

    def _analyze_trade_flows(self, df: pd.DataFrame) -> Dict:
        """Analyze trade flow patterns"""

        # Trade size distribution
        size_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        size_labels = ['Micro (<$10)', 'Small ($10-50)', 'Medium ($50-100)', 'Large ($100-500)', 'XL ($500-1000)', 'Whale (>$1000)']

        df['size_category'] = pd.cut(df['value'], bins=size_bins, labels=size_labels, right=False)
        size_distribution = df['size_category'].value_counts().to_dict()

        # Buy/sell patterns by size
        buy_sell_by_size = {}
        for size_cat in size_labels:
            cat_data = df[df['size_category'] == size_cat]
            if len(cat_data) > 0:
                buy_ratio = len(cat_data[cat_data['side'] == 'buy']) / len(cat_data)
                buy_sell_by_size[size_cat] = {
                    'buy_percentage': float(buy_ratio),
                    'sell_percentage': float(1 - buy_ratio),
                    'count': len(cat_data)
                }

        # Trade clustering (trades in short time windows)
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        clusters = len(df[df['time_diff'] < 60])  # Trades within 1 minute

        return {
            "size_distribution": size_distribution,
            "buy_sell_by_size": buy_sell_by_size,
            "trade_clusters_1min": clusters,
            "avg_time_between_trades": float(df['time_diff'].mean()) if len(df) > 1 else 0
        }

    def _compare_retail_institutional(self, df: pd.DataFrame) -> Dict:
        """Compare retail vs institutional trading patterns"""

        retail_trades = df[df['is_retail_trade'] == True]
        inst_trades = df[df['is_retail_trade'] == False]

        comparison = {
            "retail_trades_count": len(retail_trades),
            "institutional_trades_count": len(inst_trades),
            "retail_percentage": float(len(retail_trades) / len(df)) if len(df) > 0 else 0
        }

        if len(retail_trades) > 0:
            comparison["retail_avg_size"] = float(retail_trades['value'].mean())
            comparison["retail_buy_percentage"] = float(len(retail_trades[retail_trades['side'] == 'buy']) / len(retail_trades))

        if len(inst_trades) > 0:
            comparison["institutional_avg_size"] = float(inst_trades['value'].mean())
            comparison["institutional_buy_percentage"] = float(len(inst_trades[inst_trades['side'] == 'buy']) / len(inst_trades))

        return comparison

    def _analyze_temporal_trade_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze how trades are distributed over time"""

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

        hourly_distribution = df.groupby('hour').size().to_dict()
        daily_distribution = df.groupby('day_of_week').size().to_dict()

        # Peak trading hours
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None
        peak_day = max(daily_distribution.items(), key=lambda x: x[1])[0] if daily_distribution else None

        return {
            "hourly_distribution": hourly_distribution,
            "daily_distribution": daily_distribution,
            "peak_trading_hour": peak_hour,
            "peak_trading_day": peak_day,
            "trading_intensity_score": len(df) / 24  # trades per hour on average
        }

    def _analyze_price_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze price impact of different trade sizes"""

        if len(df) < 2:
            return {"insufficient_data": True}

        # Sort by timestamp and calculate price changes
        df = df.sort_values('timestamp').copy()
        df['price_change'] = df['price'].diff()

        # Large trades and their price impact
        large_trades = df[df['value'] > 1000]
        large_trade_impact = large_trades['price_change'].abs().mean() if len(large_trades) > 0 else 0

        small_trades = df[df['value'] <= 100]
        small_trade_impact = small_trades['price_change'].abs().mean() if len(small_trades) > 0 else 0

        return {
            "large_trade_avg_impact": float(large_trade_impact),
            "small_trade_avg_impact": float(small_trade_impact),
            "impact_ratio": float(large_trade_impact / small_trade_impact) if small_trade_impact > 0 else 0,
            "price_volatility": float(df['price'].std())
        }

    def _generate_category_summary(self, market_insights: List[Dict]) -> Dict:
        """Generate category-level summary statistics"""

        if not market_insights:
            return {}

        categories = defaultdict(list)

        for insight in market_insights:
            cat = insight.get('category', 'unknown')
            retail_score = insight.get('overall_retail_score', {}).get('overall_score', 0)
            categories[cat].append(retail_score)

        summary = {}
        for cat, scores in categories.items():
            summary[cat] = {
                "market_count": len(scores),
                "avg_retail_score": float(np.mean(scores)),
                "median_retail_score": float(np.median(scores)),
                "retail_score_std": float(np.std(scores)),
                "high_retail_markets": len([s for s in scores if s > 7]),
                "low_retail_markets": len([s for s in scores if s < 4])
            }

        return summary

    def _identify_behavioral_patterns(self, market_insights: List[Dict]) -> Dict:
        """Identify common behavioral patterns across markets"""

        patterns = {
            "high_retail_markets": [],
            "meme_markets": [],
            "institutional_dominant": [],
            "weekend_warriors": [],
            "evening_traders": []
        }

        for insight in market_insights:
            market_id = insight['market_id']
            retail_score = insight.get('overall_retail_score', {}).get('overall_score', 0)

            # High retail activity
            if retail_score > 7:
                patterns["high_retail_markets"].append({
                    "market_id": market_id,
                    "title": insight.get('market_title', ''),
                    "score": retail_score
                })

            # Check for specific patterns
            volume_patterns = insight.get('retail_indicators', {}).get('volume_indicators', [])
            if 'strong_weekend_activity' in volume_patterns:
                patterns["weekend_warriors"].append(market_id)

            if 'strong_evening_activity' in volume_patterns:
                patterns["evening_traders"].append(market_id)

        return patterns

    def _detect_anomalies(self, market_insights: List[Dict]) -> Dict:
        """Detect anomalous retail behavior patterns"""

        if len(market_insights) < 3:
            return {"insufficient_data": True}

        retail_scores = [
            insight.get('overall_retail_score', {}).get('overall_score', 0)
            for insight in market_insights
        ]
        mean_score = np.mean(retail_scores)
        std_score = np.std(retail_scores)

        anomalies = []
        for insight in market_insights:
            score = insight.get('overall_retail_score', {}).get('overall_score', 0)
            z_score = (score - mean_score) / std_score if std_score > 0 else 0

            if abs(z_score) > 2:  # More than 2 standard deviations
                anomalies.append({
                    "market_id": insight['market_id'],
                    "title": insight.get('market_title', ''),
                    "retail_score": score,
                    "z_score": float(z_score),
                    "anomaly_type": "high_retail" if z_score > 0 else "low_retail"
                })

        return {
            "anomalies_detected": len(anomalies),
            "anomalous_markets": anomalies,
            "anomaly_threshold": 2.0
        }

    def _generate_predictive_insights(self, market_insights: List[Dict]) -> Dict:
        """Generate predictive insights based on patterns"""

        insights = {
            "retail_market_prediction": {},
            "risk_assessment": {},
            "opportunity_identification": {}
        }

        # Simple predictive model based on current patterns
        high_retail_markets = [
            m for m in market_insights
            if m.get('overall_retail_score', {}).get('overall_score', 0) > 7
        ]

        if high_retail_markets:
            avg_score = np.mean([
                m.get('overall_retail_score', {}).get('overall_score', 0)
                for m in high_retail_markets
            ])
            insights["retail_market_prediction"] = {
                "predicted_retail_dominance": float(avg_score > 6),
                "confidence_level": "high" if len(high_retail_markets) > 5 else "medium",
                "sample_size": len(high_retail_markets)
            }

        # Risk assessment
        volatile_markets = [m for m in market_insights if m.get('retail_indicators', {}).get('volume_volatility', 0) > 1.0]
        insights["risk_assessment"] = {
            "high_volatility_markets": len(volatile_markets),
            "volatility_risk_level": "high" if len(volatile_markets) > len(market_insights) * 0.3 else "moderate"
        }

        return insights

def main():
    """Main function for command-line usage"""

    analyzer = AdvancedRetailAnalyzer()

    print("🔍 Polymarket Retail Behavior Analysis Tool")
    print("=" * 50)

    # Example analyses
    print("\n1. Comprehensive Market Analysis (Crypto Category)")
    crypto_analysis = analyzer.comprehensive_market_analysis(category="crypto", days_back=7)
    if "error" not in crypto_analysis:
        print(f"   Markets analyzed: {crypto_analysis['parameters']['markets_analyzed']}")
        print(f"   Category summary: {crypto_analysis['category_summary']}")
    else:
        print(f"   Error: {crypto_analysis['error']}")

    print("\n2. Trade Flow Analysis (Sample Market)")
    # Get a sample market ID from the database
    db_session = get_db_session()
    sample_market = db_session.query(Market).filter(Market.category == "crypto").first()
    db_session.close()

    if sample_market:
        trade_analysis = analyzer.trade_flow_analysis(sample_market.id, hours_back=24)
        if "error" not in trade_analysis:
            print(f"   Market: {sample_market.title}")
            print(f"   Total trades: {trade_analysis['total_trades']}")
            avg_trade_size = trade_analysis['trade_characteristics'].get('avg_trade_size', 0)
            retail_pct = trade_analysis['trade_characteristics'].get('retail_trade_percentage', 0)
            print(f"   Avg trade size: ${avg_trade_size:.2f}")
            print(f"   Retail trade %: {retail_pct:.1%}")
        else:
            print(f"   Error: {trade_analysis['error']}")
    else:
        print("   No crypto markets found in database")

    print("\n3. Key Insights:")
    print("   - Focus on markets with retail scores > 7 for high retail activity")
    print("   - Look for weekend and evening trading patterns")
    print("   - Monitor trade size distributions for retail vs institutional shifts")
    print("   - Watch for volume spikes as indicators of retail FOMO/hype")

if __name__ == "__main__":
    main()
