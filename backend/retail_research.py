#!/usr/bin/env python3
"""
Prediction Market Retail Behavior Analysis System

This script demonstrates how to use the enhanced prediction market dashboard
to pull data from Polymarket APIs and analyze retail trading patterns.

Features:
- Polymarket data collection with enhanced retail metrics
- Comprehensive retail behavior analysis
- Market lifecycle tracking
- Social media correlation (placeholder for future Twitter integration)
"""

import logging
from datetime import datetime
from fetch_polymarket import PolymarketCollector
from retail_analyzer import RetailBehaviorAnalyzer
from database import init_db, get_db_session
from models import Market, MarketSnapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailResearchSystem:
    """Complete system for retail behavior research in prediction markets"""

    def __init__(self):
        self.collector = PolymarketCollector()
        self.analyzer = RetailBehaviorAnalyzer()
        init_db()  # Ensure database is ready

    def collect_market_data(self, limit: int = 10):
        """Step 1: Collect fresh market data from Polymarket"""
        logger.info("🔄 Step 1: Collecting market data from Polymarket...")

        # Get database session
        db_session = get_db_session()

        try:
            # Get active markets with retail focus
            markets = self.collector.get_active_markets(limit=limit)

            if not markets:
                logger.error("❌ No markets found")
                return []

            logger.info(f"📊 Found {len(markets)} active markets")

            # Save market data and snapshots
            collected_markets = []
            for market_data in markets:
                try:
                    market_id = self.collector.save_market(market_data, db_session)
                    if market_id:
                        self.collector.save_snapshot(market_id, market_data, db_session)
                        collected_markets.append(market_id)
                        logger.info(f"✅ Collected: {market_data.get('question', 'Unknown')[:50]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to save market: {e}")

            logger.info(f"🎯 Successfully collected {len(collected_markets)} markets")
            return collected_markets

        finally:
            db_session.close()

    def analyze_retail_behavior(self, market_ids: list = None, days_back: int = 7):
        """Step 2: Analyze retail behavior patterns"""
        logger.info("🔍 Step 2: Analyzing retail behavior patterns...")

        if market_ids is None:
            # Get all markets from database
            db = get_db_session()
            try:
                markets = db.query(Market).all()
                market_ids = [m.id for m in markets]
            finally:
                db.close()

        retail_insights = []

        for market_id in market_ids:
            try:
                analysis = self.analyzer.analyze_market_retail_behavior(market_id, days_back)

                if "error" not in analysis:
                    retail_insights.append({
                        'market_id': market_id,
                        'retail_score': analysis.get('overall_retail_score', {}),
                        'key_indicators': {
                            'trade_sizes': analysis.get('retail_indicators', {}).get('trade_size_indicators', []),
                            'volume_patterns': analysis.get('retail_indicators', {}).get('volume_indicators', []),
                            'temporal_patterns': analysis.get('behavioral_patterns', {}).get('retail_patterns', [])
                        },
                        'market_health': analysis.get('market_health', {}),
                        'lifecycle': analysis.get('behavioral_patterns', {}).get('lifecycle_description', '')
                    })

                    logger.info(f"📈 Analyzed {market_id[:8]}...: Retail intensity = {analysis.get('overall_retail_score', {}).get('retail_intensity', 'unknown')}")

            except Exception as e:
                logger.error(f"❌ Failed to analyze {market_id}: {e}")

        return retail_insights

    def generate_research_report(self, retail_insights: list):
        """Step 3: Generate research report on retail behavior"""
        logger.info("📋 Step 3: Generating retail research report...")

        if not retail_insights:
            return {"error": "No retail insights available"}

        # Categorize markets by retail intensity
        high_retail = []
        medium_retail = []
        low_retail = []

        for insight in retail_insights:
            intensity = insight['retail_score'].get('retail_intensity', 'unknown')

            if intensity == 'high' or intensity == 'very_high':
                high_retail.append(insight)
            elif intensity == 'moderate':
                medium_retail.append(insight)
            else:
                low_retail.append(insight)

        # Generate patterns summary
        all_patterns = []
        for insight in retail_insights:
            all_patterns.extend(insight['key_indicators']['trade_sizes'])
            all_patterns.extend(insight['key_indicators']['volume_patterns'])
            all_patterns.extend(insight['key_indicators']['temporal_patterns'])

        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_markets_analyzed': len(retail_insights),
            'retail_intensity_distribution': {
                'high_retail_markets': len(high_retail),
                'medium_retail_markets': len(medium_retail),
                'low_retail_markets': len(low_retail),
                'retail_dominated_percentage': (len(high_retail) / len(retail_insights) * 100) if retail_insights else 0
            },
            'top_retail_patterns': top_patterns,
            'high_retail_markets_sample': high_retail[:3],  # Top 3 examples
            'research_insights': self._generate_research_insights(high_retail, medium_retail, pattern_counts)
        }

        return report

    def _generate_research_insights(self, high_retail: list, medium_retail: list, pattern_counts: dict):
        """Generate actionable research insights"""
        insights = []

        # Retail dominance insights
        retail_percentage = (len(high_retail) / (len(high_retail) + len(medium_retail)) * 100) if (high_retail or medium_retail) else 0

        if retail_percentage > 50:
            insights.append("🔥 High retail dominance detected - prediction markets showing strong retail participation")
        elif retail_percentage > 25:
            insights.append("📊 Moderate retail activity - mixed institutional and retail participation")

        # Pattern insights
        if pattern_counts.get('small_trades', 0) > len(high_retail) * 0.5:
            insights.append("💰 Small trade sizes dominate - retail traders preferring smaller positions")

        if pattern_counts.get('strong_weekend_activity', 0) > len(high_retail) * 0.3:
            insights.append("🎯 Weekend trading surge - retail activity peaks during off-hours")

        if pattern_counts.get('frequent_spikes', 0) > len(high_retail) * 0.4:
            insights.append("📈 Volatile volume patterns - retail-driven hype cycles creating price volatility")

        if pattern_counts.get('late_night_volume', 0) > len(high_retail) * 0.2:
            insights.append("🌙 After-hours activity - retail traders active outside business hours")

        # Market lifecycle insights
        meme_markets = sum(1 for m in high_retail if 'meme' in str(m.get('lifecycle', '')).lower())
        if meme_markets > len(high_retail) * 0.3:
            insights.append("🚀 Meme market dynamics - explosive growth and rapid decline patterns observed")

        return insights if insights else ["📝 Limited retail patterns detected - may need more data for conclusive insights"]

    def run_full_research_cycle(self, market_limit: int = 10, analysis_days: int = 7):
        """Run complete research cycle: collect → analyze → report"""
        logger.info("🚀 Starting complete retail research cycle...")

        # Step 1: Collect data
        market_ids = self.collect_market_data(limit=market_limit)

        if not market_ids:
            return {"error": "No markets collected"}

        # Step 2: Analyze retail behavior
        retail_insights = self.analyze_retail_behavior(market_ids, analysis_days)

        # Step 3: Generate research report
        report = self.generate_research_report(retail_insights)

        logger.info("✅ Research cycle complete!")
        return {
            'status': 'success',
            'markets_collected': len(market_ids),
            'markets_analyzed': len(retail_insights),
            'research_report': report
        }

def main():
    """Main function to demonstrate the retail research system"""
    system = RetailResearchSystem()

    print("🎯 Prediction Market Retail Behavior Research System")
    print("=" * 60)

    # Run full research cycle
    result = system.run_full_research_cycle(market_limit=10, analysis_days=7)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    # Display results
    report = result['research_report']

    print(f"\n📊 Research Results ({report['total_markets_analyzed']} markets analyzed)")
    print("-" * 40)

    dist = report['retail_intensity_distribution']
    print(f"🏪 High Retail Markets: {dist['high_retail_markets']}")
    print(f"📊 Medium Retail Markets: {dist['medium_retail_markets']}")
    print(f"🏢 Low Retail Markets: {dist['low_retail_markets']}")
    print(f"Retail Market Percentage: {dist.get('retail_dominated_percentage', 0):.1f}%")
    print(f"\n🔥 Top Retail Patterns:")
    for pattern, count in report['top_retail_patterns'][:3]:
        print(f"   • {pattern.replace('_', ' ').title()}: {count} markets")

    print(f"\n💡 Research Insights:")
    for insight in report['research_insights']:
        print(f"   {insight}")

    print(f"\n✅ Research cycle completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    main()