#!/usr/bin/env python3
"""
Retail Behavior Analysis - Key Insights Generator
Analyzes Polymarket data for retail trading patterns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

def analyze_retail_behavior():
    """Analyze retail behavior from the database"""

    # Connect to database
    conn = sqlite3.connect('prediction_markets.db')
    conn.row_factory = sqlite3.Row

    print("🔍 Polymarket Retail Behavior Analysis")
    print("=" * 50)

    # 1. Overall Market Statistics
    print("\n1. MARKET OVERVIEW")
    markets_df = pd.read_sql_query("SELECT * FROM markets", conn)
    snapshots_df = pd.read_sql_query("SELECT * FROM market_snapshots", conn)
    trades_df = pd.read_sql_query("SELECT * FROM trades", conn)

    print(f"   Total Markets: {len(markets_df)}")
    print(f"   Total Snapshots: {len(snapshots_df)}")
    print(f"   Total Trades: {len(trades_df)}")

    if len(markets_df) > 0:
        category_counts = markets_df['category'].value_counts()
        print(f"   Categories: {dict(category_counts)}")

    # 2. Retail Trade Analysis
    print("\n2. RETAIL TRADE ANALYSIS")
    if len(trades_df) > 0:
        # Trade size distribution
        trade_sizes = trades_df['value']
        print(f"   Avg trade size: ${trade_sizes.mean():.2f}")
        print(f"   Median trade size: ${trade_sizes.median():.2f}")
        print(f"   Trade size std dev: ${trade_sizes.std():.2f}")

        # Size categories
        size_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        size_labels = ['Micro (<$10)', 'Small ($10-50)', 'Medium ($50-100)', 'Large ($100-500)', 'XL ($500-1000)', 'Whale (>$1000)']
        trades_df['size_category'] = pd.cut(trade_sizes, bins=size_bins, labels=size_labels, right=False)
        size_distribution = trades_df['size_category'].value_counts()
        print(f"   Trade Size Distribution:")
        for category, count in size_distribution.items():
            percentage = (count / len(trades_df)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")

        # Buy/Sell ratio
        buy_trades = len(trades_df[trades_df['side'] == 'buy'])
        sell_trades = len(trades_df[trades_df['side'] == 'sell'])
        buy_ratio = buy_trades / (buy_trades + sell_trades) if (buy_trades + sell_trades) > 0 else 0
        print(f"   Buy ratio: {buy_ratio:.1%} ({buy_trades} buy / {sell_trades} sell)")

    # 3. Market Snapshot Analysis
    print("\n3. MARKET ACTIVITY ANALYSIS")
    if len(snapshots_df) > 0:
        # Average metrics
        avg_volume = snapshots_df['volume_24h'].mean()
        avg_trade_size = snapshots_df['avg_trade_size'].mean()
        avg_liquidity = snapshots_df['liquidity'].mean()

        print(f"   Avg 24h volume: ${avg_volume:,.2f}")
        print(f"   Avg trade size: ${avg_trade_size:,.2f}")
        print(f"   Avg liquidity: ${avg_liquidity:,.2f}")

        # Weekend vs Weekday activity
        weekend_data = snapshots_df[snapshots_df['is_weekend'] == 1]
        weekday_data = snapshots_df[snapshots_df['is_weekend'] == 0]

        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_avg_volume = weekend_data['volume_24h'].mean()
            weekday_avg_volume = weekday_data['volume_24h'].mean()
            weekend_ratio = weekend_avg_volume / weekday_avg_volume if weekday_avg_volume > 0 else 0
            print(f"   Weekend/Weekday volume ratio: {weekend_ratio:.2f}")

        # Evening activity
        evening_data = snapshots_df[snapshots_df['is_evening'] == 1]
        day_data = snapshots_df[snapshots_df['is_evening'] == 0]

        if len(evening_data) > 0 and len(day_data) > 0:
            evening_avg_volume = evening_data['volume_24h'].mean()
            day_avg_volume = day_data['volume_24h'].mean()
            evening_ratio = evening_avg_volume / day_avg_volume if day_avg_volume > 0 else 0
            print(f"   Evening/Day volume ratio: {evening_ratio:.2f}")

    # 4. Top Retail Markets
    print("\n4. TOP RETAIL MARKETS")
    if len(snapshots_df) > 0:
        # Get latest snapshot for each market
        latest_snapshots = snapshots_df.sort_values('timestamp').groupby('market_id').last().reset_index()

        # Filter for retail characteristics (small trade sizes)
        retail_markets = latest_snapshots[latest_snapshots['avg_trade_size'] < 500].copy()

        if len(retail_markets) > 0:
            # Add market titles
            retail_markets = retail_markets.merge(markets_df[['id', 'title', 'category']],
                                                 left_on='market_id', right_on='id', how='left')

            # Sort by volume
            retail_markets = retail_markets.sort_values('volume_24h', ascending=False)

            print("   Top 5 Retail Markets by Volume:")
            for i, (_, market) in enumerate(retail_markets.head(5).iterrows(), 1):
                print(f"   {i}. {market['title'][:50]}... (${market['volume_24h']:.0f}) - {market['category']}")
        else:
            print("   No markets with clear retail characteristics found")

    # 5. Category Analysis
    print("\n5. CATEGORY ANALYSIS")
    if len(markets_df) > 0 and len(snapshots_df) > 0:
        # Merge market and snapshot data
        market_stats = snapshots_df.merge(markets_df[['id', 'category']],
                                        left_on='market_id', right_on='id', how='left')

        category_stats = market_stats.groupby('category').agg({
            'volume_24h': 'mean',
            'avg_trade_size': 'mean',
            'liquidity': 'mean',
            'market_id': 'count'
        }).round(2)

        print("   Average metrics by category:")
        for category, stats in category_stats.iterrows():
            retail_score = "High" if stats['avg_trade_size'] < 500 else "Medium" if stats['avg_trade_size'] < 1000 else "Low"
            print(f"   {category.title()}: Volume=${stats['volume_24h']:.0f}, Avg Trade=${stats['avg_trade_size']:.0f}, Retail={retail_score}")

    # 6. Key Insights
    print("\n6. KEY RETAIL BEHAVIOR INSIGHTS")
    print("   • Small trade sizes (<$500) indicate high retail participation")
    print("   • Weekend trading activity suggests retail traders")
    print("   • Evening/night trading patterns are retail characteristics")
    print("   • High volume volatility indicates retail FOMO/hype cycles")
    print("   • Markets with frequent small trades are retail-dominated")
    print("   • Look for price movements driven by small trade clusters")

    # 7. Recommendations
    print("\n7. RECOMMENDATIONS FOR ANALYSIS")
    print("   • Focus on markets with avg_trade_size < $200 for pure retail analysis")
    print("   • Monitor weekend activity spikes for retail sentiment")
    print("   • Watch for trade clustering (multiple trades in short timeframes)")
    print("   • Analyze buy/sell ratios by trade size categories")
    print("   • Track volume patterns during market hours vs after-hours")

    conn.close()

if __name__ == "__main__":
    analyze_retail_behavior()
