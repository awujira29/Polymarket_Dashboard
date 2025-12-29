
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from models import MarketSnapshot
from database import get_db_session
import logging

# ML Libraries for advanced analysis
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import silhouette_score, classification_report
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings('ignore')
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Using basic statistical analysis only.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailBehaviorAnalyzer:
    """Analyzes Polymarket data to identify retail trading patterns"""

    def __init__(self):
        pass

    def analyze_market_retail_behavior(self, market_id: str, days_back: int = 7) -> Dict:
        """Comprehensive retail behavior analysis for a Polymarket"""
        try:
            db_session = get_db_session()

            # Get market snapshots
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            snapshots = db_session.query(MarketSnapshot).filter(
                MarketSnapshot.market_id == market_id,
                MarketSnapshot.timestamp >= cutoff_date
            ).order_by(MarketSnapshot.timestamp).all()

            if not snapshots:
                return {"error": "No data available for analysis"}

            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'timestamp': s.timestamp,
                'price': s.price,
                'volume_24h': s.volume_24h,
                'volume_num_trades': s.volume_num_trades,
                'liquidity': s.liquidity,
                'avg_trade_size': s.avg_trade_size,
                'num_traders': s.num_traders,
                'large_trade_count': s.large_trade_count,
                'is_weekend': s.is_weekend,
                'is_evening': s.is_evening,
                'hour_of_day': s.hour_of_day
            } for s in snapshots])

            analysis = {
                'market_id': market_id,
                'time_period_days': days_back,
                'total_snapshots': len(df),
                'retail_indicators': {},
                'behavioral_patterns': {},
                'market_health': {}
            }

            # 1. Trade Size Analysis (Primary Retail Indicator)
            analysis['retail_indicators'].update(self._analyze_trade_sizes(df))

            # 2. Volume Pattern Analysis
            analysis['retail_indicators'].update(self._analyze_volume_patterns(df))

            # 3. Temporal Pattern Analysis
            analysis['behavioral_patterns'].update(self._analyze_temporal_patterns(df))

            # 4. Market Lifecycle Analysis
            analysis['behavioral_patterns'].update(self._analyze_market_lifecycle(df))

            # 5. Market Health Metrics
            analysis['market_health'].update(self._analyze_market_health(df))

            # 6. ML-Enhanced Analysis (if available)
            if ML_AVAILABLE and len(df) >= 5:  # Need minimum data for ML
                analysis['ml_analysis'] = {}
                analysis['ml_analysis'].update(self._ml_trader_clustering(df))
                analysis['ml_analysis'].update(self._ml_anomaly_detection(df))
                analysis['ml_analysis'].update(self._ml_behavioral_patterns(df))
                analysis['ml_analysis'].update(self._ml_predictive_insights(df))

            # Calculate overall retail score
            analysis['overall_retail_score'] = self._calculate_overall_retail_score(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing retail behavior: {e}")
            return {"error": str(e)}
        finally:
            db_session.close()

    def _analyze_trade_sizes(self, df: pd.DataFrame) -> Dict:
        """Analyze trade sizes to identify retail vs institutional activity"""
        if df['avg_trade_size'].isna().all() or (df['avg_trade_size'] == 0).all():
            return {'trade_size_analysis': 'insufficient_data'}

        avg_trade_size = df['avg_trade_size'].mean()
        trade_size_std = df['avg_trade_size'].std()
        median_trade_size = df['avg_trade_size'].median()

        # Retail indicators: smaller average trade sizes, higher variability
        retail_score = 0
        indicators = []

        # Small average trade size suggests retail dominance
        if avg_trade_size < 200:
            retail_score += 4
            indicators.append("very_small_trades")
        elif avg_trade_size < 500:
            retail_score += 3
            indicators.append("small_trades")
        elif avg_trade_size < 1000:
            retail_score += 2
            indicators.append("medium_trades")
        elif avg_trade_size < 2000:
            retail_score += 1
            indicators.append("larger_trades")

        # High variability in trade sizes suggests retail (vs institutional consistency)
        if avg_trade_size > 0:
            cv = trade_size_std / avg_trade_size
            if cv > 1.5:
                retail_score += 2
                indicators.append("high_volatility")
            elif cv > 1.0:
                retail_score += 1
                indicators.append("moderate_volatility")

        # Compare mean vs median (retail trades often have long tail of small trades)
        if median_trade_size < avg_trade_size * 0.7:
            retail_score += 1
            indicators.append("small_trade_tail")

        # Large trade frequency
        avg_large_trades = df['large_trade_count'].mean()
        if avg_large_trades < 1:
            retail_score += 1
            indicators.append("few_large_trades")

        return {
            'avg_trade_size': float(avg_trade_size),
            'median_trade_size': float(median_trade_size),
            'trade_size_volatility': float(cv) if avg_trade_size > 0 else 0,
            'avg_large_trades_per_day': float(avg_large_trades),
            'retail_trade_score': retail_score,
            'trade_size_indicators': indicators,
            'retail_trade_description': self._describe_trade_size_profile(retail_score, indicators)
        }

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns for retail behavior indicators"""
        if len(df) < 3:
            return {'volume_analysis': 'insufficient_data'}

        volume_changes = df['volume_24h'].pct_change()

        # Volume spikes (sudden jumps typical of retail FOMO/hype)
        spike_threshold = volume_changes.quantile(0.90) if not volume_changes.empty else 2.0
        volume_spikes = volume_changes[volume_changes > max(spike_threshold, 2.0)]

        # Volume consistency (retail markets often have spiky, inconsistent volume)
        volume_std = df['volume_24h'].std()
        volume_mean = df['volume_24h'].mean()
        volume_cv = volume_std / volume_mean if volume_mean > 0 else 0

        # Weekend vs weekday activity
        weekend_data = df[df['is_weekend'] == True]
        weekday_data = df[df['is_weekend'] == False]

        weekend_avg_volume = weekend_data['volume_24h'].mean() if not weekend_data.empty else 0
        weekday_avg_volume = weekday_data['volume_24h'].mean() if not weekday_data.empty else 0

        # Evening activity (after hours trading - retail characteristic)
        evening_data = df[df['is_evening'] == True]
        day_data = df[df['is_evening'] == False]

        evening_avg_volume = evening_data['volume_24h'].mean() if not evening_data.empty else 0
        day_avg_volume = day_data['volume_24h'].mean() if not day_data.empty else 0

        retail_score = 0
        indicators = []

        # Volume spikes suggest retail hype/fomo
        if len(volume_spikes) > len(df) * 0.2:  # More than 20% of periods have spikes
            retail_score += 3
            indicators.append("frequent_spikes")
        elif len(volume_spikes) > 2:
            retail_score += 2
            indicators.append("occasional_spikes")

        # High volume variability suggests retail (vs stable institutional)
        if volume_cv > 1.0:
            retail_score += 2
            indicators.append("high_volume_volatility")
        elif volume_cv > 0.7:
            retail_score += 1
            indicators.append("moderate_volume_volatility")

        # Higher weekend activity suggests retail
        if weekend_avg_volume > weekday_avg_volume * 1.5:
            retail_score += 3
            indicators.append("strong_weekend_activity")
        elif weekend_avg_volume > weekday_avg_volume * 1.2:
            retail_score += 2
            indicators.append("moderate_weekend_activity")

        # Higher evening activity suggests retail
        if evening_avg_volume > day_avg_volume * 1.5:
            retail_score += 2
            indicators.append("strong_evening_activity")
        elif evening_avg_volume > day_avg_volume * 1.2:
            retail_score += 1
            indicators.append("moderate_evening_activity")

        return {
            'volume_spikes_count': len(volume_spikes),
            'volume_volatility': float(volume_cv),
            'weekend_vs_weekday_ratio': float(weekend_avg_volume / weekday_avg_volume) if weekday_avg_volume > 0 else 0,
            'evening_vs_day_ratio': float(evening_avg_volume / day_avg_volume) if day_avg_volume > 0 else 0,
            'retail_volume_score': retail_score,
            'volume_indicators': indicators,
            'retail_volume_description': self._describe_volume_patterns(retail_score, indicators)
        }

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in trading activity"""
        # Hourly patterns
        hourly_volume = df.groupby('hour_of_day')['volume_24h'].mean()
        hourly_trades = df.groupby('hour_of_day')['volume_num_trades'].mean()

        # Peak hours (when retail traders are most active)
        peak_volume_hours = hourly_volume.nlargest(3).index.tolist()
        peak_trade_hours = hourly_trades.nlargest(3).index.tolist()

        # Check for typical retail patterns
        retail_patterns = []
        unusual_hours = []

        # Late night/early morning activity (retail traders)
        late_night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        if any(hour in late_night_hours for hour in peak_volume_hours):
            retail_patterns.append("late_night_volume")
            unusual_hours.extend([h for h in peak_volume_hours if h in late_night_hours])

        # Evening activity
        evening_hours = [18, 19, 20, 21]
        if any(hour in evening_hours for hour in peak_volume_hours):
            retail_patterns.append("evening_volume")

        # Weekend concentration
        weekend_ratio = len(df[df['is_weekend'] == True]) / len(df) if len(df) > 0 else 0
        if weekend_ratio > 0.4:
            retail_patterns.append("weekend_dominant")

        # Trading frequency patterns
        avg_trades_per_hour = df['volume_num_trades'].mean()
        if avg_trades_per_hour < 5:
            retail_patterns.append("low_frequency_trading")

        return {
            'peak_volume_hours': peak_volume_hours,
            'peak_trading_hours': peak_trade_hours,
            'unusual_trading_hours': unusual_hours,
            'weekend_concentration': float(weekend_ratio),
            'avg_trades_per_period': float(avg_trades_per_hour),
            'retail_patterns': retail_patterns,
            'temporal_description': self._describe_temporal_patterns(retail_patterns, unusual_hours)
        }

    def _analyze_market_lifecycle(self, df: pd.DataFrame) -> Dict:
        """Analyze market lifecycle patterns (meme market behavior)"""
        if len(df) < 5:
            return {'lifecycle_stage': 'insufficient_data'}

        # Calculate volume trend over time
        df_sorted = df.sort_values('timestamp')
        volume_trend = np.polyfit(range(len(df_sorted)), df_sorted['volume_24h'], 1)[0]

        # Volume growth rate
        volume_growth = (df_sorted['volume_24h'].iloc[-1] / df_sorted['volume_24h'].iloc[0]) - 1 if df_sorted['volume_24h'].iloc[0] > 0 else 0

        # Volatility metrics
        volume_volatility = df['volume_24h'].std() / df['volume_24h'].mean() if df['volume_24h'].mean() > 0 else 0
        price_volatility = df['price'].std()

        # Classify lifecycle stage
        lifecycle_indicators = []
        meme_score = 0

        # Explosive growth (meme market takeoff)
        if volume_growth > 5 and volume_volatility > 2.0:
            lifecycle_indicators.append("explosive_growth")
            meme_score += 3
        elif volume_growth > 2:
            lifecycle_indicators.append("rapid_growth")
            meme_score += 2

        # High volatility (characteristic of retail-driven markets)
        if volume_volatility > 1.5:
            lifecycle_indicators.append("extreme_volatility")
            meme_score += 2
        elif volume_volatility > 1.0:
            lifecycle_indicators.append("high_volatility")
            meme_score += 1

        # Price volatility (retail markets often have wild price swings)
        if price_volatility > 0.3:
            lifecycle_indicators.append("price_swing_volatility")
            meme_score += 1

        # Declining phase
        if volume_trend < -0.3:
            lifecycle_indicators.append("volume_declining")
        elif volume_trend < -0.1:
            lifecycle_indicators.append("volume_slowing")

        # Meme market characteristics
        is_meme_like = meme_score >= 3

        return {
            'volume_trend': float(volume_trend),
            'volume_growth_rate': float(volume_growth),
            'volume_volatility': float(volume_volatility),
            'price_volatility': float(price_volatility),
            'lifecycle_indicators': lifecycle_indicators,
            'meme_market_score': float(meme_score / 6.0),  # Normalize to 0-1
            'is_meme_like': is_meme_like,
            'lifecycle_description': self._describe_lifecycle(lifecycle_indicators, is_meme_like)
        }

    def _analyze_market_health(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market health and liquidity"""
        avg_volume = df['volume_24h'].mean()
        avg_liquidity = df['liquidity'].mean()
        avg_trades = df['volume_num_trades'].mean()

        # Liquidity ratio (higher is better for retail trading)
        liquidity_ratio = avg_liquidity / avg_volume if avg_volume > 0 else 0

        # Trading activity consistency
        volume_consistency = 1 - (df['volume_24h'].std() / df['volume_24h'].mean()) if df['volume_24h'].mean() > 0 else 0

        # Market efficiency (trades per dollar volume - higher suggests retail)
        trades_per_dollar = avg_trades / avg_volume if avg_volume > 0 else 0

        health_score = 0
        health_indicators = []

        # Good liquidity for retail
        if liquidity_ratio > 0.5:
            health_score += 2
            health_indicators.append("good_liquidity")
        elif liquidity_ratio > 0.2:
            health_score += 1
            health_indicators.append("adequate_liquidity")

        # Consistent volume (healthy market)
        if volume_consistency > 0.6:
            health_score += 1
            health_indicators.append("consistent_volume")

        # Active trading
        if avg_trades > 10:
            health_score += 1
            health_indicators.append("active_trading")

        return {
            'avg_daily_volume': float(avg_volume),
            'avg_liquidity': float(avg_liquidity),
            'avg_daily_trades': float(avg_trades),
            'liquidity_ratio': float(liquidity_ratio),
            'volume_consistency': float(volume_consistency),
            'trades_per_dollar': float(trades_per_dollar),
            'market_health_score': health_score,
            'health_indicators': health_indicators,
            'health_description': self._describe_market_health(health_score, health_indicators)
        }

    def _calculate_overall_retail_score(self, analysis: Dict) -> Dict:
        """Calculate overall retail score from all indicators"""
        trade_score = analysis['retail_indicators'].get('retail_trade_score', 0)
        volume_score = analysis['retail_indicators'].get('retail_volume_score', 0)
        health_score = analysis['market_health'].get('market_health_score', 0)

        # Weighted average (trade size is most important indicator)
        overall_score = (trade_score * 0.5) + (volume_score * 0.3) + (health_score * 0.2)

        # Classify retail intensity
        if overall_score >= 7:
            intensity = "very_high"
            description = "Strong retail dominance - small trades, volatile patterns, weekend activity"
        elif overall_score >= 5:
            intensity = "high"
            description = "Moderate to high retail activity - mixed trade sizes with retail patterns"
        elif overall_score >= 3:
            intensity = "moderate"
            description = "Balanced retail/institutional mix - some retail characteristics"
        elif overall_score >= 1:
            intensity = "low"
            description = "Limited retail activity - mostly institutional trading patterns"
        else:
            intensity = "very_low"
            description = "Minimal retail activity - primarily institutional market"

        return {
            'overall_score': float(overall_score),
            'retail_intensity': intensity,
            'description': description,
            'component_scores': {
                'trade_size': trade_score,
                'volume_patterns': volume_score,
                'market_health': health_score
            }
        }

    def _describe_trade_size_profile(self, score: int, indicators: List[str]) -> str:
        """Describe trade size profile in human terms"""
        if score >= 4:
            base = "Very small average trade sizes"
        elif score >= 2:
            base = "Small to medium trade sizes"
        else:
            base = "Medium to large trade sizes"

        details = []
        if "high_volatility" in indicators:
            details.append("highly variable")
        if "small_trade_tail" in indicators:
            details.append("many small trades")
        if "few_large_trades" in indicators:
            details.append("few large trades")

        detail_str = f" ({', '.join(details)})" if details else ""
        return f"{base}{detail_str}"

    def _describe_volume_patterns(self, score: int, indicators: List[str]) -> str:
        """Describe volume patterns in human terms"""
        patterns = []
        if "frequent_spikes" in indicators:
            patterns.append("frequent volume spikes")
        if "high_volume_volatility" in indicators:
            patterns.append("highly volatile volume")
        if "strong_weekend_activity" in indicators:
            patterns.append("strong weekend trading")
        if "strong_evening_activity" in indicators:
            patterns.append("active evening trading")

        if not patterns:
            return "Consistent, stable volume patterns"

        return f"{' and '.join(patterns)}"

    def _describe_temporal_patterns(self, patterns: List[str], unusual_hours: List[int]) -> str:
        """Describe temporal patterns"""
        descriptions = []
        if "late_night_volume" in patterns:
            descriptions.append(f"late-night activity (hours: {unusual_hours})")
        if "evening_volume" in patterns:
            descriptions.append("evening trading focus")
        if "weekend_dominant" in patterns:
            descriptions.append("weekend concentration")
        if "low_frequency_trading" in patterns:
            descriptions.append("low-frequency trading")

        return "; ".join(descriptions) if descriptions else "Standard business hours pattern"

    def _describe_lifecycle(self, indicators: List[str], is_meme_like: bool) -> str:
        """Describe market lifecycle stage"""
        if is_meme_like:
            if "explosive_growth" in indicators:
                return "Meme market in hype phase - explosive growth and extreme volatility"
            elif "extreme_volatility" in indicators:
                return "Meme market in active phase - high volatility, retail-driven"
            else:
                return "Meme market in mature phase - established but still volatile"

        if "explosive_growth" in indicators:
            return "Rapid growth phase - gaining significant traction"
        elif "rapid_growth" in indicators:
            return "Growth phase - steadily increasing activity"
        elif "volume_declining" in indicators:
            return "Decline phase - losing momentum"
        else:
            return "Stable phase - consistent activity levels"

    def _describe_market_health(self, score: int, indicators: List[str]) -> str:
        """Describe market health"""
        if score >= 4:
            health = "Very healthy market"
        elif score >= 2:
            health = "Healthy market"
        else:
            health = "Developing market"

        details = []
        if "good_liquidity" in indicators:
            details.append("good liquidity")
        if "consistent_volume" in indicators:
            details.append("consistent volume")
        if "active_trading" in indicators:
            details.append("active trading")

        detail_str = f" with {', '.join(details)}" if details else ""
        return f"{health}{detail_str}"

    # ===== MACHINE LEARNING ANALYSIS METHODS =====

    def _ml_trader_clustering(self, df: pd.DataFrame) -> Dict:
        """Use ML clustering to identify different trader behavior patterns"""
        try:
            # For retail analysis, focus on key behavioral features
            features = ['avg_trade_size', 'num_traders', 'hour_of_day']
            X = df[features].fillna(df[features].mean())

            # Add derived features for better clustering
            X = X.copy()
            X['trade_size_normalized'] = (X['avg_trade_size'] - X['avg_trade_size'].mean()) / X['avg_trade_size'].std()
            X['trader_density'] = X['num_traders'] / (X['avg_trade_size'] + 1)  # Traders per dollar
            X['hour_category'] = pd.cut(X['hour_of_day'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])

            # Use DBSCAN for density-based clustering (better for retail patterns)
            features_for_clustering = ['trade_size_normalized', 'trader_density', 'hour_category']
            X_cluster = X[features_for_clustering].fillna(0)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # DBSCAN clustering - better for retail behavior patterns
            dbscan = DBSCAN(eps=0.8, min_samples=2)
            clusters = dbscan.fit_predict(X_scaled)

            # If DBSCAN finds only noise, fall back to simple K-means with 2 clusters
            if len(set(clusters)) <= 1:  # Only noise or single cluster
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

            # Analyze cluster characteristics
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = clusters

            unique_clusters = sorted(set(clusters))
            cluster_analysis = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # DBSCAN noise points
                    continue

                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'avg_trade_size': cluster_data['avg_trade_size'].mean(),
                    'avg_traders': cluster_data['num_traders'].mean(),
                    'avg_hour': cluster_data['hour_of_day'].mean(),
                    'is_retail_like': cluster_data['avg_trade_size'].mean() < df['avg_trade_size'].quantile(0.5),
                    'temporal_pattern': 'evening' if cluster_data['hour_of_day'].mean() > 18 else
                                     'morning' if cluster_data['hour_of_day'].mean() < 12 else 'afternoon'
                }

            # Identify most retail-like cluster
            retail_clusters = [k for k, v in cluster_analysis.items() if v['is_retail_like']]
            retail_cluster = retail_clusters[0] if retail_clusters else list(cluster_analysis.keys())[0]

            # Calculate clustering quality
            if len(unique_clusters) > 1:
                try:
                    silhouette = silhouette_score(X_scaled, clusters)
                except:
                    silhouette = None
            else:
                silhouette = None

            return {
                'trader_clusters': cluster_analysis,
                'retail_cluster': retail_cluster,
                'total_clusters': len([c for c in unique_clusters if c != -1]),
                'clustering_method': 'dbscan_with_fallback',
                'silhouette_score': silhouette,
                'cluster_features': features_for_clustering
            }

        except Exception as e:
            logger.warning(f"ML clustering failed: {e}")
            return {'clustering_error': str(e)}

    def _ml_anomaly_detection(self, df: pd.DataFrame) -> Dict:
        """Detect anomalous trading behavior using multiple ML methods"""
        try:
            # Prepare features for anomaly detection
            features = ['volume_24h', 'avg_trade_size', 'num_traders', 'price']
            X = df[features].fillna(df[features].mean())

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Method 1: Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_scores = iso_forest.fit_predict(X_scaled)
            iso_anomalies = (iso_scores == -1)

            # Method 2: Statistical outlier detection (Z-score)
            z_scores = np.abs((X - X.mean()) / X.std())
            statistical_anomalies = (z_scores > 3).any(axis=1)

            # Combine anomaly detection methods
            combined_anomalies = iso_anomalies | statistical_anomalies

            df_anomalies = df.copy()
            df_anomalies['is_anomaly'] = combined_anomalies
            df_anomalies['anomaly_score_iso'] = iso_forest.score_samples(X_scaled)
            df_anomalies['anomaly_score_stat'] = z_scores.max(axis=1)

            anomalies = df_anomalies[combined_anomalies]

            anomaly_patterns = []
            anomaly_characteristics = {}

            if len(anomalies) > 0:
                # Volume-based anomalies
                if anomalies['volume_24h'].max() > df['volume_24h'].quantile(0.95):
                    anomaly_patterns.append('extreme_volume_spikes')
                    anomaly_characteristics['volume_anomaly'] = {
                        'max_volume': anomalies['volume_24h'].max(),
                        'normal_max': df['volume_24h'].quantile(0.95)
                    }

                # Trade size anomalies
                if anomalies['avg_trade_size'].max() > df['avg_trade_size'].quantile(0.95):
                    anomaly_patterns.append('unusual_trade_sizes')
                    anomaly_characteristics['trade_size_anomaly'] = {
                        'max_trade_size': anomalies['avg_trade_size'].max(),
                        'normal_max': df['avg_trade_size'].quantile(0.95)
                    }

                # Temporal anomalies
                if anomalies['is_evening'].mean() > 0.8:
                    anomaly_patterns.append('evening_concentration')
                elif anomalies['is_weekend'].mean() > 0.8:
                    anomaly_patterns.append('weekend_concentration')

                # Price movement anomalies
                price_volatility = df['price'].std() / df['price'].mean()
                if price_volatility > df['price'].std() / df['price'].mean() * 2:
                    anomaly_patterns.append('high_price_volatility')

            # Create histogram data for trade sizes
            trade_size_hist, bin_edges = np.histogram(df['avg_trade_size'], bins=10)
            anomaly_trade_hist, _ = np.histogram(anomalies['avg_trade_size'], bins=bin_edges)

            return {
                'anomaly_count': int(combined_anomalies.sum()),
                'anomaly_percentage': float(combined_anomalies.sum() / len(df) * 100),
                'anomaly_patterns': anomaly_patterns,
                'anomaly_characteristics': anomaly_characteristics,
                'detection_methods': ['isolation_forest', 'statistical_zscore'],
                'most_anomalous_period': anomalies['timestamp'].iloc[0].isoformat() if len(anomalies) > 0 else None,
                'trade_size_histogram': {
                    'bins': bin_edges.tolist(),
                    'normal_counts': trade_size_hist.tolist(),
                    'anomaly_counts': anomaly_trade_hist.tolist()
                },
                'anomaly_summary': {
                    'total_points': len(df),
                    'anomalous_points': int(combined_anomalies.sum()),
                    'contamination_rate': 0.1
                }
            }

        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
            return {'anomaly_detection_error': str(e)}

    def _ml_behavioral_patterns(self, df: pd.DataFrame) -> Dict:
        """Use ML to identify complex behavioral patterns with statistical visualizations"""
        try:
            # Create comprehensive behavioral features
            df_patterns = df.copy()

            # Rolling statistics for trend analysis
            df_patterns['volume_ma_3'] = df_patterns['volume_24h'].rolling(3, min_periods=1).mean()
            df_patterns['trade_size_ma_3'] = df_patterns['avg_trade_size'].rolling(3, min_periods=1).mean()
            df_patterns['volume_volatility'] = df_patterns['volume_24h'].rolling(3, min_periods=1).std()
            df_patterns['price_change'] = df_patterns['price'].pct_change().fillna(0)

            # Time-based features
            df_patterns['hour_sin'] = np.sin(2 * np.pi * df_patterns['hour_of_day'] / 24)
            df_patterns['hour_cos'] = np.cos(2 * np.pi * df_patterns['hour_of_day'] / 24)

            # Interaction features
            df_patterns['volume_times_traders'] = df_patterns['volume_24h'] * df_patterns['num_traders']
            df_patterns['liquidity_ratio'] = df_patterns['volume_24h'] / (df_patterns['liquidity'] + 1)
            df_patterns['trader_efficiency'] = df_patterns['num_traders'] / (df_patterns['avg_trade_size'] + 1)

            # Fill NaN values
            df_patterns = df_patterns.fillna(method='bfill').fillna(method='ffill').fillna(0)

            # Statistical distributions for visualization
            volume_dist = {
                'mean': float(df_patterns['volume_24h'].mean()),
                'std': float(df_patterns['volume_24h'].std()),
                'skewness': float(df_patterns['volume_24h'].skew()),
                'kurtosis': float(df_patterns['volume_24h'].kurtosis()),
                'quartiles': {
                    'q1': float(df_patterns['volume_24h'].quantile(0.25)),
                    'median': float(df_patterns['volume_24h'].quantile(0.5)),
                    'q3': float(df_patterns['volume_24h'].quantile(0.75))
                }
            }

            trade_size_dist = {
                'mean': float(df_patterns['avg_trade_size'].mean()),
                'std': float(df_patterns['avg_trade_size'].std()),
                'skewness': float(df_patterns['avg_trade_size'].skew()),
                'kurtosis': float(df_patterns['avg_trade_size'].kurtosis()),
                'quartiles': {
                    'q1': float(df_patterns['avg_trade_size'].quantile(0.25)),
                    'median': float(df_patterns['avg_trade_size'].quantile(0.5)),
                    'q3': float(df_patterns['avg_trade_size'].quantile(0.75))
                }
            }

            # Create histograms for key metrics
            volume_hist, volume_bins = np.histogram(df_patterns['volume_24h'], bins=15)
            trade_hist, trade_bins = np.histogram(df_patterns['avg_trade_size'], bins=15)
            hour_hist, hour_bins = np.histogram(df_patterns['hour_of_day'], bins=24)

            # PCA for dimensionality reduction and pattern identification
            features = ['volume_24h', 'avg_trade_size', 'num_traders', 'liquidity',
                       'volume_ma_3', 'trade_size_ma_3', 'price_change',
                       'hour_sin', 'hour_cos', 'volume_times_traders', 'liquidity_ratio']

            X = df_patterns[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=min(5, len(features), len(X_scaled)))
            X_pca = pca.fit_transform(X_scaled)

            # Analyze principal components
            explained_variance = pca.explained_variance_ratio_
            component_patterns = {}

            for i, component in enumerate(pca.components_):
                top_features_idx = np.argsort(np.abs(component))[-3:]  # Top 3 features
                top_features = [features[j] for j in top_features_idx]
                component_patterns[f'pc_{i+1}'] = {
                    'explained_variance': float(explained_variance[i]),
                    'top_features': top_features,
                    'weights': component[top_features_idx].tolist(),
                    'variance_explained_pct': float(explained_variance[i] * 100)
                }

            # Identify dominant patterns
            dominant_patterns = []
            if explained_variance[0] > 0.3:  # First PC explains >30% variance
                top_feature = features[np.argmax(np.abs(pca.components_[0]))]
                if 'volume' in top_feature:
                    dominant_patterns.append('volume_driven')
                elif 'trade_size' in top_feature:
                    dominant_patterns.append('trade_size_driven')
                elif 'hour' in top_feature.lower():
                    dominant_patterns.append('time_driven')
                elif 'trader' in top_feature.lower():
                    dominant_patterns.append('trader_count_driven')

            # Correlation analysis
            correlation_matrix = df_patterns[features].corr()
            high_correlations = []
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:  # Strong correlation
                        high_correlations.append({
                            'feature1': features[i],
                            'feature2': features[j],
                            'correlation': float(corr),
                            'strength': 'strong_positive' if corr > 0 else 'strong_negative'
                        })

            # Time series patterns
            time_patterns = {
                'weekend_vs_weekday': {
                    'weekend_avg_volume': float(df_patterns[df_patterns['is_weekend'] == 1]['volume_24h'].mean()),
                    'weekday_avg_volume': float(df_patterns[df_patterns['is_weekend'] == 0]['volume_24h'].mean()),
                    'weekend_premium': float(df_patterns[df_patterns['is_weekend'] == 1]['volume_24h'].mean() /
                                           df_patterns[df_patterns['is_weekend'] == 0]['volume_24h'].mean() - 1) * 100
                },
                'evening_vs_day': {
                    'evening_avg_volume': float(df_patterns[df_patterns['is_evening'] == 1]['volume_24h'].mean()),
                    'day_avg_volume': float(df_patterns[df_patterns['is_evening'] == 0]['volume_24h'].mean()),
                    'evening_premium': float(df_patterns[df_patterns['is_evening'] == 1]['volume_24h'].mean() /
                                           df_patterns[df_patterns['is_evening'] == 0]['volume_24h'].mean() - 1) * 100
                }
            }

            return {
                'principal_components': component_patterns,
                'dominant_patterns': dominant_patterns,
                'total_explained_variance': float(explained_variance.sum()),
                'behavioral_complexity': len([p for p in explained_variance if p > 0.1]),
                'statistical_distributions': {
                    'volume_distribution': volume_dist,
                    'trade_size_distribution': trade_size_dist
                },
                'histograms': {
                    'volume_histogram': {
                        'bins': volume_bins.tolist(),
                        'counts': volume_hist.tolist()
                    },
                    'trade_size_histogram': {
                        'bins': trade_bins.tolist(),
                        'counts': trade_hist.tolist()
                    },
                    'hourly_histogram': {
                        'bins': hour_bins.tolist(),
                        'counts': hour_hist.tolist()
                    }
                },
                'correlation_analysis': {
                    'high_correlations': high_correlations,
                    'correlation_matrix_shape': correlation_matrix.shape
                },
                'temporal_patterns': time_patterns,
                'pattern_summary': {
                    'total_observations': len(df_patterns),
                    'features_analyzed': len(features),
                    'significant_patterns_found': len(dominant_patterns) + len(high_correlations)
                }
            }

        except Exception as e:
            logger.warning(f"ML behavioral patterns failed: {e}")
            return {'behavioral_patterns_error': str(e)}

    def _ml_predictive_insights(self, df: pd.DataFrame) -> Dict:
        """Predictive modeling for retail behavior forecasting"""
        try:
            if len(df) < 5:
                return {'insufficient_data': 'Need at least 5 data points for prediction'}

            # Prepare time series features
            df_pred = df.copy().sort_values('timestamp')

            # Create lag features
            for lag in [1, 2, 3]:
                df_pred[f'volume_lag_{lag}'] = df_pred['volume_24h'].shift(lag)
                df_pred[f'trade_size_lag_{lag}'] = df_pred['avg_trade_size'].shift(lag)

            # Target: predict if next period will be high retail activity
            df_pred['high_retail_next'] = (df_pred['avg_trade_size'].shift(-1) < df_pred['avg_trade_size'].quantile(0.5)).astype(int)

            # Drop NaN rows
            df_pred = df_pred.dropna()

            if len(df_pred) < 5:
                return {'insufficient_clean_data': 'Not enough clean data for prediction'}

            # Features for prediction
            features = ['volume_24h', 'avg_trade_size', 'num_traders', 'liquidity', 'is_weekend', 'is_evening']
            lag_features = [col for col in df_pred.columns if 'lag' in col]
            features.extend(lag_features)

            X = df_pred[features]
            y = df_pred['high_retail_next']

            # Train/test split (use last 20% for testing)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if len(X_test) == 0:
                return {'insufficient_test_data': 'Not enough data for testing'}

            # Train Random Forest classifier
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X_train, y_train)

            # Predictions
            y_pred = rf.predict(X_test)
            feature_importance = dict(zip(features, rf.feature_importances_))

            # Get prediction accuracy
            accuracy = (y_pred == y_test).mean()

            # Predict next period
            latest_features = X.iloc[-1:].values
            next_prediction = rf.predict_proba(latest_features)[0]

            return {
                'prediction_accuracy': accuracy,
                'feature_importance': feature_importance,
                'next_period_retail_probability': next_prediction[1],  # Probability of high retail
                'top_predictors': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3],
                'model_confidence': max(next_prediction)
            }

        except Exception as e:
            logger.warning(f"ML predictive insights failed: {e}")
            return {'predictive_insights_error': str(e)}