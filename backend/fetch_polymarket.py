import requests
import logging
import json
import os
import base64
import hmac
import hashlib
import time
import asyncio
import re
from pathlib import Path
from datetime import datetime, timezone
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from models import Market, MarketSnapshot
from database import get_db_session, init_db
import time
from dotenv import load_dotenv
from config import (
    RETAIL_SIZE_PERCENTILES,
    RETAIL_MIN_TRADES,
    RETAIL_FALLBACK_THRESHOLD
)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolymarketCollector:
    def __init__(self):
        self.gamma_base = "https://gamma-api.polymarket.com"
        self.clob_base = "https://clob.polymarket.com"
        self.data_api_base = "https://data-api.polymarket.com"
        self.clob_api_key = os.getenv("POLYMARKET_CLOB_API_KEY")
        self.clob_api_secret = os.getenv("POLYMARKET_CLOB_API_SECRET")
        self.clob_api_passphrase = os.getenv("POLYMARKET_CLOB_API_PASSPHRASE")
        self.clob_address = os.getenv("POLYMARKET_CLOB_ADDRESS")
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        self._trade_key_warned = False
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        # Define market categories based on tags
        self.category_mapping = {
            'politics': ['Politics', 'US Politics', 'Elections', 'Government'],
            'crypto': ['Crypto', 'Bitcoin', 'Ethereum', 'Cryptocurrency', 'DeFi'],
            'sports': [
                'Sports', 'Football', 'Basketball', 'Soccer', 'Baseball', 'Olympics',
                'NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'WNBA', 'NCAA', 'College',
                'Hockey', 'Tennis', 'Golf', 'Cricket', 'Rugby', 'Boxing', 'UFC', 'MMA',
                'Formula 1', 'F1', 'NASCAR', 'Motorsport', 'Wimbledon', 'World Cup',
                'Premier League', 'Champions League', 'La Liga', 'Serie A', 'Bundesliga'
            ],
            'business': ['Business', 'Economy', 'Finance', 'Stocks', 'Companies'],
            'entertainment': ['Entertainment', 'Movies', 'TV', 'Music', 'Celebrities'],
            'science': ['Science', 'Technology', 'AI', 'Space', 'Health'],
            'world': ['World', 'International', 'War', 'Conflict', 'Diplomacy']
        }
        self._maybe_derive_clob_creds()

    def _maybe_derive_clob_creds(self):
        if self.clob_api_key and self.clob_api_secret and self.clob_api_passphrase and self.clob_address:
            return

        if not self.private_key:
            return

        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            logger.warning("py-clob-client not installed; cannot derive CLOB API credentials.")
            return

        async def _derive():
            client = ClobClient(self.clob_base, key=self.private_key, chain_id=137)
            return await client.create_or_derive_api_key()

        try:
            creds = asyncio.run(_derive())
        except RuntimeError:
            logger.warning("Async loop running; skipping auto-derive of CLOB credentials.")
            return

        self.clob_api_key = creds.get("apiKey") or creds.get("key")
        self.clob_api_secret = creds.get("secret")
        self.clob_api_passphrase = creds.get("passphrase")

        if not self.clob_address:
            try:
                from eth_account import Account
                self.clob_address = Account.from_key(self.private_key).address
            except Exception:
                logger.warning("Unable to derive wallet address for CLOB auth.")

    def _safe_float(self, value, default=0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list):
            return self._safe_float(value[0], default) if value else default
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith('['):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list) and parsed:
                        return self._safe_float(parsed[0], default)
                except json.JSONDecodeError:
                    return default
            try:
                return float(stripped)
            except ValueError:
                return default
        return default

    def _extract_price(self, market: Dict) -> float:
        outcomes = market.get('outcomes') or []
        outcome_prices = market.get('outcomePrices') or market.get('outcome_prices') or []
        if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
            for idx, outcome in enumerate(outcomes):
                if str(outcome).strip().lower() == 'yes':
                    return self._safe_float(outcome_prices[idx], 0.0)
            prices = [self._safe_float(price, 0.0) for price in outcome_prices]
            return max(prices) if prices else 0.0

        tokens = market.get('tokens') or []
        for token in tokens:
            label = (
                token.get('outcome')
                or token.get('label')
                or token.get('name')
                or ''
            )
            if str(label).strip().lower() == 'yes':
                return self._safe_float(token.get('price', 0.0), 0.0)
        if tokens:
            prices = [self._safe_float(token.get('price', 0.0), 0.0) for token in tokens]
            return max(prices) if prices else 0.0

        if market.get('lastTradePrice') is not None:
            return self._safe_float(market.get('lastTradePrice'), 0.0)
        return self._safe_float(market.get('price', 0.0), 0.0)

    def _base64_to_bytes(self, value: str) -> bytes:
        cleaned = value.replace("-", "+").replace("_", "/")
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "+/=")
        padding = "=" * ((4 - len(cleaned) % 4) % 4)
        return base64.b64decode(cleaned + padding)

    def _build_l2_signature(self, method: str, path: str, body: str | None = None) -> str:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}"
        if body:
            message += body
        key_bytes = self._base64_to_bytes(self.clob_api_secret or "")
        digest = hmac.new(key_bytes, message.encode("utf-8"), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(digest).decode("utf-8")
        return timestamp, signature

    def get_active_markets(self, limit=50, categories=None) -> List[Dict]:
        """Fetch active, high-volume markets from Polymarket with categorization"""
        try:
            if categories:
                categories = [str(category).strip().lower() for category in categories]
            # Get events with pagination for better coverage
            url = f"{self.gamma_base}/events"
            page_size = 100
            max_pages = 10
            offset = 0
            target_per_category = max(1, limit // len(categories)) if categories else None
            max_total = limit * 5

            logger.info("Fetching active events from Polymarket...")

            # Extract and categorize markets
            categorized_markets = {
                'politics': [],
                'crypto': [],
                'sports': [],
                'business': [],
                'entertainment': [],
                'science': [],
                'world': [],
                'other': []
            }

            seen_market_ids = set()
            total_markets = 0
            for _ in range(max_pages):
                params = {
                    "limit": page_size,
                    "offset": offset,
                    "closed": False,
                    "archived": False,
                    "active": True
                }
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch events: {response.status_code}")
                    break

                events = response.json() or []
                if not events:
                    break

                logger.info(f"✅ Fetched {len(events)} events (offset {offset})")

                for event in events:
                    if 'markets' not in event or not event['markets']:
                        continue
                    event_tags = [tag.get('label', '') for tag in event.get('tags', [])]
                    event_slugs = [tag.get('slug', '') for tag in event.get('tags', [])]
                    event_category = self._categorize_event(
                        event_tags,
                        event_slugs,
                        event.get('title', '')
                    )
                    event_category = str(event_category or "other").strip().lower()

                    for market in event['markets']:
                        market_id = market.get('id')
                        if not market_id or market_id in seen_market_ids:
                            continue
                        seen_market_ids.add(market_id)
                        market_category = self._categorize_event(
                            event_tags,
                            event_slugs,
                            event.get('title', ''),
                            market.get('question', '')
                        )
                        market_category = str(market_category or "other").strip().lower()
                        if market_category == 'other' and event_category != 'other':
                            market_category = event_category
                        # Enhanced market data
                        volume = self._safe_float(market.get('volume', 0), 0.0)
                        volume_24h = self._safe_float(market.get('volume24hr', 0), 0.0)
                        liquidity = self._safe_float(market.get('liquidity', 0), 0.0)
                        price = self._extract_price(market)

                        market_data = {
                            'id': market_id,
                            'condition_id': market.get('conditionId') or market.get('condition_id') or market_id,
                            'question': market.get('question', ''),
                            'volume': volume,
                            'volume_24h': volume_24h,
                            'liquidity': liquidity,
                            'price': price,
                            'active': market.get('active', True),
                            'closed': market.get('closed', False),
                            'category': market_category,
                            'event_title': event.get('title', ''),
                            'event_tags': event_tags,
                            'created_at': market.get('createdAt'),
                            'end_date': market.get('endDate'),
                            'outcomes': market.get('outcomes', [])
                        }

                        categorized_markets[market_category].append(market_data)
                        total_markets += 1

                offset += page_size
                if categories and target_per_category:
                    if all(
                        len(categorized_markets.get(cat, [])) >= target_per_category * 3
                        for cat in categories
                    ):
                        break
                elif total_markets >= max_total:
                    break

                if len(events) < page_size:
                    break

            if total_markets == 0:
                return []

            # Select markets based on requested categories and limits
            selected_markets = []
            if categories:
                per_category_limit = max(1, limit // len(categories))
                for category in categories:
                    if category in categorized_markets:
                        categorized_markets[category].sort(
                            key=lambda m: (m.get('volume', 0) or 0, m.get('liquidity', 0) or 0),
                            reverse=True
                        )
                for category in categories:
                    if category in categorized_markets:
                        selected_markets.extend(categorized_markets[category][:per_category_limit])
            else:
                # Default: take balanced sample from all categories
                markets_per_category = max(5, limit // len(categorized_markets))
                for category_markets in categorized_markets.values():
                    category_markets.sort(
                        key=lambda m: (m.get('volume', 0) or 0, m.get('liquidity', 0) or 0),
                        reverse=True
                    )
                    selected_markets.extend(category_markets[:markets_per_category])

            # Sort by volume and limit final result
            selected_markets.sort(key=lambda x: x['volume'], reverse=True)
            final_markets = selected_markets[:limit]

            logger.info(f"📊 Found {total_markets} total markets, selected {len(final_markets)} for analysis")
            for category, markets in categorized_markets.items():
                if markets:
                    logger.info(f"  {category}: {len(markets)} markets")

            return final_markets

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def _categorize_event(
        self,
        tags: List[str],
        slugs: List[str] | None = None,
        title: str | None = None,
        question: str | None = None
    ) -> str:
        """Categorize event based on tags, slugs, and titles."""
        tag_text = ' '.join(tags).lower()
        slug_text = ' '.join(slugs or []).lower()
        title_text = (title or "").lower()
        question_text = (question or "").lower()
        tag_blob = f"{tag_text} {slug_text}".strip()
        text_blob = f"{title_text} {question_text}".strip()

        def _keyword_hits(text: str, keywords: List[str]) -> int:
            hits = 0
            for keyword in keywords:
                key = keyword.lower().strip()
                if not key:
                    continue
                if len(key) <= 3:
                    if re.search(rf"\b{re.escape(key)}\b", text):
                        hits += 1
                else:
                    if key in text:
                        hits += 1
            return hits

        best_category = "other"
        best_score = 0
        best_tag_hits = 0
        tag_hits_by_cat = {}
        text_hits_by_cat = {}
        for category, keywords in self.category_mapping.items():
            tag_hits = _keyword_hits(tag_blob, keywords)
            text_hits = _keyword_hits(text_blob, keywords)
            tag_hits_by_cat[category] = tag_hits
            text_hits_by_cat[category] = text_hits
            score = (tag_hits * 3) + text_hits
            if score > best_score or (score == best_score and tag_hits > best_tag_hits):
                best_score = score
                best_category = category
                best_tag_hits = tag_hits

        if best_score > 0:
            if best_category == "sports":
                crypto_tag_hits = tag_hits_by_cat.get("crypto", 0)
                sports_tag_hits = tag_hits_by_cat.get("sports", 0)
                if crypto_tag_hits > 0 and crypto_tag_hits >= sports_tag_hits:
                    return "crypto"
            return best_category

        combined = f"{tag_blob} {text_blob}"

        # Additional keyword matching
        if any(word in combined for word in ['president', 'election', 'democrat', 'republican', 'congress']):
            return 'politics'
        elif any(word in combined for word in ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi']):
            return 'crypto'
        elif any(word in combined for word in ['football', 'basketball', 'soccer', 'nfl', 'nba', 'mlb', 'nhl', 'ufc', 'f1']):
            return 'sports'
        elif any(word in combined for word in ['movie', 'film', 'actor', 'celebrity', 'tv show']):
            return 'entertainment'
        elif any(word in combined for word in ['ai', 'artificial intelligence', 'machine learning']):
            return 'science'

        return 'other'

    def _fetch_public_trades_pages(self, pages: int = 2, limit: int = 500, taker_only: bool = False) -> List[Dict]:
        """Fetch public trade pages (global feed) from Polymarket data API."""
        trades = []
        for page in range(pages):
            params = {
                "limit": limit,
                "offset": page * limit
            }
            if taker_only:
                params["takerOnly"] = True
            response = self.session.get(f"{self.data_api_base}/trades", params=params, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch public trades page {page}: {response.status_code}")
                break
            payload = response.json()
            if not payload:
                break
            trades.extend(payload)
            if len(payload) < limit:
                break
        return trades

    def _classify_trade_size(self, value: float) -> str:
        """Classify trade size for analysis."""
        if value < 50:
            return 'small'
        if value < 500:
            return 'medium'
        if value < 5000:
            return 'large'
        return 'xlarge'

    def _percentile(self, values: List[float], percentile: float) -> float:
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

    def _retail_threshold(
        self,
        values: List[float],
        category: str | None = None,
        min_trades: int = RETAIL_MIN_TRADES,
        fallback: float = RETAIL_FALLBACK_THRESHOLD
    ) -> tuple[float, str]:
        percentile = RETAIL_SIZE_PERCENTILES.get(category, RETAIL_SIZE_PERCENTILES.get("default", 0.3))
        if len(values) < min_trades:
            return fallback, "fixed"
        return self._percentile(values, percentile), "percentile"

    def get_trades_for_conditions(self, condition_ids: set, pages: int = 2) -> Dict[str, List[Dict]]:
        """Fetch public trades and group by conditionId."""
        grouped = {cid: [] for cid in condition_ids}
        trades = self._fetch_public_trades_pages(pages=pages)

        for trade in trades:
            condition_id = trade.get("conditionId")
            if condition_id not in grouped:
                continue

            price = self._safe_float(trade.get('price', 0), 0.0)
            quantity = self._safe_float(trade.get('size', trade.get('quantity', 0)), 0.0)
            raw_ts = trade.get('timestamp') or trade.get('created_at')
            if isinstance(raw_ts, (int, float)):
                timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc).isoformat()
            else:
                timestamp = raw_ts

            grouped[condition_id].append({
                'timestamp': timestamp,
                'price': price,
                'quantity': quantity,
                'side': trade.get('side', trade.get('taker_side', 'unknown')),
                'value': price * quantity,
                'taker': trade.get('proxyWallet') or trade.get('taker'),
                'maker': trade.get('maker')
            })

        return grouped

    def collect_comprehensive_data(self, categories=None, markets_per_category=10, include_trades=True, trade_pages: int = 2):
        """Collect comprehensive market data with trades"""
        logger.info("🚀 Starting comprehensive data collection...")

        # Get markets
        markets = self.get_active_markets(limit=markets_per_category*10, categories=categories)

        collected_data = {
            'timestamp': datetime.utcnow(),
            'categories': {},
            'total_markets': len(markets),
            'total_trades': 0
        }

        condition_trade_map = {}
        if include_trades and markets:
            condition_ids = {m.get("condition_id") for m in markets if m.get("condition_id")}
            condition_trade_map = self.get_trades_for_conditions(condition_ids, pages=trade_pages)

        # Group markets by category
        for market in markets:
            category = market['category']
            if category not in collected_data['categories']:
                collected_data['categories'][category] = []

            market_data = market.copy()

            # Get recent trades if requested
            if include_trades:
                trades = condition_trade_map.get(market.get('condition_id'), [])
                market_data['recent_trades'] = trades
                collected_data['total_trades'] += len(trades)

                # Calculate retail metrics from trades
                if trades:
                    market_data['retail_metrics'] = self._calculate_trade_metrics(trades, category)
                    trade_volume = sum(t['value'] for t in trades)
                    market_data['trade_count'] = len(trades)
                    market_data['avg_trade_size'] = trade_volume / len(trades)
                else:
                    market_data['trade_count'] = 0
                    market_data['avg_trade_size'] = 0

            collected_data['categories'][category].append(market_data)

            # Rate limiting
            time.sleep(0.1)

        logger.info(f"✅ Comprehensive data collection complete: {collected_data['total_markets']} markets, {collected_data['total_trades']} trades")
        return collected_data

    def _calculate_trade_metrics(self, trades: List[Dict], category: str | None = None) -> Dict:
        """Calculate retail behavior metrics from trade data"""
        if not trades:
            return {}

        trade_sizes = [trade['value'] for trade in trades]
        trade_sizes.sort()

        # Basic statistics
        metrics = {
            'total_trades': len(trades),
            'total_volume': sum(trade_sizes),
            'avg_trade_size': sum(trade_sizes) / len(trades),
            'median_trade_size': trade_sizes[len(trade_sizes)//2],
            'min_trade_size': min(trade_sizes),
            'max_trade_size': max(trade_sizes),
            'trade_size_std': sum((x - sum(trade_sizes)/len(trade_sizes))**2 for x in trade_sizes) ** 0.5 / len(trade_sizes)
        }

        # Retail indicators (percentile-based)
        threshold, method = self._retail_threshold(trade_sizes, category)
        small_trades = [t for t in trade_sizes if t <= threshold]
        metrics['retail_trade_count'] = len(small_trades)
        metrics['retail_percentage'] = len(small_trades) / len(trades) * 100
        metrics['retail_volume_share'] = sum(small_trades) / metrics['total_volume'] if metrics['total_volume'] > 0 else 0
        metrics['retail_threshold'] = threshold
        metrics['retail_threshold_method'] = method

        # Time-based patterns
        hour_counts = {}
        for trade in trades:
            if 'timestamp' in trade:
                try:
                    dt = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                    hour = dt.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except:
                    pass

        metrics['hourly_distribution'] = hour_counts

        return metrics
    
    def save_market(self, market_data: Dict, db_session):
        """Save or update market in database"""
        try:
            # Get market ID - try multiple possible fields
            market_id = (market_data.get("id") or
                        market_data.get("condition_id") or
                        market_data.get("conditionId"))
            
            if not market_id:
                logger.warning("⚠️  No market ID found, skipping")
                return None
            
            # Get title - try multiple fields
            title = (market_data.get("question") or 
                    market_data.get("title") or
                    market_data.get("event_title") or
                    "Unknown Market")

            category = str(market_data.get("category", "other")).strip().lower()
            
            # Check if market exists
            existing = db_session.query(Market).filter_by(id=market_id).first()
            
            if existing:
                # Update existing
                existing.title = title
                existing.description = market_data.get("description", "")
                existing.status = "active"
                existing.closed = str(market_data.get("closed", False)).lower()
                existing.archived = str(market_data.get("archived", False)).lower()
                existing.end_date_iso = market_data.get("endDateIso")
                existing.subcategory = market_data.get("subcategory")
                existing.condition_id = market_data.get("condition_id") or market_data.get("conditionId")
                existing.event_title = market_data.get("event_title")
                existing.event_tags = market_data.get("event_tags")
                existing.outcomes = market_data.get("outcomes")
                existing.category = category
            else:
                # Create new
                new_market = Market(
                    id=market_id,
                    condition_id=market_data.get("condition_id") or market_data.get("conditionId") or market_id,
                    title=title,
                    category=category,
                    subcategory=market_data.get("subcategory"),
                    description=market_data.get("description", ""),
                    status="active",
                    closed=str(market_data.get("closed", False)).lower(),
                    archived=str(market_data.get("archived", False)).lower(),
                    end_date_iso=market_data.get("endDateIso"),
                    event_title=market_data.get("event_title"),
                    event_tags=market_data.get("event_tags"),
                    outcomes=market_data.get("outcomes")
                )
                db_session.add(new_market)
            
            db_session.commit()
            logger.info(f"   💾 {title[:70]}")
            return market_id
            
        except Exception as e:
            logger.error(f"❌ Error saving market: {e}")
            db_session.rollback()
            return None
    
    def save_snapshot(self, market_id: str, market_data: Dict, db_session):
        """Save market snapshot"""
        try:
            import json
            
            # Extract price - try different structures
            price = 0.5  # Default
            
            # Try lastTradePrice first (most reliable for current price)
            if "lastTradePrice" in market_data and market_data["lastTradePrice"] is not None:
                price_str = str(market_data["lastTradePrice"])
                try:
                    price = float(price_str)
                except (ValueError, TypeError) as e:
                    logger.warning(f"   ⚠️  Could not convert lastTradePrice '{price_str}': {e}")
            
            # Try outcomePrices array (parse as JSON if it's a string)
            elif "outcomePrices" in market_data:
                outcome_prices = market_data["outcomePrices"]
                
                # If it's a string, parse as JSON
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except json.JSONDecodeError as e:
                        logger.warning(f"   ⚠️  Could not parse outcomePrices JSON: {e}")
                        outcome_prices = []
                
                # Now extract the first price (usually the "Yes" outcome)
                if isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                    price_str = str(outcome_prices[0])
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"   ⚠️  Could not convert outcomePrices[0] '{price_str}': {e}")
            
            # Try tokens array (fallback for older API)
            elif "tokens" in market_data:
                tokens = market_data["tokens"]
                if tokens and len(tokens) > 0:
                    token = tokens[0]
                    price_str = token.get("price", "0.5")
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"   ⚠️  Could not convert token price '{price_str}': {e}")
            
            # Try direct price field (fallback)
            elif "price" in market_data:
                price_str = market_data["price"]
                try:
                    price = float(price_str)
                except (ValueError, TypeError) as e:
                    logger.warning(f"   ⚠️  Could not convert direct price '{price_str}': {e}")
            
            # Extract volume and liquidity
            volume_24h = float(market_data.get("volume24hr", 0) or 
                             market_data.get("volume_24h", 0) or
                             market_data.get("volume", 0) or 0)
            
            liquidity = float(market_data.get("liquidity", 0) or 
                            market_data.get("liq", 0) or 0)
            
            # Get total volume for reference
            total_volume = float(market_data.get("volume", 0) or 0)
            
            # Extract additional trading metrics for retail analysis
            num_trades = int(market_data.get("numTrades", 0) or 
                           market_data.get("tradeCount", 0) or 0)
            
            # Estimate number of traders (rough approximation)
            # Polymarket doesn't provide this directly, so we estimate based on trade count
            estimated_traders = max(1, num_trades // 3) if num_trades > 0 else 0
            
            # Calculate average trade size
            avg_trade_size = volume_24h / max(num_trades, 1) if volume_24h > 0 else 0
            
            # Count large trades (rough estimation - trades > $1000)
            large_trade_threshold = 1000
            estimated_large_trades = max(0, int(volume_24h / large_trade_threshold) - 1) if volume_24h > large_trade_threshold else 0
            
            # Determine temporal patterns for retail analysis
            now = datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
            hour_of_day = now.hour
            is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
            is_evening = hour_of_day >= 18 or hour_of_day <= 5  # After 6 PM or before 6 AM ET
            
            snapshot = MarketSnapshot(
                market_id=market_id,
                timestamp=now,
                price=price,
                volume_24h=volume_24h,
                volume_num_trades=num_trades,
                volume_all_time=total_volume,
                liquidity=liquidity,
                num_traders=estimated_traders,
                avg_trade_size=avg_trade_size,
                large_trade_count=estimated_large_trades,
                is_weekend=is_weekend,
                is_evening=is_evening,
                hour_of_day=hour_of_day
            )
            
            db_session.add(snapshot)
            db_session.commit()
            
            # Enhanced logging with retail indicators
            weekend_indicator = "🎯" if is_weekend else ""
            evening_indicator = "🌙" if is_evening else ""
            retail_indicator = "🏪" if avg_trade_size < 500 else ""
            
            logger.info(f"   📊 Price: ${price:.3f} | 24h Vol: ${volume_24h:,.0f} | Trades: {num_trades} | Liq: ${liquidity:,.0f} {weekend_indicator}{evening_indicator}{retail_indicator}")
            
        except Exception as e:
            logger.error(f"❌ Error saving snapshot: {e}")
            import traceback
            traceback.print_exc()
            db_session.rollback()
    
    def collect_data(self):
        """Main collection routine"""
        logger.info("="*70)
        logger.info("🚀 STARTING DATA COLLECTION")
        logger.info("="*70)
        
        db_session = get_db_session()
        
        try:
            markets = self.get_active_markets(limit=10)
            
            if not markets:
                logger.error("❌ No markets returned from API!")
                logger.info("This could mean:")
                logger.info("  - API endpoint changed")
                logger.info("  - Network connectivity issue")
                logger.info("  - All markets are inactive/closed")
                return
            
            logger.info(f"\n📈 Processing {len(markets)} markets...\n")
            
            success_count = 0
            for i, market in enumerate(markets, 1):
                logger.info(f"[{i}/{len(markets)}]")
                
                market_id = self.save_market(market, db_session)
                
                if market_id:
                    self.save_snapshot(market_id, market, db_session)
                    success_count += 1
                
                logger.info("")  # Blank line between markets
            
            logger.info("="*70)
            logger.info(f"✅ COLLECTION COMPLETE! ({success_count}/{len(markets)} successful)")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            db_session.close()


def main():
    logger.info("\n🎯 Initializing Polymarket Collector\n")
    
    # Initialize database
    init_db()
    
    # Create collector and run
    collector = PolymarketCollector()
    collector.collect_data()
    
    logger.info("\n" + "="*70)
    logger.info("✨ Done! Next steps:")
    logger.info("   - Run 'python check_data.py' to view collected data")
    logger.info("   - Run 'python scheduler.py' for continuous collection")
    logger.info("   - Run 'python api.py' to start the API server")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
