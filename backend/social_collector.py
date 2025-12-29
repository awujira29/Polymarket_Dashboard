import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaCollector:
    """Placeholder for social media data collection (Twitter, Reddit, etc.)"""

    def __init__(self):
        # Twitter API v2 credentials (set these environment variables)
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

        # Reddit API credentials (optional)
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PredictionMarketResearch/1.0'
        })

    def search_twitter_mentions(self, keywords: List[str], hours_back: int = 24) -> Dict:
        """
        Search for Twitter mentions of prediction markets
        Note: Requires Twitter API v2 Bearer Token
        """
        if not self.twitter_bearer_token:
            return {
                'status': 'api_not_configured',
                'message': 'Twitter API credentials not found. Set TWITTER_BEARER_TOKEN environment variable.',
                'sample_data': self._get_sample_twitter_data(keywords)
            }

        try:
            # Twitter API v2 recent search endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"

            # Build search query
            query = " OR ".join([f'"{kw}"' for kw in keywords])
            query += " -is:retweet"  # Exclude retweets

            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics,author_id,lang',
                'start_time': (datetime.utcnow() - timedelta(hours=hours_back)).isoformat() + 'Z'
            }

            headers = {'Authorization': f'Bearer {self.twitter_bearer_token}'}

            response = self.session.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            mentions = []
            if 'data' in data:
                for tweet in data['data']:
                    mentions.append({
                        'tweet_id': tweet['id'],
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'likes': tweet['public_metrics']['like_count'],
                        'retweets': tweet['public_metrics']['retweet_count'],
                        'replies': tweet['public_metrics']['reply_count'],
                        'engagement_score': sum(tweet['public_metrics'].values())
                    })

            return {
                'status': 'success',
                'mentions_found': len(mentions),
                'total_engagement': sum(m['engagement_score'] for m in mentions),
                'mentions': mentions[:10],  # Return top 10
                'keywords_searched': keywords
            }

        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'sample_data': self._get_sample_twitter_data(keywords)
            }

    def search_reddit_mentions(self, keywords: List[str], hours_back: int = 24) -> Dict:
        """
        Search for Reddit mentions of prediction markets
        Note: Requires Reddit API credentials
        """
        if not (self.reddit_client_id and self.reddit_client_secret):
            return {
                'status': 'api_not_configured',
                'message': 'Reddit API credentials not found.',
                'sample_data': self._get_sample_reddit_data(keywords)
            }

        try:
            # Reddit API search (simplified)
            # This would require OAuth flow for full access
            # For now, return placeholder
            return {
                'status': 'api_available',
                'message': 'Reddit API integration ready (OAuth implementation needed)',
                'sample_data': self._get_sample_reddit_data(keywords)
            }

        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return {'status': 'error', 'message': str(e)}

    def correlate_social_market_activity(self, market_title: str, market_volume: float) -> Dict:
        """
        Correlate social media mentions with market activity
        This is a simplified correlation analysis
        """
        # Extract keywords from market title
        keywords = self._extract_market_keywords(market_title)

        # Get social mentions (using sample data for now)
        twitter_data = self.search_twitter_mentions(keywords, hours_back=24)
        reddit_data = self.search_reddit_mentions(keywords, hours_back=24)

        # Simple correlation analysis
        social_engagement = 0
        if twitter_data.get('status') == 'success':
            social_engagement += twitter_data.get('total_engagement', 0)

        # Hypothetical correlation with volume
        # In reality, you'd need time-series correlation analysis
        volume_correlation = 'unknown'
        if social_engagement > 100 and market_volume > 10000:
            volume_correlation = 'high'
        elif social_engagement > 50 or market_volume > 5000:
            volume_correlation = 'moderate'
        else:
            volume_correlation = 'low'

        return {
            'market_title': market_title,
            'keywords_analyzed': keywords,
            'social_engagement_score': social_engagement,
            'market_volume_24h': market_volume,
            'social_volume_correlation': volume_correlation,
            'twitter_mentions': twitter_data.get('mentions_found', 0),
            'potential_hype_indicators': self._analyze_hype_potential(social_engagement, market_volume)
        }

    def _extract_market_keywords(self, market_title: str) -> List[str]:
        """Extract searchable keywords from market title"""
        words = market_title.lower().split()
        keywords = []

        # Add full title if not too long
        if len(market_title) <= 60:
            keywords.append(market_title)

        # Add significant words
        significant_words = []
        for word in words:
            if (len(word) > 3 and
                word not in ['will', 'the', 'and', 'for', 'are', 'but', 'not', 'you',
                           'all', 'can', 'her', 'was', 'one', 'our', 'had', 'by',
                           'hot', 'but', 'some', 'what', 'said', 'each', 'which',
                           'their', 'time', 'would', 'there', 'could', 'other']):
                significant_words.append(word)

        keywords.extend(significant_words[:3])  # Top 3 significant words
        return list(set(keywords))  # Remove duplicates

    def _analyze_hype_potential(self, social_engagement: int, market_volume: float) -> List[str]:
        """Analyze potential hype indicators"""
        indicators = []

        if social_engagement > 200:
            indicators.append("high_social_engagement")
        elif social_engagement > 50:
            indicators.append("moderate_social_engagement")

        if market_volume > 50000:
            indicators.append("high_volume")
        elif market_volume > 10000:
            indicators.append("moderate_volume")

        # Combined hype analysis
        if social_engagement > 100 and market_volume > 20000:
            indicators.append("viral_potential")
        elif social_engagement > 50 and market_volume > 10000:
            indicators.append("growing_attention")

        return indicators

    def _get_sample_twitter_data(self, keywords: List[str]) -> Dict:
        """Return sample Twitter data for demonstration"""
        return {
            'sample_mentions': [
                {
                    'tweet_id': 'sample_1',
                    'text': f"Just saw this on Polymarket: {keywords[0] if keywords else 'market'}. What do you think?",
                    'engagement_score': 25,
                    'created_at': datetime.utcnow().isoformat()
                },
                {
                    'tweet_id': 'sample_2',
                    'text': f"Interesting prediction market about {keywords[0] if keywords else 'topic'} on Polymarket",
                    'engagement_score': 12,
                    'created_at': (datetime.utcnow() - timedelta(hours=2)).isoformat()
                }
            ],
            'note': 'This is sample data. Configure Twitter API for real data.'
        }

    def _get_sample_reddit_data(self, keywords: List[str]) -> Dict:
        """Return sample Reddit data for demonstration"""
        return {
            'sample_posts': [
                {
                    'post_id': 'sample_1',
                    'title': f"Discussion: {keywords[0] if keywords else 'Market'} prediction market",
                    'subreddit': 'r/predictionmarkets',
                    'score': 45,
                    'comments': 12
                }
            ],
            'note': 'This is sample data. Configure Reddit API for real data.'
        }

# Example usage functions
def analyze_market_social_sentiment(market_title: str, market_volume: float):
    """Analyze social sentiment for a market"""
    collector = SocialMediaCollector()
    return collector.correlate_social_market_activity(market_title, market_volume)

def get_platform_mentions(platform: str = "polymarket", hours_back: int = 24):
    """Get mentions of a prediction market platform"""
    collector = SocialMediaCollector()

    if platform.lower() == "polymarket":
        keywords = ["polymarket", " Polymarket", "@polymarket"]
    elif platform.lower() == "kalshi":
        keywords = ["kalshi", " Kalshi", "@kalshi"]
    else:
        keywords = [platform]

    return collector.search_twitter_mentions(keywords, hours_back)