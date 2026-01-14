#!/usr/bin/env python3
"""
Startup Idea Discovery Tool - Complaint Miner
==============================================

Based on Om Patel's playbook for finding startup ideas by mining user complaints.
"Every complaint is someone saying 'I would pay for this to not suck'"

Usage:
    python complaint_miner.py --topic "golf" --sources reddit,appstore --min_mentions 15
    python complaint_miner.py --topic "crypto" --sources all
    python complaint_miner.py --topic "meditation apps" --sources reddit,google_play

Author: Startup Idea Miner
License: MIT
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better console output: pip install rich")

# PRAW no longer required - using Reddit's public JSON endpoints

# App store scrapers
try:
    from google_play_scraper import Sort, reviews as gplay_reviews, search as gplay_search
    GPLAY_AVAILABLE = True
except ImportError:
    GPLAY_AVAILABLE = False
    print("Note: Install 'google-play-scraper' for Google Play: pip install google-play-scraper")

try:
    from app_store_scraper import AppStore
    APPSTORE_AVAILABLE = True
except ImportError:
    APPSTORE_AVAILABLE = False
    print("Note: Install 'app-store-scraper' for App Store: pip install app-store-scraper")

# NLTK for NLP
try:
    import nltk
    from nltk import ngrams, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Note: Install 'nltk' for pattern extraction: pip install nltk")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Try to import user config, fall back to defaults
try:
    from config import (
        REDDIT_CONFIG, SCRAPING_CONFIG, TOPIC_SUBREDDITS,
        TOPIC_APP_SEARCHES, TOPIC_B2B_PRODUCTS, UPWORK_SEARCH_TEMPLATES,
        COMPLAINT_KEYWORDS, OUTPUT_CONFIG
    )
except ImportError:
    # Default configurations if config.py doesn't exist
    REDDIT_CONFIG = {
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "StartupIdeaMiner/1.0"),
    }

    SCRAPING_CONFIG = {
        "request_delay": 2.0,
        "reddit_delay": 1.0,
        "appstore_delay": 1.5,
        "max_reddit_posts": 100,
        "max_reddit_comments": 500,
        "max_app_reviews": 200,
        "max_g2_reviews": 100,
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 5,
    }

    TOPIC_SUBREDDITS = {
        "crypto": ["cryptocurrency", "bitcoin", "ethereum", "defi"],
        "finance": ["personalfinance", "investing", "FinancialPlanning"],
        "fitness": ["fitness", "loseit", "running", "gym"],
        "meditation": ["Meditation", "mindfulness", "yoga"],
        "golf": ["golf", "GolfSwing"],
        "productivity": ["productivity", "Notion", "ObsidianMD"],
        "podcast": ["podcasting", "podcasts"],
        "marketing": ["marketing", "digital_marketing", "SEO"],
        "real_estate": ["realestate", "RealEstateInvesting"],
        "_default": ["mildlyinfuriating", "entrepreneur", "SomebodyMakeThis"],
    }

    TOPIC_APP_SEARCHES = {
        "crypto": ["crypto wallet", "bitcoin tracker"],
        "finance": ["budget tracker", "expense manager"],
        "meditation": ["meditation", "mindfulness"],
        "fitness": ["workout tracker", "fitness app"],
        "golf": ["golf gps", "golf swing"],
        "productivity": ["todo app", "task manager"],
        "podcast": ["podcast player", "podcast app"],
    }

    TOPIC_B2B_PRODUCTS = {
        "crm": [{"name": "Salesforce", "g2_slug": "salesforce-sales-cloud"}],
        "productivity": [{"name": "Notion", "g2_slug": "notion"}],
        "marketing": [{"name": "Mailchimp", "g2_slug": "mailchimp"}],
    }

    UPWORK_SEARCH_TEMPLATES = {
        "_default": ["{topic} weekly", "{topic} monthly", "{topic} ongoing"],
    }

    COMPLAINT_KEYWORDS = {
        "missing": ["doesn't have", "missing", "no way to", "can't", "cannot",
                    "doesn't support", "lack of", "no option"],
        "wishes": ["wish it could", "wish there was", "wish they would",
                   "should have", "needs to have"],
        "frustration": ["frustrating", "annoying", "hate when", "hate that",
                       "terrible", "awful", "horrible"],
        "complexity": ["too complex", "complicated", "confusing", "hard to use"],
        "pricing": ["too expensive", "overpriced", "not worth", "costs too much"],
        "performance": ["slow", "laggy", "crashes", "buggy", "unreliable"],
    }

    OUTPUT_CONFIG = {
        "min_mentions_threshold": 5,
        "idea_generation_threshold": 10,
        "max_ideas": 10,
        "export_csv": True,
        "export_json": True,
        "output_dir": "output",
    }


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Complaint:
    """Represents a single complaint/pain point discovered."""
    text: str
    source: str
    topic: str
    url: str = ""
    product_name: str = ""
    rating: float = 0.0
    date: str = ""
    upvotes: int = 0
    keywords_found: list = field(default_factory=list)
    category: str = ""  # missing, wish, frustration, etc.


@dataclass
class StartupIdea:
    """A generated startup idea based on complaint patterns."""
    title: str
    description: str
    pain_point: str
    mention_count: int
    sources: list
    target_audience: str = ""
    potential_revenue: str = ""
    competition_notes: str = ""


# =============================================================================
# HTTP SESSION WITH RETRY
# =============================================================================

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=SCRAPING_CONFIG["max_retries"],
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session


# =============================================================================
# CONSOLE OUTPUT HELPERS
# =============================================================================

console = Console() if RICH_AVAILABLE else None


def print_header(text: str):
    """Print a section header."""
    if RICH_AVAILABLE:
        console.print(Panel(text, style="bold blue"))
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}")


def print_success(text: str):
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] {text}")
    else:
        print(f"✓ {text}")


def print_warning(text: str):
    """Print warning message."""
    if RICH_AVAILABLE:
        console.print(f"[yellow]⚠[/yellow] {text}")
    else:
        print(f"⚠ {text}")


def print_error(text: str):
    """Print error message."""
    if RICH_AVAILABLE:
        console.print(f"[red]✗[/red] {text}")
    else:
        print(f"✗ {text}")


def print_info(text: str):
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ[/blue] {text}")
    else:
        print(f"ℹ {text}")


# =============================================================================
# PATTERN EXTRACTION (NLTK)
# =============================================================================

class PatternExtractor:
    """Extract complaint patterns using NLP techniques."""

    def __init__(self):
        self.stop_words = set()
        self._setup_nltk()

    def _setup_nltk(self):
        """Download required NLTK data."""
        if not NLTK_AVAILABLE:
            return

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                pass

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                             'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                             'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                             'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                             'through', 'during', 'before', 'after', 'above', 'below',
                             'between', 'under', 'again', 'further', 'then', 'once'}

    def extract_ngrams(self, texts: list[str], n: int = 3) -> Counter:
        """Extract n-grams from complaint texts."""
        if not NLTK_AVAILABLE:
            return self._simple_ngram_extraction(texts, n)

        all_ngrams = []
        for text in texts:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
                text_ngrams = list(ngrams(tokens, n))
                all_ngrams.extend([' '.join(ng) for ng in text_ngrams])
            except Exception:
                continue

        return Counter(all_ngrams)

    def _simple_ngram_extraction(self, texts: list[str], n: int = 3) -> Counter:
        """Simple ngram extraction without NLTK."""
        all_ngrams = []
        for text in texts:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            words = [w for w in words if len(w) > 2 and w not in self.stop_words]
            for i in range(len(words) - n + 1):
                all_ngrams.append(' '.join(words[i:i+n]))
        return Counter(all_ngrams)

    def find_complaint_patterns(self, texts: list[str]) -> dict:
        """Find complaint patterns in texts using predefined keywords."""
        patterns = defaultdict(list)

        for text in texts:
            text_lower = text.lower()
            for category, keywords in COMPLAINT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        # Extract context around the keyword
                        idx = text_lower.find(keyword)
                        start = max(0, idx - 50)
                        end = min(len(text), idx + len(keyword) + 100)
                        context = text[start:end].strip()
                        patterns[category].append({
                            'keyword': keyword,
                            'context': context,
                            'full_text': text
                        })

        return dict(patterns)

    def extract_key_phrases(self, texts: list[str], top_n: int = 20) -> list[tuple]:
        """Extract key complaint phrases."""
        # Combine bigrams and trigrams
        bigrams = self.extract_ngrams(texts, 2)
        trigrams = self.extract_ngrams(texts, 3)

        # Filter for complaint-related phrases
        complaint_indicators = {'not', 'no', 'cant', "can't", 'dont', "don't", 'doesnt',
                               "doesn't", 'missing', 'wish', 'hate', 'frustrating',
                               'annoying', 'difficult', 'hard', 'slow', 'expensive',
                               'broken', 'terrible', 'awful', 'worst', 'need', 'want'}

        filtered_phrases = []
        for phrase, count in {**bigrams, **trigrams}.items():
            words = set(phrase.split())
            if words & complaint_indicators:
                filtered_phrases.append((phrase, count))

        # Sort by count and return top N
        filtered_phrases.sort(key=lambda x: x[1], reverse=True)
        return filtered_phrases[:top_n]


# =============================================================================
# REDDIT SCRAPER
# =============================================================================

class RedditScraper:
    """Scrape complaints from Reddit using public JSON endpoints (no API key required)."""

    BASE_URL = "https://www.reddit.com"

    def __init__(self, topic: str):
        self.topic = topic
        self.session = create_session()
        # Reddit requires a descriptive User-Agent for JSON endpoints
        self.session.headers.update({
            "User-Agent": "StartupIdeaMiner/1.0 (complaint research tool)"
        })

    def get_subreddits(self) -> list[str]:
        """Get relevant subreddits for the topic."""
        topic_lower = self.topic.lower().replace(" ", "_")

        # Check exact match
        if topic_lower in TOPIC_SUBREDDITS:
            return TOPIC_SUBREDDITS[topic_lower]

        # Check partial matches
        for key, subs in TOPIC_SUBREDDITS.items():
            if key in topic_lower or topic_lower in key:
                return subs

        # Return defaults plus try topic as subreddit
        defaults = TOPIC_SUBREDDITS.get("_default", ["entrepreneur"])
        return defaults + [topic_lower]

    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a request to Reddit JSON endpoint with rate limiting."""
        try:
            # Reddit JSON endpoints require longer delays to avoid 429 errors
            time.sleep(SCRAPING_CONFIG.get("reddit_delay", 2.0))

            response = self.session.get(
                url,
                params=params,
                timeout=SCRAPING_CONFIG.get("timeout", 30)
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print_warning("Reddit rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(url, params)  # Retry once
            else:
                logger.debug(f"Reddit request failed: {response.status_code} for {url}")
                return None

        except Exception as e:
            logger.debug(f"Reddit request error: {e}")
            return None

    def search_subreddit(self, subreddit: str, query: str, limit: int = 25) -> list[dict]:
        """Search a subreddit using JSON endpoint."""
        url = f"{self.BASE_URL}/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "1",  # Restrict to subreddit
            "sort": "relevance",
            "limit": limit,
            "t": "year",  # Time filter: past year
        }

        data = self._make_request(url, params)
        if data and "data" in data and "children" in data["data"]:
            return [post["data"] for post in data["data"]["children"]]
        return []

    def get_hot_posts(self, subreddit: str, limit: int = 30) -> list[dict]:
        """Get hot posts from a subreddit using JSON endpoint."""
        url = f"{self.BASE_URL}/r/{subreddit}/hot.json"
        params = {"limit": limit}

        data = self._make_request(url, params)
        if data and "data" in data and "children" in data["data"]:
            return [post["data"] for post in data["data"]["children"]]
        return []

    def search_all_reddit(self, query: str, limit: int = 50) -> list[dict]:
        """Search all of Reddit using JSON endpoint."""
        url = f"{self.BASE_URL}/search.json"
        params = {
            "q": query,
            "sort": "relevance",
            "limit": limit,
            "t": "year",
        }

        data = self._make_request(url, params)
        if data and "data" in data and "children" in data["data"]:
            return [post["data"] for post in data["data"]["children"]]
        return []

    def search_complaints(self) -> list[Complaint]:
        """Search Reddit for complaints related to the topic."""
        complaints = []
        subreddits = self.get_subreddits()
        seen_ids = set()  # Avoid duplicates

        # Complaint search queries
        search_templates = [
            f"{self.topic} frustrating",
            f"{self.topic} hate when",
            f"{self.topic} wish someone would",
            f"{self.topic} doesn't have",
            f"{self.topic} missing feature",
            f"hate {self.topic}",
            f"{self.topic} problems",
            f"{self.topic} annoying",
        ]

        print_info(f"Searching Reddit (JSON API) in subreddits: {', '.join(subreddits[:5])}...")
        print_info("Note: Using public JSON endpoints - no API key required")

        # First, do a global Reddit search for the topic + complaint keywords
        print_info("  Performing global Reddit search...")
        for query in search_templates[:4]:
            posts = self.search_all_reddit(query, limit=25)
            for post_data in posts:
                if post_data.get("id") not in seen_ids:
                    seen_ids.add(post_data.get("id"))
                    complaint = self._process_post_data(post_data)
                    if complaint:
                        complaints.append(complaint)

        # Then search specific subreddits
        for subreddit_name in subreddits[:5]:  # Limit to top 5 subreddits
            print_info(f"  Searching r/{subreddit_name}...")

            # Search with complaint-related queries
            for query in search_templates[:3]:
                posts = self.search_subreddit(subreddit_name, query, limit=20)
                for post_data in posts:
                    if post_data.get("id") not in seen_ids:
                        seen_ids.add(post_data.get("id"))
                        complaint = self._process_post_data(post_data)
                        if complaint:
                            complaints.append(complaint)

            # Also get hot posts that might contain complaints
            hot_posts = self.get_hot_posts(subreddit_name, limit=25)
            for post_data in hot_posts:
                if post_data.get("id") not in seen_ids:
                    seen_ids.add(post_data.get("id"))
                    text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                    if self._contains_complaint_keywords(text):
                        complaint = self._process_post_data(post_data)
                        if complaint:
                            complaints.append(complaint)

        print_success(f"Found {len(complaints)} potential complaints from Reddit")
        return complaints

    def _process_post_data(self, post_data: dict) -> Optional[Complaint]:
        """Process a Reddit post JSON data into a Complaint."""
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        text = f"{title} {selftext}"

        if not self._contains_complaint_keywords(text):
            return None

        keywords = self._extract_keywords(text)
        permalink = post_data.get("permalink", "")
        created_utc = post_data.get("created_utc", 0)

        return Complaint(
            text=text[:2000],  # Limit text length
            source="reddit",
            topic=self.topic,
            url=f"https://reddit.com{permalink}" if permalink else "",
            upvotes=post_data.get("score", 0),
            date=datetime.fromtimestamp(created_utc).isoformat() if created_utc else "",
            keywords_found=keywords,
            category=self._categorize_complaint(text),
            product_name=f"r/{post_data.get('subreddit', 'unknown')}",
        )

    def _contains_complaint_keywords(self, text: str) -> bool:
        """Check if text contains complaint keywords."""
        text_lower = text.lower()
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    return True
        return False

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract complaint keywords found in text."""
        text_lower = text.lower()
        found = []
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
        return found

    def _categorize_complaint(self, text: str) -> str:
        """Categorize the complaint type."""
        text_lower = text.lower()
        for category, keywords in COMPLAINT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return "general"


# =============================================================================
# GOOGLE PLAY SCRAPER
# =============================================================================

class GooglePlayScraper:
    """Scrape complaints from Google Play Store reviews."""

    def __init__(self, topic: str):
        self.topic = topic

    def get_search_terms(self) -> list[str]:
        """Get app search terms for the topic."""
        topic_lower = self.topic.lower().replace(" ", "_")

        if topic_lower in TOPIC_APP_SEARCHES:
            return TOPIC_APP_SEARCHES[topic_lower]

        # Try partial match
        for key, searches in TOPIC_APP_SEARCHES.items():
            if key in topic_lower or topic_lower in key:
                return searches

        # Default: use topic as search term
        return [self.topic]

    def search_apps(self) -> list[dict]:
        """Search for apps related to the topic."""
        if not GPLAY_AVAILABLE:
            print_warning("google-play-scraper not available.")
            return []

        apps = []
        search_terms = self.get_search_terms()

        print_info(f"Searching Google Play for: {', '.join(search_terms)}")

        for term in search_terms:
            try:
                results = gplay_search(term, n_hits=10, lang='en', country='us')
                for app in results:
                    apps.append({
                        'app_id': app['appId'],
                        'name': app['title'],
                        'score': app.get('score', 0),
                        'search_term': term,
                    })
                time.sleep(SCRAPING_CONFIG["appstore_delay"])
            except Exception as e:
                logger.debug(f"Error searching for '{term}': {e}")
                continue

        # Remove duplicates by app_id
        seen = set()
        unique_apps = []
        for app in apps:
            if app['app_id'] not in seen:
                seen.add(app['app_id'])
                unique_apps.append(app)

        print_success(f"Found {len(unique_apps)} apps to analyze")
        return unique_apps

    def get_low_star_reviews(self, app_id: str, app_name: str) -> list[Complaint]:
        """Get 1-2 star reviews from an app."""
        if not GPLAY_AVAILABLE:
            return []

        complaints = []

        try:
            # Get reviews sorted by most relevant
            result, _ = gplay_reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
                count=SCRAPING_CONFIG["max_app_reviews"],
                filter_score_with=1,  # 1-star reviews
            )

            for review in result:
                complaint = Complaint(
                    text=review['content'][:2000],
                    source="google_play",
                    topic=self.topic,
                    product_name=app_name,
                    rating=review['score'],
                    date=review['at'].isoformat() if review.get('at') else "",
                    upvotes=review.get('thumbsUpCount', 0),
                    keywords_found=self._extract_keywords(review['content']),
                    category=self._categorize_complaint(review['content']),
                )
                complaints.append(complaint)

            # Also get 2-star reviews
            result_2star, _ = gplay_reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
                count=SCRAPING_CONFIG["max_app_reviews"] // 2,
                filter_score_with=2,
            )

            for review in result_2star:
                complaint = Complaint(
                    text=review['content'][:2000],
                    source="google_play",
                    topic=self.topic,
                    product_name=app_name,
                    rating=review['score'],
                    date=review['at'].isoformat() if review.get('at') else "",
                    upvotes=review.get('thumbsUpCount', 0),
                    keywords_found=self._extract_keywords(review['content']),
                    category=self._categorize_complaint(review['content']),
                )
                complaints.append(complaint)

            time.sleep(SCRAPING_CONFIG["appstore_delay"])

        except Exception as e:
            logger.debug(f"Error getting reviews for {app_id}: {e}")

        return complaints

    def scrape_all(self) -> list[Complaint]:
        """Scrape complaints from all relevant apps."""
        apps = self.search_apps()
        all_complaints = []

        for app in apps[:10]:  # Limit to top 10 apps
            print_info(f"  Analyzing: {app['name']}")
            complaints = self.get_low_star_reviews(app['app_id'], app['name'])
            all_complaints.extend(complaints)

        print_success(f"Collected {len(all_complaints)} reviews from Google Play")
        return all_complaints

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract complaint keywords from text."""
        text_lower = text.lower()
        found = []
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
        return found

    def _categorize_complaint(self, text: str) -> str:
        """Categorize the complaint."""
        text_lower = text.lower()
        for category, keywords in COMPLAINT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return "general"


# =============================================================================
# APP STORE SCRAPER (iOS)
# =============================================================================

class AppStoreScraper:
    """Scrape complaints from Apple App Store reviews."""

    def __init__(self, topic: str):
        self.topic = topic

    def search_apps(self) -> list[dict]:
        """Search for apps on App Store (using known apps for topic)."""
        # Note: app-store-scraper doesn't have a search function,
        # so we use predefined apps or Google Play search results
        topic_lower = self.topic.lower().replace(" ", "_")

        # Predefined popular apps by topic (app_name, app_id pairs)
        topic_apps = {
            "meditation": [
                ("Headspace", "493145008"),
                ("Calm", "571800810"),
                ("Insight Timer", "337472899"),
            ],
            "fitness": [
                ("MyFitnessPal", "341232718"),
                ("Nike Training Club", "301521403"),
                ("Strava", "426826309"),
            ],
            "finance": [
                ("Mint", "300238550"),
                ("YNAB", "1010865877"),
                ("Personal Capital", "504672168"),
            ],
            "productivity": [
                ("Notion", "1232780281"),
                ("Todoist", "585829637"),
                ("Things 3", "904237743"),
            ],
        }

        if topic_lower in topic_apps:
            return [{"name": name, "app_id": app_id} for name, app_id in topic_apps[topic_lower]]

        # For other topics, we'd need to use a different approach
        print_info(f"No predefined App Store apps for '{self.topic}'. Using Google Play data.")
        return []

    def get_low_star_reviews(self, app_name: str, app_id: str) -> list[Complaint]:
        """Get low-star reviews from App Store."""
        if not APPSTORE_AVAILABLE:
            return []

        complaints = []

        try:
            app = AppStore(country='us', app_name=app_name, app_id=app_id)
            app.review(how_many=SCRAPING_CONFIG["max_app_reviews"])

            for review in app.reviews:
                if review.get('rating', 5) <= 2:  # 1-2 star reviews
                    complaint = Complaint(
                        text=str(review.get('review', ''))[:2000],
                        source="app_store",
                        topic=self.topic,
                        product_name=app_name,
                        rating=review.get('rating', 0),
                        date=str(review.get('date', '')),
                        keywords_found=self._extract_keywords(str(review.get('review', ''))),
                        category=self._categorize_complaint(str(review.get('review', ''))),
                    )
                    complaints.append(complaint)

            time.sleep(SCRAPING_CONFIG["appstore_delay"])

        except Exception as e:
            logger.debug(f"Error getting App Store reviews for {app_name}: {e}")

        return complaints

    def scrape_all(self) -> list[Complaint]:
        """Scrape all App Store reviews for topic."""
        apps = self.search_apps()
        all_complaints = []

        for app in apps:
            print_info(f"  Analyzing App Store: {app['name']}")
            complaints = self.get_low_star_reviews(app['name'], app['app_id'])
            all_complaints.extend(complaints)

        print_success(f"Collected {len(all_complaints)} reviews from App Store")
        return all_complaints

    def _extract_keywords(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
        return found

    def _categorize_complaint(self, text: str) -> str:
        text_lower = text.lower()
        for category, keywords in COMPLAINT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return "general"


# =============================================================================
# G2/CAPTERRA SCRAPER
# =============================================================================

class G2Scraper:
    """Scrape B2B software reviews from G2."""

    def __init__(self, topic: str):
        self.topic = topic
        self.session = create_session()
        self.base_url = "https://www.g2.com"

    def get_products(self) -> list[dict]:
        """Get B2B products to analyze for the topic."""
        topic_lower = self.topic.lower().replace(" ", "_")

        if topic_lower in TOPIC_B2B_PRODUCTS:
            return TOPIC_B2B_PRODUCTS[topic_lower]

        # Try partial match
        for key, products in TOPIC_B2B_PRODUCTS.items():
            if key in topic_lower or topic_lower in key:
                return products

        print_info(f"No predefined G2 products for '{self.topic}'. Attempting search...")
        return self._search_products()

    def _search_products(self) -> list[dict]:
        """Search G2 for products (web scraping with caution)."""
        # Note: G2 may block scraping. This is a basic implementation.
        # For production, consider using G2's official API if available.
        products = []

        try:
            search_url = f"{self.base_url}/search?query={quote_plus(self.topic)}"
            response = self.session.get(search_url, timeout=SCRAPING_CONFIG["timeout"])

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                # G2's structure varies; this is a simplified extraction
                product_links = soup.select('a[href*="/products/"]')[:5]

                for link in product_links:
                    href = link.get('href', '')
                    name = link.get_text(strip=True)
                    if '/products/' in href and name:
                        slug = href.split('/products/')[-1].split('/')[0]
                        products.append({
                            'name': name,
                            'g2_slug': slug,
                        })

            time.sleep(SCRAPING_CONFIG["request_delay"])

        except Exception as e:
            logger.debug(f"G2 search error: {e}")

        return products

    def get_low_star_reviews(self, product: dict) -> list[Complaint]:
        """Scrape 1-2 star reviews from G2 product page."""
        complaints = []
        slug = product.get('g2_slug', '')

        if not slug:
            return complaints

        try:
            # G2 review URL pattern (may need adjustment)
            review_url = f"{self.base_url}/products/{slug}/reviews"

            # Add filter for low ratings if G2 supports it via URL params
            params = {'rating': '1,2'}  # This may not work; G2 uses JS filtering

            response = self.session.get(
                review_url,
                params=params,
                timeout=SCRAPING_CONFIG["timeout"]
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')

                # G2 review extraction (structure may vary)
                review_elements = soup.select('[class*="review"]')

                for elem in review_elements[:SCRAPING_CONFIG["max_g2_reviews"]]:
                    # Try to extract review text
                    text_elem = elem.select_one('[class*="review-content"], [class*="body"]')
                    rating_elem = elem.select_one('[class*="star"], [class*="rating"]')

                    if text_elem:
                        text = text_elem.get_text(strip=True)

                        # Try to determine rating (simplified)
                        rating = 0
                        if rating_elem:
                            rating_text = rating_elem.get('class', [])
                            # Attempt to extract rating from class names
                            for cls in rating_text:
                                if 'star' in cls.lower():
                                    try:
                                        rating = int(re.search(r'\d', cls).group())
                                    except (AttributeError, ValueError):
                                        pass

                        # Only include if it looks like a complaint
                        if self._contains_complaint_keywords(text):
                            complaint = Complaint(
                                text=text[:2000],
                                source="g2",
                                topic=self.topic,
                                product_name=product.get('name', ''),
                                rating=rating,
                                url=review_url,
                                keywords_found=self._extract_keywords(text),
                                category=self._categorize_complaint(text),
                            )
                            complaints.append(complaint)

            time.sleep(SCRAPING_CONFIG["request_delay"])

        except Exception as e:
            logger.debug(f"G2 review scraping error for {product.get('name')}: {e}")

        return complaints

    def scrape_all(self) -> list[Complaint]:
        """Scrape all G2 reviews for topic."""
        products = self.get_products()
        all_complaints = []

        print_info(f"Analyzing {len(products)} B2B products on G2")

        for product in products:
            print_info(f"  Checking G2: {product.get('name', 'Unknown')}")
            complaints = self.get_low_star_reviews(product)
            all_complaints.extend(complaints)

        print_success(f"Collected {len(all_complaints)} reviews from G2")
        return all_complaints

    def _contains_complaint_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    return True
        return False

    def _extract_keywords(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for keywords in COMPLAINT_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
        return found

    def _categorize_complaint(self, text: str) -> str:
        text_lower = text.lower()
        for category, keywords in COMPLAINT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        return "general"


# =============================================================================
# UPWORK SCRAPER
# =============================================================================

class UpworkScraper:
    """Analyze Upwork for repetitive job opportunities."""

    def __init__(self, topic: str):
        self.topic = topic
        self.session = create_session()

    def get_search_queries(self) -> list[str]:
        """Get Upwork search queries for the topic."""
        topic_lower = self.topic.lower().replace(" ", "_")

        if topic_lower in UPWORK_SEARCH_TEMPLATES:
            return UPWORK_SEARCH_TEMPLATES[topic_lower]

        # Use default templates with topic substitution
        defaults = UPWORK_SEARCH_TEMPLATES.get("_default", ["{topic} weekly"])
        return [q.format(topic=self.topic) for q in defaults]

    def search_jobs(self) -> list[Complaint]:
        """Search Upwork for repetitive jobs (automation opportunities).

        Note: Upwork heavily restricts scraping. This provides a framework
        for the search approach - in production you'd need Upwork API access
        or Selenium with careful rate limiting.
        """
        opportunities = []
        queries = self.get_search_queries()

        print_info(f"Searching Upwork for automation opportunities: {', '.join(queries[:3])}")
        print_warning("Note: Upwork scraping is restricted. Using search framework only.")

        # Keywords that indicate repetitive/recurring work (automation opportunities)
        recurring_keywords = [
            "weekly", "monthly", "ongoing", "repeat", "regular",
            "recurring", "continuous", "long-term", "daily", "hourly basis"
        ]

        # This is a placeholder - actual implementation would require
        # Upwork API access or careful Selenium automation
        for query in queries:
            # Create opportunity entries based on common patterns
            opportunity = Complaint(
                text=f"Search for '{query}' on Upwork - jobs matching this pattern "
                     f"indicate automation opportunities. Look for posts with keywords: "
                     f"{', '.join(recurring_keywords)}",
                source="upwork",
                topic=self.topic,
                keywords_found=recurring_keywords,
                category="automation_opportunity",
                url=f"https://www.upwork.com/search/jobs/?q={quote_plus(query)}",
            )
            opportunities.append(opportunity)

        print_success(f"Generated {len(opportunities)} Upwork search queries")
        return opportunities

    def scrape_all(self) -> list[Complaint]:
        """Get all Upwork opportunities."""
        return self.search_jobs()


# =============================================================================
# IDEA GENERATOR
# =============================================================================

class IdeaGenerator:
    """Generate startup ideas from complaint patterns."""

    def __init__(self, topic: str, complaints: list[Complaint]):
        self.topic = topic
        self.complaints = complaints
        self.pattern_extractor = PatternExtractor()

    def analyze_complaints(self) -> dict:
        """Analyze all complaints and extract patterns."""
        texts = [c.text for c in self.complaints]

        # Get key phrases
        key_phrases = self.pattern_extractor.extract_key_phrases(texts)

        # Get patterns by category
        patterns = self.pattern_extractor.find_complaint_patterns(texts)

        # Count categories
        category_counts = Counter(c.category for c in self.complaints)

        # Count sources
        source_counts = Counter(c.source for c in self.complaints)

        # Get most mentioned keywords
        all_keywords = []
        for c in self.complaints:
            all_keywords.extend(c.keywords_found)
        keyword_counts = Counter(all_keywords)

        return {
            'key_phrases': key_phrases,
            'patterns': patterns,
            'category_counts': dict(category_counts),
            'source_counts': dict(source_counts),
            'keyword_counts': dict(keyword_counts.most_common(20)),
            'total_complaints': len(self.complaints),
        }

    def generate_ideas(self, min_mentions: int = 5) -> list[StartupIdea]:
        """Generate startup ideas based on complaint patterns."""
        ideas = []
        analysis = self.analyze_complaints()

        # Group similar complaints
        complaint_groups = self._group_similar_complaints()

        for group_key, group_complaints in complaint_groups.items():
            if len(group_complaints) < min_mentions:
                continue

            # Generate idea from this complaint cluster
            idea = self._generate_idea_from_group(group_key, group_complaints)
            if idea:
                ideas.append(idea)

        # Sort by mention count
        ideas.sort(key=lambda x: x.mention_count, reverse=True)

        return ideas[:OUTPUT_CONFIG["max_ideas"]]

    def _group_similar_complaints(self) -> dict:
        """Group complaints by similarity/theme."""
        groups = defaultdict(list)

        for complaint in self.complaints:
            # Group by category + primary keyword
            if complaint.keywords_found:
                key = f"{complaint.category}:{complaint.keywords_found[0]}"
            else:
                key = complaint.category

            groups[key].append(complaint)

        return dict(groups)

    def _generate_idea_from_group(self, group_key: str, complaints: list[Complaint]) -> Optional[StartupIdea]:
        """Generate a startup idea from a group of similar complaints."""
        if not complaints:
            return None

        # Extract common themes from complaint texts
        texts = [c.text for c in complaints]
        common_phrases = self.pattern_extractor.extract_key_phrases(texts, top_n=5)

        # Determine pain point
        category = group_key.split(':')[0] if ':' in group_key else group_key
        pain_point = self._extract_pain_point(texts, category)

        # Generate title and description
        title, description = self._create_idea_content(category, pain_point, common_phrases)

        # Get sources involved
        sources = list(set(c.source for c in complaints))

        return StartupIdea(
            title=title,
            description=description,
            pain_point=pain_point,
            mention_count=len(complaints),
            sources=sources,
            target_audience=self._infer_target_audience(category),
            potential_revenue=self._estimate_potential(len(complaints)),
        )

    def _extract_pain_point(self, texts: list[str], category: str) -> str:
        """Extract the main pain point from complaint texts."""
        # Look for common patterns
        pain_indicators = {
            'missing': 'lack of essential feature',
            'wishes': 'unmet user need',
            'frustration': 'poor user experience',
            'complexity': 'overcomplicated workflow',
            'pricing': 'value-price mismatch',
            'performance': 'reliability/speed issues',
            'support': 'inadequate customer service',
        }

        base_pain = pain_indicators.get(category, 'user dissatisfaction')

        # Try to extract specific pain from first few texts
        sample_text = ' '.join(texts[:5]).lower()

        # Look for specific complaint phrases
        if 'offline' in sample_text:
            return f"{base_pain} - no offline functionality"
        elif 'integration' in sample_text:
            return f"{base_pain} - missing integrations"
        elif 'mobile' in sample_text or 'app' in sample_text:
            return f"{base_pain} - poor mobile experience"
        elif 'price' in sample_text or 'expensive' in sample_text:
            return f"{base_pain} - overpriced for value"
        elif 'slow' in sample_text or 'crash' in sample_text:
            return f"{base_pain} - performance problems"

        return base_pain

    def _create_idea_content(self, category: str, pain_point: str,
                             phrases: list[tuple]) -> tuple[str, str]:
        """Create idea title and description."""
        # Build descriptive title
        phrase_text = phrases[0][0] if phrases else category
        title = f"{self.topic.title()} Solution: Address '{phrase_text}'"

        # Build description
        if category == 'missing':
            description = (f"Build a focused {self.topic} tool that includes the features "
                          f"users are desperately asking for. Key opportunity: {pain_point}.")
        elif category == 'complexity':
            description = (f"Create a simpler, more intuitive {self.topic} solution for "
                          f"users frustrated with existing complex tools. Focus on doing "
                          f"one thing well.")
        elif category == 'pricing':
            description = (f"Launch an affordable {self.topic} alternative with "
                          f"transparent pricing. Target users priced out of existing solutions.")
        elif category == 'performance':
            description = (f"Build a fast, reliable {self.topic} tool that prioritizes "
                          f"stability and speed over feature bloat.")
        elif category == 'wishes':
            description = (f"Develop a {self.topic} product that fulfills unmet user desires. "
                          f"Key feature opportunity: {pain_point}.")
        else:
            description = (f"Address the core {self.topic} pain point: {pain_point}. "
                          f"Build what existing solutions are failing to deliver.")

        return title, description

    def _infer_target_audience(self, category: str) -> str:
        """Infer target audience from topic and category."""
        topic_audiences = {
            'crypto': 'cryptocurrency traders and investors',
            'finance': 'personal finance enthusiasts and small businesses',
            'meditation': 'mindfulness practitioners and stress-management seekers',
            'fitness': 'fitness enthusiasts and gym-goers',
            'golf': 'golfers looking to improve their game',
            'productivity': 'knowledge workers and teams',
            'podcast': 'podcast creators and listeners',
            'marketing': 'marketers and small business owners',
            'real_estate': 'real estate professionals and home buyers',
        }

        topic_lower = self.topic.lower()
        for key, audience in topic_audiences.items():
            if key in topic_lower:
                return audience

        return f'{self.topic} users and enthusiasts'

    def _estimate_potential(self, mention_count: int) -> str:
        """Estimate revenue potential based on mention frequency."""
        if mention_count >= 100:
            return "High potential ($50k+ MRR achievable)"
        elif mention_count >= 50:
            return "Strong potential ($10-50k MRR)"
        elif mention_count >= 20:
            return "Moderate potential ($1-10k MRR)"
        else:
            return "Early signal - validate further"


# =============================================================================
# MAIN COMPLAINT MINER CLASS
# =============================================================================

class ComplaintMiner:
    """Main class orchestrating the complaint mining process."""

    def __init__(self, topic: str, sources: list[str], min_mentions: int = 5):
        self.topic = topic
        self.sources = sources
        self.min_mentions = min_mentions
        self.complaints: list[Complaint] = []
        self.ideas: list[StartupIdea] = []

    def run(self) -> tuple[pd.DataFrame, list[StartupIdea]]:
        """Run the complete complaint mining pipeline."""
        print_header(f"🔍 Mining Complaints for: {self.topic.upper()}")
        print_info(f"Sources: {', '.join(self.sources)}")
        print_info(f"Minimum mentions threshold: {self.min_mentions}")
        print("")

        # Collect complaints from each source
        for source in self.sources:
            self._collect_from_source(source)

        if not self.complaints:
            print_warning("No complaints found. Try adjusting your topic or sources.")
            return pd.DataFrame(), []

        # Create DataFrame
        df = self._create_dataframe()

        # Generate ideas
        print_header("💡 Generating Startup Ideas")
        generator = IdeaGenerator(self.topic, self.complaints)
        self.ideas = generator.generate_ideas(self.min_mentions)

        return df, self.ideas

    def _collect_from_source(self, source: str):
        """Collect complaints from a specific source."""
        source = source.lower().strip()

        print_header(f"📥 Collecting from: {source.upper()}")

        try:
            if source == 'reddit':
                scraper = RedditScraper(self.topic)
                complaints = scraper.search_complaints()

            elif source == 'google_play':
                scraper = GooglePlayScraper(self.topic)
                complaints = scraper.scrape_all()

            elif source == 'app_store' or source == 'appstore':
                scraper = AppStoreScraper(self.topic)
                complaints = scraper.scrape_all()

            elif source == 'g2':
                scraper = G2Scraper(self.topic)
                complaints = scraper.scrape_all()

            elif source == 'upwork':
                scraper = UpworkScraper(self.topic)
                complaints = scraper.scrape_all()

            elif source == 'all':
                # Run all scrapers
                for sub_source in ['reddit', 'google_play', 'g2', 'upwork']:
                    self._collect_from_source(sub_source)
                return

            else:
                print_warning(f"Unknown source: {source}")
                return

            self.complaints.extend(complaints)
            print_success(f"Added {len(complaints)} complaints from {source}")

        except Exception as e:
            print_error(f"Error collecting from {source}: {e}")
            logger.exception(f"Source collection error: {source}")

    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from complaints."""
        data = []
        for c in self.complaints:
            data.append({
                'complaint_text': c.text,
                'source': c.source,
                'topic': c.topic,
                'product_name': c.product_name,
                'rating': c.rating,
                'date': c.date,
                'upvotes': c.upvotes,
                'category': c.category,
                'keywords_found': ', '.join(c.keywords_found),
                'url': c.url,
            })

        return pd.DataFrame(data)


# =============================================================================
# OUTPUT AND REPORTING
# =============================================================================

def print_summary(df: pd.DataFrame, ideas: list[StartupIdea], topic: str):
    """Print summary of findings."""
    print_header("📊 RESULTS SUMMARY")

    if df.empty:
        print_warning("No data collected.")
        return

    # Basic stats
    print(f"\nTopic: {topic}")
    print(f"Total complaints collected: {len(df)}")
    print(f"Sources analyzed: {df['source'].nunique()}")
    print(f"Products mentioned: {df['product_name'].nunique()}")

    # Source breakdown
    print("\n📈 Complaints by Source:")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  • {source}: {count}")

    # Category breakdown
    print("\n🏷️ Complaint Categories:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.head(10).items():
        print(f"  • {category}: {count}")

    # Top keywords
    if 'keywords_found' in df.columns:
        all_keywords = []
        for kw_str in df['keywords_found'].dropna():
            all_keywords.extend([k.strip() for k in kw_str.split(',') if k.strip()])

        if all_keywords:
            print("\n🔑 Top Pain Point Keywords:")
            keyword_counts = Counter(all_keywords).most_common(10)
            for keyword, count in keyword_counts:
                print(f"  • \"{keyword}\": {count} mentions")

    # Generated ideas
    if ideas:
        print_header("💡 GENERATED STARTUP IDEAS")
        for i, idea in enumerate(ideas, 1):
            print(f"\n{'='*60}")
            print(f"IDEA #{i}: {idea.title}")
            print(f"{'='*60}")
            print(f"Pain Point: {idea.pain_point}")
            print(f"Mentions: {idea.mention_count}")
            print(f"Sources: {', '.join(idea.sources)}")
            print(f"Target Audience: {idea.target_audience}")
            print(f"Potential: {idea.potential_revenue}")
            print(f"\nDescription:\n{idea.description}")


def export_results(df: pd.DataFrame, ideas: list[StartupIdea], topic: str, output_dir: str):
    """Export results to files."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = re.sub(r'[^a-zA-Z0-9]+', '_', topic.lower())

    # Export complaints CSV
    if OUTPUT_CONFIG.get("export_csv", True) and not df.empty:
        csv_path = output_path / f"complaints_{topic_slug}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print_success(f"Exported complaints to: {csv_path}")

    # Export ideas JSON
    if OUTPUT_CONFIG.get("export_json", True) and ideas:
        ideas_data = []
        for idea in ideas:
            ideas_data.append({
                'title': idea.title,
                'description': idea.description,
                'pain_point': idea.pain_point,
                'mention_count': idea.mention_count,
                'sources': idea.sources,
                'target_audience': idea.target_audience,
                'potential_revenue': idea.potential_revenue,
            })

        json_path = output_path / f"ideas_{topic_slug}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(ideas_data, f, indent=2)
        print_success(f"Exported ideas to: {json_path}")

    # Export full analysis JSON
    analysis_path = output_path / f"analysis_{topic_slug}_{timestamp}.json"
    analysis_data = {
        'topic': topic,
        'timestamp': timestamp,
        'total_complaints': len(df),
        'source_breakdown': df['source'].value_counts().to_dict() if not df.empty else {},
        'category_breakdown': df['category'].value_counts().to_dict() if not df.empty else {},
        'ideas_count': len(ideas),
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print_success(f"Exported analysis to: {analysis_path}")


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Startup Idea Discovery Tool - Mine user complaints to find product opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complaint_miner.py --topic "golf" --sources reddit,google_play
  python complaint_miner.py --topic "crypto" --sources all --min_mentions 20
  python complaint_miner.py --topic "meditation apps" --sources reddit,app_store
  python complaint_miner.py --topic "CRM software" --sources g2,reddit

Available sources:
  reddit       - Reddit posts and comments (requires API credentials)
  google_play  - Google Play Store 1-2 star reviews
  app_store    - Apple App Store 1-2 star reviews
  g2           - G2 B2B software reviews
  upwork       - Upwork job posts (automation opportunities)
  all          - All available sources

Tips:
  - Set Reddit API credentials in config.py or environment variables
  - Use specific topics like "golf swing analysis" for better results
  - Higher --min_mentions values produce more validated ideas
        """
    )

    parser.add_argument(
        '--topic', '-t',
        type=str,
        required=True,
        help='Target topic/niche to analyze (e.g., "golf", "crypto", "meditation apps")'
    )

    parser.add_argument(
        '--sources', '-s',
        type=str,
        default='reddit,google_play',
        help='Comma-separated list of sources to scrape (default: reddit,google_play)'
    )

    parser.add_argument(
        '--min_mentions', '-m',
        type=int,
        default=OUTPUT_CONFIG.get("min_mentions_threshold", 5),
        help=f'Minimum mentions to consider significant (default: {OUTPUT_CONFIG.get("min_mentions_threshold", 5)})'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_CONFIG.get("output_dir", "output"),
        help='Output directory for exported files (default: output)'
    )

    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip exporting results to files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--list-topics',
        action='store_true',
        help='List preconfigured topics with subreddit/app mappings'
    )

    return parser.parse_args()


def list_preconfigured_topics():
    """List all preconfigured topics."""
    print_header("📋 Preconfigured Topics")

    print("\n🔴 Reddit Subreddit Mappings:")
    for topic, subreddits in TOPIC_SUBREDDITS.items():
        if topic != '_default':
            print(f"  • {topic}: r/{', r/'.join(subreddits[:3])}...")

    print("\n📱 App Store Search Terms:")
    for topic, searches in TOPIC_APP_SEARCHES.items():
        print(f"  • {topic}: {', '.join(searches[:2])}...")

    print("\n🏢 B2B Products (G2):")
    for topic, products in TOPIC_B2B_PRODUCTS.items():
        names = [p['name'] for p in products[:3]]
        print(f"  • {topic}: {', '.join(names)}...")

    print("\nNote: You can use any topic - these are just optimized presets.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_topics:
        list_preconfigured_topics()
        return

    # Parse sources
    sources = [s.strip() for s in args.sources.split(',')]

    # Run the miner
    miner = ComplaintMiner(
        topic=args.topic,
        sources=sources,
        min_mentions=args.min_mentions
    )

    df, ideas = miner.run()

    # Print summary
    print_summary(df, ideas, args.topic)

    # Export results
    if not args.no_export:
        print_header("📁 Exporting Results")
        export_results(df, ideas, args.topic, args.output)

    print_header("✅ COMPLETE")
    print(f"\nTotal complaints analyzed: {len(df)}")
    print(f"Startup ideas generated: {len(ideas)}")

    if ideas:
        print("\n🚀 Next Steps:")
        print("  1. Review generated ideas and select top candidates")
        print("  2. Validate with 10 potential customers (Week 1)")
        print("  3. Build MVP for the most promising idea (Week 2)")
        print("  4. Launch to complainers from your data (Week 3)")
        print("  5. Iterate based on feedback (Week 4)")
        print("\n  Remember: Speed is everything. Others see this data too!")


if __name__ == "__main__":
    main()
