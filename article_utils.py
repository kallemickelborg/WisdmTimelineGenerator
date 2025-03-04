"""
Common utility functions for article processing.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from utils import logger

# Domain type patterns for classification
DOMAIN_PATTERNS = {
    "left_leaning": [
        "nytimes.com",
        "cnn.com",
        "msnbc.com",
        "washingtonpost.com",
        "huffpost.com",
        "vox.com",
        "slate.com",
        "theatlantic.com",
        "thedailybeast.com",
        "motherjones.com",
        "buzzfeednews.com",
        "nbcnews.com",
        "theintercept.com",
        "democracynow.org",
        "salon.com",
        "vanityfair.com",
        "npr.org",
        "theguardian.com",
        "politico.com",
        "axios.com",
        "vice.com",
    ],
    "right_leaning": [
        "foxnews.com",
        "nypost.com",
        "breitbart.com",
        "dailycaller.com",
        "washingtontimes.com",
        "nationalreview.com",
        "theblaze.com",
        "townhall.com",
        "dailywire.com",
        "thefederalist.com",
        "newsmax.com",
        "oann.com",
        "redstate.com",
        "washingtonexaminer.com",
        "theamericanconservative.com",
        "spectator.org",
        "reason.com",
        "freebeacon.com",
        "realclearpolitics.com",
        "dailymail.co.uk",
    ],
    "center": [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "wsj.com",
        "thehill.com",
        "bloomberg.com",
        "usatoday.com",
        "c-span.org",
        "economist.com",
        "financialtimes.com",
        "foreignpolicy.com",
        "csmonitor.com",
        "marketwatch.com",
        "independent.co.uk",
        "euronews.com",
        "foreignaffairs.com",
        "lawfareblog.com",
        "upi.com",
        "afp.com",
    ],
}


def calculate_content_similarity(text1: str, text2: str) -> float:
    """Calculate content similarity between two texts using TF-IDF and cosine similarity."""
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except Exception as e:
        logger.warning(f"Error calculating content similarity: {str(e)}")
        return 0.0


def is_date_close(date1: str, date2: str, max_days_diff: int = 3) -> bool:
    """Check if two dates are within a specified number of days of each other."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return abs((d1 - d2).days) <= max_days_diff
    except Exception as e:
        logger.warning(f"Error comparing dates: {str(e)}")
        return False


def infer_domain_type(domain: str) -> str:
    """Infer the type of domain (left_leaning, right_leaning, center)."""
    for domain_type, patterns in DOMAIN_PATTERNS.items():
        if any(pattern in domain.lower() for pattern in patterns):
            return domain_type
    return "unknown"


def calculate_political_agreement(article: Dict) -> float:
    """Calculate a political agreement score for an article (-1.0 to 1.0)."""
    try:
        # Extract domain from URL
        url = article.get("url", "")
        if not url:
            return 0.0  # Neutral if no URL

        # Remove protocol and www prefix to get domain
        domain = url.split("//")[-1].split("/")[0]
        domain = domain.replace("www.", "")

        # Determine domain type
        domain_type = infer_domain_type(domain)

        # Return score based on domain type
        if domain_type == "left_leaning":
            return -0.7  # Left-leaning
        elif domain_type == "right_leaning":
            return 0.7  # Right-leaning
        elif domain_type == "center":
            return 0.0  # Center/neutral
        else:
            return 0.0  # Unknown, assume neutral
    except Exception as e:
        logger.warning(f"Error calculating political agreement: {str(e)}")
        return 0.0  # Default to neutral


def process_article(article: Dict) -> Optional[Dict]:
    """Process a raw article from GNews API into a standardized format."""
    try:
        # Validate required fields
        if not article.get("title"):
            logger.warning("Article missing required field: title")
            return None

        if not article.get("source") or not article["source"].get("name"):
            logger.warning("Article missing required field: source name")
            return None

        if not article.get("publishedAt"):
            logger.warning("Article missing required field: publication date")
            return None

        # Extract source URL to infer domain type
        source_url = article.get("url", "")
        domain = (
            source_url.split("//")[-1].split("/")[0].replace("www.", "")
            if source_url
            else ""
        )
        domain_type = infer_domain_type(domain)

        # Calculate political agreement
        political_agreement = calculate_political_agreement(article)

        # Extract content (or use description if no content)
        content = article.get("content", "")
        if not content or len(content) < 50:  # If content is too short or missing
            content = article.get("description", "")

        # Return processed article
        return {
            "title": article.get("title", ""),
            "source_name": article["source"].get("name", "Unknown Source"),
            "pubDate": article.get("publishedAt", datetime.now().isoformat()),
            "url": source_url,
            "content": content,
            "domain_type": domain_type,
            "political_agreement": political_agreement,
            "image": article.get("image"),
        }
    except Exception as e:
        logger.warning(f"Error processing article: {str(e)}")
        return None
