"""
Common utility functions for timeline generation and updates.
"""

import hashlib
import logging
import os
from typing import Dict, Any, Set
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def generate_topic_hash(topic_statement: str) -> str:
    """Generate a consistent hash for a topic statement."""
    return hashlib.md5(topic_statement.strip().lower().encode()).hexdigest()[:10]


def get_processed_articles_filename(topic_statement: str) -> str:
    """
    Generate a consistent filename for storing processed articles based on topic.
    """
    topic_hash = generate_topic_hash(topic_statement)
    return f"processed_articles_{topic_hash}.json"


def load_processed_articles(filename: str) -> Dict[str, Set[str]]:
    """Load previously processed articles from a JSON file."""
    if not os.path.exists(filename):
        return {"processed_urls": set(), "processed_titles": set()}

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "processed_urls": set(data.get("processed_urls", [])),
                "processed_titles": set(data.get("processed_titles", [])),
            }
    except Exception as e:
        logger.warning(f"Error loading processed articles: {str(e)}")
        return {"processed_urls": set(), "processed_titles": set()}


def save_processed_articles(
    processed_articles: Dict[str, Set[str]], filename: str
) -> None:
    """Save processed articles to a JSON file."""
    try:
        data = {
            "processed_urls": list(processed_articles["processed_urls"]),
            "processed_titles": list(processed_articles["processed_titles"]),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(
            f"Saved {len(processed_articles['processed_urls'])} processed articles to {filename}"
        )
    except Exception as e:
        logger.error(f"Error saving processed articles: {str(e)}")
