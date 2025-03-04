import os
import re
import dspy
import logging
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode
import time
import json
import hashlib
from utils import (
    logger,
    generate_topic_hash,
    get_processed_articles_filename,
    load_processed_articles,
    save_processed_articles,
)
from article_utils import (
    DOMAIN_PATTERNS,
    calculate_content_similarity,
    is_date_close,
    infer_domain_type,
    calculate_political_agreement,
    process_article,
)
from models import Source, Perspective, Event, Timeline
from dspy_signatures import QueryGenerator, GenerateTimeline, PerspectiveSynthesis

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )
    raise EnvironmentError("OpenAI API key not found.")

gpt4o_mini = dspy.LM("openai/gpt-4o-mini", max_tokens=8192, api_key=openai_api_key)
dspy.configure(lm=gpt4o_mini)

# ============================================================================
# GNews API Integration
# ============================================================================


def generate_topic_hash(topic_statement: str) -> str:
    """Generate a consistent hash for a topic statement."""
    return hashlib.md5(topic_statement.strip().lower().encode()).hexdigest()[:10]


def get_processed_articles_filename(topic_statement: str) -> str:
    """
    Generate a consistent filename for storing processed articles based on topic.
    This function should be identical to the one in updateTimelineEvents.py
    """
    topic_hash = generate_topic_hash(topic_statement)
    return f"processed_articles_{topic_hash}.json"


class GNewsAPI:
    """Handler for GNews API interactions."""

    BASE_URL = "https://gnews.io/api/v4"
    VALID_CATEGORIES = {
        "general",
        "world",
        "nation",
        "business",
        "technology",
        "entertainment",
        "sports",
        "science",
        "health",
    }

    def __init__(self):
        """Initialize the GNews API client."""
        self.api_key = os.getenv("GNEWS_API_KEY")
        if not self.api_key:
            logger.error(
                "GNews API key not found. Please set the GNEWS_API_KEY environment variable."
            )
            raise EnvironmentError("GNews API key not found.")

        self.headers = {"User-Agent": "TimelineGenerator/1.0"}
        self.processed_articles = {"processed_urls": set(), "processed_titles": set()}
        self.processed_articles_file = None

        logger.info("GNewsAPI initialized successfully")

    def _infer_domain_type(self, domain: str) -> str:
        """Proxy to article_utils.infer_domain_type."""
        return infer_domain_type(domain)

    def _calculate_political_agreement(self, article: Dict) -> float:
        """Proxy to article_utils.calculate_political_agreement."""
        return calculate_political_agreement(article)

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Proxy to article_utils.calculate_content_similarity."""
        return calculate_content_similarity(text1, text2)

    def _initialize_article_tracking(self, topic_statement: str):
        """Initialize tracking for processed articles for a specific topic."""
        self.processed_articles_file = get_processed_articles_filename(topic_statement)
        self.processed_articles = load_processed_articles(self.processed_articles_file)
        logger.debug(
            f"Initialized article tracking for topic: {topic_statement[:30]}... - "
            f"Found {len(self.processed_articles['processed_urls'])} processed URLs"
        )

    def _save_processed_articles(self):
        """Save the set of processed articles to disk."""
        save_processed_articles(self.processed_articles, self.processed_articles_file)

    def _load_processed_articles(self) -> Dict[str, Set[str]]:
        """Load previously processed articles from a JSON file."""
        return load_processed_articles(self.processed_articles_file)

    def _is_article_processed(self, article: Dict) -> bool:
        """Check if an article has already been processed in previous runs."""
        article_url = article.get("url", "")
        article_title = article.get("title", "")

        if article_url and article_url in self.processed_articles["processed_urls"]:
            return True

        if (
            article_title
            and article_title in self.processed_articles["processed_titles"]
        ):
            return True

        return False

    def _mark_article_processed(self, article: Dict):
        """Mark an article as processed for this topic."""
        if article.get("url"):
            self.processed_articles["processed_urls"].add(article["url"])
        if article.get("title"):
            self.processed_articles["processed_titles"].add(article["title"])

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """Make a request to the GNews API."""
        params["apikey"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            params = {k: v for k, v in params.items() if v is not None}

            safe_params = params.copy()
            if "apikey" in safe_params:
                api_key = safe_params["apikey"]
                safe_params["apikey"] = f"{api_key[:4]}...{api_key[-4:]}"
            query_url = f"{url}?{urlencode(safe_params)}"
            logger.info(f"Making API request to: {query_url}")

            response = requests.get(
                url, params=params, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GNews API request failed: {str(e)}")
            logger.error(f"Failed URL: {url}?{urlencode(params)}")
            raise

    def generate_query(
        self, topic_statement: str, background_info: str
    ) -> Dict[str, str]:
        """Generate optimized search query parameters for GNews API."""
        self._initialize_article_tracking(topic_statement)

        try:
            main_terms = []

            full_context = f"{topic_statement} {background_info}"

            if '"' in topic_statement:
                quoted_terms = [
                    term for term in topic_statement.split('"') if term.strip()
                ]
                main_terms.extend(
                    [f'"{term}"' for term in quoted_terms if len(term.split()) > 1]
                )

            key_terms = topic_statement.replace('"', "").split()
            key_terms = [
                term
                for term in key_terms
                if term.lower()
                not in {
                    "a",
                    "an",
                    "the",
                    "in",
                    "on",
                    "at",
                    "by",
                    "for",
                    "to",
                    "of",
                    "with",
                    "and",
                }
            ]

            for term in key_terms:
                if term[0].isupper() or term.isupper():
                    main_terms.append(f'"{term}"')

            full_context_lower = full_context.lower()

            term_counts = {}

            domain_indicators = {
                "politics": [
                    "election",
                    "vote",
                    "president",
                    "political",
                    "government",
                    "policy",
                    "congress",
                ],
                "economy": [
                    "economy",
                    "economic",
                    "recession",
                    "inflation",
                    "market",
                    "stock",
                    "financial",
                ],
                "trade": [
                    "tariff",
                    "trade",
                    "import",
                    "export",
                    "trade war",
                    "trade agreement",
                    "trade deficit",
                ],
                "military": [
                    "war",
                    "military",
                    "troops",
                    "defense",
                    "attack",
                    "invasion",
                    "conflict",
                    "army",
                ],
                "technology": [
                    "tech",
                    "technology",
                    "digital",
                    "software",
                    "hardware",
                    "AI",
                    "innovation",
                ],
                "health": [
                    "health",
                    "disease",
                    "virus",
                    "pandemic",
                    "medical",
                    "treatment",
                    "vaccine",
                ],
                "climate": [
                    "climate",
                    "environment",
                    "carbon",
                    "emission",
                    "warming",
                    "pollution",
                    "sustainable",
                ],
                "education": [
                    "education",
                    "school",
                    "university",
                    "student",
                    "learning",
                    "teaching",
                    "academic",
                ],
                "social": [
                    "social",
                    "society",
                    "community",
                    "rights",
                    "equality",
                    "discrimination",
                    "justice",
                ],
                "culture": [
                    "culture",
                    "entertainment",
                    "media",
                    "art",
                    "music",
                    "film",
                    "book",
                ],
                "sports": [
                    "sports",
                    "game",
                    "team",
                    "athlete",
                    "championship",
                    "tournament",
                    "olympics",
                ],
            }

            domain_relevance = {domain: 0 for domain in domain_indicators}

            for domain, indicators in domain_indicators.items():
                for indicator in indicators:
                    if indicator in full_context_lower:
                        count = full_context_lower.count(indicator)
                        domain_relevance[domain] += count
                        term_counts[indicator] = count

            relevant_domains = sorted(
                domain_relevance.items(), key=lambda x: x[1], reverse=True
            )[:2]
            relevant_domains = [
                domain for domain, score in relevant_domains if score > 0
            ]

            logger.info(f"Detected relevant domains: {relevant_domains}")

            important_domain_terms = []
            for domain in relevant_domains:
                domain_terms = sorted(
                    [
                        (term, count)
                        for term, count in term_counts.items()
                        if term in domain_indicators[domain]
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                important_domain_terms.extend([term for term, _ in domain_terms[:2]])

            for term in important_domain_terms:
                if " " in term:
                    main_terms.append(f'"{term}"')
                else:
                    main_terms.append(term)

            entity_pattern = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b"
            import re

            potential_entities = re.findall(entity_pattern, full_context)

            entity_counts = {}
            for entity in potential_entities:
                if len(entity.split()) > 1:
                    entity_counts[entity] = full_context.lower().count(entity.lower())

            top_entities = sorted(
                entity_counts.items(), key=lambda x: x[1], reverse=True
            )[:2]
            for entity, _ in top_entities:
                if entity not in [term.strip('"') for term in main_terms]:
                    main_terms.append(f'"{entity}"')

            if len(main_terms) < 2:
                remaining_terms = [
                    term for term in key_terms if f'"{term}"' not in main_terms
                ]
                main_terms.extend(remaining_terms[:2])

            main_terms = main_terms[:4]
            search_query = " AND ".join(main_terms)

            logger.info(f"Generated search query: {search_query}")

            query_params = {
                "q": search_query,
                "lang": "en",
                "country": "us",
                "max": "10",
                "sortby": "relevance",
                "expand": "content",
            }

            return query_params

        except Exception as e:
            logger.error(f"Error generating query parameters: {str(e)}")
            terms = [term for term in topic_statement.split() if term[0].isupper()]
            if not terms:
                terms = topic_statement.split()[:3]
            search_query = " AND ".join(f'"{term}"' for term in terms)
            return {
                "q": search_query,
                "lang": "en",
                "country": "us",
                "max": "10",
                "sortby": "relevance",
                "expand": "content",
            }

    def fetch_diverse_articles(
        self, query_params: Dict[str, str], min_articles: int = 20
    ) -> List[Dict]:
        """Fetch diverse articles from GNews API with pagination."""
        try:
            collected_articles = []

            articles_per_page = 10
            total_pages = min(20, (min_articles // articles_per_page) + 1)

            for page in range(1, total_pages + 1):
                current_params = query_params.copy()
                current_params["page"] = str(page)

                response = self._make_request("search", current_params)
                total_articles = response.get("totalArticles", 0)
                logger.info(f"Total articles available: {total_articles}")
                logger.info(f"Fetching page {page} of {total_pages}")

                articles = response.get("articles", [])
                if not articles:
                    logger.warning(f"No articles found in response for page {page}")
                    break

                logger.info(
                    f"Received {len(articles)} articles from API on page {page}"
                )

                for article in articles:
                    try:
                        if self._is_article_processed(article):
                            logger.info(
                                f"Skipping previously processed article: {article.get('title', '')[:50]}..."
                            )
                            continue

                        if not article.get("title"):
                            logger.warning("Skipping article without title")
                            continue

                        if not article.get("source", {}).get("name"):
                            logger.warning("Skipping article without source name")
                            continue

                        if not article.get("publishedAt"):
                            logger.warning("Skipping article without publication date")
                            continue

                        if not self._is_duplicate(article, collected_articles):
                            source_url = article.get("source", {}).get("url", "")
                            article["domain_type"] = self._infer_domain_type(source_url)

                            article["political_agreement"] = (
                                self._calculate_political_agreement(article)
                            )

                            content = article.get("content", "")
                            if not content or content == "null":
                                content = article.get("description", "")

                            processed_article = {
                                "title": article["title"],
                                "description": article.get("description", ""),
                                "content": content,
                                "source_name": article["source"]["name"],
                                "source_url": article["source"]["url"],
                                "url": article["url"],
                                "pubDate": article["publishedAt"],
                                "domain_type": article["domain_type"],
                                "political_agreement": article["political_agreement"],
                            }

                            collected_articles.append(processed_article)
                            self._mark_article_processed(processed_article)
                            logger.debug(
                                f"Added article: {processed_article['title'][:50]}..."
                            )

                    except Exception as e:
                        logger.warning(f"Error processing article: {str(e)}")
                        continue

                if page < total_pages:
                    time.sleep(1.5)

                if len(collected_articles) >= min_articles:
                    logger.info(f"Reached minimum article count ({min_articles})")
                    break

            logger.info(f"Total articles collected: {len(collected_articles)}")

            self._save_processed_articles()

            return collected_articles

        except Exception as e:
            logger.error(f"Error fetching articles: {str(e)}")
            return []

    def _is_duplicate(self, article: Dict, existing_articles: List[Dict]) -> bool:
        """Check if an article is a duplicate based on content similarity."""
        try:
            new_content = " ".join(
                filter(
                    None,
                    [
                        str(article.get("description", "")),
                        str(article.get("content", "")),
                        str(article.get("title", "")),
                    ],
                )
            ).lower()

            if not new_content:
                return False

            for existing in existing_articles:
                try:
                    existing_content = " ".join(
                        filter(
                            None,
                            [
                                str(existing.get("description", "")),
                                str(existing.get("content", "")),
                                str(existing.get("title", "")),
                            ],
                        )
                    ).lower()

                    if (
                        existing_content
                        and self._calculate_content_similarity(
                            new_content, existing_content
                        )
                        > 0.8
                    ):
                        return True
                except Exception as e:
                    logger.warning(f"Error comparing with existing article: {str(e)}")
                    continue

            return False
        except Exception as e:
            logger.warning(f"Error in duplicate detection: {str(e)}")
            return False

    def _is_article_relevant(
        self, article: Dict, topic_statement: str, background_info: str
    ) -> bool:
        """
        Check if an article is relevant to the specific topic we're interested in.
        This helps filter out articles that mention the search terms but aren't about the topic.
        """
        try:
            article_text = " ".join(
                filter(
                    None,
                    [
                        str(article.get("title", "")).lower(),
                        str(article.get("description", "")).lower(),
                        str(article.get("content", "")).lower(),
                    ],
                )
            )

            if len(article_text) < 50:
                return False

            topic_lower = topic_statement.lower()
            background_lower = background_info.lower()
            combined_context = f"{topic_lower} {background_lower}"

            key_terms = set()
            for text in [topic_lower, background_lower]:
                words = text.replace(",", " ").replace(".", " ").split()
                # Single word terms
                key_terms.update(
                    [
                        w
                        for w in words
                        if len(w) > 3
                        and w
                        not in {
                            "this",
                            "that",
                            "with",
                            "from",
                            "have",
                            "been",
                            "were",
                            "they",
                            "their",
                            "what",
                            "when",
                            "where",
                        }
                    ]
                )

                for i in range(len(words) - 1):
                    if len(words[i]) > 2 and len(words[i + 1]) > 2:
                        key_terms.add(f"{words[i]} {words[i+1]}")

            required_terms = []

            domain_indicators = {
                "politics": [
                    "election",
                    "vote",
                    "president",
                    "political",
                    "government",
                    "policy",
                    "congress",
                ],
                "economy": [
                    "economy",
                    "economic",
                    "recession",
                    "inflation",
                    "market",
                    "stock",
                    "financial",
                ],
                "trade": [
                    "tariff",
                    "trade",
                    "import",
                    "export",
                    "trade war",
                    "trade agreement",
                    "trade deficit",
                ],
                "military": [
                    "war",
                    "military",
                    "troops",
                    "defense",
                    "attack",
                    "invasion",
                    "conflict",
                    "army",
                ],
                "technology": [
                    "tech",
                    "technology",
                    "digital",
                    "software",
                    "hardware",
                    "AI",
                    "innovation",
                ],
                "health": [
                    "health",
                    "disease",
                    "virus",
                    "pandemic",
                    "medical",
                    "treatment",
                    "vaccine",
                ],
                "climate": [
                    "climate",
                    "environment",
                    "carbon",
                    "emission",
                    "warming",
                    "pollution",
                    "sustainable",
                ],
                "education": [
                    "education",
                    "school",
                    "university",
                    "student",
                    "learning",
                    "teaching",
                    "academic",
                ],
                "social": [
                    "social",
                    "society",
                    "community",
                    "rights",
                    "equality",
                    "discrimination",
                    "justice",
                ],
                "culture": [
                    "culture",
                    "entertainment",
                    "media",
                    "art",
                    "music",
                    "film",
                    "book",
                ],
                "sports": [
                    "sports",
                    "game",
                    "team",
                    "athlete",
                    "championship",
                    "tournament",
                    "olympics",
                ],
            }

            domain_relevance = {domain: 0 for domain in domain_indicators}

            for domain, indicators in domain_indicators.items():
                for indicator in indicators:
                    if indicator in combined_context:
                        domain_relevance[domain] += combined_context.count(indicator)

            relevant_domains = sorted(
                domain_relevance.items(), key=lambda x: x[1], reverse=True
            )[:2]
            relevant_domains = [
                domain for domain, score in relevant_domains if score > 0
            ]

            critical_domain_terms = {
                "trade": ["trade", "tariff", "economic"],
                "politics": ["government", "political", "election"],
                "military": ["military", "war", "conflict"],
                "economy": ["economic", "market", "financial"],
                "technology": ["technology", "tech", "digital"],
                "health": ["health", "medical", "disease"],
                "climate": ["climate", "environment", "emission"],
                "education": ["education", "school", "university"],
                "social": ["social", "community", "rights"],
                "culture": ["culture", "art", "media"],
                "sports": ["sports", "game", "team"],
            }

            for domain in relevant_domains:
                if domain in critical_domain_terms:
                    required_terms.extend(critical_domain_terms[domain][:2])

            entities = re.findall(r"\b[A-Z][a-z]+\b", topic_statement)
            if entities:
                required_terms.extend([entity.lower() for entity in entities[:2]])

            required_terms = list(set(required_terms))

            required_term_threshold = max(1, min(len(required_terms) // 3, 2))
            passes_required = (
                not required_terms
                or sum(1 for term in required_terms if term in article_text)
                >= required_term_threshold
            )

            matching_terms = sum(1 for term in key_terms if term in article_text)
            relevance_threshold = max(2, min(len(key_terms) // 4, 5))

            is_relevant = matching_terms >= relevance_threshold and passes_required

            if not is_relevant:
                logger.debug(
                    f"Article rejected as irrelevant: {article.get('title')[:50]}..."
                )

            return is_relevant

        except Exception as e:
            logger.warning(f"Error checking article relevance: {str(e)}")
            return False

    def _is_source_duplicate(
        self, source: Dict, existing_sources: List[Source]
    ) -> bool:
        """Check if a source already exists in the sources list."""
        for existing in existing_sources:
            if (
                source.get("title") == existing.title
                and source.get("publication") == existing.publication
                and source.get("date") == existing.date
                and source.get("image_url") == existing.image_url
            ):
                return True
        return False


# ============================================================================
# Timeline Generator
# ============================================================================


class TimelineGenerator:
    """Generates comprehensive timelines from news articles."""

    def __init__(self):
        """Initialize the timeline generator."""
        self.generator = dspy.ChainOfThought(GenerateTimeline)
        self.news_api = GNewsAPI()
        self.perspective_generator = dspy.ChainOfThought(PerspectiveSynthesis)
        logger.info("TimelineGenerator initialized successfully")

    def _clean_source(self, source_data: dict) -> dict:
        """Clean and validate source data before creating Source object."""
        if source_data.get("url") and (
            "..." in source_data["url"] or len(source_data["url"]) < 10
        ):
            logger.warning(f"Removing incomplete URL: {source_data['url']}")
            source_data["url"] = None
        return source_data

    def _convert_article_to_source(self, article: Dict) -> Dict:
        """Convert a GNews article to our Source format."""
        try:
            return {
                "title": str(article.get("title", "")),
                "author": None,
                "publication": str(article.get("source_name", "Unknown Source")),
                "date": str(article.get("pubDate", "")).split("T")[0],
                "url": article.get("url"),
                "doi": None,
                "image_url": article.get("image"),
            }
        except Exception as e:
            logger.warning(f"Error converting article to source: {str(e)}")
            return {
                "title": str(article.get("title", "Unknown Title")),
                "author": None,
                "publication": str(article.get("source_name", "Unknown Source")),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "url": None,
                "doi": None,
                "image_url": None,
            }

    def _analyze_article_perspective(self, article: Dict) -> Dict:
        """Analyze an article to extract its perspective and key points."""
        try:
            content = " ".join(
                filter(
                    None,
                    [
                        str(article.get("title", "")),
                        str(article.get("description", "")),
                        str(article.get("content", "")),
                    ],
                )
            )

            key_points = []
            quotes = []

            if article.get("title"):
                key_points.append(article["title"])

            if article.get("description"):
                quotes.append(article["description"])

            if article.get("content") and article["content"] != article.get(
                "description"
            ):
                quotes.append(article["content"])

            return {
                "content": content,
                "key_points": key_points,
                "quotes": quotes,
                "political_agreement": article.get("political_agreement", 0.5),
                "source": self._convert_article_to_source(article),
            }
        except Exception as e:
            logger.warning(f"Error analyzing article perspective: {str(e)}")
            return None

    def _consolidate_perspective(
        self, articles: List[Dict], perspective_type: str
    ) -> Dict:
        """Consolidate multiple articles into a single coherent perspective."""
        if not articles:
            return None

        analyzed_articles = []
        for article in articles:
            analysis = self._analyze_article_perspective(article)
            if analysis:
                analyzed_articles.append(analysis)

        if not analyzed_articles:
            return None

        try:
            combined_content = " ".join(a["content"] for a in analyzed_articles)

            all_key_points = []
            all_quotes = []
            sources = []

            for analysis in analyzed_articles:
                all_key_points.extend(analysis["key_points"])
                all_quotes.extend(analysis["quotes"])
                sources.append(analysis["source"])

            synthesizer = dspy.ChainOfThought(PerspectiveSynthesis)
            synthesizer.temperature = 0.3
            result = synthesizer(
                content=combined_content,
                key_points=all_key_points,
                perspective_type=perspective_type,
            )

            source_quotes = []
            for source, quotes in zip(
                sources, [a["quotes"] for a in analyzed_articles]
            ):
                if quotes:
                    source_quotes.append(
                        {
                            "source": source,
                            "quote": quotes[0],
                        }
                    )

            sorted_sources = sorted(
                source_quotes,
                key=lambda x: abs(0.5 - x["source"].get("political_agreement", 0.5)),
                reverse=True,
            )

            final_sources = []
            final_quotes = []
            for item in sorted_sources[:3]:
                final_sources.append(item["source"])
                final_quotes.append(item["quote"])

            return {
                "viewpoint": result.synthesized_viewpoint,
                "sources": final_sources,
                "quotes": final_quotes,
            }

        except Exception as e:
            logger.error(
                f"Error consolidating {perspective_type} perspective: {str(e)}"
            )
            return None

    def generate(self, topic_statement: str, background_info: str) -> Timeline:
        """Generate a factual, well-sourced timeline for the given topic."""
        try:
            query_params = self.news_api.generate_query(
                topic_statement, background_info
            )
            articles = self.news_api.fetch_diverse_articles(query_params)

            if not articles:
                raise ValueError("No relevant articles found through GNews")

            relevant_articles = [
                article
                for article in articles
                if self.news_api._is_article_relevant(
                    article, topic_statement, background_info
                )
            ]

            if not relevant_articles:
                raise ValueError("No relevant articles found after filtering")

            articles_by_date = {}
            for article in relevant_articles:
                date = article["pubDate"].split("T")[0]
                if date not in articles_by_date:
                    articles_by_date[date] = {
                        "left": [],
                        "center": [],
                        "right": [],
                        "all": [],
                    }

                score = article.get("political_agreement", 0.5)
                if score < 0.4:
                    articles_by_date[date]["left"].append(article)
                elif score > 0.6:
                    articles_by_date[date]["right"].append(article)
                else:
                    articles_by_date[date]["center"].append(article)

                articles_by_date[date]["all"].append(article)

            events_data = []
            for date, date_articles in articles_by_date.items():
                has_enough_articles = len(date_articles["all"]) >= 2

                if has_enough_articles:
                    main_article = (
                        date_articles["center"][0]
                        if date_articles["center"]
                        else date_articles["all"][0]
                    )

                    left_perspective = self._consolidate_perspective(
                        date_articles["left"]
                        or date_articles["center"]
                        or [main_article],
                        "left-leaning",
                    )
                    right_perspective = self._consolidate_perspective(
                        date_articles["right"]
                        or date_articles["center"]
                        or [main_article],
                        "right-leaning",
                    )

                    if left_perspective and right_perspective:
                        left_sources = [
                            Source(**source) for source in left_perspective["sources"]
                        ]
                        right_sources = [
                            Source(**source) for source in right_perspective["sources"]
                        ]

                        event_data = {
                            "date": date,
                            "event": main_article["title"],
                            "left_perspective": {
                                "viewpoint": left_perspective["viewpoint"],
                                "sources": left_sources,
                                "quotes": left_perspective["quotes"],
                            },
                            "right_perspective": {
                                "viewpoint": right_perspective["viewpoint"],
                                "sources": right_sources,
                                "quotes": right_perspective["quotes"],
                            },
                        }
                        events_data.append(event_data)

            if not events_data:
                logger.warning(
                    "No events created with standard criteria, trying with more lenient criteria..."
                )

                for date, date_articles in articles_by_date.items():
                    if len(date_articles["all"]) > 0:
                        main_article = date_articles["all"][0]

                        source_obj = Source(
                            **self._convert_article_to_source(main_article)
                        )

                        left_perspective = {
                            "viewpoint": f"Perspective on {main_article['title']}",
                            "sources": [source_obj],
                            "quotes": [
                                main_article.get(
                                    "description", "No description available"
                                )
                            ],
                        }

                        right_perspective = {
                            "viewpoint": f"Alternative view on {main_article['title']}",
                            "sources": [source_obj],
                            "quotes": [
                                main_article.get("content", "No content available")[
                                    :200
                                ]
                            ],
                        }

                        event_data = {
                            "date": date,
                            "event": main_article["title"],
                            "left_perspective": left_perspective,
                            "right_perspective": right_perspective,
                        }

                        events_data.append(event_data)

                        if len(events_data) >= 3:
                            break

            if not events_data:
                raise ValueError("Could not create any valid events")

            result = self.generator(
                topic_statement=topic_statement, background_info=background_info
            )

            sources_consulted = []
            for article in relevant_articles:
                try:
                    source_data = self._convert_article_to_source(article)
                    source_obj = Source(**source_data)
                    sources_consulted.append(source_obj)
                except Exception as e:
                    logger.warning(
                        f"Skipping invalid source: {source_data.get('title', 'Unknown')} - {str(e)}"
                    )

            if not sources_consulted:
                raise ValueError(
                    "No valid sources could be created from the generated data"
                )

            valid_events = []
            for event_data in events_data:
                try:
                    event_obj = Event(**event_data)
                    valid_events.append(event_obj)
                except Exception as e:
                    logger.warning(
                        f"Skipping invalid event from {event_data.get('date', 'Unknown date')}: {str(e)}"
                    )

            if not valid_events:
                raise ValueError(
                    "No valid events could be created from the generated data"
                )

            valid_events.sort(key=lambda x: x.date)

            timeline = Timeline(
                topic_statement=topic_statement,
                summary=result.topic_summary,
                timeline=valid_events,
                sources_consulted=sources_consulted,
            )

            return timeline

        except Exception as e:
            logger.error(f"Error generating timeline: {str(e)}")
            raise ValueError(
                f"Failed to generate valid timeline with verifiable sources: {str(e)}"
            )


# ============================================================================
# Perspective Synthesis
# ============================================================================


class PerspectiveSynthesis(dspy.Signature):
    """Synthesize a coherent perspective from multiple sources."""

    content = dspy.InputField(desc="Combined content from multiple articles")
    key_points = dspy.InputField(desc="List of key points from the articles")
    perspective_type = dspy.InputField(
        desc="The type of perspective to synthesize (left-leaning or right-leaning)"
    )

    synthesized_viewpoint = dspy.OutputField(
        desc="A coherent viewpoint that represents the collective perspective while maintaining political leaning"
    )


if __name__ == "__main__":
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    topic_statement = "The US and China are in a trade war"
    background_information = """The US and China have been in a trade war since 2018."""

    try:
        generator = TimelineGenerator()
        timeline = generator.generate(topic_statement, background_information)

        output_file = "timeline.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(timeline.model_dump_json(indent=2))

        logger.info(f"Timeline successfully generated and saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate timeline: {str(e)}")
