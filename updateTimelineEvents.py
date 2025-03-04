import os
import dspy
import logging
import json
from typing import List, Dict, Any, Set
from datetime import datetime
import requests
from timelineGenerator import GNewsAPI
from models import Source, Perspective, Event, Timeline
from dspy_signatures import PerspectiveSynthesis
import time
from utils import (
    logger,
    generate_topic_hash,
    get_processed_articles_filename,
    load_processed_articles,
    save_processed_articles,
)
from article_utils import (
    calculate_content_similarity,
    is_date_close,
    infer_domain_type,
    calculate_political_agreement,
    process_article,
)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class TimelineUpdater:
    """Updates an existing timeline with new events from GNews API."""

    def __init__(
        self,
        timeline_file: str = "timeline.json",
        topic_statement: str = None,
    ):
        """Initialize the timeline updater."""
        self.timeline_file = timeline_file
        self.news_api = GNewsAPI()
        self.perspective_generator = dspy.ChainOfThought(PerspectiveSynthesis)
        self.processed_articles = {"processed_urls": set(), "processed_titles": set()}
        self.processed_articles_file = None

        if topic_statement:
            self._initialize_article_tracking(topic_statement)

        logger.info("TimelineUpdater initialized")

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

    def load_timeline(self) -> Timeline:
        """Load the existing timeline from JSON file."""
        try:
            with open(self.timeline_file, "r", encoding="utf-8") as f:
                timeline_data = json.load(f)

            timeline = Timeline(**timeline_data)
            logger.info(
                f"Successfully loaded timeline with {len(timeline.timeline)} events"
            )
            return timeline
        except FileNotFoundError:
            logger.error(f"Timeline file '{self.timeline_file}' not found.")
            raise
        except Exception as e:
            logger.error(f"Error loading timeline: {str(e)}")
            raise

    def save_timeline(self, timeline: Timeline):
        """Save the updated timeline back to the JSON file."""
        try:
            timeline.timeline.sort(key=lambda x: x.date)

            with open(self.timeline_file, "w", encoding="utf-8") as f:
                f.write(timeline.model_dump_json(indent=2))
            logger.info(f"Timeline updated with {len(timeline.timeline)} events")
        except Exception as e:
            logger.error(f"Error saving timeline: {str(e)}")
            raise

    def _analyze_article_perspective(self, article: Dict) -> Dict:
        """Analyze an article to extract its perspective and key points."""
        try:
            title = str(article.get("title", ""))
            description = str(article.get("description", ""))
            content = str(article.get("content", ""))

            full_text = ""
            if content and len(content) > 20 and content != "null":
                full_text = f"{title} {description} {content}"
            else:
                full_text = f"{title} {description}"

            key_points = []
            quotes = []

            if title:
                key_points.append(title)

            if description and len(description) > 10:
                quotes.append(description)

            if (
                content
                and len(content) > 20
                and content != "null"
                and content != description
            ):
                if len(content) > 300:
                    quotes.append(content[:300] + "...")
                else:
                    quotes.append(content)

            return {
                "content": full_text,
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

            result = self.perspective_generator(
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

    def _is_source_duplicate(
        self, source: Dict, existing_sources: List[Source]
    ) -> bool:
        """Check if a source already exists in the timeline."""
        for existing in existing_sources:
            if (
                source.get("title") == existing.title
                and source.get("publication") == existing.publication
                and source.get("date") == existing.date
                and source.get("image_url") == existing.image_url
            ):
                return True
        return False

    def _is_event_duplicate(
        self,
        event_date: str,
        event_title: str,
        existing_events: List[Event],
        content: str = None,
    ) -> bool:
        """
        Check if an event with the same date and title already exists.
        If content is provided, also checks for content similarity.
        """

        for existing in existing_events:
            if (
                existing.date == event_date
                and existing.event.lower() == event_title.lower()
            ):
                return True

        if content:
            for existing in existing_events:
                date_diff = abs(
                    (
                        datetime.strptime(existing.date, "%Y-%m-%d")
                        - datetime.strptime(event_date, "%Y-%m-%d")
                    ).days
                )

                if date_diff <= 1:
                    title_similarity = self._calculate_content_similarity(
                        existing.event.lower(), event_title.lower()
                    )

                    if title_similarity > 0.7:
                        return True

        return False

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text contents."""
        return calculate_content_similarity(text1, text2)

    def _is_article_relevant(
        self, article: Dict, topic_statement: str, background_info: str
    ) -> bool:
        """
        Check if an article is relevant to the specific topic we're interested in.
        This delegates to the news_api's relevance checker for consistency.
        """
        try:
            return self.news_api._is_article_relevant(
                article, topic_statement, background_info
            )
        except Exception as e:
            logger.warning(f"Error checking article relevance: {str(e)}")
            return True

    def _process_article(self, article: Dict) -> Dict:
        """Process a raw article from GNews API."""
        return process_article(article)

    def generate_new_events(self, topic_statement: str, background_info: str) -> Dict:
        """Generate new events from GNews API to add to the timeline."""
        try:
            self._initialize_article_tracking(topic_statement)

            timeline = self.load_timeline()

            existing_event_titles = {event.event.lower() for event in timeline.timeline}
            existing_event_dates = {event.date for event in timeline.timeline}

            query_params = self.news_api.generate_query(
                topic_statement, background_info
            )

            articles = []
            min_articles_target = 50
            articles_per_page = 10

            logger.info(
                f"Starting article collection for topic: {topic_statement[:50]}..."
            )

            first_page_params = query_params.copy()
            first_page_params["page"] = "1"

            try:
                response = self.news_api._make_request("search", first_page_params)
                total_available_articles = response.get("totalArticles", 0)

                if total_available_articles == 0:
                    logger.warning("No articles found for the topic")
                    return timeline

                total_pages = min(
                    20,
                    (total_available_articles + articles_per_page - 1)
                    // articles_per_page,
                )
                logger.info(
                    f"Found {total_available_articles} total articles across {total_pages} pages"
                )

                first_page_articles = response.get("articles", [])
                for article in first_page_articles:
                    if self._is_article_processed(article):
                        logger.info(
                            f"Skipping previously processed article: {article.get('title', '')[:50]}..."
                        )
                        continue

                    if not any(
                        a.get("title") == article.get("title") for a in articles
                    ):
                        processed_article = self._process_article(article)
                        if processed_article:
                            articles.append(processed_article)
                            self._mark_article_processed(article)

                logger.info(f"Collected {len(articles)} new articles from page 1")

                for page in range(2, total_pages + 1):
                    if len(articles) >= min_articles_target:
                        logger.info(
                            f"Reached target article count ({min_articles_target}), stopping pagination"
                        )
                        break

                    logger.info(f"Fetching page {page} of {total_pages}")
                    page_params = query_params.copy()
                    page_params["page"] = str(page)

                    try:
                        response = self.news_api._make_request("search", page_params)
                        page_articles = response.get("articles", [])

                        if not page_articles:
                            logger.warning(f"No articles found on page {page}")
                            break

                        new_articles_count = 0
                        for article in page_articles:
                            if self._is_article_processed(article):
                                logger.info(
                                    f"Skipping previously processed article: {article.get('title', '')[:50]}..."
                                )
                                continue

                            if not any(
                                a.get("title") == article.get("title") for a in articles
                            ):
                                processed_article = self._process_article(article)
                                if processed_article:
                                    articles.append(processed_article)
                                    new_articles_count += 1
                                    self._mark_article_processed(article)

                        logger.info(
                            f"Collected {new_articles_count} new articles from page {page}"
                        )

                        if page < total_pages:
                            time.sleep(1.5)

                    except Exception as e:
                        logger.warning(f"Error fetching page {page}: {str(e)}")
                        break

            except Exception as e:
                logger.error(f"Error with initial API request: {str(e)}")
                return timeline

            logger.info(f"Total unique articles collected: {len(articles)}")

            self._save_processed_articles()

            relevant_articles = [
                article
                for article in articles
                if self._is_article_relevant(article, topic_statement, background_info)
            ]
            logger.info(
                f"Articles deemed relevant to topic: {len(relevant_articles)}/{len(articles)}"
            )

            if relevant_articles:
                articles = relevant_articles

            if not articles:
                logger.warning("No new relevant articles found")
                return timeline

            articles_by_date = {}
            for article in articles:
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

            new_events = []
            all_new_sources = []
            events_created = 0

            for date, date_articles in articles_by_date.items():
                if len(date_articles["all"]) >= 2:
                    main_article = (
                        date_articles["center"][0]
                        if date_articles["center"]
                        else date_articles["all"][0]
                    )

                    if (
                        date in existing_event_dates
                        and main_article["title"].lower() in existing_event_titles
                    ):
                        logger.info(
                            f"Skipping duplicate event from {date}: {main_article['title'][:50]}..."
                        )
                        continue

                    if self._is_event_duplicate(
                        date,
                        main_article["title"],
                        timeline.timeline,
                        main_article["content"],
                    ):
                        logger.info(
                            f"Skipping duplicate event from {date}: {main_article['title'][:50]}..."
                        )
                        continue

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

                        try:
                            event = Event(**event_data)
                            new_events.append(event)
                            events_created += 1

                            logger.info(
                                f"Created new event for {date}: {main_article['title'][:50]}..."
                            )

                            all_sources = []
                            all_sources.extend(
                                [
                                    self._convert_article_to_source(a)
                                    for a in date_articles["all"]
                                ]
                            )

                            for source in all_sources:
                                if not self._is_source_duplicate(
                                    source, timeline.sources_consulted
                                ):
                                    try:
                                        source_obj = Source(**source)
                                        all_new_sources.append(source_obj)
                                    except Exception as e:
                                        logger.warning(f"Invalid source: {str(e)}")

                        except Exception as e:
                            logger.warning(f"Error creating event: {str(e)}")

            if events_created == 0:
                logger.warning(
                    "No events created with standard criteria, trying with more lenient criteria..."
                )

                for date, date_articles in articles_by_date.items():
                    if len(date_articles["all"]) > 0:
                        main_article = date_articles["all"][0]

                        if self._is_event_duplicate(
                            date,
                            main_article["title"],
                            timeline.timeline,
                            main_article["content"],
                        ):
                            continue

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

                        try:
                            event = Event(**event_data)
                            new_events.append(event)
                            events_created += 1

                            if not self._is_source_duplicate(
                                self._convert_article_to_source(main_article),
                                timeline.sources_consulted,
                            ):
                                all_new_sources.append(source_obj)

                            logger.info(
                                f"Created simplified event for {date}: {main_article['title'][:50]}..."
                            )

                        except Exception as e:
                            logger.warning(f"Error creating simplified event: {str(e)}")

            if new_events:
                new_events.sort(key=lambda x: x.date)
                timeline.timeline.extend(new_events)

                timeline.sources_consulted.extend(all_new_sources)
                timeline.timeline.sort(key=lambda x: x.date)
                self.save_timeline(timeline)

                logger.info(f"Added {len(new_events)} new events to timeline")
                return timeline
            else:
                logger.info("No new events to add to timeline")
                return timeline

        except Exception as e:
            logger.error(f"Error generating new events: {str(e)}")
            raise


if __name__ == "__main__":
    topic_statement = "The US and China are in a trade war"
    background_information = """The US and China have been in a trade war since 2018."""

    try:
        updater = TimelineUpdater()
        updated_timeline = updater.generate_new_events(
            topic_statement, background_information
        )
        logger.info(f"Timeline now has {len(updated_timeline.timeline)} events total")
    except Exception as e:
        logger.error(f"Failed to update timeline: {str(e)}")
