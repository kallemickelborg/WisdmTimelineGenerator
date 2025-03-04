"""
Data models for the timeline generation system.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from urllib.parse import urlparse


class Source(BaseModel):
    """A verifiable source with citation information."""

    title: str = Field(..., description="Title of the source document")
    author: Optional[str] = Field(None, description="Author(s) of the source")
    publication: str = Field(..., description="Publication/Organization name")
    date: str = Field(..., description="Publication date (YYYY-MM-DD)")
    url: Optional[str] = Field(None, description="URL to the source (if available)")
    doi: Optional[str] = Field(None, description="DOI if academic paper")
    image_url: Optional[str] = Field(
        None, description="URL to the article's image (if available)"
    )

    @field_validator("date")
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator("url")
    def validate_url(cls, v):
        """Validate URL format if provided."""
        if v is None:
            return v
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")


class Perspective(BaseModel):
    """A perspective on an event with required source attribution."""

    viewpoint: str = Field(..., description="The perspective's viewpoint")
    sources: List[Source] = Field(
        ..., min_items=1, description="Multiple sources supporting this perspective"
    )
    quotes: List[str] = Field(
        ..., description="Relevant quotes from sources supporting this perspective"
    )


class Event(BaseModel):
    """A historical event with factual information and sourced perspectives."""

    date: str = Field(..., description="The date when the event occurred (YYYY-MM-DD)")
    event: str = Field(..., description="Factual description of the event")
    left_perspective: Perspective = Field(
        ..., description="Left-leaning perspective with multiple sources"
    )
    right_perspective: Perspective = Field(
        ..., description="Right-leaning perspective with multiple sources"
    )

    @field_validator("date")
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class Timeline(BaseModel):
    """A complete timeline with metadata and sourced events."""

    topic_statement: str
    summary: str
    timeline: List[Event]
    sources_consulted: List[Source] = Field(
        ..., description="List of all sources consulted in the research"
    )
