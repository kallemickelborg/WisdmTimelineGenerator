"""
DSPy signatures for timeline generation.
"""

import dspy


class QueryGenerator(dspy.Signature):
    """Generate optimized search queries for GNews API."""

    topic = dspy.InputField(desc="The main topic to search for")
    context = dspy.InputField(desc="Additional context to help refine the search")

    search_query = dspy.OutputField(desc="Main search keywords")
    category = dspy.OutputField(
        desc="Best category (general, world, nation, business, technology, entertainment, sports, science, health)"
    )


class GenerateTimeline(dspy.Signature):
    """Generate a comprehensive, factually accurate timeline with verifiable sources."""

    topic_statement = dspy.InputField(desc="The main topic or event to analyze")
    background_info = dspy.InputField(
        desc="Additional context or background information"
    )
    topic_summary = dspy.OutputField(
        desc="Comprehensive, objective summary of the topic with key source citations"
    )
    sources_consulted = dspy.OutputField(
        desc="""List of REAL, VERIFIABLE sources consulted, in the following format:
    [
        {
            "title": "Exact title of the source document",
            "author": "Author name(s) if available",
            "publication": "Name of publication/organization",
            "date": "YYYY-MM-DD",
            "url": "Direct URL to source if available",
            "doi": "DOI if academic paper"
        },
        ...
    ]
    
    CRITICAL SOURCE REQUIREMENTS:
    1. ONLY include REAL, VERIFIABLE sources that actually exist
    2. All sources must have accurate publication dates
    3. URLs must be real and accessible
    4. DOIs must be valid for academic papers
    5. NO FABRICATED or APPROXIMATED sources allowed
    6. If unsure about a source detail, omit it rather than guess"""
    )

    events = dspy.OutputField(
        desc="""List of key events in chronological order, with each event structured as:
    [
        {
            "date": "YYYY-MM-DD",
            "event": "Factual description of what happened",
            "left_perspective": {
                "viewpoint": "Description from left-leaning viewpoint",
                "source": {
                    "title": "Exact title of source",
                    "author": "Author name(s)",
                    "publication": "Publication name",
                    "date": "YYYY-MM-DD",
                    "url": "Direct URL"
                },
                "quote": "Exact quote from the source"
            },
            "right_perspective": {
                "viewpoint": "Description from right-leaning viewpoint",
                "source": {
                    "title": "Exact title of source",
                    "author": "Author name(s)",
                    "publication": "Publication name",
                    "date": "YYYY-MM-DD",
                    "url": "Direct URL"
                },
                "quote": "Exact quote from the source"
            }
        }
    ]
    
    STRICT REQUIREMENTS:
    1. ONLY include events with REAL, VERIFIABLE sources
    2. ALL sources must actually exist and be accessible
    3. ALL quotes must be real and verifiable
    4. Dates must be specific and accurate"""
    )


class PerspectiveSynthesis(dspy.Signature):
    """Synthesize a coherent perspective from multiple sources."""

    content = dspy.InputField(desc="Combined content from multiple articles")
    key_points = dspy.InputField(desc="List of key points from the articles")
    perspective_type = dspy.InputField(
        desc="The type of perspective to synthesize (left-leaning or right-leaning)"
    )

    synthesized_viewpoint = dspy.OutputField(
        desc="A coherent, unified perspective synthesized from the sources that represents the political perspective well"
    )
    representative_quotes = dspy.OutputField(
        desc="List of 2-3 key quotes that best represent this perspective"
    )
