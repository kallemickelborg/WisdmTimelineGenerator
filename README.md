# Timeline Generation System

A system for generating comprehensive, balanced timelines for topics by gathering and analyzing news articles from GNews API.

## To-Do

- [ ] Refactor timelineGenerator.py and updateTimelineEvents.py for readability
- [ ] Improve perspective analysis with more nuanced political spectrum
- [ ] Create documentation for API endpoints
- [ ] Add support for timeline version control

## Project Structure

The project is organized into several modules:

- **timelineGenerator.py**: Main script for generating new timelines from scratch
- **updateTimelineEvents.py**: Script for updating existing timelines with new events
- **models.py**: Pydantic data models for the timeline system
- **utils.py**: Common utility functions for file operations and tracking
- **article_utils.py**: Utility functions for article processing and analysis
- **dspy_signatures.py**: DSPy signatures for timeline generation using LLMs

## Setup

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   export GNEWS_API_KEY=your_gnews_api_key
   export OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Generating a New Timeline

```python
from timelineGenerator import TimelineGenerator

generator = TimelineGenerator()

topic_statement = "US-China Trade War"
background_info = """The US and China have been engaged in a trade war since 2018,
with both sides imposing tariffs on hundreds of billions of dollars of goods."""

timeline = generator.generate(topic_statement, background_info)

print(f"Generated {len(timeline.timeline)} events")
```

### Updating an Existing Timeline

```python
from updateTimelineEvents import TimelineUpdater

updater = TimelineUpdater(timeline_file="timeline.json")

topic_statement = "US-China Trade War"
background_info = """The US and China have been engaged in a trade war since 2018,
with both sides imposing tariffs on hundreds of billions of dollars of goods."""

updated_timeline = updater.generate_new_events(topic_statement, background_info)

print(f"Timeline now has {len(updated_timeline.timeline)} events")
```

## Requirements

- Python 3.8+
- GNews API key
- OpenAI API key for GPT-4o-mini
- Required Python packages (see requirements.txt)
