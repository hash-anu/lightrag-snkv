from __future__ import annotations

import os
from dataclasses import dataclass, field


def parse_lookback(s: str) -> int:
    """Convert a lookback string to integer days.

    "Ny" -> N * 365  (years)
    "Nw" -> N * 7    (weeks)
    "Nd" -> N        (days)
    """
    s = s.strip().lower()
    if s.endswith("y"):
        return int(s[:-1]) * 365
    if s.endswith("w"):
        return int(s[:-1]) * 7
    if s.endswith("d"):
        return int(s[:-1])
    raise ValueError(
        f"Unrecognised lookback format {s!r}. Use e.g. '1y', '2y', '4w', '52w', '90d'."
    )


@dataclass
class HNConfig:
    # How far back to fetch on the very first run only.
    # Subsequent runs fetch incrementally from the last run timestamp.
    lookback: str = field(
        default_factory=lambda: os.environ.get("HN_LOOKBACK", "4w")
    )

    # "daily" | "weekly" | None (run-once)
    schedule: str | None = field(
        default_factory=lambda: os.environ.get("HN_SCHEDULE") or None
    )

    # LightRAG working directory for the HN knowledge graph (isolated from main RAG)
    working_dir: str = field(
        default_factory=lambda: os.environ.get("HN_WORKING_DIR", "./hn_rag_storage")
    )

    # Skip stories below this HN upvote score
    min_score: int = field(
        default_factory=lambda: int(os.environ.get("HN_MIN_SCORE", "10"))
    )

    # Stories sent to rag.ainsert() per call
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get("HN_BATCH_SIZE", "10"))
    )

    # HTTP timeout in seconds for Algolia API requests
    fetch_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HN_FETCH_TIMEOUT", "15"))
    )

    # Algolia tag filter — comma-separated OR list
    # ask_hn = Ask HN posts, show_hn = Show HN posts, story = all stories
    tags: str = field(
        default_factory=lambda: os.environ.get("HN_TAGS", "ask_hn,show_hn")
    )

    # Results per Algolia page (Algolia max is 1000)
    hits_per_page: int = 100
