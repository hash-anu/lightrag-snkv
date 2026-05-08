from __future__ import annotations

import html
import logging
import re
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
_ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{}"


def _build_tags_filter(tags: str) -> str:
    """Convert "ask_hn,show_hn" -> "(ask_hn,show_hn)" for Algolia OR syntax."""
    parts = [t.strip() for t in tags.split(",") if t.strip()]
    if len(parts) == 1:
        return parts[0]
    return "(" + ",".join(parts) + ")"


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r" {2,}", " ", text).strip()


async def iter_stories(
    client: httpx.AsyncClient,
    since_ts: int,
    until_ts: int,
    tags: str,
    min_score: int,
    hits_per_page: int = 100,
) -> AsyncIterator[dict]:
    """Yield HN story dicts from Algolia between [since_ts, until_ts).

    Paginates automatically. Skips stories where points < min_score.
    """
    tags_filter = _build_tags_filter(tags)

    page = 0
    total_fetched = 0
    total_yielded = 0

    while True:
        # Build URL manually — httpx would URL-encode parentheses in tags=(ask_hn,show_hn)
        # which breaks Algolia's OR filter syntax.
        url = (
            f"{_ALGOLIA_SEARCH_URL}"
            f"?tags={tags_filter}"
            f"&numericFilters=created_at_i>={since_ts},created_at_i<{until_ts}"
            f"&hitsPerPage={hits_per_page}"
            f"&page={page}"
        )
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Algolia search failed (page %d): %s", page, exc)
            break

        hits = data.get("hits", [])
        if not hits:
            break

        if page == 0:
            logger.info(
                "Algolia found %d total stories matching tags=%s in window.",
                data.get("nbHits", 0),
                tags_filter,
            )

        total_fetched += len(hits)
        for story in hits:
            score = story.get("points") or 0
            if score < min_score:
                continue
            total_yielded += 1
            yield story

        nb_pages = data.get("nbPages", 1)
        page += 1
        if page >= nb_pages:
            break

    logger.info(
        "Algolia pagination done. Fetched %d stories across %d pages, %d passed min_score filter.",
        total_fetched, page, total_yielded,
    )


async def fetch_item(story_id: str, client: httpx.AsyncClient) -> dict | None:
    """Fetch full item with all nested comments from Algolia items API.

    Returns the item dict (with 'children' list) or None on failure.
    """
    try:
        resp = await client.get(_ALGOLIA_ITEM_URL.format(story_id))
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch item %s: %s", story_id, exc)
        return None


def _collect_comments(children: list[dict], out: list[str]) -> None:
    """Recursively collect all comment texts (depth-first)."""
    for child in children:
        if child.get("type") != "comment":
            continue
        text = _strip_html(child.get("text") or "").strip()
        if text:
            author = child.get("author") or "unknown"
            out.append(f"**{author}**: {text}")
        sub = child.get("children") or []
        if sub:
            _collect_comments(sub, out)


def build_document(story: dict, item: dict | None) -> str | None:
    """Format a HN story + all its comments as a LightRAG-ingestible document.

    Returns None only if there is truly no content (no body and no comments).

    Format:
        # {title}

        Author: {author}
        Score: {points}
        Date: {created_at}
        HN ID: {id}

        {body}              ← story_text if present

        ## Discussion
        **user1**: ...
        **user2**: ...
    """
    title = story.get("title") or "Untitled"
    author = story.get("author") or "unknown"
    points = story.get("points") or 0
    created_at = story.get("created_at") or ""
    oid = story.get("objectID") or ""

    body = _strip_html(story.get("story_text") or "").strip()

    comments: list[str] = []
    if item:
        _collect_comments(item.get("children") or [], comments)

    if not body and not comments:
        return None

    lines: list[str] = [
        f"# {title}",
        "",
        f"Author: {author}",
        f"Score: {points}",
        f"Date: {created_at}",
        f"HN ID: {oid}",
    ]

    if body:
        lines += ["", body]

    if comments:
        lines += ["", "## Discussion", ""]
        lines.extend(comments)

    return "\n".join(lines)
