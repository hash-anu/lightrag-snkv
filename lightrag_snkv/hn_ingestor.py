from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from lightrag import LightRAG
    from lightrag_snkv.hn_config import HNConfig

_SERVER_URL = os.environ.get("HN_SERVER_URL", "http://localhost:9621")

from lightrag_snkv.hn_config import parse_lookback
from lightrag_snkv.hn_fetcher import build_document, fetch_item, iter_stories

logger = logging.getLogger(__name__)

_SCHEDULE_SECONDS = {
    "daily": 86_400,
    "weekly": 604_800,
}


class HNState:
    """Persistent ingestion state stored in {working_dir}/hn_state.json.

    Schema:
        {
          "last_run_ts": 1710000000,
          "ingested_ids": ["12345", "67890"]
        }

    last_run_ts is set to until_ts at the START of each run so a mid-run
    crash re-fetches the same window rather than silently skipping stories.
    ingested_ids deduplicates stories near the timestamp boundary.
    """

    def __init__(self, working_dir: str) -> None:
        self._path = Path(working_dir) / "hn_state.json"
        self._data: dict = {"last_run_ts": None, "ingested_ids": []}
        self._ingested_set: set[str] = set()

    def load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                self._ingested_set = set(self._data.get("ingested_ids", []))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read hn_state.json (%s); starting fresh.", exc)
                self._data = {"last_run_ts": None, "ingested_ids": []}
                self._ingested_set = set()

    def save(self) -> None:
        """Atomically write state via temp-file rename."""
        self._data["ingested_ids"] = list(self._ingested_set)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        try:
            tmp.replace(self._path)
        except OSError:
            # Windows fallback: direct write
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.unlink(missing_ok=True)

    @property
    def last_run_ts(self) -> int | None:
        return self._data.get("last_run_ts")

    def set_last_run_ts(self, ts: int) -> None:
        self._data["last_run_ts"] = ts

    def is_ingested(self, story_id: str) -> bool:
        return story_id in self._ingested_set

    def mark_ingested(self, story_id: str) -> None:
        self._ingested_set.add(story_id)


class HNIngestor:
    """Fetches HN stories and inserts them into a LightRAG knowledge graph.

    Usage:
        ingestor = HNIngestor(rag, cfg)
        await ingestor.run_once()       # fetch + insert, then return
        await ingestor.run_daemon()     # loop forever on configured schedule
    """

    def __init__(self, rag: "LightRAG", cfg: "HNConfig") -> None:
        self._rag = rag
        self._cfg = cfg
        self._state = HNState(cfg.working_dir)
        self._state.load()

    async def run_once(self) -> int:
        """Fetch new stories and insert into LightRAG. Returns count ingested."""
        since_ts, until_ts = self._compute_fetch_window()
        self._state.set_last_run_ts(until_ts)

        logger.info(
            "HN ingestion run: fetching stories from %d to %d (tags=%s, min_score=%d)",
            since_ts,
            until_ts,
            self._cfg.tags,
            self._cfg.min_score,
        )

        ingested = 0
        batch_docs: list[str] = []
        batch_ids: list[str] = []

        async with httpx.AsyncClient(timeout=self._cfg.fetch_timeout) as client:
            async for story in iter_stories(
                client=client,
                since_ts=since_ts,
                until_ts=until_ts,
                tags=self._cfg.tags,
                min_score=self._cfg.min_score,
                hits_per_page=self._cfg.hits_per_page,
            ):
                oid = str(story.get("objectID", ""))

                if self._state.is_ingested(oid):
                    continue

                item = await fetch_item(oid, client)
                doc = build_document(story, item)
                if doc is None:
                    logger.debug("Skipping story %s — no body or comments.", oid)
                    continue

                batch_docs.append(doc)
                batch_ids.append(oid)

                if len(batch_docs) >= self._cfg.batch_size:
                    ingested += await self._flush(batch_docs, batch_ids)
                    batch_docs, batch_ids = [], []

        if batch_docs:
            ingested += await self._flush(batch_docs, batch_ids)

        logger.info("HN ingestion run complete. Stories inserted: %d", ingested)
        return ingested

    async def run_daemon(self) -> None:
        """Loop forever, calling run_once() on the configured schedule."""
        sleep_secs = int(os.environ.get("HN_INTERVAL_SECONDS") or
                         _SCHEDULE_SECONDS.get(self._cfg.schedule or "", 86_400))
        logger.info("HN daemon started (schedule=%s, sleep=%ds)", self._cfg.schedule, sleep_secs)

        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("HN run_once() failed; will retry at next interval.")
            logger.info("HN daemon sleeping %d seconds.", sleep_secs)
            await asyncio.sleep(sleep_secs)

    def _compute_fetch_window(self) -> tuple[int, int]:
        until_ts = int(time.time())
        if self._state.last_run_ts is not None:
            since_ts = self._state.last_run_ts
        else:
            days = parse_lookback(self._cfg.lookback)
            since_ts = until_ts - days * 86_400
        return since_ts, until_ts

    async def _flush(self, docs: list[str], ids: list[str]) -> int:
        """Insert a batch into LightRAG via the server API so the WebUI shows progress."""
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{_SERVER_URL}/documents/texts",
                    json={"texts": docs},
                )
                resp.raise_for_status()
        except Exception:
            logger.exception(
                "POST /documents/texts failed for batch of %d stories; will retry next run.", len(docs)
            )
            return 0

        for oid in ids:
            self._state.mark_ingested(oid)
        self._state.save()
        return len(ids)
