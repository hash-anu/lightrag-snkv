"""Ingest Hacker News posts into a LightRAG+SNKV knowledge graph.

Run from the lightrag-snkv directory:

    python ingest_hn.py                        # run-once, defaults from .env
    python ingest_hn.py --lookback 1y          # backfill past year
    python ingest_hn.py --schedule daily       # daemon: fetch every 24h
    python ingest_hn.py --schedule weekly --min-score 50
    python ingest_hn.py --help
"""
import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from lightrag_snkv import register

register()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Hacker News posts into a LightRAG+SNKV knowledge graph."
    )
    p.add_argument(
        "--lookback",
        default=None,
        metavar="PERIOD",
        help=(
            "How far back to fetch on the first run. "
            "Format: 1y (1 year), 2y (2 years), 4w (4 weeks), 90d (90 days). "
            "Default: HN_LOOKBACK env var or '4w'."
        ),
    )
    p.add_argument(
        "--schedule",
        choices=["daily", "weekly"],
        default=None,
        help=(
            "Run as a daemon on this schedule. "
            "'daily' fetches every 24 hours; 'weekly' every 7 days. "
            "Omit to run once and exit."
        ),
    )
    p.add_argument(
        "--working-dir",
        dest="working_dir",
        default=None,
        metavar="DIR",
        help="Directory for the HN knowledge graph. Default: HN_WORKING_DIR env or ./hn_rag_storage.",
    )
    p.add_argument(
        "--min-score",
        dest="min_score",
        type=int,
        default=None,
        metavar="N",
        help="Skip posts below this HN upvote score. Default: HN_MIN_SCORE env or 10.",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Posts sent to the knowledge graph in one batch. Default: HN_BATCH_SIZE env or 10.",
    )
    p.add_argument(
        "--tags",
        default=None,
        metavar="TAGS",
        help=(
            "Post types to fetch, comma-separated. "
            "ask_hn = Ask HN discussions, show_hn = Show HN posts. "
            "Default: HN_TAGS env or 'ask_hn,show_hn'."
        ),
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    from lightrag import LightRAG
    from bench.llm_env import get_llm_and_embed_funcs
    from lightrag_snkv.hn_config import HNConfig
    from lightrag_snkv.hn_ingestor import HNIngestor

    cfg = HNConfig()

    if args.lookback is not None:
        cfg.lookback = args.lookback
    if args.schedule is not None:
        cfg.schedule = args.schedule
    if args.working_dir is not None:
        cfg.working_dir = args.working_dir
    if args.min_score is not None:
        cfg.min_score = args.min_score
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.tags is not None:
        cfg.tags = args.tags

    os.makedirs(cfg.working_dir, exist_ok=True)

    llm_func, embed_func = get_llm_and_embed_funcs()

    rag = LightRAG(
        working_dir=cfg.working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()
    try:
        ingestor = HNIngestor(rag, cfg)
        if cfg.schedule:
            await ingestor.run_daemon()
        else:
            n = await ingestor.run_once()
            logger.info("Done. Stories inserted this run: %d", n)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
