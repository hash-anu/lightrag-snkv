"""Start LightRAG web server with SNKV storage backends.

Run from the lightrag-snkv directory:
    python server.py

The .env file in this directory is loaded automatically.

If HN_SCHEDULE is set in .env (daily or weekly), the Hacker News ingestion
daemon starts automatically in the background alongside the web server.
"""
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # load .env before any lightrag imports

# Register SNKV backends with LightRAG's storage registry
from lightrag_snkv import register
register()

# Override storage backends to use SNKV
os.environ["LIGHTRAG_KV_STORAGE"] = "SNKVKVStorage"
os.environ["LIGHTRAG_VECTOR_STORAGE"] = "SNKVVectorStorage"
os.environ["LIGHTRAG_GRAPH_STORAGE"] = "SNKVGraphStorage"
os.environ["LIGHTRAG_DOC_STATUS_STORAGE"] = "SNKVDocStatusStorage"

logger = logging.getLogger(__name__)


async def _run() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9621"))

    # Import the FastAPI app that lightrag_server builds
    from lightrag.api.lightrag_server import app

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))

    tasks: list[asyncio.Task] = [asyncio.create_task(server.serve())]

    hn_schedule = os.environ.get("HN_SCHEDULE")
    if hn_schedule:
        from lightrag import LightRAG
        from bench.llm_env import get_llm_and_embed_funcs
        from lightrag_snkv.hn_config import HNConfig
        from lightrag_snkv.hn_ingestor import HNIngestor

        cfg = HNConfig()
        os.makedirs(cfg.working_dir, exist_ok=True)

        llm_func, embed_func = get_llm_and_embed_funcs()
        hn_rag = LightRAG(
            working_dir=cfg.working_dir,
            llm_model_func=llm_func,
            embedding_func=embed_func,
            kv_storage="SNKVKVStorage",
            vector_storage="SNKVVectorStorage",
            graph_storage="SNKVGraphStorage",
            doc_status_storage="SNKVDocStatusStorage",
        )
        await hn_rag.initialize_storages()

        ingestor = HNIngestor(hn_rag, cfg)

        async def _hn_daemon() -> None:
            try:
                await ingestor.run_daemon()
            finally:
                await hn_rag.finalize_storages()

        tasks.append(asyncio.create_task(_hn_daemon()))
        logger.info("HN ingestion daemon started (schedule=%s).", hn_schedule)

    await asyncio.gather(*tasks)


def main() -> None:
    # Fallback: if lightrag_server exposes its own `app` we use the async path above.
    # If the import fails (older lightrag version), fall back to the original blocking call.
    try:
        asyncio.run(_run())
    except ImportError:
        # lightrag version does not export `app` directly — use original entry point
        from lightrag.api.lightrag_server import main as _main
        _main()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
