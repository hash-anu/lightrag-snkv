"""Start LightRAG web server with SNKV storage backends.

Run from the lightrag-snkv directory:
    python server.py

The .env file in this directory is loaded automatically.

If HN_SCHEDULE is set in .env (daily or weekly), the Hacker News ingestion
daemon starts automatically in the background alongside the web server.
"""
import asyncio
import datetime
import json
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

_DATE_PROMPT = "Today's date is {date}. Use it to interpret relative time references like 'this week', 'recently', or 'today' against document timestamps."
_QUERY_PATHS = {"/query", "/query/stream"}


def _add_date_middleware(app) -> None:
    """Inject today's date into user_prompt for every query request."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class DateInjectionMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.method == "POST" and request.url.path in _QUERY_PATHS:
                try:
                    body = await request.body()
                    data = json.loads(body)
                    if not data.get("user_prompt"):
                        data["user_prompt"] = _DATE_PROMPT.format(
                            date=datetime.date.today().isoformat()
                        )
                    # Replace the request body with the modified payload
                    modified = json.dumps(data).encode()

                    async def receive():
                        return {"type": "http.request", "body": modified, "more_body": False}

                    request = Request(request.scope, receive)
                except Exception:
                    pass  # leave request unchanged on any parse error
            return await call_next(request)

    app.add_middleware(DateInjectionMiddleware)


async def _run() -> None:
    import uvicorn
    from lightrag.api.lightrag_server import get_application

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9621"))

    app = get_application()
    _add_date_middleware(app)

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


def _start_hn_daemon_thread() -> None:
    """Start the HN ingestion daemon in a background thread with its own event loop."""
    import threading
    from lightrag import LightRAG
    from bench.llm_env import get_llm_and_embed_funcs
    from lightrag_snkv.hn_config import HNConfig
    from lightrag_snkv.hn_ingestor import HNIngestor

    cfg = HNConfig()
    os.makedirs(cfg.working_dir, exist_ok=True)

    def _run_daemon() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _daemon():
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
            try:
                await ingestor.run_daemon()
            finally:
                await hn_rag.finalize_storages()

        loop.run_until_complete(_daemon())

    t = threading.Thread(target=_run_daemon, daemon=True, name="hn-daemon")
    t.start()
    logger.info("HN ingestion daemon started in background thread (schedule=%s).", cfg.schedule)


def main() -> None:
    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.warning("Primary startup failed (%s); falling back to lightrag _main().", exc)
        if os.environ.get("HN_SCHEDULE"):
            _start_hn_daemon_thread()
        from lightrag.api.lightrag_server import main as _main
        _main()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
