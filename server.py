"""Start the LightRAG web server with SNKV storage backends.

Run from the lightrag-snkv directory:
    python server.py

The .env file in this directory is loaded automatically. All four LightRAG
storage types (KV, vector, graph, doc status) are served by SNKV, so no
external database is needed.
"""
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


def main() -> None:
    from lightrag.api.lightrag_server import main as _main
    _main()


if __name__ == "__main__":
    main()
