"""Start LightRAG web server with SNKV storage backends.

Run from the lightrag-snkv directory:
    python server.py

The .env file in this directory is loaded automatically.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # load .env before anything else

# Register SNKV backends with LightRAG's storage registry
from lightrag_snkv import register
register()

# Override storage backends to use SNKV
os.environ["LIGHTRAG_KV_STORAGE"] = "SNKVKVStorage"
os.environ["LIGHTRAG_VECTOR_STORAGE"] = "SNKVVectorStorage"
os.environ["LIGHTRAG_GRAPH_STORAGE"] = "SNKVGraphStorage"
os.environ["LIGHTRAG_DOC_STATUS_STORAGE"] = "SNKVDocStatusStorage"

import uvicorn
from lightrag.api.lightrag_server import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9621"))
    uvicorn.run(app, host=host, port=port)
