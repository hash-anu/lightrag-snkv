"""Load LLM and embedding functions from environment variables (test helper).

Re-exports the bench helper so both share the same resolution logic.
"""
from bench.llm_env import get_llm_and_embed_funcs  # noqa: F401
