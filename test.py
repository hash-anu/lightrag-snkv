import asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag_snkv import register, get_llm_and_embed_funcs

load_dotenv("/path/to/your/lightrag/.env")   # load your existing LightRAG .env

llm_func, embed_func = get_llm_and_embed_funcs()  # reads LLM_BINDING, LLM_MODEL, EMBEDDING_BINDING, etc.

register()

async def main():
    rag = LightRAG(
        working_dir="./my_rag",
        llm_model_func=llm_func,
        embedding_func=embed_func,
        kv_storage="SNKVKVStorage",
        vector_storage="SNKVVectorStorage",
        graph_storage="SNKVGraphStorage",
        doc_status_storage="SNKVDocStatusStorage",
    )

    await rag.initialize_storages()
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
