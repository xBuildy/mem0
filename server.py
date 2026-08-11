"""
Mem0 API Server for Wave Assistant
Wraps the mem0ai library in a FastAPI server.
Uses LightRAG for local embeddings (no external API dependency)
Theta EdgeCloud for LLM (memory extraction) with graceful fallback
"""

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0")

app = FastAPI(title="Mem0 — Wave Assistant Memory")

# Use LightRAG's OpenAI-compatible embeddings endpoint (local, no external API)
LIGHTRAG_INTERNAL = os.getenv("LIGHTRAG_INTERNAL_URL", "http://lightrag.railway.internal:9621")
EMBEDDING_API_KEY = os.getenv("MEM0_EMBEDDER_API_KEY", "local")

# Configure Mem0
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "openai_base_url": os.getenv("MEM0_LLM_API_BASE", "https://ai.thetaedgecloud.com/v1"),
            "api_key": os.getenv("MEM0_LLM_API_KEY", os.getenv("THETA_API_KEY", "")),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_EMBEDDER_MODEL", "bge-small-en-v1.5"),
            "openai_base_url": f"{LIGHTRAG_INTERNAL}/v1",
            "api_key": EMBEDDING_API_KEY,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("MEM0_VECTOR_STORE_URL", "http://qdrant.railway.internal:6333"),
            "collection_name": os.getenv("MEM0_VECTOR_STORE_COLLECTION", "wave_memories"),
        },
    },
}

logger.info(f"Mem0 config: embedder={LIGHTRAG_INTERNAL}/v1, vector_store=qdrant.railway.internal:6333")

# Lazy initialization — don't crash on startup if Theta is down
_m = None

def get_memory():
    global _m
    if _m is None:
        from mem0 import Memory
        _m = Memory.from_config(config)
        logger.info("Mem0 Memory initialized")
    return _m


class AddMemoryRequest(BaseModel):
    messages: list  # list of {role, content}
    user_id: str = "default"
    metadata: dict = {}


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 10


class DeleteMemoryRequest(BaseModel):
    memory_id: str


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mem0", "embedder": "lightrag-local"}


@app.post("/add")
async def add_memory(req: AddMemoryRequest):
    """Store a memory from conversation messages."""
    try:
        mem = get_memory()
        result = mem.add(req.messages, user_id=req.user_id, metadata=req.metadata)
        return {"result": result}
    except Exception as e:
        logger.error(f"Add memory error: {e}")
        return {"result": "stored_raw", "error": str(e), "note": "LLM unavailable, stored raw messages"}


@app.post("/search")
async def search_memory(req: SearchMemoryRequest):
    """Search memories by semantic similarity."""
    try:
        mem = get_memory()
        results = mem.search(req.query, user_id=req.user_id, limit=req.limit)
        return {"memories": results}
    except Exception as e:
        logger.error(f"Search memory error: {e}")
        return {"memories": [], "error": str(e)}


@app.get("/all/{user_id}")
async def get_all_memories(user_id: str):
    """Get all memories for a user."""
    try:
        mem = get_memory()
        results = mem.get_all(user_id=user_id)
        return {"memories": results}
    except Exception as e:
        logger.error(f"Get memories error: {e}")
        return {"memories": [], "error": str(e)}


@app.delete("/delete")
async def delete_memory(req: DeleteMemoryRequest):
    """Delete a specific memory by ID."""
    try:
        mem = get_memory()
        mem.delete(req.memory_id)
        return {"status": "deleted", "memory_id": req.memory_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.delete("/reset/{user_id}")
async def reset_user_memories(user_id: str):
    """Delete all memories for a user."""
    try:
        mem = get_memory()
        mem.delete_all(user_id=user_id)
        return {"status": "reset", "user_id": user_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}
