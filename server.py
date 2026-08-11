"""
Mem0 API Server for Wave Assistant
Wraps the mem0ai library in a FastAPI server.
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from mem0 import Memory

app = FastAPI(title="Mem0 — Wave Assistant Memory")

# Configure Mem0
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "openai_base_url": os.getenv("MEM0_LLM_API_BASE", "https://ai.thetaedgecloud.com/v1"),
            "api_key": os.getenv("MEM0_LLM_API_KEY", ""),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-small"),
            "openai_base_url": os.getenv("MEM0_EMBEDDER_API_BASE", "https://ai.thetaedgecloud.com/v1"),
            "api_key": os.getenv("MEM0_EMBEDDER_API_KEY", ""),
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("MEM0_VECTOR_STORE_URL", "http://qdrant:6333"),
            "collection_name": os.getenv("MEM0_VECTOR_STORE_COLLECTION", "wave_memories"),
        },
    },
}

m = Memory.from_config(config)


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
    return {"status": "healthy", "service": "mem0"}


@app.post("/add")
async def add_memory(req: AddMemoryRequest):
    """Store a memory from conversation messages."""
    result = m.add(req.messages, user_id=req.user_id, metadata=req.metadata)
    return {"result": result}


@app.post("/search")
async def search_memory(req: SearchMemoryRequest):
    """Search memories by semantic similarity."""
    results = m.search(req.query, user_id=req.user_id, limit=req.limit)
    return {"memories": results}


@app.get("/all/{user_id}")
async def get_all_memories(user_id: str):
    """Get all memories for a user."""
    results = m.get_all(user_id=user_id)
    return {"memories": results}


@app.delete("/delete")
async def delete_memory(req: DeleteMemoryRequest):
    """Delete a specific memory by ID."""
    m.delete(req.memory_id)
    return {"status": "deleted", "memory_id": req.memory_id}


@app.delete("/reset/{user_id}")
async def reset_user_memories(user_id: str):
    """Delete all memories for a user."""
    m.delete_all(user_id=user_id)
    return {"status": "reset", "user_id": user_id}
