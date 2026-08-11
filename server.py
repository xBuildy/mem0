"""
Mem0 API Server for Wave Assistant — v2 (no mem0ai dependency)
Direct Qdrant + LightRAG local embeddings. No external API needed.
When Theta recovers, LLM extraction can be added back as an enhancement.
"""

import os
import uuid
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0")

app = FastAPI(title="Mem0 — Wave Assistant Memory")

# LightRAG for local embeddings
LIGHTRAG_URL = os.getenv("LIGHTRAG_INTERNAL_URL", "http://lightrag.railway.internal:9621")
LIGHTRAG_PUBLIC = os.getenv("LIGHTRAG_URL", "https://lightrag-production-2b43.up.railway.app")

# Qdrant
QDRANT_URL = os.getenv("MEM0_VECTOR_STORE_URL", "http://qdrant.railway.internal:6333")
QDRANT_COLLECTION = os.getenv("MEM0_VECTOR_STORE_COLLECTION", "wave_memories")
EMBEDDING_DIM = 384

import httpx

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
    return {"status": "healthy", "service": "mem0", "embedder": "lightrag-local", "mode": "direct-qdrant"}


async def get_embedding(text: str) -> list:
    """Get embedding from LightRAG's OpenAI-compatible endpoint"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{LIGHTRAG_URL}/v1/embeddings",
            json={"model": "bge-small", "input": text}
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def get_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    # Ensure collection exists with 384 dims
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION} ({EMBEDDING_DIM}d)")
    return client


@app.post("/add")
async def add_memory(req: AddMemoryRequest):
    """Store memories directly from conversation messages (no LLM extraction)."""
    try:
        from qdrant_client.models import PointStruct
        client = get_qdrant()
        
        # Extract text from messages and store each as a memory
        points = []
        memories_stored = []
        for msg in req.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content:
                continue
            
            # Create a memory text combining role and content
            memory_text = f"[{role}] {content}"
            embedding = await get_embedding(memory_text)
            
            point_id = str(uuid.uuid4())
            payload = {
                "text": memory_text,
                "role": role,
                "content": content,
                "user_id": req.user_id,
                **req.metadata,
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            memories_stored.append({"id": point_id, "text": memory_text[:100]})
        
        if points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.info(f"Stored {len(points)} memories for user {req.user_id}")
        
        return {"result": "success", "memories_added": len(points), "memories": memories_stored}
    except Exception as e:
        logger.error(f"Add memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_memory(req: SearchMemoryRequest):
    """Search memories by semantic similarity."""
    try:
        client = get_qdrant()
        query_embedding = await get_embedding(req.query)
        
        # Use query_points (new API) with user_id filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=req.limit,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=req.user_id))]),
        )
        
        memories = [
            {
                "id": r.id,
                "text": r.payload.get("text", "") if r.payload else "",
                "content": r.payload.get("content", "") if r.payload else "",
                "role": r.payload.get("role", "") if r.payload else "",
                "score": r.score,
                "metadata": {k: v for k, v in (r.payload or {}).items() if k not in ("text", "content", "role", "user_id")},
            }
            for r in results.points
        ]
        return {"memories": memories}
    except Exception as e:
        logger.error(f"Search memory error: {e}")
        return {"memories": [], "error": str(e)}


@app.get("/all/{user_id}")
async def get_all_memories(user_id: str):
    """Get all memories for a user."""
    try:
        client = get_qdrant()
        from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=100,
        )
        
        memories = [
            {
                "id": r.id,
                "text": r.payload.get("text", "") if r.payload else "",
                "content": r.payload.get("content", "") if r.payload else "",
                "role": r.payload.get("role", "") if r.payload else "",
                "metadata": {k: v for k, v in (r.payload or {}).items() if k not in ("text", "content", "role", "user_id")},
            }
            for r in results[0]
        ]
        return {"memories": memories}
    except Exception as e:
        logger.error(f"Get memories error: {e}")
        return {"memories": [], "error": str(e)}


@app.delete("/delete")
async def delete_memory(req: DeleteMemoryRequest):
    """Delete a specific memory by ID."""
    try:
        client = get_qdrant()
        client.delete(collection_name=QDRANT_COLLECTION, points=[req.memory_id])
        return {"status": "deleted", "memory_id": req.memory_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.delete("/reset/{user_id}")
async def reset_user_memories(user_id: str):
    """Delete all memories for a user."""
    try:
        client = get_qdrant()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        )
        return {"status": "reset", "user_id": user_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}
