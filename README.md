# Mem0 — Long-Term Memory for Wave Assistant

Self-hosted memory layer that stores user preferences, facts, and conversation context.

## What It Does

- Extracts important facts/preferences from conversations automatically
- Stores them as vector embeddings in Qdrant
- Recalls relevant memories when Wave Assistant processes a query
- Learns over time: user's coding style, preferred tools, project context, habits

## Architecture

```
Wave Assistant backend function
        ↓
   Mem0 API (:8012)
    ├── /add → Extract & store memory from messages
    ├── /search → Semantic recall of relevant memories
    ├── /all/{user_id} → Get all memories for a user
    └── LLM calls → Theta EdgeCloud GLM-5.2 (for memory extraction)
        ↓
   Qdrant (:6333) — vector storage
```

## Railway Deployment

```
Railway Dashboard → spirited-playfulness project
+ New Service → Docker Image (or GitHub repo deploy)
  Source: this folder (has Dockerfile + server.py)
  Name: mem0
  Port: 8012
  Volume: /app/data (persistent disk, min 1GB)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| THETA_API_KEY | Theta EdgeCloud API key |
| MEM0_VECTOR_STORE_URL | Qdrant endpoint (internal Railway URL) |
| MEM0_LLM_MODEL | LLM model on Theta EdgeCloud |

Set `THETA_API_KEY` to your Theta EdgeCloud API key.
Set `MEM0_VECTOR_STORE_URL` to `http://qdrant.railway.internal:6333`.

## API Endpoints

### Add Memory
```bash
curl -X POST https://mem0-production.up.railway.app/add \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I prefer dark mode and use Surge for coding"},
      {"role": "assistant", "content": "Noted! Dark mode and Surge."}
    ],
    "user_id": "eddie"
  }'
```

### Search Memories
```bash
curl -X POST https://mem0-production.up.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What editor does Eddie prefer?",
    "user_id": "eddie"
  }'
# Returns: {"memories": [{"id":"...", "memory":"Eddie prefers dark mode and uses Surge for coding", ...}]}
```

### Get All Memories
```bash
curl https://mem0-production.up.railway.app/all/eddie
```

### Delete Memory
```bash
curl -X DELETE https://mem0-production.up.railway.app/delete \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "abc123"}'
```

## Verify After Deploy

```bash
# Health check
curl https://mem0-production.up.railway.app/health
# Should return: {"status":"healthy","service":"mem0"}

# Add test memory
curl -X POST https://mem0-production.up.railway.app/add \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Wave OS uses Theta EdgeCloud for GPU compute"}],"user_id":"test"}'

# Search for it
curl -X POST https://mem0-production.up.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What GPU platform does Wave OS use?","user_id":"test"}'
```

## Cost

- Railway compute: ~$0.05/hr when active
- Theta EdgeCloud: TFUEL per LLM call (minimal — only for memory extraction)
- Qdrant storage: shared with LightRAG

## Wave Assistant Integration

Wave Assistant backend function will:
1. After each conversation → POST messages to Mem0 `/add` (extracts facts)
2. Before generating response → POST query to Mem0 `/search` (recall context)
3. Combine Mem0 memories + LightRAG knowledge → send to GLM-5.2
4. GLM-5.2 generates response with full context awareness
