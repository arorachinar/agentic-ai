# AGENTS.md — L45 Autonomous Research Agent

## Architecture (matches diagram)
```
API Gateway (POST /research)
     ↓
Orchestrator  [budget · circuit-breaker · trace]
     ↓
Planner  →  Retriever (hybrid search · ChromaDB 30-doc corpus)
                ↓
           Validator  (cosine gate ≥0.72 · pre-filter)
                ↓  ← replan on quality failure
           Synthesizer (hop-attributed output)
                ↓
       Evaluation Pipeline (faithfulness · relevancy · recall · precision)
                ↓
          /traces  /prod_ready  [trace timeline · eval scores · PROD_READY]

Storage:
  ChromaDB → 30-doc ArXiv corpus (./chroma_db)
  SQLite   → trace store         (./traces.db)
  Redis    → PROD_READY flag     (graceful fallback if unavailable)
```

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Optional Redis (for PROD_READY flag)
docker run -d -p 6379:6379 redis
```

## Run
```bash
# CLI mode (runs 2 demo questions)
python agent.py

# API server mode
python agent.py serve
# → POST http://localhost:8000/research
# → GET  http://localhost:8000/traces
# → GET  http://localhost:8000/prod_ready/{run_id}
# → GET  http://localhost:8000/health
```

## Run via Codex CLI
```bash
# Non-interactive
codex exec --dangerously-bypass-approvals-and-sandbox "Run agent.py"

# Delegate a feature
codex "Add a --question CLI flag to agent.py so I can pass questions inline"
codex "Add a Streamlit dashboard that polls /traces every 5 seconds"
codex "Make the executor hops run in parallel with asyncio.gather()"
```

## Key Config
| Variable       | Default           | Description                    |
|----------------|-------------------|--------------------------------|
| MODEL          | codex-mini-latest | OpenAI model                   |
| MAX_PAPERS     | 30                | ArXiv papers to seed           |
| COSINE_GATE    | 0.72              | Validator similarity threshold |
| CONFIDENCE_MIN | 6                 | Replan trigger (1–10)          |
| MAX_HOPS       | 4                 | Circuit-breaker: max steps     |
| MAX_RETRIES    | 2                 | Retries per hop                |
