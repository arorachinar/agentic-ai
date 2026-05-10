"""
L45 · Autonomous Research Agent — System Architecture
======================================================
Matches the architecture diagram exactly:

  API Gateway (POST /research)
       ↓
  Orchestrator  [budget · circuit-breaker · trace]
       ↓
  Planner  →  Retriever (hybrid search · ChromaDB)
                  ↓
             Validator  (cosine gate ≥0.72)
                  ↓  replan on quality failure ↑
             Synthesizer (hop-attributed)
                  ↓
         Evaluation Pipeline (faithfulness · relevancy · recall · precision)
                  ↓
            API  [trace timeline · eval scores · PROD_READY flag]

Storage:
  ChromaDB  — 30-doc corpus
  SQLite    — trace store
  Redis     — PROD_READY flag

Run API:
    python agent.py serve

Run CLI:
    python agent.py

Run via Codex CLI:
    codex exec --dangerously-bypass-approvals-and-sandbox "Run agent.py"
"""

import os
import json
import sqlite3
import textwrap
import time
import sys
from datetime import datetime
from typing import Optional

import arxiv
import chromadb
import numpy as np
import uvicorn
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL            = "codex-mini-latest"
MAX_PAPERS       = 30                   # diagram: 30-doc corpus
COSINE_GATE      = 0.72                 # validator threshold from diagram
CONFIDENCE_MIN   = 6                    # replan threshold (1–10)
MAX_HOPS         = 4                    # circuit-breaker: max plan steps
MAX_RETRIES      = 2                    # circuit-breaker: retries per step
DB_PATH          = "./chroma_db"
SQLITE_PATH      = "./traces.db"
REDIS_HOST       = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.environ.get("REDIS_PORT", 6379))

ARXIV_TOPICS = [
    "large language models agents",
    "retrieval augmented generation",
    "transformer architecture attention",
    "LangChain autonomous agents",
    "chain of thought reasoning",
]

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
app           = FastAPI(title="L45 · Autonomous Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis (graceful fallback if not running) ──
try:
    import redis as redis_lib
    redis_client    = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client    = None
    REDIS_AVAILABLE = False
    print("  ⚠  Redis unavailable — PROD_READY flag uses in-memory fallback.")

PROD_READY_STORE: dict = {}  # in-memory fallback


# ─────────────────────────────────────────────
# SQLITE — TRACE STORE
# ─────────────────────────────────────────────
def init_sqlite():
    con = sqlite3.connect(SQLITE_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            question     TEXT NOT NULL,
            plan         TEXT,
            executions   TEXT,
            eval_scores  TEXT,
            final_answer TEXT,
            prod_ready   INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()


def save_trace_sqlite(run_id: str, trace: dict):
    con = sqlite3.connect(SQLITE_PATH)
    con.execute(
        """INSERT INTO traces
           (run_id, timestamp, question, plan, executions, eval_scores, final_answer, prod_ready)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            trace["timestamp"],
            trace["question"],
            json.dumps(trace.get("plan", [])),
            json.dumps(trace.get("executions", [])),
            json.dumps(trace.get("eval_scores", {})),
            trace.get("final_answer", ""),
            int(trace.get("prod_ready", False)),
        ),
    )
    con.commit()
    con.close()


def get_traces_sqlite(limit: int = 20) -> list[dict]:
    con  = sqlite3.connect(SQLITE_PATH)
    rows = con.execute(
        "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    keys = ["id", "run_id", "timestamp", "question", "plan",
            "executions", "eval_scores", "final_answer", "prod_ready"]
    return [dict(zip(keys, r)) for r in rows]


# ─────────────────────────────────────────────
# REDIS — PROD_READY FLAG
# ─────────────────────────────────────────────
def set_prod_ready(run_id: str, value: bool):
    if REDIS_AVAILABLE:
        redis_client.set(f"prod_ready:{run_id}", int(value))
    else:
        PROD_READY_STORE[run_id] = value


def get_prod_ready(run_id: str) -> bool:
    if REDIS_AVAILABLE:
        val = redis_client.get(f"prod_ready:{run_id}")
        return bool(int(val)) if val is not None else False
    return PROD_READY_STORE.get(run_id, False)


# ─────────────────────────────────────────────
# CORPUS SEEDING
# ─────────────────────────────────────────────
_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    chroma = chromadb.PersistentClient(path=DB_PATH)
    col    = chroma.get_or_create_collection(
        name="ai_research",
        embedding_function=DefaultEmbeddingFunction(),
    )
    if col.count() == 0:
        _seed_corpus(col)
    else:
        print(f"  ✓  Corpus ready ({col.count()} docs).\n")
    _collection = col
    return col


def _seed_corpus(collection):
    print(f"  📚  Seeding {MAX_PAPERS} ArXiv papers...")
    per_topic = max(1, MAX_PAPERS // len(ARXIV_TOPICS))
    ids, docs, metas, seen = [], [], [], set()

    for topic in ARXIV_TOPICS:
        search = arxiv.Search(
            query=topic,
            max_results=per_topic,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in search.results():
            if paper.entry_id in seen:
                continue
            seen.add(paper.entry_id)
            ids.append(paper.entry_id)
            docs.append(f"Title: {paper.title}\n\nAbstract: {paper.summary}")
            metas.append({
                "title":  paper.title,
                "source": paper.entry_id,
                "year":   str(paper.published.year),
                "topic":  topic,
            })

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"  ✓  Seeded {len(ids)} papers into ChromaDB.\n")


# ─────────────────────────────────────────────
# PLANNER  (hop sequence + replan logic)
# ─────────────────────────────────────────────
def plan(question: str) -> list[str]:
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent(f"""
                    You are a research planner. Decompose the question into
                    2-{MAX_HOPS} atomic sub-questions for multi-hop retrieval.
                    Each sub-question must be specific and self-contained.
                    Sub-questions should build on each other.
                    Return ONLY a valid JSON array of strings. No preamble.
                """),
            },
            {"role": "user", "content": f"Question: {question}"},
        ],
        temperature=0.3,
    )
    raw   = response.choices[0].message.content.strip()
    steps = json.loads(raw)
    return steps[:MAX_HOPS]  # circuit-breaker: cap hops


def replan_step(original_step: str, context: str) -> str:
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "A research sub-question returned low-quality results. "
                    "Rewrite it with different keywords to improve retrieval. "
                    "Return ONLY the rewritten question string."
                ),
            },
            {
                "role": "user",
                "content": f"Original: {original_step}\nContext: {context or 'None.'}\nRewrite:",
            },
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# RETRIEVER  (hybrid search · ChromaDB)
# ─────────────────────────────────────────────
def retrieve(query: str, collection, n: int = 4):
    """Hybrid: semantic embedding search + keyword title boost."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n * 2, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    raw_docs  = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    query_terms = set(query.lower().split())
    scored = []
    for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists):
        title_terms  = set(meta.get("title", "").lower().split())
        keyword_hits = len(query_terms & title_terms)
        boosted      = dist - (0.05 * keyword_hits)
        scored.append((boosted, doc, meta, dist))

    scored.sort(key=lambda x: x[0])
    top = scored[:n]
    return [x[1] for x in top], [x[2] for x in top], [x[3] for x in top]


# ─────────────────────────────────────────────
# VALIDATOR  (cosine gate ≥0.72 · pre-filter)
# ─────────────────────────────────────────────
def validate_retrieval(distances: list[float]) -> tuple[bool, float]:
    """Convert ChromaDB L2 distances to similarity; apply cosine gate."""
    if not distances:
        return False, 0.0
    best_sim = max(1 / (1 + d) for d in distances)
    return best_sim >= COSINE_GATE, round(best_sim, 3)


# ─────────────────────────────────────────────
# EXECUTOR  (retrieve → validate → answer → score)
# ─────────────────────────────────────────────
def execute_step(sub_question: str, collection, context_so_far: str):
    docs, metas, distances = retrieve(sub_question, collection)
    gate_passed, cosine_score = validate_retrieval(distances)
    docs_text = "\n\n---\n\n".join(docs)

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent("""
                    You are a precise research assistant. Answer using ONLY the
                    provided documents and accumulated context.
                    After your answer output on a new line:
                    CONFIDENCE: <integer 1-10>
                    Score honestly — lower if docs only partially support the answer.
                """),
            },
            {
                "role": "user",
                "content": (
                    f"Accumulated context:\n{context_so_far or 'None yet.'}\n\n"
                    f"Retrieved documents:\n{docs_text}\n\n"
                    f"Sub-question: {sub_question}"
                ),
            },
        ],
        temperature=0.2,
    )

    full       = response.choices[0].message.content.strip()
    confidence = 7

    if "CONFIDENCE:" in full:
        answer_lines = []
        for line in full.split("\n"):
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = int(line.split(":")[1].strip())
                except ValueError:
                    pass
            else:
                answer_lines.append(line)
        answer = "\n".join(answer_lines).strip()
    else:
        answer = full

    return answer, confidence, metas, cosine_score, gate_passed


# ─────────────────────────────────────────────
# SYNTHESIZER  (hop-attributed)
# ─────────────────────────────────────────────
def synthesize(question: str, context: str, all_sources: list[dict]) -> str:
    sources_text = "\n".join(
        f"- [{m['title']}] ({m['year']}) — {m['source']}" for m in all_sources
    )
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent("""
                    You are a senior research analyst producing a hop-attributed answer.
                    - Answer in 3-5 paragraphs.
                    - For each key claim, cite which hop (Step N) and paper it came from.
                    - End with a "Key Sources" section.
                    - Be precise — no padding.
                """),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Multi-hop research:\n{context}\n\n"
                    f"Available sources:\n{sources_text}"
                ),
            },
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# EVALUATION PIPELINE
# faithfulness · relevancy · recall · precision · EvaluationGate
# ─────────────────────────────────────────────
def evaluate(question: str, answer: str, sources: list[dict]) -> dict:
    sources_text = "\n".join(f"- {m['title']}" for m in sources[:6])
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent("""
                    You are an evaluation judge (RAGAS-style). Score 0.0–1.0:
                    - faithfulness: every claim supported by sources?
                    - relevancy:    directly addresses the question?
                    - recall:       covers all key aspects?
                    - precision:    free of irrelevant content?
                    Return ONLY valid JSON. Example:
                    {"faithfulness": 0.9, "relevancy": 0.85, "recall": 0.8, "precision": 0.9}
                """),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nAnswer: {answer}\n\nSources:\n{sources_text}"
                ),
            },
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"faithfulness": 0.0, "relevancy": 0.0, "recall": 0.0, "precision": 0.0}

    # EvaluationGate: PROD_READY only if all dimensions ≥ 0.75
    scores["prod_ready"] = all(
        v >= 0.75 for k, v in scores.items() if k != "prod_ready"
    )
    return scores


# ─────────────────────────────────────────────
# ORCHESTRATOR  (budget · circuit-breaker · trace)
# ─────────────────────────────────────────────
def run_agent(question: str) -> dict:
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    col    = get_collection()

    trace = {
        "run_id":       run_id,
        "question":     question,
        "model":        MODEL,
        "timestamp":    datetime.now().isoformat(),
        "plan":         [],
        "executions":   [],
        "eval_scores":  {},
        "final_answer": "",
        "prod_ready":   False,
    }

    print("\n" + "═" * 64)
    print(f"  🔍  QUESTION\n  {question}")
    print(f"  🆔  {run_id}")
    print("═" * 64)

    # ── PLAN ──────────────────────────────────
    print("\n  🧠  PLANNER — generating hop sequence...")
    steps        = plan(question)
    trace["plan"] = steps
    for i, s in enumerate(steps, 1):
        print(f"    Hop {i}: {s}")

    # ── EXECUTE LOOP ────────────────────────────
    print("\n  ⚡  ORCHESTRATOR — executing hops...\n")
    context, all_sources = "", []

    for i, step in enumerate(steps, 1):
        print(f"  ┌─ Hop {i}/{len(steps)}: {step}")
        retries = 0

        while retries <= MAX_RETRIES:
            answer, confidence, metas, cosine_score, gate_passed = execute_step(
                step, col, context
            )
            gate_icon = "✓" if gate_passed else f"✗ (< {COSINE_GATE})"
            print(f"  │  Cosine gate: {cosine_score} {gate_icon}")
            print(f"  │  Confidence:  {confidence}/10")

            if gate_passed and confidence >= CONFIDENCE_MIN:
                break

            retries += 1
            if retries > MAX_RETRIES:
                print(f"  │  ⚠  Max retries — proceeding with best result.")
                break

            print(f"  │  ↺  Replanning (attempt {retries})...")
            step = replan_step(step, context)
            print(f"  │     → {step}")

        replanned = retries > 0
        print(f"  └─ {'✓' if gate_passed else '~'} {'[replanned]' if replanned else ''}\n")

        context += f"\n\nHop {i} — {step}:\n{answer}"
        all_sources.extend(metas)
        trace["executions"].append({
            "hop":            i,
            "step":           step,
            "replanned":      replanned,
            "cosine_score":   cosine_score,
            "gate_passed":    gate_passed,
            "confidence":     confidence,
            "answer_preview": answer[:200],
            "sources":        [m["title"] for m in metas],
        })

    # ── SYNTHESIZE ──────────────────────────────
    print("  ✅  SYNTHESIZER — building hop-attributed answer...")
    final                 = synthesize(question, context, all_sources)
    trace["final_answer"] = final

    # ── EVALUATE ────────────────────────────────
    print("  📊  EVALUATION PIPELINE — scoring...\n")
    eval_scores          = evaluate(question, final, all_sources)
    trace["eval_scores"] = eval_scores
    trace["prod_ready"]  = eval_scores.get("prod_ready", False)

    print(f"  Faithfulness : {eval_scores.get('faithfulness', 0):.2f}")
    print(f"  Relevancy    : {eval_scores.get('relevancy', 0):.2f}")
    print(f"  Recall       : {eval_scores.get('recall', 0):.2f}")
    print(f"  Precision    : {eval_scores.get('precision', 0):.2f}")
    print(f"  PROD_READY   : {'✅ YES' if trace['prod_ready'] else '❌ NO'}")

    # ── REDIS + SQLITE ───────────────────────────
    set_prod_ready(run_id, trace["prod_ready"])
    save_trace_sqlite(run_id, trace)

    print("\n" + "═" * 64)
    print("  FINAL ANSWER\n")
    print(textwrap.fill(final, width=64, subsequent_indent="  "))
    print("═" * 64)
    print(f"\n  📄 Trace → SQLite  |  Run ID: {run_id}\n")

    return trace


# ─────────────────────────────────────────────
# API GATEWAY  (POST /research)
# ─────────────────────────────────────────────
class ResearchRequest(BaseModel):
    question: str

class ResearchResponse(BaseModel):
    run_id:       str
    question:     str
    plan:         list[str]
    final_answer: str
    eval_scores:  dict
    prod_ready:   bool
    executions:   list[dict]


@app.post("/research", response_model=ResearchResponse)
def research_endpoint(req: ResearchRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    trace = run_agent(req.question)
    return ResearchResponse(**{k: trace[k] for k in ResearchResponse.model_fields})


@app.get("/traces")
def list_traces(limit: int = 20):
    """Trace timeline for the React Dashboard."""
    return get_traces_sqlite(limit)


@app.get("/prod_ready/{run_id}")
def check_prod_ready(run_id: str):
    return {"run_id": run_id, "prod_ready": get_prod_ready(run_id)}


@app.get("/health")
def health():
    return {"status": "ok", "redis": REDIS_AVAILABLE, "corpus": get_collection().count()}


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
def main():
    init_sqlite()
    get_collection()

    questions = [
        "How did the Transformer architecture influence the design of modern LLM agent frameworks like LangChain?",
        "What is retrieval-augmented generation and why is it important for reducing hallucinations in LLMs?",
    ]
    for q in questions:
        run_agent(q)
        print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        init_sqlite()
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
    else:
        main()
