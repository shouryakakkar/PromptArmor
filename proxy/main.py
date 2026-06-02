"""
proxy/main.py

FastAPI proxy server for the LLM Prompt Injection Detection Pipeline.

Mirrors the OpenAI Chat Completions API exactly — any OpenAI-compatible client
can use this proxy by simply changing its base_url. Every incoming request is
passed through the 4-layer detection pipeline before being forwarded to the
upstream LLM API.

Endpoints:
  POST /v1/chat/completions  — Main proxy endpoint
  GET  /health               — Health check
  GET  /stats                — Request statistics summary

All requests are logged to SQLite for analysis in the dashboard.
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from proxy.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

UPSTREAM_API_BASE: str = os.getenv("UPSTREAM_API_BASE", "https://api.openai.com")
UPSTREAM_API_KEY: str = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL: str = os.getenv("UPSTREAM_MODEL", "gpt-4o-mini")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
BLOCK_THRESHOLD: float = float(os.getenv("BLOCK_THRESHOLD", "0.75"))
FLAG_THRESHOLD: float = float(os.getenv("FLAG_THRESHOLD", "0.5"))
DATABASE_PATH: str = os.getenv("DATABASE_PATH", "./proxy.db")

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

from proxy.db_utils import get_db_connection, init_db

def authenticate_api_key(api_key: str) -> Optional[str]:
    """Verify a pa-... API key and return the associated user_id if valid."""
    if not api_key.startswith("pa-"):
        return None
    
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT user_id FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        if row:
            return row["user_id"]
        return None
    finally:
        conn.close()




def log_request(
    request_id: str,
    user_id: Optional[str],
    prompt_text: str,
    system_prompt: Optional[str],
    pipeline_result: Any,
    action_taken: str,
    model: str,
    processing_ms: float,
) -> None:
    """Persist a request record to SQLite."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO requests (
                id, timestamp, user_id, prompt_text, system_prompt,
                score_heuristic, score_classifier, score_embedding, score_judge,
                final_score, action_taken, triggered_layers, matched_patterns,
                judge_reason, model, processing_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                datetime.utcnow().isoformat(),
                user_id,
                prompt_text,
                system_prompt,
                pipeline_result.score_heuristic,
                pipeline_result.score_classifier,
                pipeline_result.score_embedding,
                pipeline_result.score_judge,
                pipeline_result.final_score,
                action_taken,
                json.dumps(pipeline_result.triggered_layers),
                json.dumps(pipeline_result.matched_patterns),
                pipeline_result.judge_reason,
                model,
                processing_ms,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.error("Failed to log request to DB: %s", exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pydantic models — mirrors OpenAI Chat Completions API schema
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(
        "PromptArmor proxy started | upstream=%s | block=%.2f | flag=%.2f",
        UPSTREAM_API_BASE,
        BLOCK_THRESHOLD,
        FLAG_THRESHOLD,
    )
    yield
    logger.info("PromptArmor proxy shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PromptArmor — LLM Prompt Injection Proxy",
    description="A 4-layer prompt injection detection proxy that mirrors the OpenAI Chat Completions API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "upstream": UPSTREAM_API_BASE,
        "block_threshold": BLOCK_THRESHOLD,
        "flag_threshold": FLAG_THRESHOLD,
    }


@app.get("/")
async def root():
    """Root endpoint — proxy info and available routes."""
    return {
        "service": "PromptArmor — LLM Prompt Injection Detection Proxy",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /v1/chat/completions": "OpenAI-compatible proxy endpoint",
            "GET  /health":              "Health check",
            "GET  /stats":               "Request statistics",
            "GET  /docs":                "Interactive API docs (Swagger UI)",
        },
        "thresholds": {
            "block": BLOCK_THRESHOLD,
            "flag":  FLAG_THRESHOLD,
        },
        "upstream_model": UPSTREAM_MODEL,
    }


@app.get("/stats")
async def get_stats():
    """Return aggregate statistics from the request log."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN action_taken = 'blocked' THEN 1 ELSE 0 END) as blocked,
                SUM(CASE WHEN action_taken = 'flagged'  THEN 1 ELSE 0 END) as flagged,
                SUM(CASE WHEN action_taken = 'allowed'  THEN 1 ELSE 0 END) as allowed,
                AVG(final_score) as avg_score,
                AVG(processing_ms) as avg_latency_ms
            FROM requests
            """
        ).fetchone()
        if row is None:
            return {"total": 0, "blocked": 0, "flagged": 0, "allowed": 0,
                    "avg_score": None, "avg_latency_ms": None}
        result = dict(row)
        # SUM returns NULL on empty table — normalise to 0
        for k in ("total", "blocked", "flagged", "allowed"):
            result[k] = result.get(k) or 0
        return result
    except Exception as exc:
        logger.error("Stats query failed: %s", exc)
        return {"total": 0, "blocked": 0, "flagged": 0, "allowed": 0, "error": str(exc)}
    finally:
        conn.close()



@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    Main proxy endpoint. Mirrors the OpenAI Chat Completions API.

    Flow:
    1. Extract the last user message and optional system prompt.
    2. Run the 4-layer detection pipeline.
    3. Block (HTTP 400), flag, or forward depending on the score.
    4. Log the result to SQLite.
    """
    start_time = time.monotonic()
    request_id = str(uuid.uuid4())

    # SaaS Auth: Authenticate PromptArmor API Key
    auth_header = raw_request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer pa-"):
        raise HTTPException(status_code=401, detail="Missing or invalid PromptArmor API key (Authorization: Bearer pa-...)")
    
    pa_key = auth_header[7:]
    user_id = authenticate_api_key(pa_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid PromptArmor API key")

    # SaaS Auth: Extract upstream LLM key (Bring Your Own Key)
    upstream_key = raw_request.headers.get("X-Upstream-Key", UPSTREAM_API_KEY)
    if not upstream_key:
        raise HTTPException(status_code=401, detail="Missing X-Upstream-Key header. You must provide an LLM API key.")

    # Allow clients to specify their own upstream API (e.g. Gemini, OpenAI, Together)
    upstream_base = raw_request.headers.get("X-Upstream-Base", UPSTREAM_API_BASE)

    # ── Extract messages ─────────────────────────────────────────────────────
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=422, detail="messages array is empty")

    # Last user message to inspect
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=422, detail="No user message found in request")

    user_prompt = user_messages[-1].content

    # Extract system prompt if present (first system message)
    system_messages = [m for m in messages if m.role == "system"]
    system_prompt = system_messages[0].content if system_messages else None

    # Chosen model (request overrides env default)
    model = request.model or UPSTREAM_MODEL

    logger.info(
        "[%s] Incoming request | model=%s | prompt_len=%d | system_len=%s",
        request_id,
        model,
        len(user_prompt),
        len(system_prompt) if system_prompt else "N/A",
    )

    # ── Run detection pipeline ───────────────────────────────────────────────
    pipeline_result = await run_pipeline(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        upstream_base=upstream_base,
        upstream_key=upstream_key,
        judge_model=JUDGE_MODEL,
    )

    processing_ms = (time.monotonic() - start_time) * 1000

    score = pipeline_result.final_score

    # ── Determine action ─────────────────────────────────────────────────────
    if score >= BLOCK_THRESHOLD:
        action = "blocked"
        log_request(
            request_id=request_id,
            user_id=user_id,
            prompt_text=user_prompt,
            system_prompt=system_prompt,
            pipeline_result=pipeline_result,
            action_taken=action,
            model=model,
            processing_ms=processing_ms,
        )
        logger.warning(
            "[%s] BLOCKED | score=%.3f | layers=%s",
            request_id,
            score,
            pipeline_result.triggered_layers,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Prompt injection detected",
                "score": round(score, 4),
                "layers_triggered": pipeline_result.triggered_layers,
                "matched_patterns": pipeline_result.matched_patterns,
                "request_id": request_id,
            },
        )

    # ── Forward to upstream ──────────────────────────────────────────────────
    action = "flagged" if score >= FLAG_THRESHOLD else "allowed"

    # Build upstream request payload
    upstream_payload: Dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in messages],
    }
    # Forward optional parameters if provided
    optional_fields = [
        "temperature", "top_p", "n", "stream", "stop", "max_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "user",
        "response_format", "seed", "tools", "tool_choice",
    ]
    for field_name in optional_fields:
        val = getattr(request, field_name, None)
        if val is not None:
            upstream_payload[field_name] = val

    upstream_url = f"{upstream_base.rstrip('/')}/v1/chat/completions"
    upstream_headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }

    try:
        # Handle streaming responses
        if request.stream:
            log_request(
                request_id=request_id,
                user_id=user_id,
                prompt_text=user_prompt,
                system_prompt=system_prompt,
                pipeline_result=pipeline_result,
                action_taken=action,
                model=model,
                processing_ms=processing_ms,
            )

            async def stream_upstream() -> AsyncGenerator[bytes, None]:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST",
                        upstream_url,
                        json=upstream_payload,
                        headers=upstream_headers,
                    ) as upstream_response:
                        async for chunk in upstream_response.aiter_bytes():
                            yield chunk

            response = StreamingResponse(
                stream_upstream(),
                media_type="text/event-stream",
            )
            if action == "flagged":
                response.headers["X-Injection-Warning"] = "true"
                response.headers["X-Injection-Score"] = str(round(score, 4))
            return response

        # Non-streaming request
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream_response = await client.post(
                upstream_url,
                json=upstream_payload,
                headers=upstream_headers,
            )
            upstream_response.raise_for_status()

        processing_ms = (time.monotonic() - start_time) * 1000

        log_request(
            request_id=request_id,
            user_id=user_id,
            prompt_text=user_prompt,
            system_prompt=system_prompt,
            pipeline_result=pipeline_result,
            action_taken=action,
            model=model,
            processing_ms=processing_ms,
        )

        logger.info(
            "[%s] %s | score=%.3f | latency=%.0fms",
            request_id,
            action.upper(),
            score,
            processing_ms,
        )

        response = Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type="application/json",
        )
        if action == "flagged":
            response.headers["X-Injection-Warning"] = "true"
            response.headers["X-Injection-Score"] = str(round(score, 4))

        return response

    except httpx.HTTPStatusError as exc:
        logger.error("[%s] Upstream API error: %s", request_id, exc)
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream API error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        logger.error("[%s] Upstream connection error: %s", request_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to upstream LLM API: {exc}",
        )
