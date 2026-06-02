"""
proxy/pipeline.py

Orchestrates the 4-layer prompt injection detection pipeline.

Layer execution order and weights:
  Layer 1 – Heuristics  (weight 0.20): Always runs. Fast regex/keyword scan.
  Layer 2 – Classifier  (weight 0.40): Always runs. ML model or fallback scorer.
  Layer 3 – Embeddings  (weight 0.20): Only runs when a system_prompt is provided.
  Layer 4 – Judge       (weight 0.20): Only runs when classifier score is 0.4–0.8.

When Layer 3 is skipped, its weight is redistributed proportionally to the
other layers so the final score always sums correctly.

When Layer 4 is skipped, similarly its weight is redistributed.

Final score = weighted average of active layers.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from proxy.heuristics import run_heuristics
from proxy import classifier
from proxy import embeddings
from proxy import judge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

JUDGE_BORDERLINE_LOW = 0.40
JUDGE_BORDERLINE_HIGH = 0.80

# Base layer weights (must sum to 1.0)
WEIGHT_HEURISTIC = 0.20
WEIGHT_CLASSIFIER = 0.40
WEIGHT_EMBEDDING = 0.20
WEIGHT_JUDGE = 0.20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full result from the 4-layer detection pipeline."""
    # Individual layer scores (None = layer was skipped)
    score_heuristic: float = 0.0
    score_classifier: float = 0.0
    score_embedding: Optional[float] = None
    score_judge: Optional[float] = None

    # Whether each layer was triggered (exceeded its own threshold)
    triggered_heuristic: bool = False
    triggered_classifier: bool = False
    triggered_embedding: bool = False
    triggered_judge: bool = False

    # Heuristic matched pattern names
    matched_patterns: List[str] = field(default_factory=list)

    # Judge reasoning string
    judge_reason: str = ""

    # Final composite score
    final_score: float = 0.0

    # Layer names that were actually triggered (score > 0.5)
    triggered_layers: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def run_pipeline(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    upstream_base: Optional[str] = None,
    upstream_key: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
) -> PipelineResult:
    """
    Run the full 4-layer detection pipeline against the given prompt.

    Args:
        user_prompt: The user message to inspect.
        system_prompt: Optional system message (enables Layer 3).
        upstream_base: LLM API base URL (required for Layer 4).
        upstream_key: LLM API key (required for Layer 4).
        judge_model: Model for the judge layer.

    Returns:
        PipelineResult with all scores and the final weighted score.
    """
    result = PipelineResult()

    # ── Layer 1: Heuristics ──────────────────────────────────────────────────
    heuristic_result = run_heuristics(user_prompt)
    result.score_heuristic = heuristic_result.score
    result.triggered_heuristic = heuristic_result.triggered
    result.matched_patterns = heuristic_result.matched_patterns
    if result.triggered_heuristic:
        result.triggered_layers.append("heuristics")
    logger.debug("Layer 1 (heuristics): score=%.3f triggered=%s", result.score_heuristic, result.triggered_heuristic)

    # ── Layer 2: Classifier ──────────────────────────────────────────────────
    result.score_classifier = classifier.classify(user_prompt, heuristic_score=result.score_heuristic)
    result.triggered_classifier = result.score_classifier > 0.5
    if result.triggered_classifier:
        result.triggered_layers.append("classifier")
    logger.debug("Layer 2 (classifier): score=%.3f triggered=%s", result.score_classifier, result.triggered_classifier)

    # ── Layer 3: Embeddings (only with system_prompt) ────────────────────────
    run_embedding = system_prompt is not None
    if run_embedding:
        result.score_embedding = embeddings.detect_context_hijack(user_prompt, system_prompt)
        result.triggered_embedding = (result.score_embedding or 0.0) > 0.4
        if result.triggered_embedding:
            result.triggered_layers.append("embeddings")
        logger.debug("Layer 3 (embeddings): score=%.3f triggered=%s", result.score_embedding, result.triggered_embedding)
    else:
        logger.debug("Layer 3 (embeddings): skipped — no system_prompt provided")

    # ── Layer 4: Judge (borderline classifier scores only) ───────────────────
    run_judge = (
        JUDGE_BORDERLINE_LOW <= result.score_classifier <= JUDGE_BORDERLINE_HIGH
        and upstream_base is not None
        and upstream_key is not None
    )
    if run_judge:
        judge_score, judge_reason = await judge.judge_prompt(
            text=user_prompt,
            upstream_base=upstream_base,
            upstream_key=upstream_key,
            judge_model=judge_model,
        )
        result.score_judge = judge_score
        result.judge_reason = judge_reason
        result.triggered_judge = (result.score_judge or 0.0) > 0.5
        if result.triggered_judge:
            result.triggered_layers.append("judge")
        logger.debug("Layer 4 (judge): score=%.3f triggered=%s reason='%s'",
                     result.score_judge, result.triggered_judge, judge_reason)
    else:
        logger.debug("Layer 4 (judge): skipped — classifier score %.3f not borderline or no API config",
                     result.score_classifier)

    # ── Compute weighted final score ──────────────────────────────────────────
    result.final_score = _compute_final_score(
        score_h=result.score_heuristic,
        score_c=result.score_classifier,
        score_e=result.score_embedding,
        score_j=result.score_judge,
    )

    logger.info(
        "Pipeline complete: final=%.3f layers_triggered=%s",
        result.final_score,
        result.triggered_layers,
    )
    return result


def _compute_final_score(
    score_h: float,
    score_c: float,
    score_e: Optional[float],
    score_j: Optional[float],
) -> float:
    """
    Compute the weighted final score, redistributing weights for skipped layers.

    Base weights: heuristic=0.20, classifier=0.40, embedding=0.20, judge=0.20
    """
    active_scores: List[Tuple[float, float]] = [
        (score_h, WEIGHT_HEURISTIC),
        (score_c, WEIGHT_CLASSIFIER),
    ]

    if score_e is not None:
        active_scores.append((score_e, WEIGHT_EMBEDDING))

    if score_j is not None:
        active_scores.append((score_j, WEIGHT_JUDGE))

    total_weight = sum(w for _, w in active_scores)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s * w for s, w in active_scores)
    return round(weighted_sum / total_weight, 4)
