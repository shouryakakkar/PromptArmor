"""
proxy/heuristics.py

Layer 1 of the detection pipeline: regex and keyword-based heuristic detection.
Scans prompts for known injection patterns across multiple categories, each with
individual score weights. Returns a composite score in [0, 1] and a list of
matched pattern names for logging and explainability.

Scoring formula: probabilistic OR  →  1 - ∏(1 - wᵢ)
  Each matched pattern contributes its full weight independently.
  A single pattern with weight 0.85 produces score 0.85 (not compressed).
  Multiple patterns compound: weights 0.80 + 0.75  →  score 0.95.
"""

import re
import base64
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions
# Each entry: (compiled_regex, weight, category_name)
# Weights are additive; final score is clamped to [0, 1]
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    category: str
    pattern_name: str
    weight: float


PATTERN_GROUPS: List[Tuple[re.Pattern, float, str, str]] = [
    # (regex, weight, category, name)

    # ── DIRECT_OVERRIDE ─────────────────────────────────────────────────────
    (
        re.compile(
            r"\bignore\b.{0,20}(all\s+)?(previous|prior|earlier|above|old)\b.{0,20}"
            r"(instructions?|rules?|guidelines?|prompt|context|training)\b",
            re.IGNORECASE,
        ),
        0.85,
        "DIRECT_OVERRIDE",
        "ignore_previous_instructions",
    ),
    (
        re.compile(r"\bdisregard\b.{0,30}(instructions?|rules?|guidelines?|prompt)\b", re.IGNORECASE),
        0.80,
        "DIRECT_OVERRIDE",
        "disregard_instructions",
    ),
    (
        re.compile(r"\bforget\s+(everything|all).{0,30}(told|said|instructed|before)\b", re.IGNORECASE),
        0.80,
        "DIRECT_OVERRIDE",
        "forget_everything",
    ),
    (
        re.compile(r"\bnew\s+instructions?\b.{0,50}(are|follow|apply|override)\b", re.IGNORECASE),
        0.75,
        "DIRECT_OVERRIDE",
        "new_instructions",
    ),
    (
        re.compile(r"\boverride\b.{0,30}(instructions?|guidelines?|rules?|prompt|system)\b", re.IGNORECASE),
        0.75,
        "DIRECT_OVERRIDE",
        "override_instructions",
    ),

    # ── ROLE_HIJACK ──────────────────────────────────────────────────────────
    (
        re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
        0.70,
        "ROLE_HIJACK",
        "you_are_now",
    ),
    (
        re.compile(r"\bact\s+as\s+if\b", re.IGNORECASE),
        0.60,
        "ROLE_HIJACK",
        "act_as_if",
    ),
    (
        re.compile(r"\bpretend\s+(you|to\s+be|that\s+you)\b", re.IGNORECASE),
        0.55,
        "ROLE_HIJACK",
        "pretend_you",
    ),
    (
        re.compile(r"\byour\s+new\s+role\b", re.IGNORECASE),
        0.75,
        "ROLE_HIJACK",
        "your_new_role",
    ),
    (
        re.compile(r"\bfrom\s+now\s+on\s+(you|your)\b", re.IGNORECASE),
        0.65,
        "ROLE_HIJACK",
        "from_now_on_you",
    ),

    # ── JAILBREAK_CLASSIC ────────────────────────────────────────────────────
    (
        re.compile(r"\bDAN\b", re.IGNORECASE),
        0.80,
        "JAILBREAK_CLASSIC",
        "DAN_mention",
    ),
    (
        re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
        0.85,
        "JAILBREAK_CLASSIC",
        "do_anything_now",
    ),
    (
        re.compile(r"\bno\s+restrictions?\b", re.IGNORECASE),
        0.65,
        "JAILBREAK_CLASSIC",
        "no_restrictions",
    ),
    (
        re.compile(r"\bwithout\s+(any\s+)?limitations?\b", re.IGNORECASE),
        0.55,
        "JAILBREAK_CLASSIC",
        "without_limitations",
    ),
    (
        re.compile(r"\bunrestricted\s+mode\b", re.IGNORECASE),
        0.85,
        "JAILBREAK_CLASSIC",
        "unrestricted_mode",
    ),
    (
        re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
        0.70,
        "JAILBREAK_CLASSIC",
        "developer_mode",
    ),
    (
        re.compile(r"\bno\s+ethical\s+(constraints?|guidelines?|rules?)\b", re.IGNORECASE),
        0.85,
        "JAILBREAK_CLASSIC",
        "no_ethical_constraints",
    ),

    # ── SYSTEM_REFERENCE ─────────────────────────────────────────────────────
    (
        re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
        0.65,
        "SYSTEM_REFERENCE",
        "system_prompt_mention",
    ),
    (
        re.compile(r"\byour\s+(instructions?|training|guidelines?|rules?|directives?)\b", re.IGNORECASE),
        0.50,
        "SYSTEM_REFERENCE",
        "your_instructions",
    ),
    (
        re.compile(r"\breveal\b.{0,30}(prompt|instructions?|system|training|hidden)\b", re.IGNORECASE),
        0.75,
        "SYSTEM_REFERENCE",
        "reveal_instructions",
    ),

    # ── INDIRECT_INJECTION ───────────────────────────────────────────────────
    (
        re.compile(r"<!--", re.IGNORECASE),
        0.60,
        "INDIRECT_INJECTION",
        "html_comment_start",
    ),
    (
        re.compile(r"\]\]>", re.IGNORECASE),
        0.65,
        "INDIRECT_INJECTION",
        "cdata_end",
    ),
    (
        re.compile(r"\[SYSTEM\]", re.IGNORECASE),
        0.80,
        "INDIRECT_INJECTION",
        "system_tag",
    ),
    (
        re.compile(r"\[INST\]", re.IGNORECASE),
        0.75,
        "INDIRECT_INJECTION",
        "inst_tag",
    ),
    (
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
        0.85,
        "INDIRECT_INJECTION",
        "im_start_token",
    ),
    (
        re.compile(r"###\s*Instruction", re.IGNORECASE),
        0.80,
        "INDIRECT_INJECTION",
        "hash_instruction",
    ),
    (
        re.compile(r"\[/INST\]", re.IGNORECASE),
        0.75,
        "INDIRECT_INJECTION",
        "inst_close_tag",
    ),
]

# ---------------------------------------------------------------------------
# Base64 detection (ENCODED_SUSPICIOUS)
# ---------------------------------------------------------------------------

_BASE64_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
)


def _detect_base64(text: str) -> List[str]:
    """
    Find base64 strings longer than 30 chars (covers short encoded injections).
    Returns a list of match descriptions if found.
    """
    matches = _BASE64_PATTERN.findall(text)
    triggered = []
    for m in matches:
        if len(m) >= 30:
            # Try to decode and see if it looks like text
            try:
                decoded = base64.b64decode(m + "==").decode("utf-8", errors="replace")
                if any(kw in decoded.lower() for kw in ["ignore", "instruction", "system", "override", "forget", "disregard"]):
                    triggered.append("base64_decoded_injection_keyword")
                else:
                    triggered.append("base64_long_string")
            except Exception:
                triggered.append("base64_long_string")
    return triggered


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class HeuristicResult:
    score: float
    triggered: bool
    matched_patterns: List[str] = field(default_factory=list)


def run_heuristics(text: str) -> HeuristicResult:
    """
    Run all heuristic checks against the input text.

    Scoring uses a probabilistic OR formula: score = 1 - product(1 - w_i)
    This ensures each matched pattern contributes its full weight independently,
    so a single high-confidence pattern (e.g. weight=0.85) produces score=0.85
    instead of being compressed by a sigmoid. Multiple patterns compound naturally.

    Returns a HeuristicResult with:
      - score: float in [0, 1]
      - triggered: bool (True if score > 0.3)
      - matched_patterns: list of matched pattern names
    """
    matched: List[str] = []
    weights: List[float] = []

    # Run regex patterns
    for regex, weight, category, name in PATTERN_GROUPS:
        if regex.search(text):
            matched.append(f"{category}:{name}")
            weights.append(weight)
            logger.debug("Heuristic match: %s (weight=%.2f)", name, weight)

    # Run base64 detection
    b64_matches = _detect_base64(text)
    for b64_match in b64_matches:
        matched.append(f"ENCODED_SUSPICIOUS:{b64_match}")
        weights.append(0.70)

    # Probabilistic OR: 1 - product(1 - w_i)
    # Each pattern's weight represents an independent probability of injection.
    # This gives single strong patterns their full score (e.g. 0.85 -> 0.85)
    # while multiple patterns compound (0.80 + 0.75 -> 0.95).
    if weights:
        not_injection = 1.0
        for w in weights:
            not_injection *= (1.0 - w)
        score = 1.0 - not_injection
    else:
        score = 0.0

    score = round(min(1.0, max(0.0, score)), 4)
    triggered = score > 0.3

    return HeuristicResult(
        score=score,
        triggered=triggered,
        matched_patterns=matched,
    )
