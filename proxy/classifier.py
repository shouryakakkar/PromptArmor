"""
proxy/classifier.py

Layer 2 of the detection pipeline: ML-based sequence classification.
Attempts to load a fine-tuned DistilBERT model from ./models/classifier/.
If the model is not found, falls back to a rule-based scorer that mimics a
classifier output (suitable for development/testing without training).

NOTE: Train a real model with training/finetune.py for production use.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache — loaded once, reused on subsequent calls
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_loaded: bool = False
_use_fallback: bool = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier")


def _load_model() -> None:
    """
    Attempt to load the fine-tuned DistilBERT classifier from disk.
    Sets module-level flags so loading only happens once.
    """
    global _model, _tokenizer, _model_loaded, _use_fallback

    if _model_loaded:
        return

    model_dir = os.path.abspath(MODEL_PATH)

    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        logger.warning(
            "Classifier model not found at %s. "
            "Using fallback rule-based scorer. "
            "Run training/finetune.py to train a real model.",
            model_dir,
        )
        _use_fallback = True
        _model_loaded = True
        return

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        logger.info("Loading classifier from %s …", model_dir)
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.eval()
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        logger.info("Classifier loaded successfully (device=%s)", device)
        _use_fallback = False
    except Exception as exc:
        logger.error("Failed to load classifier model: %s — falling back to rule-based scorer.", exc)
        _use_fallback = True

    _model_loaded = True


# ---------------------------------------------------------------------------
# Fallback scorer
# ---------------------------------------------------------------------------

def _fallback_score(text: str, heuristic_score: float) -> float:
    """
    Improved rule-based classifier fallback for when the ML model is not available.

    Uses a piecewise linear mapping to approximate a trained classifier's output
    distribution. A real DistilBERT classifier pushes clear injections (high
    heuristic) to 0.85–0.95; this fallback mirrors that behaviour.

    Mapping:
      heuristic 0.0 – 0.2  →  classifier 0.00 – 0.35  (clean territory)
      heuristic 0.2 – 0.5  →  classifier 0.35 – 0.72  (borderline)
      heuristic 0.5 – 1.0  →  classifier 0.72 – 0.96  (injection territory)

    NOTE: Train a real model with training/finetune.py for production use.
    The fallback exists only so the proxy can run end-to-end without training.
    """
    # Seed on text content so results are deterministic per prompt
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    noise = rng.uniform(-0.03, 0.03)

    if heuristic_score >= 0.5:
        # Clear injection: map [0.5, 1.0] → [0.72, 0.96]
        scaled = 0.72 + (heuristic_score - 0.5) * 0.48
    elif heuristic_score >= 0.2:
        # Borderline: map [0.2, 0.5] → [0.35, 0.72]
        scaled = 0.35 + (heuristic_score - 0.2) * (0.37 / 0.3)
    else:
        # Clean: map [0.0, 0.2] → [0.0, 0.35]
        scaled = heuristic_score * 1.75

    return float(min(1.0, max(0.0, scaled + noise)))



# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify(text: str, heuristic_score: float = 0.0) -> float:
    """
    Return injection confidence score in [0, 1] for class 1 (injection).

    Args:
        text: The user prompt text to classify.
        heuristic_score: Score from Layer 1, used by the fallback scorer.

    Returns:
        Confidence float between 0 (clean) and 1 (injection).
    """
    _load_model()

    if _use_fallback:
        return _fallback_score(text, heuristic_score)

    try:
        import torch  # type: ignore

        device = next(_model.parameters()).device
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits

        # Softmax over logits → probabilities; index 1 = injection class
        probs = torch.softmax(logits, dim=-1)
        injection_prob = probs[0][1].item()
        return float(injection_prob)

    except Exception as exc:
        logger.error("Classifier inference failed: %s — returning heuristic fallback.", exc)
        return _fallback_score(text, heuristic_score)
