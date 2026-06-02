"""
benchmarks/eval.py

Offline evaluation of the full detection pipeline against a labeled test set.

Runs each sample directly through the pipeline (no HTTP overhead), computes
per-layer and full-pipeline metrics, and generates:
  - Precision, Recall, F1 per layer and overall
  - Confusion matrix
  - ROC-AUC curve saved to benchmarks/roc_curve.png
  - Latency stats: p50, p90, p99 in milliseconds
  - Results saved to benchmarks/results.json

Usage:
  python -m benchmarks.eval
  python -m benchmarks.eval --test-csv training/data/test.csv --threshold 0.75
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_PATH = "benchmarks/results.json"
ROC_CURVE_PATH = "benchmarks/roc_curve.png"
DEFAULT_TEST_CSV = "training/data/test.csv"
DEFAULT_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_csv(path: str) -> Tuple[List[str], List[int]]:
    """Load test CSV with 'text' and 'label' columns."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    logger.info("Loaded %d test samples from %s", len(texts), path)
    return texts, labels


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def evaluate_sample(text: str) -> Tuple[float, float, float, Optional[float], Optional[float], float]:
    """
    Run a single text through the pipeline and return per-layer scores + latency.
    Returns: (score_h, score_c, score_e, score_j, final_score, latency_ms)
    Note: No upstream API is used — judge layer will be skipped (no API key).
    """
    from proxy.pipeline import run_pipeline

    start = time.monotonic()
    result = await run_pipeline(
        user_prompt=text,
        system_prompt="You are a helpful assistant.",  # Provide system_prompt to enable embedding layer
        upstream_base=None,  # No API = judge layer skipped
        upstream_key=None,
    )
    latency_ms = (time.monotonic() - start) * 1000

    return (
        result.score_heuristic,
        result.score_classifier,
        result.score_embedding,
        result.score_judge,
        result.final_score,
        latency_ms,
    )


async def run_evaluation(
    texts: List[str],
    labels: List[int],
    threshold: float,
) -> Dict:
    """Run full evaluation and return all scores + metrics."""
    scores_h, scores_c, scores_e, scores_j, scores_final, latencies = [], [], [], [], [], []

    print(f"Evaluating {len(texts)} samples…")
    for i, (text, label) in enumerate(zip(texts, labels)):
        score_h, score_c, score_e, score_j, final, latency = await evaluate_sample(text)
        scores_h.append(score_h)
        scores_c.append(score_c)
        scores_e.append(score_e if score_e is not None else 0.0)
        scores_j.append(score_j if score_j is not None else 0.5)
        scores_final.append(final)
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(texts)}")

    return {
        "scores_h": scores_h,
        "scores_c": scores_c,
        "scores_e": scores_e,
        "scores_j": scores_j,
        "scores_final": scores_final,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    scores: List[float],
    labels: List[int],
    threshold: float,
    layer_name: str,
) -> Dict:
    """Compute precision, recall, F1 for a given score list and threshold."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # type: ignore

    preds = [1 if s >= threshold else 0 for s in scores]
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    tp = sum(1 for p_, l in zip(preds, labels) if p_ == 1 and l == 1)
    fp = sum(1 for p_, l in zip(preds, labels) if p_ == 1 and l == 0)
    tn = sum(1 for p_, l in zip(preds, labels) if p_ == 0 and l == 0)
    fn = sum(1 for p_, l in zip(preds, labels) if p_ == 0 and l == 1)

    return {
        "layer": layer_name,
        "threshold": threshold,
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_latency_stats(latencies: List[float]) -> Dict:
    """Compute p50, p90, p99 latency percentiles."""
    arr = np.array(latencies)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def plot_roc_curve(scores: List[float], labels: List[int], output_path: str) -> None:
    """Generate and save an ROC curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc  # type: ignore

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#6366f1", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#6366f1")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("PromptArmor Pipeline — ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("ROC curve saved to %s", output_path)
        print(f"\n📊 ROC curve saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not installed — skipping ROC curve plot")
    except Exception as exc:
        logger.error("Failed to generate ROC curve: %s", exc)


def print_confusion_matrix(metrics: Dict) -> None:
    """Pretty-print confusion matrix."""
    print(f"\n{'─'*40}")
    print(f"  Confusion Matrix — {metrics['layer']}")
    print(f"{'─'*40}")
    print(f"               Predicted Clean  Predicted Injection")
    print(f"  Actual Clean       {metrics['tn']:>5}              {metrics['fp']:>5}")
    print(f"  Actual Injection   {metrics['fn']:>5}              {metrics['tp']:>5}")


# ---------------------------------------------------------------------------
# Main evaluation orchestration
# ---------------------------------------------------------------------------

async def main_async(test_csv: str, threshold: float) -> None:
    if not Path(test_csv).exists():
        print(f"\n⚠ Test CSV not found at {test_csv}")
        print("  Run 'python -m training.dataset' first to generate the dataset.")
        return

    texts, labels = load_test_csv(test_csv)
    eval_data = await run_evaluation(texts, labels, threshold)

    # ── Per-layer metrics ────────────────────────────────────────────────────
    layer_threshold = 0.5  # Per-layer binary threshold
    layer_metrics = []
    for layer_name, scores in [
        ("heuristics", eval_data["scores_h"]),
        ("classifier", eval_data["scores_c"]),
        ("embeddings", eval_data["scores_e"]),
        ("pipeline_final", eval_data["scores_final"]),
    ]:
        m = compute_metrics(scores, labels, layer_threshold, layer_name)
        layer_metrics.append(m)

    # ── Full pipeline at block threshold ─────────────────────────────────────
    pipeline_metrics = compute_metrics(
        eval_data["scores_final"], labels, threshold, f"pipeline_final@{threshold}"
    )

    # ── Latency stats ─────────────────────────────────────────────────────────
    latency_stats = compute_latency_stats(eval_data["latencies"])

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PROMPTARMOR EVALUATION RESULTS")
    print(f"Test set: {test_csv}  |  Block threshold: {threshold}")
    print("="*65)
    print(f"\n{'Layer':<22} {'Precision':>11} {'Recall':>9} {'F1':>7} {'AUC':>8}")
    print("-"*65)
    for m in layer_metrics:
        print(f"{m['layer']:<22} {m['precision']:>11.4f} {m['recall']:>9.4f} {m['f1']:>7.4f} {m['roc_auc']:>8.4f}")

    print(f"\n{'─'*65}")
    print(f"PIPELINE @ threshold={threshold}")
    print(f"  Precision: {pipeline_metrics['precision']:.4f}")
    print(f"  Recall:    {pipeline_metrics['recall']:.4f}")
    print(f"  F1:        {pipeline_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {pipeline_metrics['roc_auc']:.4f}")
    print_confusion_matrix(pipeline_metrics)

    print(f"\n{'─'*65}")
    print("LATENCY STATISTICS (per request, no LLM call)")
    print(f"  p50: {latency_stats['p50_ms']:.1f}ms  |  p90: {latency_stats['p90_ms']:.1f}ms  |  p99: {latency_stats['p99_ms']:.1f}ms")
    print(f"  mean: {latency_stats['mean_ms']:.1f}ms  |  min: {latency_stats['min_ms']:.1f}ms  |  max: {latency_stats['max_ms']:.1f}ms")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    plot_roc_curve(eval_data["scores_final"], labels, ROC_CURVE_PATH)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("benchmarks", exist_ok=True)
    output = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "test_set": test_csv,
        "num_samples": len(texts),
        "block_threshold": threshold,
        "layer_metrics": layer_metrics,
        "pipeline_metrics_at_threshold": pipeline_metrics,
        "latency_stats": latency_stats,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_PATH}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate PromptArmor pipeline on a labeled test set")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Path to test CSV file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Block threshold (default 0.75)")
    args = parser.parse_args()

    asyncio.run(main_async(args.test_csv, args.threshold))


if __name__ == "__main__":
    main()
