"""
Quality Evaluation — measure helpfulness, coherence, and win rates.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Helpfulness test prompts (benign, diverse tasks)
QUALITY_PROMPTS = [
    "Explain quantum computing to a 10-year-old.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "What are the main causes of climate change and what can individuals do about it?",
    "Summarize the French Revolution in 3 bullet points.",
    "Write a haiku about artificial intelligence.",
    "Compare and contrast TCP and UDP protocols.",
    "What's the difference between a virus and a bacterium?",
    "Help me plan a week-long trip to Japan on a budget.",
    "Explain the concept of recursion with a simple example.",
    "Write a professional email declining a job offer politely.",
    "What are the pros and cons of electric vehicles?",
    "Explain how vaccines work in simple terms.",
    "Write a short story about a robot learning to paint.",
    "How does the stock market work?",
    "Give me a recipe for a simple pasta dish with ingredients I likely have at home.",
]


def evaluate_response_quality(
    response: str,
) -> dict[str, float]:
    """
    Heuristic quality metrics for a single response.

    Returns scores for:
    - length_score: Penalizes very short or very long responses
    - coherence_score: Basic coherence heuristic (sentence structure)
    - informativeness: Vocabulary richness
    """
    words = response.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]

    # Length score: bell curve around 50-300 words
    n = len(words)
    if n < 10:
        length_score = 0.1
    elif n < 50:
        length_score = n / 50
    elif n <= 300:
        length_score = 1.0
    elif n <= 500:
        length_score = 1.0 - (n - 300) / 400
    else:
        length_score = 0.5

    # Coherence: ratio of sentences to words (well-structured = moderate ratio)
    if len(sentences) == 0 or len(words) == 0:
        coherence_score = 0.0
    else:
        avg_sentence_len = len(words) / len(sentences)
        if 8 <= avg_sentence_len <= 25:
            coherence_score = 1.0
        elif avg_sentence_len < 8:
            coherence_score = avg_sentence_len / 8
        else:
            coherence_score = max(0.3, 1.0 - (avg_sentence_len - 25) / 50)

    # Informativeness: unique word ratio
    unique_words = set(w.lower() for w in words)
    informativeness = len(unique_words) / max(1, len(words))

    return {
        "length_score": round(min(1.0, max(0.0, length_score)), 3),
        "coherence_score": round(min(1.0, max(0.0, coherence_score)), 3),
        "informativeness": round(min(1.0, max(0.0, informativeness)), 3),
    }


def evaluate_quality(
    generate_fn,
    *,
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate model quality on a set of helpful prompts.

    Parameters
    ----------
    generate_fn : callable
        Function that takes a prompt and returns a response.
    prompts : list of str, optional
        Custom prompts. Uses built-in quality prompts by default.

    Returns
    -------
    dict with aggregated quality metrics.
    """
    if prompts is None:
        prompts = QUALITY_PROMPTS

    logger.info(f"Running quality evaluation on {len(prompts)} prompts...")

    all_scores = []
    details = []

    for prompt in prompts:
        response = generate_fn(prompt)
        scores = evaluate_response_quality(response)
        all_scores.append(scores)
        details.append({
            "prompt": prompt,
            "response": response[:500],
            **scores,
        })

    # Aggregate
    metrics = {}
    for key in all_scores[0]:
        values = [s[key] for s in all_scores]
        metrics[f"mean_{key}"] = round(sum(values) / len(values), 4)
        metrics[f"min_{key}"] = round(min(values), 4)

    # Overall quality score (weighted average)
    overall = (
        0.3 * metrics["mean_length_score"]
        + 0.4 * metrics["mean_coherence_score"]
        + 0.3 * metrics["mean_informativeness"]
    )
    metrics["overall_quality"] = round(overall, 4)
    metrics["num_prompts"] = len(prompts)
    metrics["details"] = details

    logger.info(f"Quality evaluation complete:")
    logger.info(f"  Overall quality score: {overall:.4f}")
    logger.info(f"  Coherence: {metrics['mean_coherence_score']:.4f}")
    logger.info(f"  Informativeness: {metrics['mean_informativeness']:.4f}")

    return metrics


def compute_win_rate(
    generate_fn_a,
    generate_fn_b,
    *,
    prompts: list[str] | None = None,
    judge: str = "heuristic",
) -> dict[str, Any]:
    """
    Compare two models (A vs B) on the same prompts.

    Parameters
    ----------
    generate_fn_a, generate_fn_b : callable
        Generation functions for model A and B.
    prompts : list of str
        Prompts to compare on.
    judge : str
        "heuristic" (default) — use quality scores to judge.

    Returns
    -------
    dict with win rates for A and B.
    """
    if prompts is None:
        prompts = QUALITY_PROMPTS

    logger.info(f"Computing win rate on {len(prompts)} prompts...")

    a_wins = 0
    b_wins = 0
    ties = 0
    comparisons = []

    for prompt in prompts:
        resp_a = generate_fn_a(prompt)
        resp_b = generate_fn_b(prompt)

        scores_a = evaluate_response_quality(resp_a)
        scores_b = evaluate_response_quality(resp_b)

        # Simple overall score comparison
        overall_a = sum(scores_a.values()) / len(scores_a)
        overall_b = sum(scores_b.values()) / len(scores_b)

        if overall_a > overall_b + 0.05:
            a_wins += 1
            winner = "A"
        elif overall_b > overall_a + 0.05:
            b_wins += 1
            winner = "B"
        else:
            ties += 1
            winner = "tie"

        comparisons.append({
            "prompt": prompt,
            "response_a": resp_a[:300],
            "response_b": resp_b[:300],
            "scores_a": scores_a,
            "scores_b": scores_b,
            "winner": winner,
        })

    total = len(prompts)
    results = {
        "model_a_win_rate": round(a_wins / total, 4),
        "model_b_win_rate": round(b_wins / total, 4),
        "tie_rate": round(ties / total, 4),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": total,
        "comparisons": comparisons,
    }

    logger.info(f"Win rate: A={a_wins}/{total}, B={b_wins}/{total}, Ties={ties}/{total}")
    return results
