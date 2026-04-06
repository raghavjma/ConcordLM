"""
Benchmark — automated evaluation harness combining safety and quality metrics.

Generates a comprehensive JSON report comparing models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from concordlm.eval.quality import evaluate_quality
from concordlm.eval.safety import evaluate_safety

logger = logging.getLogger(__name__)


def _create_generate_fn(model_path: str, **generation_kwargs):
    """Create a generation function from a model path."""
    from concordlm.models.loader import load_model_for_inference

    model, tokenizer = load_model_for_inference(model_path)

    default_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    default_kwargs.update(generation_kwargs)

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **default_kwargs)
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def run_benchmark(
    model_path: str,
    *,
    output_dir: str = "./outputs/eval",
    max_safety_samples: int | None = None,
    max_quality_samples: int | None = None,
    compare_model_path: str | None = None,
) -> dict[str, Any]:
    """
    Run the full evaluation benchmark on a model.

    Parameters
    ----------
    model_path : str
        Path to the model to evaluate.
    output_dir : str
        Where to save the evaluation report.
    max_safety_samples : int, optional
        Limit safety prompts (for faster testing).
    max_quality_samples : int, optional
        Limit quality prompts.
    compare_model_path : str, optional
        If provided, also evaluate this model and compute win-rates.

    Returns
    -------
    dict  Full evaluation report.
    """
    logger.info("=" * 60)
    logger.info("  ConcordLM — Evaluation Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")

    # Create generation function
    generate_fn = _create_generate_fn(model_path)

    # --- Safety evaluation ---
    logger.info("\n--- Safety Evaluation ---")
    safety_prompts = None
    if max_safety_samples:
        from concordlm.eval.safety import SAFETY_PROMPTS
        safety_prompts = SAFETY_PROMPTS[:max_safety_samples]
    safety_results = evaluate_safety(generate_fn, prompts=safety_prompts)

    # --- Quality evaluation ---
    logger.info("\n--- Quality Evaluation ---")
    quality_prompts = None
    if max_quality_samples:
        from concordlm.eval.quality import QUALITY_PROMPTS
        quality_prompts = QUALITY_PROMPTS[:max_quality_samples]
    quality_results = evaluate_quality(generate_fn, prompts=quality_prompts)

    # --- Win-rate comparison ---
    win_rate_results = None
    if compare_model_path:
        from concordlm.eval.quality import compute_win_rate
        logger.info(f"\n--- Win-Rate Comparison vs {compare_model_path} ---")
        compare_fn = _create_generate_fn(compare_model_path)
        win_rate_results = compute_win_rate(generate_fn, compare_fn)

    # --- Build report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "summary": {
            "safety_refusal_rate": safety_results["refusal_rate"],
            "quality_overall": quality_results["overall_quality"],
            "quality_coherence": quality_results["mean_coherence_score"],
            "quality_informativeness": quality_results["mean_informativeness"],
        },
        "safety": {k: v for k, v in safety_results.items() if k != "details"},
        "quality": {k: v for k, v in quality_results.items() if k != "details"},
    }

    if win_rate_results:
        report["summary"]["win_rate_vs_baseline"] = win_rate_results["model_a_win_rate"]
        report["win_rate"] = {
            k: v for k, v in win_rate_results.items() if k != "comparisons"
        }

    # --- Save report ---
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nEvaluation report saved to: {report_path}")

    # Also save detailed results
    full_report = {
        **report,
        "safety_details": safety_results.get("details", []),
        "quality_details": quality_results.get("details", []),
    }
    if win_rate_results:
        full_report["win_rate_details"] = win_rate_results.get("comparisons", [])

    detailed_path = os.path.join(output_dir, "eval_report_detailed.json")
    with open(detailed_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Detailed report saved to: {detailed_path}")

    # --- Print summary ---
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Safety refusal rate:  {report['summary']['safety_refusal_rate']:.1%}")
    logger.info(f"  Quality overall:      {report['summary']['quality_overall']:.4f}")
    logger.info(f"  Quality coherence:    {report['summary']['quality_coherence']:.4f}")
    logger.info(f"  Informativeness:      {report['summary']['quality_informativeness']:.4f}")
    if "win_rate_vs_baseline" in report["summary"]:
        logger.info(f"  Win rate vs baseline: {report['summary']['win_rate_vs_baseline']:.1%}")
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# Side-by-side generation samples
# ---------------------------------------------------------------------------


def generate_side_by_side(
    model_path_a: str,
    model_path_b: str,
    prompts: list[str] | None = None,
    output_path: str | None = None,
) -> list[dict[str, str]]:
    """
    Generate side-by-side responses from two models for visual comparison.
    """
    from concordlm.eval.quality import QUALITY_PROMPTS

    if prompts is None:
        prompts = QUALITY_PROMPTS[:5]

    gen_a = _create_generate_fn(model_path_a)
    gen_b = _create_generate_fn(model_path_b)

    samples = []
    for prompt in prompts:
        samples.append({
            "prompt": prompt,
            "response_a": gen_a(prompt),
            "response_b": gen_b(prompt),
        })

    if output_path:
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Side-by-side samples saved to: {output_path}")

    return samples


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Evaluation Benchmark"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/eval",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to a baseline model for win-rate comparison",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of prompts for faster evaluation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_benchmark(
        args.model,
        output_dir=args.output_dir,
        max_safety_samples=args.max_samples,
        max_quality_samples=args.max_samples,
        compare_model_path=args.compare,
    )


if __name__ == "__main__":
    main()
