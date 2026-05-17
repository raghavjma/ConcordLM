"""
Pipeline Orchestrator — chain SFT → DPO → RLHF stages automatically.

Handles inter-stage checkpoint propagation:
  - SFT produces a checkpoint → passed to DPO as sft_model_path
  - DPO produces a checkpoint → passed to RLHF as dpo_model_path
  - RLHF produces the final aligned model
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from concordlm.config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


def run_full_pipeline(
    config_path: str,
    stages: list[str],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute multiple alignment stages sequentially.

    Parameters
    ----------
    config_path : str
        Path to the base YAML config (or stage-specific config).
    stages : list of str
        Stages to run in order. Valid: ["sft", "dpo", "rlhf"].
    overrides : list of str, optional
        Dot-notation config overrides.

    Returns
    -------
    dict  Pipeline manifest with stage results and checkpoint paths.
    """
    valid_stages = {"sft", "dpo", "rlhf"}
    for stage in stages:
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Valid: {valid_stages}")

    logger.info("=" * 60)
    logger.info("  ConcordLM — Full Alignment Pipeline")
    logger.info(f"  Stages: {' → '.join(stages)}")
    logger.info("=" * 60)

    manifest = {
        "started_at": datetime.now().isoformat(),
        "stages_requested": stages,
        "stages_completed": [],
        "checkpoints": {},
        "errors": {},
    }

    # Track checkpoint paths across stages
    sft_checkpoint = None
    dpo_checkpoint = None

    for stage in stages:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Starting stage: {stage.upper()}")
        logger.info(f"{'=' * 60}\n")

        try:
            # Load stage-specific config
            stage_config_path = Path(config_path).parent / f"{stage}.yaml"
            if stage_config_path.exists():
                config = load_config(str(stage_config_path), overrides=overrides or [])
            else:
                config = load_config(config_path, overrides=overrides or [])

            # Propagate checkpoint paths from prior stages
            if stage == "dpo" and sft_checkpoint:
                config.sft_model_path = sft_checkpoint
                logger.info(f"Propagating SFT checkpoint → DPO: {sft_checkpoint}")

            if stage == "rlhf":
                if dpo_checkpoint:
                    config.dpo_model_path = dpo_checkpoint
                    logger.info(f"Propagating DPO checkpoint → RLHF: {dpo_checkpoint}")
                elif sft_checkpoint:
                    config.sft_model_path = sft_checkpoint
                    logger.info(f"Propagating SFT checkpoint → RLHF: {sft_checkpoint}")

            # Run the stage
            if stage == "sft":
                from concordlm.trainers.sft import run_sft
                sft_checkpoint = run_sft(config)
                manifest["checkpoints"]["sft"] = sft_checkpoint

            elif stage == "dpo":
                from concordlm.trainers.dpo import run_dpo
                dpo_checkpoint = run_dpo(config)
                manifest["checkpoints"]["dpo"] = dpo_checkpoint

            elif stage == "rlhf":
                from concordlm.trainers.rlhf import run_rlhf
                rlhf_checkpoint = run_rlhf(config)
                manifest["checkpoints"]["rlhf"] = rlhf_checkpoint

            manifest["stages_completed"].append(stage)

        except Exception as e:
            logger.error(f"Stage {stage} failed: {e}", exc_info=True)
            manifest["errors"][stage] = str(e)
            break  # Stop pipeline on failure

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["success"] = len(manifest["errors"]) == 0

    # Save manifest
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nPipeline manifest saved to: {manifest_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Stages completed: {manifest['stages_completed']}")
    for stage, path in manifest["checkpoints"].items():
        logger.info(f"  {stage.upper()} checkpoint: {path}")
    if manifest["errors"]:
        for stage, err in manifest["errors"].items():
            logger.error(f"  {stage.upper()} error: {err}")
    logger.info(f"  Success: {manifest['success']}")
    logger.info("=" * 60)

    return manifest
