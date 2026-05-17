#!/usr/bin/env python3
"""
ConcordLM — Run Full Alignment Pipeline

Chain multiple training stages: SFT → DPO → RLHF

Usage:
    python scripts/run_pipeline.py --stages sft,dpo,rlhf
    python scripts/run_pipeline.py --stages sft,dpo
    python scripts/run_pipeline.py --stages dpo,rlhf --override model.name=Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import logging

from concordlm.trainers.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ConcordLM — Run Full Alignment Pipeline"
    )
    parser.add_argument(
        "--stages", type=str, default="sft,dpo",
        help="Comma-separated stages to run (e.g., 'sft,dpo,rlhf')",
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs",
        help="Directory containing stage config YAML files",
    )
    parser.add_argument(
        "--override", type=str, action="append", default=[],
        help="Config overrides in dot-notation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stages = [s.strip() for s in args.stages.split(",")]
    # Use base config as starting point
    config_path = f"{args.config_dir}/base.yaml"

    manifest = run_full_pipeline(
        config_path=config_path,
        stages=stages,
        overrides=args.override,
    )

    if not manifest["success"]:
        exit(1)


if __name__ == "__main__":
    main()
