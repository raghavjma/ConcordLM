"""
ConcordLM Web Dashboard — FastAPI Backend

Provides:
- REST API for pipeline configuration, training, evaluation
- WebSocket for live training log streaming
- Static file serving for the frontend SPA
- Chat endpoint for inference
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ConcordLM Dashboard",
    description="End-to-end LLM Alignment Pipeline — SFT → DPO → RLHF",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# In-memory state
_active_jobs: dict[str, dict[str, Any]] = {}
_training_logs: dict[str, list[str]] = {}
_connected_websockets: dict[str, list[WebSocket]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TrainingRequest(BaseModel):
    stage: str  # "sft" | "dpo" | "rlhf"
    config_overrides: dict[str, Any] = {}
    sft_model_path: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    model_path: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class DatasetBuildRequest(BaseModel):
    pairs: list[dict[str, str]]
    output_name: str = "custom_preferences.jsonl"


class EvalRequest(BaseModel):
    model_path: str
    compare_model_path: Optional[str] = None
    max_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def get_status():
    """Get overall pipeline status."""
    # Check for existing checkpoints
    sft_exists = (OUTPUTS_DIR / "sft" / "checkpoint-final").exists()
    dpo_exists = (OUTPUTS_DIR / "dpo" / "checkpoint-final").exists()
    rlhf_exists = (OUTPUTS_DIR / "rlhf" / "ppo-final").exists()
    reward_exists = (OUTPUTS_DIR / "reward_model" / "checkpoint-final").exists()

    # Check for eval reports
    eval_report = None
    eval_path = OUTPUTS_DIR / "eval" / "eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)

    return {
        "pipeline": {
            "sft": {"completed": sft_exists, "path": str(OUTPUTS_DIR / "sft" / "checkpoint-final") if sft_exists else None},
            "dpo": {"completed": dpo_exists, "path": str(OUTPUTS_DIR / "dpo" / "checkpoint-final") if dpo_exists else None},
            "rlhf": {"completed": rlhf_exists, "path": str(OUTPUTS_DIR / "rlhf" / "ppo-final") if rlhf_exists else None},
            "reward_model": {"completed": reward_exists},
        },
        "active_jobs": {jid: {"stage": j["stage"], "status": j["status"], "started_at": j["started_at"]} for jid, j in _active_jobs.items()},
        "eval_report": eval_report,
    }


@app.get("/api/configs")
async def get_configs():
    """Get all available configurations."""
    configs = {}
    for cfg_file in CONFIGS_DIR.glob("*.yaml"):
        import yaml
        with open(cfg_file) as f:
            configs[cfg_file.stem] = yaml.safe_load(f)
    return configs


@app.get("/api/configs/{stage}")
async def get_config(stage: str):
    """Get configuration for a specific stage."""
    cfg_path = CONFIGS_DIR / f"{stage}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"Config not found: {stage}")
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@app.post("/api/configs/{stage}")
async def update_config(stage: str, config: dict[str, Any]):
    """Update configuration for a specific stage."""
    cfg_path = CONFIGS_DIR / f"{stage}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"Config not found: {stage}")
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return {"status": "ok", "message": f"Config '{stage}' updated"}


# ---------------------------------------------------------------------------
# Training management
# ---------------------------------------------------------------------------


@app.post("/api/train")
async def start_training(req: TrainingRequest):
    """Launch a training job in the background."""
    job_id = str(uuid.uuid4())[:8]

    # Build command
    script_map = {"sft": "train_sft.py", "dpo": "train_dpo.py", "rlhf": "train_rlhf.py"}
    if req.stage not in script_map:
        raise HTTPException(400, f"Invalid stage: {req.stage}")

    script = PROJECT_ROOT / "scripts" / script_map[req.stage]
    cmd = [sys.executable, str(script), "--config", str(CONFIGS_DIR / f"{req.stage}.yaml")]

    # Add overrides
    for key, value in req.config_overrides.items():
        cmd.extend(["--override", f"{key}={value}"])

    if req.sft_model_path and req.stage == "dpo":
        cmd.extend(["--sft-model", req.sft_model_path])

    _active_jobs[job_id] = {
        "stage": req.stage,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "command": " ".join(cmd),
        "process": None,
    }
    _training_logs[job_id] = []

    # Launch in background
    asyncio.create_task(_run_training_process(job_id, cmd))

    return {"job_id": job_id, "status": "started", "stage": req.stage}


async def _run_training_process(job_id: str, cmd: list[str]):
    """Run a training process and stream logs."""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        _active_jobs[job_id]["process"] = process

        async for line_bytes in process.stdout:
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            _training_logs[job_id].append(line)

            # Broadcast to connected websockets
            for ws in _connected_websockets.get(job_id, []):
                try:
                    await ws.send_json({"type": "log", "job_id": job_id, "message": line})
                except Exception:
                    pass

        await process.wait()
        _active_jobs[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        _active_jobs[job_id]["finished_at"] = datetime.now().isoformat()
        _active_jobs[job_id]["return_code"] = process.returncode

        # Notify websockets of completion
        for ws in _connected_websockets.get(job_id, []):
            try:
                await ws.send_json({
                    "type": "status",
                    "job_id": job_id,
                    "status": _active_jobs[job_id]["status"],
                })
            except Exception:
                pass

    except Exception as e:
        _active_jobs[job_id]["status"] = "failed"
        _active_jobs[job_id]["error"] = str(e)
        _training_logs[job_id].append(f"ERROR: {e}")


@app.get("/api/train/{job_id}")
async def get_job_status(job_id: str):
    """Get status and logs for a training job."""
    if job_id not in _active_jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    job = _active_jobs[job_id]
    return {
        "job_id": job_id,
        "stage": job["stage"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job.get("finished_at"),
        "logs": _training_logs.get(job_id, [])[-100:],  # Last 100 lines
        "log_count": len(_training_logs.get(job_id, [])),
    }


@app.post("/api/train/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job."""
    if job_id not in _active_jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    job = _active_jobs[job_id]
    if job.get("process") and job["status"] == "running":
        job["process"].terminate()
        job["status"] = "stopped"
        return {"status": "stopped"}
    return {"status": job["status"]}


# ---------------------------------------------------------------------------
# WebSocket for live logs
# ---------------------------------------------------------------------------


@app.websocket("/ws/logs/{job_id}")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for streaming training logs."""
    await websocket.accept()

    if job_id not in _connected_websockets:
        _connected_websockets[job_id] = []
    _connected_websockets[job_id].append(websocket)

    try:
        # Send existing logs
        for line in _training_logs.get(job_id, []):
            await websocket.send_json({"type": "log", "job_id": job_id, "message": line})

        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _connected_websockets:
            _connected_websockets[job_id].remove(websocket)


# ---------------------------------------------------------------------------
# Chat / Inference
# ---------------------------------------------------------------------------

_loaded_generators: dict[str, Any] = {}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with a trained model."""
    model_path = req.model_path

    if not Path(model_path).exists():
        raise HTTPException(404, f"Model not found: {model_path}")

    try:
        # Lazy-load generator
        if model_path not in _loaded_generators:
            from concordlm.inference.generate import Generator
            _loaded_generators[model_path] = Generator(
                model_path,
                system_prompt=req.system_prompt,
            )

        generator = _loaded_generators[model_path]
        response = generator.generate(
            req.message,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return {"response": response, "model_path": model_path}

    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.post("/api/evaluate")
async def run_evaluation(req: EvalRequest):
    """Run evaluation on a model."""
    job_id = str(uuid.uuid4())[:8]

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--model", req.model_path,
        "--output-dir", str(OUTPUTS_DIR / "eval"),
    ]
    if req.compare_model_path:
        cmd.extend(["--compare", req.compare_model_path])
    if req.max_samples:
        cmd.extend(["--max-samples", str(req.max_samples)])

    _active_jobs[job_id] = {
        "stage": "eval",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "command": " ".join(cmd),
    }
    _training_logs[job_id] = []
    asyncio.create_task(_run_training_process(job_id, cmd))

    return {"job_id": job_id, "status": "started"}


@app.get("/api/evaluate/report")
async def get_eval_report():
    """Get the latest evaluation report."""
    report_path = OUTPUTS_DIR / "eval" / "eval_report.json"
    if not report_path.exists():
        raise HTTPException(404, "No evaluation report found")
    with open(report_path) as f:
        return json.load(f)


@app.get("/api/evaluate/report/detailed")
async def get_eval_report_detailed():
    """Get the detailed evaluation report."""
    report_path = OUTPUTS_DIR / "eval" / "eval_report_detailed.json"
    if not report_path.exists():
        raise HTTPException(404, "No detailed report found")
    with open(report_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


@app.post("/api/dataset/build")
async def build_dataset(req: DatasetBuildRequest):
    """Build a custom preference dataset from provided pairs."""
    from concordlm.data.preference_dataset import build_preference_dataset_from_pairs

    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / req.output_name

    try:
        result_path = build_preference_dataset_from_pairs(req.pairs, output_path)
        return {"status": "ok", "path": str(result_path), "num_pairs": len(req.pairs)}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/dataset/list")
async def list_datasets():
    """List available datasets."""
    data_dir = PROJECT_ROOT / "data"
    datasets = []
    if data_dir.exists():
        for f in data_dir.glob("*.jsonl"):
            stat = f.stat()
            with open(f) as fh:
                line_count = sum(1 for _ in fh)
            datasets.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "num_pairs": line_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return datasets


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@app.get("/api/models")
async def list_models():
    """List available model checkpoints."""
    models = []
    if OUTPUTS_DIR.exists():
        for stage_dir in OUTPUTS_DIR.iterdir():
            if not stage_dir.is_dir():
                continue
            for ckpt_dir in stage_dir.iterdir():
                if ckpt_dir.is_dir() and (ckpt_dir / "config.json").exists() or \
                   (ckpt_dir / "adapter_config.json").exists():
                    models.append({
                        "name": f"{stage_dir.name}/{ckpt_dir.name}",
                        "path": str(ckpt_dir),
                        "stage": stage_dir.name,
                        "has_adapter": (ckpt_dir / "adapter_config.json").exists(),
                    })
    return models


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main():
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(WEB_DIR)],
    )


if __name__ == "__main__":
    main()
