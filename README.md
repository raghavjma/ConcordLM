# ConcordLM 🔮

An end-to-end LLM alignment pipeline designed for aligning open-source models (Llama 3, Mistral, Qwen) using Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). Includes a premium web dashboard for managing the ml-pipeline.

## Features

- **Modular Pipeline:** Train models using 3 distinct stages: SFT → DPO → RLHF (PPO) 
- **Web Dashboard:** FastAPI and Vanilla JS powered dark-mode GUI to track progress, edit configs, and launch training runs.
- **Resource Efficient:** Full support for QLoRA (4-bit NF4) enabling training on consumer GPUs (≥24GB VRAM for 7B models).
- **Datasets & Evaluation:** Standardized data loading, automated safety and quality evaluation bench, and a custom dataset builder.
- **Interactive Chat:** Test your aligned models directly in the UI.

## Tech Stack

**ML Backend:** PyTorch, Hugging Face Transformers, TRL (Transformer Reinforcement Learning), PEFT, BitsAndBytes.
**Web Frontend:** HTML, CSS (Glassmorphism design), Vanilla JS, FastAPI, Uvicorn, WebSockets.

## Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/raghavjma/ConcordLM.git
cd ConcordLM

# Install with development dependencies
pip install -e ".[dev]"
```

*Note: Set your `HF_TOKEN` environment variable if you plan to use gated models like Llama 3.*

## Usage

### 1. Web Dashboard (Recommended)

Start the dashboard to control the pipeline from your browser:

```bash
concordlm-web
# or
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000`.

### 2. CLI

You can also run training stages directly via command line, using dot-notation to override configuration values in `configs/`:

```bash
# SFT
concordlm-sft --override model.name=mistralai/Mistral-7B-Instruct-v0.3 training.max_steps=500

# DPO
concordlm-dpo --override dpo.beta=0.1

# RLHF
concordlm-rlhf
```

### 3. CLI Evaluation & Generation

```bash
# Evaluate a checkpoint's safety and quality
concordlm-eval --model_path ./outputs/dpo/checkpoint-final --compare ./outputs/sft/checkpoint-final

# Interactive chat in the terminal
concordlm-generate --model_path ./outputs/dpo/checkpoint-final --interactive
```

## Configuration
The project uses a hierarchical YAML config system.
- `configs/base.yaml`: Global defaults (Model, Quantization, Generic Training Params).
- `configs/sft.yaml`: SFT specific parameters.
- `configs/dpo.yaml`: DPO specific parameters.
- `configs/rlhf.yaml`: RLHF (+ Reward Model) specific parameters.

## License

MIT License