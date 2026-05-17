# ConcordLM — Project State & Handover

## Overview
**ConcordLM** is an end-to-end LLM alignment pipeline designed for aligning open-source models (Llama 3, Mistral, Qwen) using Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). 

## Recent Milestones Achieved
- **Interactive Upgrades:** Upgraded the ConcordLM dashboard to support real-time, stateful, and streaming multi-turn conversations in the UI.
- **Visual Pedigree:** Implemented dynamic visualization of the model's alignment pedigree within the dashboard, allowing users to track the progression from base model through the SFT, DPO, and RLHF stages.
- **Dashboard Infrastructure:** Robust integration between the FastAPI backend and Vanilla JS (Glassmorphism design) frontend for direct inference and model interaction.

## Tech Stack Used
- **ML Backend:** PyTorch, Hugging Face Transformers, TRL, PEFT, BitsAndBytes (QLoRA).
- **Web Frontend:** HTML, CSS (Vanilla), JS, FastAPI, WebSockets.

## How to Resume Later
- **Dashboard:** Navigate to the folder and run `concordlm-web` or `python -m uvicorn web.app:app --host 0.0.0.0 --port 8000`.
- **CLI Chat:** Run `concordlm-generate --model_path <path> --interactive`.

*(Note: The AI assistant natively retains all deep knowledge of this workspace in its Conversation Logs and Knowledge Items for future context!)*
