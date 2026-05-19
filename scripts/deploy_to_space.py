#!/usr/bin/env python3
"""ConcordLM — Hugging Face Space Deployer"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Deploy ConcordLM Dashboard to Hugging Face Spaces")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Space repo ID (e.g., your_username/ConcordLM-Dashboard)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face write token")
    args = parser.parse_args()

    from huggingface_hub import HfApi, login
    
    print("Logging into Hugging Face...")
    login(token=args.token)
    api = HfApi()

    print(f"Creating Hugging Face Space '{args.repo_id}' (Docker SDK)...")
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
    except Exception as e:
        print(f"Warning/Error creating space: {e}")

    print("\nUploading source code to Space (ignoring outputs/ and pycache)...")
    
    # Upload everything in the project root except the ignored folders
    api.upload_folder(
        folder_path=".",
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["outputs/*", "data/*", "__pycache__/*", "*.pyc", ".git/*", ".venv/*"],
        commit_message="Deploy ConcordLM Dashboard"
    )
    
    print("\n✅ Deployment complete!")
    print(f"Your dashboard is currently building and will be live at: https://huggingface.co/spaces/{args.repo_id.replace('/', '-')}")

if __name__ == "__main__":
    main()
