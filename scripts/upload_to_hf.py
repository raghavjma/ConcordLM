#!/usr/bin/env python3
"""ConcordLM — Hugging Face Uploader"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Upload ConcordLM model to Hugging Face")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local model (e.g., ./outputs/rlhf/grpo-final-1250)")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID (e.g., your_username/concordlm-0.5b-rlhf)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face write token")
    parser.add_argument("--merge", action="store_true", help="Merge PEFT adapter into base model before uploading")
    args = parser.parse_args()

    from huggingface_hub import HfApi, login
    
    print("Logging into Hugging Face...")
    login(token=args.token)
    api = HfApi()

    print(f"Creating repository '{args.repo_id}' if it doesn't exist...")
    api.create_repo(repo_id=args.repo_id, exist_ok=True)

    if args.merge:
        print("\nMerging PEFT adapter into base model...")
        import torch
        from transformers import AutoTokenizer
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Merging weights (this may take a minute)...")
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        print(f"\nPushing merged model to {args.repo_id}...")
        model.push_to_hub(args.repo_id, token=args.token)
        tokenizer.push_to_hub(args.repo_id, token=args.token)
    else:
        print(f"\nPushing PEFT adapter directory directly to {args.repo_id}...")
        api.upload_folder(
            folder_path=args.model_path,
            repo_id=args.repo_id,
            commit_message="Upload ConcordLM RLHF Adapter"
        )
    
    print("\nUpload complete! View your model at: https://huggingface.co/" + args.repo_id)

if __name__ == "__main__":
    main()
