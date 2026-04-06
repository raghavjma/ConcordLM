"""
Generation utilities — interactive chat, batch generation, sampling.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation engine
# ---------------------------------------------------------------------------


class Generator:
    """
    Text generation engine with configurable sampling parameters.

    Supports LoRA adapter loading (merged or on-the-fly) and chat-template formatting.
    """

    def __init__(
        self,
        model_path: str,
        *,
        merge_adapter: bool = True,
        dtype: str = "bfloat16",
        device: str = "auto",
        system_prompt: str | None = None,
    ):
        from concordlm.models.loader import load_model_for_inference

        self.model, self.tokenizer = load_model_for_inference(
            model_path,
            merge_adapter=merge_adapter,
            dtype=dtype,
            device=device,
        )
        self.system_prompt = system_prompt
        self.device = next(self.model.parameters()).device
        logger.info(f"Generator ready on device: {self.device}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response for a single prompt.

        The prompt is formatted using the model's chat template.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[str]:
        """Generate responses for a batch of prompts (sequentially)."""
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}...")
            responses.append(self.generate(prompt, **kwargs))
        return responses

    def interactive_chat(self, **kwargs):
        """
        Start an interactive chat session in the terminal.

        Type 'quit' or 'exit' to stop. Type 'reset' to clear system prompt.
        """
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()

        console.print(
            Panel.fit(
                "[bold cyan]ConcordLM Interactive Chat[/bold cyan]\n"
                "[dim]Type 'quit' to exit, 'reset' to clear context[/dim]",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input.lower() == "reset":
                console.print("[dim]Context reset.[/dim]")
                continue

            response = self.generate(user_input, **kwargs)
            console.print(f"\n[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(response))


# ---------------------------------------------------------------------------
# Batch generation from file
# ---------------------------------------------------------------------------


def batch_generate_from_file(
    model_path: str,
    input_path: str,
    output_path: str,
    **kwargs,
) -> None:
    """
    Generate responses for prompts from a JSONL file.

    Input format: one JSON object per line with a ``prompt`` field.
    Output: same objects with an added ``response`` field.
    """
    generator = Generator(model_path)

    input_data = []
    with open(input_path) as f:
        for line in f:
            input_data.append(json.loads(line))

    logger.info(f"Generating {len(input_data)} responses...")

    with open(output_path, "w") as f:
        for item in input_data:
            prompt = item.get("prompt", "")
            response = generator.generate(prompt, **kwargs)
            item["response"] = response
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Responses saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for generation."""
    parser = argparse.ArgumentParser(
        description="ConcordLM — Text Generation"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--mode", type=str, choices=["chat", "batch", "single"], default="chat",
        help="Generation mode: 'chat' (interactive), 'batch' (from file), 'single' (one prompt)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt for single generation mode",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="Input JSONL file for batch mode",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output JSONL file for batch mode",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="System prompt to prepend",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.mode == "chat":
        generator = Generator(args.model, system_prompt=args.system_prompt)
        generator.interactive_chat(**gen_kwargs)

    elif args.mode == "single":
        if not args.prompt:
            print("Error: --prompt is required for single mode")
            sys.exit(1)
        generator = Generator(args.model, system_prompt=args.system_prompt)
        response = generator.generate(args.prompt, **gen_kwargs)
        print(f"\n{response}")

    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print("Error: --input-file and --output-file are required for batch mode")
            sys.exit(1)
        batch_generate_from_file(
            args.model, args.input_file, args.output_file,
            **gen_kwargs,
        )


if __name__ == "__main__":
    main()
