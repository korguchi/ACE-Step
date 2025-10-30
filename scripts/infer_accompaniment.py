#!/usr/bin/env python3
"""Inference script for accompaniment generation."""

import argparse
from pathlib import Path

from acestep.pipeline_ace_step import ACEStepPipeline


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Generate accompaniment from vocal track"
    )
    parser.add_argument(
        "--vocal_path",
        type=str,
        required=True,
        help="Path to vocal track file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt (tags) for generation, e.g., 'electronic, piano, 120bpm'",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to pretrained ACE-Step checkpoint directory",
    )
    parser.add_argument(
        "--lora_weight",
        type=float,
        default=0.8,
        help="LoRA weight (0.0-1.0)",
    )
    parser.add_argument(
        "--ref_strength",
        type=float,
        default=0.5,
        help="Reference strength for Image2Image mode (0.0-1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=15.0,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=60,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    # Validate paths
    vocal_path = Path(args.vocal_path)
    if not vocal_path.exists():
        raise FileNotFoundError(f"Vocal track not found: {vocal_path}")

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Accompaniment generation:")
    print(f"  Vocal: {vocal_path}")
    print(f"  Prompt: {args.prompt}")
    print(f"  LoRA: {lora_path}")
    print(f"  LoRA weight: {args.lora_weight}")
    print(f"  Reference strength: {args.ref_strength}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Inference steps: {args.infer_steps}")
    print(f"  Output: {output_dir}")

    # Initialize pipeline
    print("\nLoading model...")
    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Load LoRA
    print("Loading LoRA adapter...")
    pipeline.load_lora(str(lora_path), args.lora_weight)

    # Generate
    print("\nGenerating accompaniment...")
    output_paths = pipeline.text2music_diffusion_process(
        prompt=args.prompt,
        lyrics="[instrumental]",
        duration=None,  # Use vocal duration
        audio2audio_enable=True,
        ref_audio_input=str(vocal_path),
        ref_audio_strength=args.ref_strength,
        guidance_scale=args.guidance_scale,
        infer_steps=args.infer_steps,
        output_dir=str(output_dir),
        task="audio2audio",
    )

    print(f"\n{'=' * 60}")
    print("Generation completed!")
    print(f"{'=' * 60}")
    print(f"Output file: {output_paths[0]}")


if __name__ == "__main__":
    main()
