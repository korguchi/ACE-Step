#!/usr/bin/env python3
"""Compare multiple LoRA models for accompaniment generation."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from acestep.pipeline_ace_step import ACEStepPipeline


def generate_comparison_html(
    output_dir: Path, lora_configs: list[str], test_samples: list[dict]
) -> None:
    """
    Generate comparison HTML page.

    Args:
        output_dir: Output directory
        lora_configs: List of LoRA config paths
        test_samples: List of test samples
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accompaniment Generation Comparison</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            audio {
                width: 300px;
            }
            .sample-key {
                font-weight: bold;
                color: #4CAF50;
            }
            .prompt {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Accompaniment Generation Comparison</h1>
        <table>
            <tr>
                <th>Sample</th>
                <th>Vocal (Input)</th>
    """

    # LoRA column headers
    for config_path in lora_configs:
        lora_name = Path(config_path).stem
        html += f"<th>{lora_name}</th>"

    html += "</tr>"

    # Sample rows
    for sample in test_samples:
        html += f"""
        <tr>
            <td>
                <div class="sample-key">{sample["key"]}</div>
                <div class="prompt">{sample["prompt"]}</div>
            </td>
            <td><audio controls><source src="{sample["vocal_path"]}" type="audio/mpeg"></audio></td>
        """

        for config_path in lora_configs:
            lora_name = Path(config_path).stem
            audio_path = output_dir / lora_name / f"{sample['key']}_generated.mp3"
            rel_path = audio_path.relative_to(output_dir)
            html += f'<td><audio controls><source src="{rel_path}" type="audio/mpeg"></audio></td>'

        html += "</tr>"

    html += """
        </table>
        <div style="text-align: center; margin-top: 20px; color: #666;">
            <p>Generated with ACE-Step Accompaniment Generation System</p>
        </div>
    </body>
    </html>
    """

    with open(output_dir / "comparison.html", "w") as f:
        f.write(html)


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare multiple LoRA models")
    parser.add_argument(
        "--test_set",
        type=str,
        required=True,
        help="Path to test set JSON file",
    )
    parser.add_argument(
        "--lora_configs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to LoRA config JSON files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to pretrained ACE-Step checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison",
        help="Output directory for comparison",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=15.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=60,
        help="Inference steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    args = parser.parse_args()

    # Load test set
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_set_path}")

    with open(test_set_path) as f:
        test_samples = json.load(f)

    print("Comparison configuration:")
    print(f"  Test samples: {len(test_samples)}")
    print(f"  LoRA configs: {len(args.lora_configs)}")
    print(f"  Output: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize pipeline
    print("\nLoading model...")
    pipeline = ACEStepPipeline(checkpoint_dir=args.checkpoint_dir, device=args.device)

    # Process each LoRA
    for lora_config_path in args.lora_configs:
        with open(lora_config_path) as f:
            lora_config = json.load(f)

        lora_name = Path(lora_config_path).stem
        lora_output_dir = output_dir / lora_name
        lora_output_dir.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing: {lora_name}")
        print(f"{'=' * 60}")

        # Load LoRA
        pipeline.load_lora(
            lora_config["lora_path"], lora_config.get("lora_weight", 0.8)
        )

        # Process samples
        for sample in tqdm(test_samples, desc=f"Generating with {lora_name}"):
            try:
                output_paths = pipeline.text2music_diffusion_process(
                    prompt=sample["prompt"],
                    lyrics="[instrumental]",
                    audio2audio_enable=True,
                    ref_audio_input=sample["vocal_path"],
                    ref_audio_strength=lora_config.get("ref_strength", 0.5),
                    guidance_scale=args.guidance_scale,
                    infer_steps=args.infer_steps,
                    output_dir=str(lora_output_dir),
                    filename_prefix=sample["key"],
                )
            except Exception as e:
                print(f"  Error processing {sample['key']}: {e}")
                continue

    # Generate comparison HTML
    print(f"\n{'=' * 60}")
    print("Generating comparison HTML...")
    print(f"{'=' * 60}")
    generate_comparison_html(output_dir, args.lora_configs, test_samples)

    print("\nComparison completed!")
    print("Open the following file in your browser:")
    print(f"  {output_dir / 'comparison.html'}")


if __name__ == "__main__":
    main()
