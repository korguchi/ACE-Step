#!/usr/bin/env python3
"""Training script for Image2Image accompaniment generation."""

import argparse
import json
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from acestep.config_schemas import TrainingConfig
from trainer_accompaniment import AccompanimentTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train accompaniment generation model (Image2Image mode)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to pretrained ACE-Step checkpoint directory",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps from config (for testing)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Override max_steps if specified
    if args.max_steps is not None:
        config_dict["max_steps"] = args.max_steps

    # Validate config
    config = TrainingConfig(**config_dict)

    print("Training configuration:")
    print(f"  Mode: {config.accompaniment.mode}")
    print(f"  Output type: {config.accompaniment.output_type}")
    print(f"  Reference strength: {config.accompaniment.reference_strength}")
    print(f"  LoRA rank: {config.lora.r}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Accumulate batches: {config.accumulate_grad_batches}")

    # Initialize trainer
    model = AccompanimentTrainer(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        lora_config_path=None,  # Use config.lora
        num_workers=args.num_workers,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_top_k=-1,  # Save all checkpoints
        filename="epoch={epoch}-step={step}",
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=config.logger_dir,
        name=config.exp_name,
    )

    # PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        max_steps=config.max_steps,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.5,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=10,
        check_val_every_n_epoch=None,  # No validation
        enable_checkpointing=True,
        enable_model_summary=True,
    )

    print("\nStarting training...")
    print(f"Logs will be saved to: {logger.log_dir}")

    # Start training
    trainer.fit(model)

    print("\nTraining completed!")
    print(f"Final checkpoint saved to: {logger.log_dir}/checkpoints/")


if __name__ == "__main__":
    main()
