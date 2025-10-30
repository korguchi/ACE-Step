#!/usr/bin/env python3
"""Optuna hyperparameter search for accompaniment generation."""

import argparse
import json
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from pytorch_lightning import Trainer

from acestep.config_schemas import OptunaConfig, TrainingConfig
from trainer_accompaniment import AccompanimentTrainer


def objective(
    trial: optuna.Trial, base_config: TrainingConfig, optuna_cfg: OptunaConfig
) -> float:
    """
    Optuna objective function.

    Args:
        trial: Optuna trial
        base_config: Base training configuration
        optuna_cfg: Optuna configuration

    Returns:
        Final loss value
    """
    # Sample hyperparameters based on mode
    if base_config.accompaniment.mode == "image2image":
        # Image2Image search space
        reference_strength = trial.suggest_float("reference_strength", 0.1, 0.9)
        lora_rank = trial.suggest_int("lora_rank", 64, 512, step=64)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    else:
        # ControlNet search space
        control_encoder_depth = trial.suggest_int(
            "control_encoder_depth", 4, 12, step=2
        )
        controlnet_scale = trial.suggest_float("controlnet_scale", 0.5, 2.0)
        lora_rank = trial.suggest_int("lora_rank", 64, 512, step=64)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Create trial config
    trial_config = base_config.model_copy(deep=True)
    trial_config.lora.r = lora_rank
    trial_config.learning_rate = learning_rate

    if base_config.accompaniment.mode == "image2image":
        trial_config.accompaniment.reference_strength = reference_strength
    else:
        trial_config.accompaniment.control_encoder_depth = control_encoder_depth
        trial_config.accompaniment.controlnet_scale = controlnet_scale

    # Short training (1000 steps for faster search)
    trial_config.max_steps = 1000
    trial_config.checkpoint_every_n_steps = 10000  # Don't save during search

    # Initialize trainer
    model = AccompanimentTrainer(
        config=trial_config,
        checkpoint_dir=base_config.dataset_path.replace(
            "/data/accompaniment_dataset", "/checkpoints"
        ),
        num_workers=2,  # Reduced for speed
    )

    # Pruning callback
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="train/denoising_loss"
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=trial_config.max_steps,
        callbacks=[pruning_callback],
        logger=False,  # Disable logging during search
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=0.5,
        accumulate_grad_batches=trial_config.accumulate_grad_batches,
    )

    # Train
    trainer.fit(model)

    # Return final loss
    if "train/denoising_loss" in trainer.callback_metrics:
        return trainer.callback_metrics["train/denoising_loss"].item()
    else:
        # Trial was pruned or failed
        raise optuna.TrialPruned()


def main():
    """Main search function."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base training config JSON file",
    )
    parser.add_argument(
        "--optuna_config",
        type=str,
        required=True,
        help="Path to Optuna config JSON file",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_accompaniment.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="accompaniment_search",
        help="Optuna study name",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        config_dict = json.load(f)
    base_config = TrainingConfig(**config_dict)

    with open(args.optuna_config) as f:
        optuna_dict = json.load(f)
    optuna_cfg = OptunaConfig(**optuna_dict)

    print("Optuna search configuration:")
    print(f"  Mode: {base_config.accompaniment.mode}")
    print(f"  Number of trials: {optuna_cfg.n_trials}")
    print(f"  Pruner: {optuna_cfg.pruner_type}")
    print(f"  Storage: {args.storage}")
    print(f"  Study name: {args.study_name}")

    # Configure pruner
    if optuna_cfg.pruner_type == "median":
        pruner = MedianPruner(
            n_startup_trials=optuna_cfg.pruner_n_startup_trials,
            n_warmup_steps=optuna_cfg.pruner_n_warmup_steps,
        )
    elif optuna_cfg.pruner_type == "successive_halving":
        pruner = SuccessiveHalvingPruner()
    else:
        pruner = None

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    print("\nStarting optimization...")

    # Optimize
    study.optimize(
        lambda trial: objective(trial, base_config, optuna_cfg),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
        show_progress_bar=True,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("Optimization completed!")
    print(f"{'=' * 60}")
    print("\nBest trial:")
    print(f"  Value (loss): {study.best_value:.6f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Create best config
    best_config = base_config.model_copy(deep=True)
    best_config.lora.r = study.best_params["lora_rank"]
    best_config.learning_rate = study.best_params["learning_rate"]

    if base_config.accompaniment.mode == "image2image":
        best_config.accompaniment.reference_strength = study.best_params[
            "reference_strength"
        ]
    else:
        best_config.accompaniment.control_encoder_depth = study.best_params[
            "control_encoder_depth"
        ]
        best_config.accompaniment.controlnet_scale = study.best_params[
            "controlnet_scale"
        ]

    # Restore original max_steps
    best_config.max_steps = base_config.max_steps
    best_config.checkpoint_every_n_steps = base_config.checkpoint_every_n_steps

    # Save best config
    output_path = Path(args.config).parent / f"{args.study_name}_best.json"
    with open(output_path, "w") as f:
        json.dump(best_config.model_dump(), f, indent=2)

    print(f"\nBest config saved to: {output_path}")
    print("\nYou can now train with:")
    print("  uv run python scripts/train_image2image.py \\")
    print(f"    --config {output_path} \\")
    print("    --checkpoint_dir ./checkpoints \\")
    print("    --devices 1")


if __name__ == "__main__":
    main()
