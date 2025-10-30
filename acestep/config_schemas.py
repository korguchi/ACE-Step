"""Pydantic configuration schemas for accompaniment generation system."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LoRAConfig(BaseModel):
    """LoRA configuration schema."""

    r: int = Field(256, ge=1, le=1024, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha scaling factor")
    target_modules: List[str] = Field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"],
        description="Module names to apply LoRA",
    )
    use_rslora: bool = Field(True, description="Use Rank-Stabilized LoRA")
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0, description="LoRA dropout rate")

    @field_validator("target_modules")
    @classmethod
    def validate_modules(cls, v: List[str]) -> List[str]:
        """Validate target module names."""
        valid = {
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "speaker_embedder",
            "genre_embedder",
            "lyric_proj",
            "linear_q",
            "linear_k",
            "linear_v",
        }
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid target modules: {invalid}")
        return v


class AccompanimentConfig(BaseModel):
    """Accompaniment generation configuration schema."""

    mode: Literal["image2image", "controlnet"] = Field(
        "image2image", description="Generation mode"
    )
    output_type: Literal["accompaniment", "mix"] = Field(
        "accompaniment", description="Output type: accompaniment only or full mix"
    )

    # Image2Image settings
    reference_strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Reference strength (0=ignore, 1=full copy)",
    )

    # ControlNet settings
    control_encoder_depth: int = Field(
        6, ge=2, le=12, description="Control encoder depth"
    )
    control_encoder_dim: int = Field(
        1536, ge=512, le=2560, description="Control encoder hidden dimension"
    )
    controlnet_scale: float = Field(
        1.0, ge=0.0, le=2.0, description="ControlNet scale factor"
    )
    injection_layers: List[int] = Field(
        default_factory=lambda: [0, 4, 8, 12],
        description="Transformer layers to inject control signals",
    )


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    # LoRA & Accompaniment
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    accompaniment: AccompanimentConfig = Field(default_factory=AccompanimentConfig)

    # Dataset
    dataset_path: str = Field(
        "./data/accompaniment_dataset", description="Dataset root path"
    )
    max_duration: float = Field(
        240.0, gt=0.0, description="Maximum audio duration (seconds)"
    )

    # Training hyperparameters
    learning_rate: float = Field(1e-4, gt=0.0, description="Learning rate")
    max_steps: int = Field(200000, ge=1, description="Maximum training steps")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    weight_decay: float = Field(1e-2, ge=0.0, description="Weight decay")
    batch_size: int = Field(1, ge=1, description="Batch size per device")
    accumulate_grad_batches: int = Field(
        1, ge=1, description="Gradient accumulation batches"
    )

    # SSL constraint
    ssl_coeff: float = Field(1.0, ge=0.0, description="SSL loss coefficient")
    ssl_depths: List[int] = Field(
        default_factory=lambda: [8, 8], description="SSL feature depths"
    )

    # Checkpoint
    checkpoint_every_n_steps: int = Field(
        2000, ge=1, description="Checkpoint save interval"
    )
    exp_name: str = Field("accompaniment_lora", description="Experiment name")
    logger_dir: str = Field("./exps/logs/", description="Logger directory")

    # Flow matching
    shift: float = Field(3.0, gt=0.0, description="Flow matching shift parameter")


class OptunaConfig(BaseModel):
    """Optuna hyperparameter search configuration schema."""

    n_trials: int = Field(50, ge=1, description="Number of trials")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")

    # Pruner
    pruner_type: Literal["median", "successive_halving", "hyperband"] = Field(
        "median", description="Pruner type"
    )
    pruner_n_startup_trials: int = Field(
        5, ge=0, description="Number of startup trials without pruning"
    )
    pruner_n_warmup_steps: int = Field(
        100, ge=0, description="Warmup steps before pruning"
    )

    # Search space
    search_space: dict = Field(
        default_factory=lambda: {
            "reference_strength": {"type": "float", "low": 0.1, "high": 0.9},
            "lora_rank": {"type": "int", "low": 64, "high": 512, "step": 64},
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        },
        description="Hyperparameter search space definition",
    )
