"""Trainer for accompaniment generation with Image2Image and ControlNet modes."""

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

from acestep.accompaniment_dataset import AccompanimentDataset
from acestep.config_schemas import TrainingConfig
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class AccompanimentTrainer(LightningModule):
    """Trainer for accompaniment generation (Image2Image + ControlNet)."""

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_dir: str,
        lora_config_path: Optional[str] = None,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize accompaniment trainer.

        Args:
            config: Training configuration (Pydantic model)
            checkpoint_dir: Pretrained model directory
            lora_config_path: LoRA config path (None to use config.lora)
            num_workers: Number of dataloader workers
        """
        super().__init__()

        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.num_workers = num_workers

        # Mode detection
        self.use_image2image = config.accompaniment.mode == "image2image"

        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=config.shift,
        )

        # Load models
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(checkpoint_dir)

        # Transformer with gradient checkpointing
        transformers = acestep_pipeline.ace_step_transformer.float().cpu()
        transformers.enable_gradient_checkpointing()

        # Load LoRA
        if lora_config_path is not None:
            from peft import LoraConfig

            with open(lora_config_path, encoding="utf-8") as f:
                lora_config_dict = json.load(f)
            lora_config = LoraConfig(**lora_config_dict)
        else:
            from peft import LoraConfig

            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.lora_dropout,
                use_rslora=config.lora.use_rslora,
            )

        adapter_name = config.exp_name
        transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
        self.adapter_name = adapter_name

        self.transformers = transformers

        # VAE (frozen)
        self.dcae = acestep_pipeline.music_dcae.float().cpu()
        self.dcae.requires_grad_(False)

        # Text encoder (frozen)
        self.text_encoder_model = acestep_pipeline.text_encoder_model.float().cpu()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer

        # ControlNet encoder (if needed)
        if not self.use_image2image:
            from acestep.models.control_encoder import ControlEncoder

            self.control_encoder = ControlEncoder(
                depth=config.accompaniment.control_encoder_depth,
                dim=config.accompaniment.control_encoder_dim,
            )
            self.control_encoder.train()

        # Training mode
        self.transformers.train()

        # Dataset placeholder
        self.train_dataset: Optional[AccompanimentDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup dataset."""
        self.train_dataset = AccompanimentDataset(
            train=True,
            train_dataset_path=self.config.dataset_path,
            output_type=self.config.accompaniment.output_type,
            max_duration=self.config.max_duration,
            minibatch_size=self.config.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        assert self.train_dataset is not None, "Dataset not initialized"
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Dataset already handles batching
            num_workers=self.num_workers,
            collate_fn=AccompanimentDataset.collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Collect parameters to optimize
        params = []

        # LoRA parameters from transformer
        for name, param in self.transformers.named_parameters():
            if param.requires_grad:
                params.append(param)

        # ControlNet parameters (if applicable)
        if not self.use_image2image:
            for param in self.control_encoder.parameters():
                if param.requires_grad:
                    params.append(param)

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Linear warmup scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def preprocess(self, batch: dict):
        """
        Preprocess batch: audio → latents.

        Args:
            batch: Batch dict from dataloader

        Returns:
            Tuple of processed tensors
        """
        device = self.device

        # Encode reference (vocal) and target (inst/mix)
        with torch.no_grad():
            ref_latents = self.dcae.encode(
                batch["reference_wavs"].to(device), batch["wav_lengths"].to(device)
            ).latent_dist.sample()

            target_latents = self.dcae.encode(
                batch["target_wavs"].to(device), batch["wav_lengths"].to(device)
            ).latent_dist.sample()

            # Text encoding (genre/prompt)
            text_inputs = self.text_tokenizer(
                batch["prompts"],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            genre_embeds = self.text_encoder_model(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )[0]

        return (
            ref_latents,
            target_latents,
            genre_embeds,
            batch["speaker_embs"].to(device),
            batch["lyric_token_ids"].to(device),
            batch["lyric_masks"].to(device),
            batch["wav_lengths"].to(device),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps with logit-normal distribution.

        Args:
            batch_size: Batch size
            device: Device

        Returns:
            Timesteps tensor (B,)
        """
        u = torch.randn(batch_size, device=device)
        u = torch.sigmoid(u)
        timesteps = u * self.scheduler.config.num_train_timesteps
        timesteps = timesteps.long()
        timesteps = torch.clamp(
            timesteps, 0, self.scheduler.config.num_train_timesteps - 1
        )
        return timesteps

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Get sigma values from timesteps.

        Args:
            timesteps: Timesteps (B,)
            n_dim: Number of dimensions to expand to
            dtype: Data type

        Returns:
            Sigmas (B, 1, 1, 1)
        """
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()

        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def create_length_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create length mask.

        Args:
            lengths: Lengths tensor (B,)
            max_len: Maximum length

        Returns:
            Mask (B, max_len)
        """
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        return mask.float()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Batch dict
            batch_idx: Batch index

        Returns:
            Loss scalar
        """
        # Preprocess
        (
            ref_latents,
            target_latents,
            genre_embeds,
            speaker_embeds,
            lyric_tokens,
            lyric_masks,
            lengths,
        ) = self.preprocess(batch)

        device = self.device
        bsz = target_latents.shape[0]

        # Sample noise
        noise = torch.randn_like(target_latents)

        # Sample timesteps
        timesteps = self.sample_timesteps(bsz, device)

        # Get sigmas
        sigmas = self.get_sigmas(timesteps, target_latents.ndim, target_latents.dtype)

        # Add noise
        if self.use_image2image:
            # Image2Image: mix reference with noise
            strength = self.config.accompaniment.reference_strength
            noisy_input = ref_latents * (1.0 - sigmas) * strength + noise * sigmas
        else:
            # ControlNet: pure noise
            noisy_input = noise * sigmas + target_latents * (1.0 - sigmas)

        # Forward
        if self.use_image2image:
            # Standard forward
            model_output = self.transformers(
                noisy_input,
                timesteps,
                encoder_text_hidden_states=genre_embeds,
                speaker_embeds=speaker_embeds,
                lyric_token_idx=lyric_tokens,
                lyric_attention_mask=lyric_masks,
            ).sample
        else:
            # ControlNet: inject control signals
            control_signals = self.control_encoder(ref_latents, timesteps)

            model_output = self.transformers(
                noisy_input,
                timesteps,
                encoder_text_hidden_states=genre_embeds,
                speaker_embeds=speaker_embeds,
                lyric_token_idx=lyric_tokens,
                lyric_attention_mask=lyric_masks,
                block_controlnet_hidden_states=[control_signals]
                * len(self.config.accompaniment.injection_layers),
                controlnet_scale=self.config.accompaniment.controlnet_scale,
            ).sample

        # Preconditioning (Flow matching)
        model_pred = model_output * (-sigmas) + noisy_input

        # Loss
        loss = F.mse_loss(model_pred, target_latents, reduction="none")

        # Masking (ignore padding)
        # Convert audio lengths to latent lengths (downsampling factor = 8 for time)
        latent_lengths = lengths // (48000 // 8)  # Rough approximation
        mask = self.create_length_mask(latent_lengths, target_latents.shape[-1])
        mask = mask[:, None, None, :]  # (B, 1, 1, T)
        loss = (loss * mask).sum() / mask.sum()

        # Logging
        self.log(
            "train/denoising_loss", loss, on_step=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train/learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True
        )

        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        """
        Save checkpoint callback - only save LoRA adapters.

        Args:
            checkpoint: Checkpoint dict

        Returns:
            Modified checkpoint dict
        """
        log_dir = self.logger.log_dir
        epoch = self.current_epoch
        step = self.global_step

        checkpoint_name = f"epoch={epoch}-step={step}_lora"
        checkpoint_dir = os.path.join(log_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA adapter
        self.transformers.save_pretrained(checkpoint_dir)

        # Save ControlNet encoder (if applicable)
        if not self.use_image2image:
            torch.save(
                self.control_encoder.state_dict(),
                os.path.join(checkpoint_dir, "control_encoder.pth"),
            )

        # Save config
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        logger.info(f"Saved LoRA checkpoint to {checkpoint_dir}")

        # Return empty dict (don't save full model state)
        return {}
