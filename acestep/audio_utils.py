"""Audio processing utilities for accompaniment generation."""

from typing import Optional

import torch
import torch.nn.functional as F


class AudioMixer:
    """Utility for blending audio in latent space."""

    @staticmethod
    def blend_latents(
        generated: torch.Tensor,
        reference: torch.Tensor,
        mix_ratio: float = 0.8,
        frequency_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Blend generated latents with reference latents.

        Args:
            generated: (B, 8, 16, T) Generated accompaniment latents
            reference: (B, 8, 16, T) Reference accompaniment latents
            mix_ratio: 0=full ref, 1=full generated
            frequency_mask: (8, 16) Frequency band mask (None for uniform blending)

        Returns:
            (B, 8, 16, T) Blended latents
        """
        if frequency_mask is not None:
            mask = frequency_mask[None, :, :, None]  # (1, 8, 16, 1)
            return (
                generated * mask * mix_ratio
                + reference * mask * (1 - mix_ratio)
                + reference * (1 - mask)
            )
        else:
            return generated * mix_ratio + reference * (1 - mix_ratio)

    @staticmethod
    def create_frequency_mask(
        preserve_low: bool = True, preserve_high: bool = False
    ) -> torch.Tensor:
        """
        Create frequency-selective mask.

        Args:
            preserve_low: Preserve low frequencies (bass, kick) from reference
            preserve_high: Preserve high frequencies from reference

        Returns:
            (8, 16) Mask (1=use generated, 0=keep reference)
        """
        mask = torch.ones(8, 16)

        if preserve_low:
            mask[:, :4] = 0.0  # Low frequency band

        if preserve_high:
            mask[:, 12:] = 0.0  # High frequency band

        return mask


class StructurePreserver:
    """Utility for computing structure preservation losses."""

    @staticmethod
    def temporal_alignment_loss(
        pred: torch.Tensor, target: torch.Tensor, vocal_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal structure preservation loss.

        Measures correlation between vocal energy curve and accompaniment.

        Args:
            pred: (B, 8, 16, T) Predicted accompaniment
            target: (B, 8, 16, T) Target accompaniment
            vocal_latents: (B, 8, 16, T) Vocal latent vectors

        Returns:
            Scalar loss
        """
        # Energy (L2 norm of each frame)
        vocal_energy = vocal_latents.pow(2).sum(dim=(1, 2))  # (B, T)
        pred_energy = pred.pow(2).sum(dim=(1, 2))  # (B, T)
        target_energy = target.pow(2).sum(dim=(1, 2))  # (B, T)

        # Normalize
        vocal_energy = F.normalize(vocal_energy, dim=1)
        pred_energy = F.normalize(pred_energy, dim=1)
        target_energy = F.normalize(target_energy, dim=1)

        # Correlation loss (align pred-vocal correlation with target-vocal correlation)
        pred_corr = (vocal_energy * pred_energy).sum(dim=1).mean()
        target_corr = (vocal_energy * target_energy).sum(dim=1).mean()

        return F.mse_loss(pred_corr, target_corr)
