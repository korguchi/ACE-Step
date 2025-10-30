"""Control encoder for ControlNet-based accompaniment generation."""

import torch
import torch.nn as nn
from diffusers.models.attention import Attention
from diffusers.models.normalization import RMSNorm


class ControlEncoder(nn.Module):
    """
    Vocal latent → Control signal converter.

    Lightweight Transformer (~50% size of ACEStepTransformer)
    """

    def __init__(
        self,
        in_channels: int = 8,
        patch_height: int = 16,
        dim: int = 1536,
        depth: int = 6,
        num_heads: int = 12,
        out_dim: int = 2560,
    ):
        """
        Initialize control encoder.

        Args:
            in_channels: Input channels (DCAE=8)
            patch_height: Patch height (DCAE=16)
            dim: Internal dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            out_dim: Output dimension (ACEStep inner_dim=2560)
        """
        super().__init__()

        self.in_channels = in_channels
        self.patch_height = patch_height
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_dim = out_dim

        # Patch embedding
        self.patch_embed = nn.Conv1d(
            in_channels * patch_height, dim, kernel_size=1, bias=True
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [ControlBlock(dim, num_heads) for _ in range(depth)]
        )

        # Time embedding projection
        self.time_proj = nn.Linear(512, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, out_dim)
        self.out_norm = RMSNorm(out_dim)

    def forward(
        self, vocal_latents: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            vocal_latents: (B, 8, 16, T) Vocal latent vectors
            timesteps: (B,) Timesteps

        Returns:
            control_signals: (B, T, 2560) Control signals
        """
        B, C, H, W = vocal_latents.shape

        # Flatten patch dimension: (B, C, H, W) → (B, C*H, W)
        x = vocal_latents.flatten(1, 2)

        # Patch embed: (B, C*H, W) → (B, dim, W) → (B, W, dim)
        hidden = self.patch_embed(x).transpose(1, 2)

        # Time embedding
        time_emb = self.get_time_embedding(timesteps, hidden.shape[1])  # (B, W, dim)

        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, time_emb)

        # Output projection
        control = self.out_proj(hidden)  # (B, W, out_dim)
        control = self.out_norm(control)

        return control

    def get_time_embedding(self, timesteps: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Create time embedding.

        Args:
            timesteps: (B,) Timesteps
            seq_len: Sequence length

        Returns:
            time_emb: (B, seq_len, dim)
        """
        device = timesteps.device
        half_dim = 256

        # Sinusoidal embedding
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, 512)

        # Expand to sequence: (B, 512) → (B, seq_len, 512)
        emb = emb[:, None, :].expand(-1, seq_len, -1)

        # Project to dim: (B, seq_len, 512) → (B, seq_len, dim)
        return self.time_proj(emb)


class ControlBlock(nn.Module):
    """Single transformer block for ControlEncoder."""

    def __init__(self, dim: int, num_heads: int):
        """
        Initialize control block.

        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=False,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, dim) Input hidden states
            time_emb: (B, T, dim) Time embedding

        Returns:
            output: (B, T, dim) Output hidden states
        """
        # Self-attention with time embedding
        residual = x
        x = self.norm1(x + time_emb)
        x = self.attn(x)
        x = residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
