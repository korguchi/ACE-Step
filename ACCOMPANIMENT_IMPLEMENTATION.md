# ACE-Step 伴奏生成システム実装方針書

**プロジェクト目標**: ボーカルトラックをリファレンス入力として、伴奏を生成するLoRAファインチューニングシステムの実装

**最終更新**: 2025-10-30

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ設計](#アーキテクチャ設計)
3. [実装フェーズ](#実装フェーズ)
4. [設定ファイル仕様](#設定ファイル仕様)
5. [データセット要件](#データセット要件)
6. [LoRA層選択戦略](#lora層選択戦略)
7. [Optuna探索設定](#optuna探索設定)
8. [コードインターフェース設計](#コードインターフェース設計)
9. [使用例](#使用例)
10. [評価プロトコル](#評価プロトコル)
11. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 背景
ACE-Stepは既にリファレンスオーディオを初期ノイズとして利用する機能（Image2Image方式）を持つが、ボーカルから伴奏を生成する特化型システムは未実装。本プロジェクトでは以下の要件を満たすシステムを構築する:

1. **複数の実装方式**
   - **Image2Image方式**: ボーカル潜在ベクトルをノイズに混合（`noisy_image = gt_latents * (1 - σ) + noise * σ`）
   - **ControlNet方式**: ボーカル潜在ベクトルをLoRAの条件付け信号として入力

2. **設定のバリエーション**
   - Image2Image: リファレンス強度（0.1, 0.3, 0.5, 0.7）
   - ControlNet: 条件付け層の選択（初期層/中間層/特定embedder）

3. **出力モード**
   - **伴奏のみ**: 分離済みインストトラックを教師データとして学習
   - **ミックス**: ボーカル+伴奏の完全ミックスを生成

4. **実験管理**
   - 設定ファイルによる全パラメータ管理
   - Optunaによる自動ハイパーパラメータ探索
   - TensorBoardロギングと可視化

### 技術スタック
- **ベースモデル**: ACE-Step Diffusion Transformer
- **ファインチューニング**: LoRA (PEFT library)
- **オーディオVAE**: DCAE (Deep Compression AutoEncoder)
- **スケジューラ**: Flow Matching Euler Discrete
- **実験管理**: Optuna, TensorBoard
- **環境**: uv (Python管理), ruff (Lint), ty (型チェック)

---

## アーキテクチャ設計

### システム全体図

```
┌─────────────────────────────────────────────────────────┐
│ データセット層                                             │
│  [vocal.mp3] + [inst.mp3/mix.mp3] + [prompt.txt]       │
└─────────────────┬───────────────────────────────────────┘
                  │ AccompanimentDataset
                  ↓
┌─────────────────────────────────────────────────────────┐
│ DCAE Encoder                                             │
│  Vocal: (2, T) → (8, 16, T/8)                          │
│  Target: (2, T) → (8, 16, T/8)                         │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ↓                   ↓
┌───────────────┐   ┌───────────────────┐
│ Image2Image   │   │ ControlNet        │
│ 方式          │   │ 方式              │
│               │   │                   │
│ Vocal Latent  │   │ Control Encoder   │
│     ↓         │   │      ↓            │
│ Mix with      │   │ Control Signal    │
│ Noise         │   │      ↓            │
│     ↓         │   │ Inject to LoRA    │
└───────┬───────┘   └─────────┬─────────┘
        │                     │
        └─────────┬───────────┘
                  ↓
        ┌─────────────────────┐
        │ ACEStepTransformer  │
        │ + LoRA Adapters     │
        └─────────┬───────────┘
                  ↓
        ┌─────────────────────┐
        │ DCAE Decoder        │
        │ (8, 16, T/8) →      │
        │ (2, T) audio        │
        └─────────────────────┘
```

### 主要コンポーネント

#### 1. 設定管理 (Pydantic Schemas)
- **LoRAConfig**: rank, alpha, target_modules, use_rslora
- **AccompanimentConfig**: 方式選択、強度、出力モード
- **TrainingConfig**: 学習率、ステップ数、SSL係数
- **OptunaConfig**: 探索空間、trial数、pruner設定

#### 2. データセット拡張
```python
class AccompanimentDataset(Text2MusicDataset):
    """
    ボーカル（入力）+ インスト/ミックス（教師）のペア読み込み

    期待ファイル構造:
    - {key}_vocal.mp3: ボーカルトラック
    - {key}_inst.mp3: インストトラック（伴奏のみモード）
    - {key}_mix.mp3: ミックストラック（ミックスモード）
    - {key}_prompt.txt: タグ（"electronic, piano, 120bpm"）
    """
```

#### 3. Image2Image実装
既存の`add_latents_noise`機構を拡張:
```python
# trainer_accompaniment.py
def run_step(batch):
    vocal_latents = batch["reference_latents"]  # ボーカルのlatent
    target_latents = batch["target_latents"]    # インスト or ミックスのlatent

    # リファレンス強度に応じたノイズ混合
    sigma = get_sigmas(timesteps)
    noisy_input = vocal_latents * (1 - sigma) + noise * sigma

    # Diffusion forward
    model_pred = transformer(noisy_input, timesteps, conditions)

    # Loss (target_latentsに向けて学習)
    loss = F.mse_loss(model_pred, target_latents)
```

#### 4. ControlNet実装
新規コントロールエンコーダを追加:
```python
# acestep/models/control_encoder.py
class ControlEncoder(nn.Module):
    """
    ボーカルlatentをコントロール信号に変換

    アーキテクチャ:
    - 入力: (B, 8, 16, T) vocal latents
    - 軽量Transformer (6層, dim=1536)
    - 出力: (B, T, 2560) control signals
    """

    def forward(self, vocal_latents, timesteps):
        # Patch embed
        hidden = self.patch_embed(vocal_latents)

        # Transformer encoding
        for block in self.blocks:
            hidden = block(hidden, timesteps)

        return hidden  # (B, T, 2560)
```

トレーナーでの統合:
```python
# trainer_accompaniment.py (ControlNet mode)
def run_step(batch):
    vocal_latents = batch["reference_latents"]
    target_latents = batch["target_latents"]

    # コントロール信号生成
    control_signals = control_encoder(vocal_latents, timesteps)

    # ノイズから開始（vocal_latentsは混ぜない）
    noisy_input = noise * sigma

    # Diffusion forward with control injection
    model_pred = transformer(
        noisy_input,
        timesteps,
        conditions,
        block_controlnet_hidden_states=control_signals,
        controlnet_scale=cfg.controlnet_scale
    )

    loss = F.mse_loss(model_pred, target_latents)
```

---

## 実装フェーズ

### Phase 1: 基盤インフラ整備

**目標**: 設定システム、データセット、ユーティリティの実装

#### 1.1 Pydantic設定スキーマ (`acestep/config_schemas.py`)

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List

class LoRAConfig(BaseModel):
    """LoRA設定"""
    r: int = Field(256, ge=1, le=1024, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha")
    target_modules: List[str] = Field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"],
        description="LoRAを適用するモジュール名"
    )
    use_rslora: bool = Field(True, description="Rank-Stabilized LoRA")
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("target_modules")
    @classmethod
    def validate_modules(cls, v):
        valid = {
            "to_q", "to_k", "to_v", "to_out.0",
            "speaker_embedder", "genre_embedder", "lyric_proj",
            "linear_q", "linear_k", "linear_v"
        }
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"無効なモジュール: {invalid}")
        return v

class AccompanimentConfig(BaseModel):
    """伴奏生成設定"""
    mode: Literal["image2image", "controlnet"] = "image2image"
    output_type: Literal["accompaniment", "mix"] = "accompaniment"

    # Image2Image設定
    reference_strength: float = Field(
        0.5, ge=0.0, le=1.0,
        description="リファレンス強度 (0=無視, 1=完全コピー)"
    )

    # ControlNet設定
    control_encoder_depth: int = Field(6, ge=2, le=12)
    control_encoder_dim: int = Field(1536, ge=512, le=2560)
    controlnet_scale: float = Field(1.0, ge=0.0, le=2.0)
    injection_layers: List[int] = Field(
        default_factory=lambda: [0, 4, 8, 12],
        description="コントロール信号を注入する層"
    )

class TrainingConfig(BaseModel):
    """学習設定"""
    # LoRA & Accompaniment
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    accompaniment: AccompanimentConfig = Field(default_factory=AccompanimentConfig)

    # データセット
    dataset_path: str = "./data/accompaniment_dataset"
    max_duration: float = Field(240.0, gt=0.0)

    # 学習ハイパーパラメータ
    learning_rate: float = Field(1e-4, gt=0.0)
    max_steps: int = Field(200000, ge=1)
    warmup_steps: int = Field(100, ge=0)
    weight_decay: float = Field(1e-2, ge=0.0)
    batch_size: int = Field(1, ge=1)
    accumulate_grad_batches: int = Field(1, ge=1)

    # SSL制約
    ssl_coeff: float = Field(1.0, ge=0.0)
    ssl_depths: List[int] = Field([8, 8])

    # チェックポイント
    checkpoint_every_n_steps: int = Field(2000, ge=1)
    exp_name: str = "accompaniment_lora"
    logger_dir: str = "./exps/logs/"

    # Flow matching
    shift: float = Field(3.0, gt=0.0)

class OptunaConfig(BaseModel):
    """Optuna探索設定"""
    n_trials: int = Field(50, ge=1)
    timeout: Optional[int] = Field(None, description="秒単位のタイムアウト")

    # Pruner
    pruner_type: Literal["median", "successive_halving", "hyperband"] = "median"
    pruner_n_startup_trials: int = Field(5, ge=0)
    pruner_n_warmup_steps: int = Field(100, ge=0)

    # 探索空間
    search_space: dict = Field(
        default_factory=lambda: {
            "reference_strength": {"type": "float", "low": 0.1, "high": 0.9},
            "lora_rank": {"type": "int", "low": 64, "high": 512, "step": 64},
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3}
        }
    )
```

#### 1.2 拡張データセット (`acestep/accompaniment_dataset.py`)

```python
import os
import torch
from typing import Optional, Literal
from .text2music_dataset import Text2MusicDataset

class AccompanimentDataset(Text2MusicDataset):
    """
    伴奏生成用データセット

    期待ファイル構造:
        data/
        ├── song001_vocal.mp3       # ボーカルトラック（入力）
        ├── song001_inst.mp3        # インストトラック（教師: accompanimentモード）
        ├── song001_mix.mp3         # ミックストラック（教師: mixモード）
        ├── song001_prompt.txt      # タグ: "electronic, piano, 120bpm"
        └── song001_lyrics.txt      # "[instrumental]" 固定
    """

    def __init__(
        self,
        output_type: Literal["accompaniment", "mix"] = "accompaniment",
        reference_cache_size: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_type = output_type
        self.ref_cache = {}
        self.cache_size = reference_cache_size

    def get_reference_audio(self, key: str) -> Optional[torch.Tensor]:
        """ボーカルトラックを取得（キャッシュ付き）"""
        if key in self.ref_cache:
            return self.ref_cache[key]

        vocal_path = os.path.join(self.data_dir, f"{key}_vocal.mp3")
        if not os.path.exists(vocal_path):
            return None

        vocal_audio = self.get_audio({"filename": vocal_path})

        if vocal_audio is not None and len(self.ref_cache) < self.cache_size:
            self.ref_cache[key] = vocal_audio

        return vocal_audio

    def get_target_audio(self, key: str) -> Optional[torch.Tensor]:
        """教師データ（インスト or ミックス）を取得"""
        suffix = "_inst.mp3" if self.output_type == "accompaniment" else "_mix.mp3"
        target_path = os.path.join(self.data_dir, f"{key}{suffix}")

        if not os.path.exists(target_path):
            return None

        return self.get_audio({"filename": target_path})

    def process(self, item):
        """データ処理（親クラスから拡張）"""
        key = item["keys"]

        # ボーカル取得
        vocal_audio = self.get_reference_audio(key)
        if vocal_audio is None:
            return []

        # ターゲット取得
        target_audio = self.get_target_audio(key)
        if target_audio is None:
            return []

        # プロンプト処理
        tags = item.get("tags", [])
        prompt = ", ".join(tags) if tags else "instrumental"

        # 歌詞は固定
        lyrics = "[instrumental]"

        # Lyric tokenization
        lyric_token_ids, lyric_masks, _ = self.process_lyric(lyrics)

        return [{
            "key": key,
            "reference_wav": vocal_audio,     # ボーカル
            "target_wav": target_audio,       # インスト or ミックス
            "prompt": prompt,
            "lyric_token_ids": lyric_token_ids,
            "lyric_masks": lyric_masks,
            "speaker_emb": torch.zeros(512)   # 未使用
        }]

    @staticmethod
    def collate_fn(batch):
        """バッチ処理"""
        max_len = max(x["reference_wav"].shape[-1] for x in batch)

        # Pad audio
        reference_wavs = []
        target_wavs = []
        wav_lengths = []

        for item in batch:
            ref_len = item["reference_wav"].shape[-1]
            target_len = item["target_wav"].shape[-1]

            # 長さを揃える（短い方に合わせる）
            min_len = min(ref_len, target_len)

            ref_wav = item["reference_wav"][:, :min_len]
            target_wav = item["target_wav"][:, :min_len]

            # Pad to max_len
            pad_len = max_len - min_len
            if pad_len > 0:
                ref_wav = torch.nn.functional.pad(ref_wav, (0, pad_len))
                target_wav = torch.nn.functional.pad(target_wav, (0, pad_len))

            reference_wavs.append(ref_wav)
            target_wavs.append(target_wav)
            wav_lengths.append(min_len)

        # Stack
        reference_wavs = torch.stack(reference_wavs)
        target_wavs = torch.stack(target_wavs)
        wav_lengths = torch.tensor(wav_lengths)

        # Pad lyrics
        max_lyric_len = max(x["lyric_token_ids"].shape[0] for x in batch)
        lyric_token_ids = []
        lyric_masks = []

        for item in batch:
            tokens = item["lyric_token_ids"]
            masks = item["lyric_masks"]

            pad_len = max_lyric_len - len(tokens)
            if pad_len > 0:
                tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
                masks = torch.cat([masks, torch.zeros(pad_len, dtype=torch.bool)])

            lyric_token_ids.append(tokens)
            lyric_masks.append(masks)

        return {
            "keys": [x["key"] for x in batch],
            "reference_wavs": reference_wavs,
            "target_wavs": target_wavs,
            "wav_lengths": wav_lengths,
            "prompts": [x["prompt"] for x in batch],
            "lyric_token_ids": torch.stack(lyric_token_ids),
            "lyric_masks": torch.stack(lyric_masks),
            "speaker_embs": torch.stack([x["speaker_emb"] for x in batch])
        }
```

#### 1.3 オーディオユーティリティ (`acestep/audio_utils.py`)

```python
import torch
import torch.nn.functional as F

class AudioMixer:
    """潜在空間でのオーディオブレンド"""

    @staticmethod
    def blend_latents(
        generated: torch.Tensor,
        reference: torch.Tensor,
        mix_ratio: float = 0.8,
        frequency_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成潜在ベクトルとリファレンスをブレンド

        Args:
            generated: (B, 8, 16, T) 生成された伴奏
            reference: (B, 8, 16, T) リファレンス伴奏
            mix_ratio: 0=full ref, 1=full generated
            frequency_mask: (8, 16) 周波数帯域マスク（Noneなら均一混合）

        Returns:
            (B, 8, 16, T) ブレンド後の潜在ベクトル
        """
        if frequency_mask is not None:
            mask = frequency_mask[None, :, :, None]  # (1, 8, 16, 1)
            return (
                generated * mask * mix_ratio +
                reference * mask * (1 - mix_ratio) +
                reference * (1 - mask)
            )
        else:
            return generated * mix_ratio + reference * (1 - mix_ratio)

    @staticmethod
    def create_frequency_mask(
        preserve_low: bool = True,
        preserve_high: bool = False
    ) -> torch.Tensor:
        """
        周波数選択的マスク生成

        Args:
            preserve_low: 低周波（ベース・キック）をリファレンスから保持
            preserve_high: 高周波をリファレンスから保持

        Returns:
            (8, 16) マスク（1=生成を使用, 0=リファレンスを保持）
        """
        mask = torch.ones(8, 16)

        if preserve_low:
            mask[:, :4] = 0.0  # 低周波帯域

        if preserve_high:
            mask[:, 12:] = 0.0  # 高周波帯域

        return mask

class StructurePreserver:
    """構造保存損失の計算"""

    @staticmethod
    def temporal_alignment_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        vocal_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        時間的構造の保存損失

        ボーカルのエネルギーカーブと伴奏の相関を計算

        Args:
            pred: (B, 8, 16, T) 予測伴奏
            target: (B, 8, 16, T) 教師伴奏
            vocal_latents: (B, 8, 16, T) ボーカル潜在ベクトル

        Returns:
            Scalar loss
        """
        # エネルギー（各フレームのL2ノルム）
        vocal_energy = vocal_latents.pow(2).sum(dim=(1, 2))  # (B, T)
        pred_energy = pred.pow(2).sum(dim=(1, 2))            # (B, T)
        target_energy = target.pow(2).sum(dim=(1, 2))        # (B, T)

        # 正規化
        vocal_energy = F.normalize(vocal_energy, dim=1)
        pred_energy = F.normalize(pred_energy, dim=1)
        target_energy = F.normalize(target_energy, dim=1)

        # 相関損失（ボーカルとの相関を教師に近づける）
        pred_corr = (vocal_energy * pred_energy).sum(dim=1).mean()
        target_corr = (vocal_energy * target_energy).sum(dim=1).mean()

        return F.mse_loss(pred_corr, target_corr)
```

#### 1.4 設定ファイルテンプレート作成

**ファイル構成**:
```
config/accompaniment/
├── image2image_base.json
├── controlnet_base.json
└── optuna_search.json
```

---

### Phase 2: Image2Image方式実装

**目標**: リファレンス強度によるノイズ混合と学習

#### 2.1 トレーナー実装 (`trainer_accompaniment.py`)

```python
import torch
import torch.nn.functional as F
from trainer import Pipeline as BasePipeline
from acestep.config_schemas import TrainingConfig
from acestep.accompaniment_dataset import AccompanimentDataset

class AccompanimentTrainer(BasePipeline):
    """伴奏生成トレーナー（Image2Image + ControlNet対応）"""

    def __init__(self, config: TrainingConfig, **kwargs):
        self.config = config

        # 親クラス初期化
        super().__init__(
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            **kwargs
        )

        # モード判定
        self.use_image2image = (config.accompaniment.mode == "image2image")

        # ControlNetモードならエンコーダ初期化
        if not self.use_image2image:
            from acestep.models.control_encoder import ControlEncoder
            self.control_encoder = ControlEncoder(
                depth=config.accompaniment.control_encoder_depth,
                dim=config.accompaniment.control_encoder_dim
            )

    def setup(self, stage=None):
        """データセットセットアップ"""
        self.train_dataset = AccompanimentDataset(
            dataset_path=self.config.dataset_path,
            output_type=self.config.accompaniment.output_type,
            max_duration=self.config.max_duration
        )

    def preprocess(self, batch, train=True):
        """前処理: オーディオ → 潜在ベクトル"""
        # リファレンス（ボーカル）エンコード
        ref_latents, ref_lengths = self.dcae.encode(
            batch["reference_wavs"],
            batch["wav_lengths"],
            sr=48000
        )

        # ターゲット（インスト or ミックス）エンコード
        target_latents, target_lengths = self.dcae.encode(
            batch["target_wavs"],
            batch["wav_lengths"],
            sr=48000
        )

        # テキスト・歌詞エンコード（親クラスのロジック）
        genre_embeds = self.umt5.encode(batch["prompts"])
        speaker_embeds = batch["speaker_embs"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_masks = batch["lyric_masks"]

        return (
            ref_latents, target_latents, ref_lengths,
            genre_embeds, speaker_embeds, lyric_token_ids, lyric_masks
        )

    def run_step(self, batch, batch_idx):
        """学習ステップ"""
        # 前処理
        (ref_latents, target_latents, lengths,
         genre_embeds, speaker_embeds, lyric_tokens, lyric_masks) = self.preprocess(batch)

        # デバイス配置
        device = self.device
        ref_latents = ref_latents.to(device)
        target_latents = target_latents.to(device)

        # ノイズ生成
        noise = torch.randn_like(target_latents)

        # タイムステップサンプリング
        bsz = target_latents.shape[0]
        timesteps = self.sample_timesteps(bsz, device)

        # Sigma計算
        sigmas = self.get_sd3_sigmas(timesteps, device, target_latents.ndim, target_latents.dtype)

        # ノイズ付加
        if self.use_image2image:
            # Image2Image: リファレンスをノイズに混合
            strength = self.config.accompaniment.reference_strength
            noisy_input = ref_latents * (1.0 - sigmas) * strength + noise * sigmas
        else:
            # ControlNet: 純粋なノイズから開始
            noisy_input = noise * sigmas + target_latents * (1.0 - sigmas)

        # Transformer forward
        if self.use_image2image:
            # 通常のforward
            model_pred = self.transformers(
                noisy_input,
                timesteps,
                encoder_text_hidden_states=genre_embeds,
                speaker_embeds=speaker_embeds,
                lyric_token_idx=lyric_tokens,
                lyric_attention_mask=lyric_masks
            ).sample
        else:
            # ControlNet: コントロール信号注入
            control_signals = self.control_encoder(ref_latents, timesteps)

            model_pred = self.transformers(
                noisy_input,
                timesteps,
                encoder_text_hidden_states=genre_embeds,
                speaker_embeds=speaker_embeds,
                lyric_token_idx=lyric_tokens,
                lyric_attention_mask=lyric_masks,
                block_controlnet_hidden_states=control_signals,
                controlnet_scale=self.config.accompaniment.controlnet_scale
            ).sample

        # Preconditioning (Flow matching)
        model_pred = model_pred * (-sigmas) + noisy_input

        # 損失計算
        loss = F.mse_loss(model_pred, target_latents, reduction="none")

        # マスキング（パディング部分を無視）
        mask = self.create_length_mask(lengths, target_latents.shape[-1])
        mask = mask[:, None, None, :]  # (B, 1, 1, T)
        loss = (loss * mask).sum() / mask.sum()

        # SSL損失（オプション）
        if self.config.ssl_coeff > 0:
            ssl_loss = self.compute_ssl_loss(model_pred, target_latents)
            loss = loss + self.config.ssl_coeff * ssl_loss

        # ロギング
        self.log("train/denoising_loss", loss, on_step=True, prog_bar=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"])

        return loss

    def create_length_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """長さマスク生成"""
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        return mask.float()

    def on_save_checkpoint(self, checkpoint):
        """LoRAアダプタのみ保存"""
        log_dir = self.logger.log_dir
        epoch = self.current_epoch
        step = self.global_step

        checkpoint_name = f"epoch={epoch}-step={step}_lora"
        checkpoint_dir = os.path.join(log_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # LoRA保存
        self.transformers.save_lora_adapter(
            checkpoint_dir,
            adapter_name=self.config.exp_name
        )

        # ControlNetエンコーダ保存（該当モードのみ）
        if not self.use_image2image:
            torch.save(
                self.control_encoder.state_dict(),
                os.path.join(checkpoint_dir, "control_encoder.pth")
            )

        return {}
```

#### 2.2 学習スクリプト (`scripts/train_image2image.py`)

```python
#!/usr/bin/env python3
"""Image2Image方式の伴奏生成トレーニング"""

import argparse
import json
from pathlib import Path
from acestep.config_schemas import TrainingConfig
from trainer_accompaniment import AccompanimentTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="設定ファイルパス")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="事前学習済みモデルディレクトリ")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    args = parser.parse_args()

    # 設定読み込み
    with open(args.config) as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)

    # トレーナー初期化
    model = AccompanimentTrainer(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        lora_config_path=None  # 設定から読み込む
    )

    # PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_top_k=-1
    )

    logger = TensorBoardLogger(
        save_dir=config.logger_dir,
        name=config.exp_name
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        max_steps=config.max_steps,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.5,
        accumulate_grad_batches=config.accumulate_grad_batches
    )

    # 学習開始
    trainer.fit(model)

if __name__ == "__main__":
    main()
```

---

### Phase 3: ControlNet方式実装

#### 3.1 コントロールエンコーダ (`acestep/models/control_encoder.py`)

```python
import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.attention import Attention
from diffusers.models.normalization import RMSNorm

class ControlEncoder(nn.Module):
    """
    ボーカル潜在ベクトルをコントロール信号に変換

    ACEStepTransformerの軽量版（約50%サイズ）
    """

    def __init__(
        self,
        in_channels: int = 8,
        patch_height: int = 16,
        dim: int = 1536,
        depth: int = 6,
        num_heads: int = 12,
        out_dim: int = 2560
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv1d(
            in_channels * patch_height,
            dim,
            kernel_size=1
        )

        # RoPE position embedding
        self.rope_theta = 1_000_000

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ControlBlock(dim, num_heads)
            for _ in range(depth)
        ])

        # Output projection
        self.out_proj = nn.Linear(dim, out_dim)
        self.out_norm = RMSNorm(out_dim)

    def forward(
        self,
        vocal_latents: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vocal_latents: (B, 8, 16, T) ボーカル潜在ベクトル
            timesteps: (B,) タイムステップ

        Returns:
            control_signals: (B, T, 2560) コントロール信号
        """
        B, C, H, W = vocal_latents.shape

        # Flatten patch dimension
        x = vocal_latents.flatten(1, 2)  # (B, C*H, W)

        # Patch embed
        hidden = self.patch_embed(x)  # (B, dim, W)
        hidden = hidden.transpose(1, 2)  # (B, W, dim)

        # Time embedding (simplified)
        time_emb = self.get_time_embedding(timesteps, hidden.shape[1])  # (B, W, dim)

        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, time_emb)

        # Output projection
        control = self.out_proj(hidden)  # (B, W, 2560)
        control = self.out_norm(control)

        return control

    def get_time_embedding(self, timesteps: torch.Tensor, seq_len: int) -> torch.Tensor:
        """簡易時間埋め込み"""
        # Sinusoidal embedding
        device = timesteps.device
        half_dim = 256
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, 512)

        # Expand to sequence
        emb = emb[:, None, :].expand(-1, seq_len, -1)  # (B, W, 512)

        # Project to dim
        if not hasattr(self, "time_proj"):
            self.time_proj = nn.Linear(512, self.patch_embed.out_channels).to(device)

        return self.time_proj(emb)

class ControlBlock(nn.Module):
    """ControlEncoderの1ブロック"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=False
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Self-attention
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
```

#### 3.2 条件付け層選択戦略

**推奨設定**:

| 戦略 | 対象層 | 目的 | 推奨ランク |
|------|-------|------|----------|
| **構造重視** | Transformer初期層 (0-7) + `lyric_proj` | タイミング・リズム同期 | 256 |
| **スタイル重視** | Transformer中間層 (8-15) + `genre_embedder` | ジャンル・楽器編成 | 128 |
| **バランス** | 全Attention層 + `genre_embedder` + `lyric_proj` | 総合的制御 | 192 |
| **最小限** | `genre_embedder`のみ | ジャンル上書き | 64 |

**実装例（config）**:
```json
{
    "lora": {
        "r": 256,
        "lora_alpha": 32,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "lyric_proj"
        ]
    },
    "accompaniment": {
        "mode": "controlnet",
        "control_encoder_depth": 6,
        "controlnet_scale": 1.0,
        "injection_layers": [0, 2, 4, 6]
    }
}
```

---

### Phase 4: Optuna統合

#### 4.1 探索スクリプト (`scripts/optuna_search.py`)

```python
#!/usr/bin/env python3
"""Optunaによるハイパーパラメータ探索"""

import optuna
from optuna.pruners import MedianPruner
import torch
from pathlib import Path
import json
from acestep.config_schemas import TrainingConfig, OptunaConfig
from trainer_accompaniment import AccompanimentTrainer
from pytorch_lightning import Trainer

def objective(trial: optuna.Trial, base_config: TrainingConfig, optuna_cfg: OptunaConfig):
    """Optuna objective関数"""

    # ハイパーパラメータサンプリング
    if base_config.accompaniment.mode == "image2image":
        # Image2Image探索空間
        reference_strength = trial.suggest_float("reference_strength", 0.1, 0.9)
        lora_rank = trial.suggest_int("lora_rank", 64, 512, step=64)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    else:
        # ControlNet探索空間
        control_encoder_depth = trial.suggest_int("control_encoder_depth", 4, 12, step=2)
        controlnet_scale = trial.suggest_float("controlnet_scale", 0.5, 2.0)
        lora_rank = trial.suggest_int("lora_rank", 64, 512, step=64)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)

    # 設定更新
    trial_config = base_config.model_copy(deep=True)
    trial_config.lora.r = lora_rank
    trial_config.learning_rate = learning_rate

    if base_config.accompaniment.mode == "image2image":
        trial_config.accompaniment.reference_strength = reference_strength
    else:
        trial_config.accompaniment.control_encoder_depth = control_encoder_depth
        trial_config.accompaniment.controlnet_scale = controlnet_scale

    # 短縮学習（1000ステップ）
    trial_config.max_steps = 1000
    trial_config.checkpoint_every_n_steps = 10000  # 保存しない

    # トレーナー
    model = AccompanimentTrainer(
        config=trial_config,
        checkpoint_dir="./checkpoints"
    )

    # Pruning callback
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="train/denoising_loss"
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=trial_config.max_steps,
        callbacks=[pruning_callback],
        logger=False,  # ロギング無効
        enable_checkpointing=False
    )

    # 学習
    trainer.fit(model)

    # 最終損失を返す
    return trainer.callback_metrics["train/denoising_loss"].item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--optuna_config", type=str, required=True)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--study_name", type=str, default="accompaniment_search")
    args = parser.parse_args()

    # 設定読み込み
    with open(args.config) as f:
        config_dict = json.load(f)
    base_config = TrainingConfig(**config_dict)

    with open(args.optuna_config) as f:
        optuna_dict = json.load(f)
    optuna_cfg = OptunaConfig(**optuna_dict)

    # Pruner設定
    if optuna_cfg.pruner_type == "median":
        pruner = MedianPruner(
            n_startup_trials=optuna_cfg.pruner_n_startup_trials,
            n_warmup_steps=optuna_cfg.pruner_n_warmup_steps
        )
    else:
        pruner = None

    # Study作成
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )

    # 乱数固定
    torch.manual_seed(42)

    # 最適化
    study.optimize(
        lambda trial: objective(trial, base_config, optuna_cfg),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout
    )

    # 結果出力
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # ベスト設定保存
    best_config = base_config.model_copy(deep=True)
    best_config.lora.r = study.best_params["lora_rank"]
    best_config.learning_rate = study.best_params["learning_rate"]

    if base_config.accompaniment.mode == "image2image":
        best_config.accompaniment.reference_strength = study.best_params["reference_strength"]
    else:
        best_config.accompaniment.control_encoder_depth = study.best_params["control_encoder_depth"]
        best_config.accompaniment.controlnet_scale = study.best_params["controlnet_scale"]

    output_path = Path(args.config).parent / f"{args.study_name}_best.json"
    with open(output_path, "w") as f:
        json.dump(best_config.model_dump(), f, indent=2)

    print(f"\nBest config saved to: {output_path}")

if __name__ == "__main__":
    main()
```

---

### Phase 5: 推論・評価

#### 5.1 推論スクリプト (`scripts/infer_accompaniment.py`)

```python
#!/usr/bin/env python3
"""伴奏生成推論"""

import argparse
import torch
from pathlib import Path
from acestep.pipeline_ace_step import ACEStepPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal_path", type=str, required=True, help="ボーカルトラックパス")
    parser.add_argument("--prompt", type=str, required=True, help="プロンプト（タグ）")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRAチェックポイントパス")
    parser.add_argument("--lora_weight", type=float, default=0.8, help="LoRA重み")
    parser.add_argument("--ref_strength", type=float, default=0.5, help="リファレンス強度（Image2Imageのみ）")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--guidance_scale", type=float, default=15.0)
    parser.add_argument("--infer_steps", type=int, default=60)
    args = parser.parse_args()

    # パイプライン初期化
    pipeline = ACEStepPipeline(
        checkpoint_dir="./checkpoints",
        device="cuda"
    )

    # LoRA読み込み
    pipeline.load_lora(args.lora_path, args.lora_weight)

    # 推論
    output_paths = pipeline.text2music_diffusion_process(
        prompt=args.prompt,
        lyrics="[instrumental]",
        duration=None,  # ボーカルと同じ長さ
        audio2audio_enable=True,
        ref_audio_input=args.vocal_path,
        ref_audio_strength=args.ref_strength,
        guidance_scale=args.guidance_scale,
        infer_steps=args.infer_steps,
        output_dir=args.output_dir,
        task="audio2audio"
    )

    print(f"Generated accompaniment: {output_paths[0]}")

if __name__ == "__main__":
    main()
```

#### 5.2 比較評価ツール (`scripts/compare_models.py`)

```python
#!/usr/bin/env python3
"""複数LoRAモデルの比較評価"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm
from acestep.pipeline_ace_step import ACEStepPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, required=True, help="テストセットJSONパス")
    parser.add_argument("--lora_configs", type=str, nargs="+", required=True, help="LoRA設定パス（複数可）")
    parser.add_argument("--output_dir", type=str, default="./comparison")
    args = parser.parse_args()

    # テストセット読み込み
    with open(args.test_set) as f:
        test_samples = json.load(f)

    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # パイプライン
    pipeline = ACEStepPipeline(checkpoint_dir="./checkpoints", device="cuda")

    # 各LoRAで推論
    for lora_config_path in args.lora_configs:
        with open(lora_config_path) as f:
            lora_config = json.load(f)

        lora_name = Path(lora_config_path).stem
        lora_output_dir = output_dir / lora_name
        lora_output_dir.mkdir(exist_ok=True)

        # LoRA読み込み
        pipeline.load_lora(lora_config["lora_path"], lora_config["lora_weight"])

        # サンプルごとに推論
        for sample in tqdm(test_samples, desc=f"Processing {lora_name}"):
            output_paths = pipeline.text2music_diffusion_process(
                prompt=sample["prompt"],
                lyrics="[instrumental]",
                audio2audio_enable=True,
                ref_audio_input=sample["vocal_path"],
                ref_audio_strength=lora_config.get("ref_strength", 0.5),
                guidance_scale=15.0,
                infer_steps=60,
                output_dir=str(lora_output_dir),
                filename_prefix=sample["key"]
            )

    # 比較HTML生成
    generate_comparison_html(output_dir, args.lora_configs, test_samples)
    print(f"\nComparison HTML generated: {output_dir / 'comparison.html'}")

def generate_comparison_html(output_dir: Path, lora_configs: list, test_samples: list):
    """比較用HTMLページ生成"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accompaniment Generation Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            audio { width: 300px; }
        </style>
    </head>
    <body>
        <h1>Accompaniment Generation Comparison</h1>
        <table>
            <tr>
                <th>Sample</th>
                <th>Vocal (Input)</th>
    """

    # LoRA列ヘッダ
    for config_path in lora_configs:
        lora_name = Path(config_path).stem
        html += f"<th>{lora_name}</th>"

    html += "</tr>"

    # サンプル行
    for sample in test_samples:
        html += f"""
        <tr>
            <td><b>{sample['key']}</b><br>{sample['prompt']}</td>
            <td><audio controls><source src="{sample['vocal_path']}" type="audio/mpeg"></audio></td>
        """

        for config_path in lora_configs:
            lora_name = Path(config_path).stem
            audio_path = output_dir / lora_name / f"{sample['key']}_generated.mp3"
            html += f'<td><audio controls><source src="{audio_path}" type="audio/mpeg"></audio></td>'

        html += "</tr>"

    html += """
        </table>
    </body>
    </html>
    """

    with open(output_dir / "comparison.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    main()
```

---

## 設定ファイル仕様

### Image2Image基本設定 (`config/accompaniment/image2image_base.json`)

```json
{
    "lora": {
        "r": 128,
        "lora_alpha": 32,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "genre_embedder",
            "lyric_proj"
        ],
        "use_rslora": true,
        "lora_dropout": 0.0
    },
    "accompaniment": {
        "mode": "image2image",
        "output_type": "accompaniment",
        "reference_strength": 0.5
    },
    "dataset_path": "./data/accompaniment_dataset",
    "max_duration": 240.0,
    "learning_rate": 1e-4,
    "max_steps": 200000,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "batch_size": 1,
    "accumulate_grad_batches": 4,
    "ssl_coeff": 1.0,
    "ssl_depths": [8, 8],
    "checkpoint_every_n_steps": 2000,
    "exp_name": "accompaniment_image2image",
    "logger_dir": "./exps/logs/",
    "shift": 3.0
}
```

### ControlNet基本設定 (`config/accompaniment/controlnet_base.json`)

```json
{
    "lora": {
        "r": 256,
        "lora_alpha": 32,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "lyric_proj"
        ],
        "use_rslora": true,
        "lora_dropout": 0.0
    },
    "accompaniment": {
        "mode": "controlnet",
        "output_type": "accompaniment",
        "control_encoder_depth": 6,
        "control_encoder_dim": 1536,
        "controlnet_scale": 1.0,
        "injection_layers": [0, 2, 4, 6, 8]
    },
    "dataset_path": "./data/accompaniment_dataset",
    "max_duration": 240.0,
    "learning_rate": 1e-4,
    "max_steps": 200000,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "batch_size": 1,
    "accumulate_grad_batches": 4,
    "ssl_coeff": 1.0,
    "ssl_depths": [8, 8],
    "checkpoint_every_n_steps": 2000,
    "exp_name": "accompaniment_controlnet",
    "logger_dir": "./exps/logs/",
    "shift": 3.0
}
```

### Optuna探索設定 (`config/accompaniment/optuna_search.json`)

```json
{
    "n_trials": 50,
    "timeout": null,
    "pruner_type": "median",
    "pruner_n_startup_trials": 5,
    "pruner_n_warmup_steps": 100,
    "search_space": {
        "reference_strength": {
            "type": "float",
            "low": 0.1,
            "high": 0.9
        },
        "lora_rank": {
            "type": "int",
            "low": 64,
            "high": 512,
            "step": 64
        },
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-3
        },
        "controlnet_scale": {
            "type": "float",
            "low": 0.5,
            "high": 2.0
        },
        "control_encoder_depth": {
            "type": "int",
            "low": 4,
            "high": 12,
            "step": 2
        }
    }
}
```

---

## データセット要件

### ファイル命名規則

```
data/accompaniment_dataset/
├── song001_vocal.mp3       # ボーカルトラック（必須）
├── song001_inst.mp3        # インストトラック（accompanimentモード用）
├── song001_mix.mp3         # ミックストラック（mixモード用）
├── song001_prompt.txt      # タグ（必須）
└── song001_lyrics.txt      # 歌詞（任意、なければ[instrumental]）
```

### プロンプトフォーマット (`*_prompt.txt`)

```
electronic, piano, synthesizer, 120bpm, energetic, minor key
```

- カンマ区切りのタグ
- ジャンル、楽器、BPM、雰囲気、キーなど
- 順序はランダムシャッフルされる（データ拡張）

### 歌詞フォーマット (`*_lyrics.txt`)

```
[instrumental]
```

- 伴奏生成では固定値を推奨
- 構造タグ（[Verse], [Chorus]等）は使用可能だが、タイミング制御が難しい

### データ前処理推奨

1. **音源分離**: Demucs、UVR等でボーカル・インスト分離
2. **音量正規化**: ラウドネス正規化（-14 LUFS推奨）
3. **トリミング**: 無音部分の削除
4. **サンプリングレート**: 48kHz（ACE-Step標準）
5. **ステレオ**: モノラルは自動変換されるが、ステレオ推奨

---

## LoRA層選択戦略

### 全利用可能モジュール

| カテゴリ | モジュール名 | 説明 | 推奨度 |
|---------|------------|------|-------|
| **Transformer Attention** | `to_q`, `to_k`, `to_v`, `to_out.0` | メイン注意機構 | ⭐⭐⭐ |
| **条件付けEmbedder** | `genre_embedder` | ジャンル埋め込み（768→2560） | ⭐⭐⭐ |
| | `speaker_embedder` | 話者埋め込み（512→2560） | ⭐ |
| | `lyric_proj` | 歌詞投影（1024→2560） | ⭐⭐⭐ |
| **Lyric Encoder** | `linear_q`, `linear_k`, `linear_v` | 歌詞エンコーダ注意 | ⭐⭐ |
| **時間条件** | `timestep_embedder.linear_1/2` | タイムステップMLP | ⭐ |

### 推奨組み合わせ

#### パターンA: 構造重視（Image2Image推奨）
```json
{
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "lyric_proj"
    ],
    "r": 128
}
```
- **目的**: ボーカルのタイミング・リズムに同期
- **効果**: ビート一致、構成保存

#### パターンB: スタイル重視（ControlNet推奨）
```json
{
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "genre_embedder"
    ],
    "r": 256
}
```
- **目的**: ジャンル・楽器編成の柔軟な制御
- **効果**: プロンプトへの高い応答性

#### パターンC: バランス型（汎用）
```json
{
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "genre_embedder",
        "lyric_proj"
    ],
    "r": 192
}
```
- **目的**: 構造とスタイルの両立
- **効果**: 総合的な品質向上

#### パターンD: 最小限（高速実験用）
```json
{
    "target_modules": ["genre_embedder"],
    "r": 64
}
```
- **目的**: 概念検証、高速イテレーション
- **効果**: 学習高速化、メモリ削減

---

## Optuna探索設定

### 探索戦略

#### Image2Image探索空間
```python
{
    "reference_strength": {"type": "float", "low": 0.1, "high": 0.9},
    "lora_rank": {"type": "int", "low": 64, "high": 512, "step": 64},
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "lora_alpha": {"type": "int", "low": 16, "high": 128, "step": 16}
}
```

#### ControlNet探索空間
```python
{
    "control_encoder_depth": {"type": "int", "low": 4, "high": 12, "step": 2},
    "controlnet_scale": {"type": "float", "low": 0.5, "high": 2.0},
    "lora_rank": {"type": "int", "low": 64, "high": 512, "step": 64},
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "control_encoder_dim": {"type": "categorical", "choices": [1024, 1536, 2048]}
}
```

### Pruner推奨設定

```json
{
    "pruner_type": "median",
    "pruner_n_startup_trials": 5,
    "pruner_n_warmup_steps": 100
}
```

- **MedianPruner**: 中央値を下回るtrialを早期終了
- **n_startup_trials**: 最初の5 trialは枝刈りしない（初期化安定化）
- **n_warmup_steps**: 100ステップ経過後に枝刈り開始

---

## コードインターフェース設計

### AccompanimentTrainerクラス

```python
class AccompanimentTrainer(Pipeline):
    """伴奏生成トレーナー"""

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_dir: str,
        lora_config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            config: 学習設定（Pydanticモデル）
            checkpoint_dir: 事前学習済みモデルディレクトリ
            lora_config_path: LoRA設定パス（Noneならconfig.loraを使用）
        """
        pass

    def preprocess(self, batch, train=True):
        """前処理: オーディオ → 潜在ベクトル"""
        pass

    def run_step(self, batch, batch_idx):
        """学習ステップ"""
        pass

    def on_save_checkpoint(self, checkpoint):
        """チェックポイント保存（LoRAのみ）"""
        pass
```

### AccompanimentDatasetクラス

```python
class AccompanimentDataset(Text2MusicDataset):
    """伴奏生成データセット"""

    def __init__(
        self,
        dataset_path: str,
        output_type: Literal["accompaniment", "mix"] = "accompaniment",
        reference_cache_size: int = 1000,
        **kwargs
    ):
        """
        Args:
            dataset_path: データセットルートパス
            output_type: 出力モード（伴奏のみ or ミックス）
            reference_cache_size: リファレンスキャッシュサイズ
        """
        pass

    def get_reference_audio(self, key: str) -> Optional[torch.Tensor]:
        """ボーカルトラック取得"""
        pass

    def get_target_audio(self, key: str) -> Optional[torch.Tensor]:
        """教師データ取得"""
        pass

    def process(self, item) -> List[dict]:
        """データ処理"""
        pass
```

### ControlEncoderクラス

```python
class ControlEncoder(nn.Module):
    """ボーカルlatent → コントロール信号変換"""

    def __init__(
        self,
        in_channels: int = 8,
        patch_height: int = 16,
        dim: int = 1536,
        depth: int = 6,
        num_heads: int = 12,
        out_dim: int = 2560
    ):
        """
        Args:
            in_channels: 入力チャンネル数（DCAE=8）
            patch_height: パッチ高さ（DCAE=16）
            dim: 内部次元
            depth: Transformerブロック数
            num_heads: 注意ヘッド数
            out_dim: 出力次元（ACEStepのinner_dim）
        """
        pass

    def forward(
        self,
        vocal_latents: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vocal_latents: (B, 8, 16, T)
            timesteps: (B,)

        Returns:
            control_signals: (B, T, 2560)
        """
        pass
```

---

## 使用例

### 1. Image2Image学習

```bash
# 1. データセット準備
mkdir -p data/accompaniment_dataset
# ファイル配置: song001_vocal.mp3, song001_inst.mp3, song001_prompt.txt

# 2. 学習実行
uv run python scripts/train_image2image.py \
    --config config/accompaniment/image2image_base.json \
    --checkpoint_dir ./checkpoints/ace_step_pretrained \
    --devices 1 \
    --precision bf16

# 3. TensorBoard監視
tensorboard --logdir exps/logs/
```

### 2. ControlNet学習

```bash
# 1. 設定編集（必要に応じて）
vim config/accompaniment/controlnet_base.json

# 2. 学習実行
uv run python scripts/train_controlnet.py \
    --config config/accompaniment/controlnet_base.json \
    --checkpoint_dir ./checkpoints/ace_step_pretrained \
    --devices 1 \
    --precision bf16
```

### 3. Optuna探索

```bash
# 1. 探索空間設定
vim config/accompaniment/optuna_search.json

# 2. 探索実行
uv run python scripts/optuna_search.py \
    --config config/accompaniment/image2image_base.json \
    --optuna_config config/accompaniment/optuna_search.json \
    --storage sqlite:///optuna_accompaniment.db \
    --study_name image2image_search

# 3. Optuna Dashboard
optuna-dashboard sqlite:///optuna_accompaniment.db
```

### 4. 推論

```bash
# 1. 単一ファイル推論
uv run python scripts/infer_accompaniment.py \
    --vocal_path data/test/vocal.mp3 \
    --prompt "electronic, piano, 120bpm" \
    --lora_path exps/logs/.../checkpoints/epoch=10-step=20000_lora \
    --lora_weight 0.8 \
    --ref_strength 0.5 \
    --output_dir outputs/

# 2. バッチ推論（比較評価）
uv run python scripts/compare_models.py \
    --test_set data/test_set.json \
    --lora_configs \
        config/lora_image2image_0.3.json \
        config/lora_image2image_0.5.json \
        config/lora_controlnet.json \
    --output_dir comparison/
```

### 5. 品質ゲート（コード変更後）

```bash
# 1. Lint & Format
uv run ruff check --fix acestep/ scripts/
uv run ruff format acestep/ scripts/

# 2. 型チェック
uv run ty check acestep/ scripts/

# 3. テスト実行
uv run pytest tests/ -v
```

---

## 評価プロトコル

### 主観評価ガイドライン

#### 評価軸（5段階評価）

| 軸 | 1点 | 3点 | 5点 |
|----|-----|-----|-----|
| **音質** | ノイズ多い、破綻 | 許容範囲 | クリア、プロ品質 |
| **ボーカル一致** | 全く無関係 | 部分的に一致 | 完全同期 |
| **プロンプト反映** | 無視 | 部分的反映 | 完全一致 |
| **音楽性** | 不協和、リズム崩れ | 聴ける | 自然、感動的 |
| **ボーカル漏れ** | 大量混入 | 微量混入 | 完全除去 |

#### 評価シート例（JSON）

```json
{
    "sample_id": "song001",
    "model": "image2image_0.5",
    "scores": {
        "audio_quality": 4,
        "vocal_alignment": 5,
        "prompt_adherence": 3,
        "musicality": 4,
        "vocal_leakage": 4
    },
    "comments": "リズム同期は完璧。ピアノの音色がプロンプトと若干ずれる。",
    "overall": 4.0
}
```

#### A/Bテスト手順

1. **ブラインド比較**: 比較HTMLでモデル名を隠蔽
2. **ランダム順序**: サンプル順序をシャッフル
3. **複数評価者**: 最低3名で評価
4. **統計検定**: Wilcoxon符号順位検定で有意差確認

---

## トラブルシューティング

### よくある問題と対処法

#### 1. Out of Memory (OOM)

**症状**: CUDA out of memory エラー

**対処法**:
```bash
# バッチサイズ削減
"batch_size": 1,
"accumulate_grad_batches": 8  # 実質バッチサイズ=8

# LoRAランク削減
"lora.r": 64

# Gradient checkpointing有効化（trainer.py line 71で既に有効）
```

#### 2. ボーカル漏れ（Vocal Leakage）

**症状**: 生成伴奏にボーカルが混入

**原因**: リファレンス強度が高すぎる

**対処法**:
```json
// Image2Imageの場合
"reference_strength": 0.3  // 0.5 → 0.3に削減

// ControlNetに切り替え（より明示的制御）
"mode": "controlnet"
```

#### 3. プロンプト無視

**症状**: プロンプトと異なる楽器・ジャンル

**原因**: `genre_embedder`がLoRA対象外

**対処法**:
```json
"target_modules": [
    "to_q", "to_k", "to_v", "to_out.0",
    "genre_embedder"  // 追加
]
```

#### 4. リズムずれ

**症状**: ボーカルとビートが合わない

**原因**: `lyric_proj`がLoRA対象外、またはリファレンス強度不足

**対処法**:
```json
// Image2Image: 強度増加
"reference_strength": 0.7

// ControlNet: lyric_proj追加
"target_modules": ["...", "lyric_proj"]
```

#### 5. 学習不安定（Loss発散）

**症状**: 損失が増加し続ける

**対処法**:
```json
// 学習率削減
"learning_rate": 5e-5,

// Gradient clipping強化
"gradient_clip_val": 0.3,

// LoRA alpha削減（安定化）
"lora_alpha": 16
```

#### 6. 音質劣化

**症状**: 生成音声がこもる、ノイジー

**原因**: SSL損失の重みが大きすぎる、またはLoRA過学習

**対処法**:
```json
// SSL係数削減
"ssl_coeff": 0.5,

// 早期停止（2000ステップごとに評価）
"checkpoint_every_n_steps": 2000
```

#### 7. Optuna探索が遅い

**症状**: Trial完了に時間がかかりすぎる

**対処法**:
```python
# 短縮学習ステップ（optuna_search.py内）
trial_config.max_steps = 500  # 1000 → 500

# Pruner活性化
"pruner_n_warmup_steps": 50  # 早期枝刈り
```

---

## 実装チェックリスト

### Phase 1: 基盤
- [ ] `acestep/config_schemas.py` 作成（Pydanticスキーマ）
- [ ] `acestep/accompaniment_dataset.py` 作成
- [ ] `acestep/audio_utils.py` 作成
- [ ] `config/accompaniment/` ディレクトリ作成
- [ ] `config/accompaniment/image2image_base.json` 作成
- [ ] `config/accompaniment/controlnet_base.json` 作成
- [ ] `config/accompaniment/optuna_search.json` 作成

### Phase 2: Image2Image
- [ ] `trainer_accompaniment.py` 作成（Image2Imageモード）
- [ ] `scripts/train_image2image.py` 作成
- [ ] テストデータで動作確認（1000ステップ）
- [ ] TensorBoardロギング確認

### Phase 3: ControlNet
- [ ] `acestep/models/control_encoder.py` 作成
- [ ] `trainer_accompaniment.py` にControlNetモード追加
- [ ] `scripts/train_controlnet.py` 作成
- [ ] テストデータで動作確認

### Phase 4: Optuna
- [ ] `scripts/optuna_search.py` 作成
- [ ] MedianPruner動作確認
- [ ] ベスト設定自動保存確認

### Phase 5: 推論・評価
- [ ] `scripts/infer_accompaniment.py` 作成
- [ ] `scripts/compare_models.py` 作成
- [ ] 比較HTML生成確認
- [ ] 評価シートテンプレート作成

### ドキュメント
- [ ] `ACCOMPANIMENT_IMPLEMENTATION.md` 作成（本文書）
- [ ] `CLAUDE.md` 更新（プロジェクト固有規約追記）
- [ ] `README.md` 更新（使用例追加）

---

## 依存関係追加

### pyproject.toml更新

```toml
[project]
dependencies = [
    # 既存依存...
    "pydantic>=2.0.0",
    "optuna>=3.0.0",
    "optuna-dashboard>=0.15.0"
]

[tool.uv.scripts]
train-image2image = "python scripts/train_image2image.py"
train-controlnet = "python scripts/train_controlnet.py"
optuna-search = "python scripts/optuna_search.py"
infer = "python scripts/infer_accompaniment.py"
compare = "python scripts/compare_models.py"

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]

[tool.ty]
strict = true
```

---

## まとめ

本実装方針書では、ACE-Stepをベースとした伴奏生成システムの完全な設計を提示しました。

**主要成果物**:
1. **Image2Image方式**: リファレンス強度による簡便な制御
2. **ControlNet方式**: 明示的条件付けによる高度な制御
3. **統一設定システム**: Pydanticによる型安全な設定管理
4. **Optuna統合**: 自動ハイパーパラメータ探索
5. **評価フレームワーク**: 比較HTMLとA/Bテストプロトコル

**推奨実装順序**:
Phase 1 → Phase 2 → Phase 4（Image2Image探索） → Phase 3 → Phase 4（ControlNet探索） → Phase 5

**推定開発時間**: 約7.5時間（設計完了済み、コーディングのみ）

本方針に従うことで、高品質で保守性の高い伴奏生成システムが実現できます。
