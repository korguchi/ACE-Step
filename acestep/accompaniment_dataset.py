"""Accompaniment generation dataset."""

import os
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger

from acestep.text2music_dataset import Text2MusicDataset


class AccompanimentDataset(Text2MusicDataset):
    """
    Dataset for accompaniment generation.

    Expected file structure:
        data/
        ├── song001_vocal.mp3       # Vocal track (input)
        ├── song001_inst.mp3        # Instrumental track (teacher: accompaniment mode)
        ├── song001_mix.mp3         # Mixed track (teacher: mix mode)
        ├── song001_prompt.txt      # Tags: "electronic, piano, 120bpm"
        └── song001_lyrics.txt      # Lyrics: "[instrumental]" (fixed)
    """

    def __init__(
        self,
        output_type: Literal["accompaniment", "mix"] = "accompaniment",
        reference_cache_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize the accompaniment dataset.

        Args:
            output_type: Output type - "accompaniment" (inst only) or "mix" (vocal+inst)
            reference_cache_size: Maximum number of reference audios to cache
            **kwargs: Additional arguments passed to parent Text2MusicDataset
        """
        self.output_type = output_type
        self.ref_cache: dict[str, torch.Tensor] = {}
        self.cache_size = reference_cache_size

        super().__init__(**kwargs)

    def get_reference_audio(self, key: str) -> Optional[torch.Tensor]:
        """
        Get vocal track with caching.

        Args:
            key: Sample key

        Returns:
            Vocal audio tensor (2, T) or None if not found
        """
        if key in self.ref_cache:
            return self.ref_cache[key]

        vocal_path = os.path.join(self.train_dataset_path, f"{key}_vocal.mp3")
        if not os.path.exists(vocal_path):
            logger.warning(f"Vocal track not found: {vocal_path}")
            return None

        try:
            vocal_audio, sr = torchaudio.load(vocal_path)

            # Convert to stereo
            if vocal_audio.shape[0] == 1:
                vocal_audio = vocal_audio.repeat(2, 1)

            # Resample to 48kHz if needed
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                vocal_audio = resampler(vocal_audio)

            # Clip audio to max duration
            max_samples = int(self.max_duration * 48000)
            if vocal_audio.shape[1] > max_samples:
                vocal_audio = vocal_audio[:, :max_samples]

            # Cache if space available
            if len(self.ref_cache) < self.cache_size:
                self.ref_cache[key] = vocal_audio

            return vocal_audio

        except Exception as e:
            logger.error(f"Failed to load vocal track {vocal_path}: {e}")
            return None

    def get_target_audio(self, key: str) -> Optional[torch.Tensor]:
        """
        Get target audio (inst or mix).

        Args:
            key: Sample key

        Returns:
            Target audio tensor (2, T) or None if not found
        """
        suffix = "_inst.mp3" if self.output_type == "accompaniment" else "_mix.mp3"
        target_path = os.path.join(self.train_dataset_path, f"{key}{suffix}")

        if not os.path.exists(target_path):
            logger.warning(f"Target track not found: {target_path}")
            return None

        try:
            target_audio, sr = torchaudio.load(target_path)

            # Convert to stereo
            if target_audio.shape[0] == 1:
                target_audio = target_audio.repeat(2, 1)

            # Resample to 48kHz if needed
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                target_audio = resampler(target_audio)

            # Clip audio to max duration
            max_samples = int(self.max_duration * 48000)
            if target_audio.shape[1] > max_samples:
                target_audio = target_audio[:, :max_samples]

            return target_audio

        except Exception as e:
            logger.error(f"Failed to load target track {target_path}: {e}")
            return None

    def process(self, item: dict) -> List[dict]:
        """
        Process a single dataset item.

        Args:
            item: Dataset item containing keys, tags, lyrics

        Returns:
            List containing processed sample dict, or empty list if loading fails
        """
        key = item["keys"]

        # Load vocal track
        vocal_audio = self.get_reference_audio(key)
        if vocal_audio is None:
            return []

        # Load target track
        target_audio = self.get_target_audio(key)
        if target_audio is None:
            return []

        # Ensure same length
        min_len = min(vocal_audio.shape[1], target_audio.shape[1])
        vocal_audio = vocal_audio[:, :min_len]
        target_audio = target_audio[:, :min_len]

        # Process prompt
        tags = item.get("tags", [])
        prompt = ", ".join(tags) if tags else "instrumental"

        # Fixed lyrics for instrumental
        lyrics = "[instrumental]"

        # Lyric tokenization
        lyric_token_ids, lyric_masks, _ = self.process_lyric(lyrics)

        return [
            {
                "key": key,
                "reference_wav": vocal_audio,
                "target_wav": target_audio,
                "prompt": prompt,
                "lyric_token_ids": lyric_token_ids,
                "lyric_masks": lyric_masks,
                "speaker_emb": torch.zeros(512),  # Unused
            }
        ]

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        Collate batch samples.

        Args:
            batch: List of processed samples

        Returns:
            Batched dictionary
        """
        max_len = max(x["reference_wav"].shape[-1] for x in batch)

        # Pad audio
        reference_wavs = []
        target_wavs = []
        wav_lengths = []

        for item in batch:
            ref_wav = item["reference_wav"]
            target_wav = item["target_wav"]

            # Ensure same length
            min_len = min(ref_wav.shape[-1], target_wav.shape[-1])
            ref_wav = ref_wav[:, :min_len]
            target_wav = target_wav[:, :min_len]

            # Pad to max_len
            pad_len = max_len - min_len
            if pad_len > 0:
                ref_wav = F.pad(ref_wav, (0, pad_len))
                target_wav = F.pad(target_wav, (0, pad_len))

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
            "speaker_embs": torch.stack([x["speaker_emb"] for x in batch]),
        }
