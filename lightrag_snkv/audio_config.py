from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    # Directory scanned for .mp3 files to ingest.
    scan_dir: str = field(
        default_factory=lambda: os.environ.get("AUDIO_SCAN_DIR", "./audio_inputs")
    )

    # faster-whisper model size: tiny, base, small, medium, large-v3, etc.
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("AUDIO_WHISPER_MODEL", "base")
    )

    # Force a transcription language (e.g. "en"). Leave unset to auto-detect per file.
    language: str | None = field(
        default_factory=lambda: os.environ.get("AUDIO_LANGUAGE") or None
    )

    # Device for faster-whisper: "cpu" or "cuda".
    device: str = field(
        default_factory=lambda: os.environ.get("AUDIO_DEVICE", "cpu")
    )

    # "daily" | "weekly" | None (run-once). Re-checks scan_dir for new files.
    schedule: str | None = field(
        default_factory=lambda: os.environ.get("AUDIO_SCHEDULE") or None
    )
