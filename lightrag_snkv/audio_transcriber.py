from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_TIMESTAMP_GROUP_SECONDS = 30

_model_cache: dict[tuple[str, str], object] = {}


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _get_model(model_size: str, device: str):
    """Load (and cache) a faster-whisper model. First use downloads weights from Hugging Face."""
    key = (model_size, device)
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model %r on %s...", model_size, device)
        _model_cache[key] = WhisperModel(model_size, device=device)
    return _model_cache[key]


def _transcribe_sync(path: str, model_size: str, device: str, language: str | None) -> tuple[str, str]:
    """Blocking transcription — run via asyncio.to_thread from the caller.

    Groups segments into ~30s "[MM:SS] ..." windows, same convention used
    elsewhere in this project, so answers can cite an approximate timestamp.

    Returns (timestamped_text, detected_language_code).
    """
    model = _get_model(model_size, device)
    segments, info = model.transcribe(path, language=language)

    lines: list[str] = []
    group_start: float | None = None
    group_texts: list[str] = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if group_start is None:
            group_start = seg.start
        elif seg.start - group_start >= _TIMESTAMP_GROUP_SECONDS and group_texts:
            lines.append(f"[{_format_timestamp(group_start)}] {' '.join(group_texts)}")
            group_start = seg.start
            group_texts = []
        group_texts.append(text)

    if group_texts and group_start is not None:
        lines.append(f"[{_format_timestamp(group_start)}] {' '.join(group_texts)}")

    return "\n".join(lines), info.language


async def transcribe_file(
    path: str, model_size: str = "base", device: str = "cpu", language: str | None = None
) -> tuple[str, str] | None:
    """Transcribe an audio file. Returns (timestamped_text, language_code) or None if empty/failed."""
    try:
        text, lang = await asyncio.to_thread(_transcribe_sync, path, model_size, device, language)
    except Exception:
        logger.exception("Transcription failed for %s", path)
        return None
    if not text:
        return None
    return text, lang


def build_document(filename: str, transcript_text: str, language_code: str) -> str:
    """Format a transcribed audio file as a LightRAG-ingestible document."""
    lines = [
        f"# {filename}",
        "",
        f"Source file: {filename}",
        f"Language: {language_code}",
        "",
        "## Transcript",
        "",
        transcript_text,
    ]
    return "\n".join(lines)
