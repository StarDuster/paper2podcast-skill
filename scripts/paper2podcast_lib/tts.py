"""TTS rendering: split → render (per-turn or multi-speaker) → cache → MP3.

Owns the per-segment metadata cache that lets resume_tts.py / re-invocations
skip already-rendered turns.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from . import gemini as gemini_mod
from .gemini import call_gemini_async, is_rate_limited_error
from .prompts import (
    SPEAKER_NAMES,
    _build_tts_header,
    _tts_sample_context,
    speaker_name_for,
)
from .runtime import abort, log_error, log_info, log_warn
from .validation import ensure_file, ensure_non_empty_text


# ---------------------------------------------------------------------------
# Text builders / transcript splitting
# ---------------------------------------------------------------------------

def build_tts_text(entries, lang="zh", segment_position: str = "middle", tts_model: str | None = None):
    """Build the text input for a Gemini TTS multi-speaker conversation segment."""
    if not entries:
        return ""

    lines = [_build_tts_header(segment_position)]
    for entry in entries:
        lines.append(f"{speaker_name_for(entry['speaker_id'])}: {entry['dialog']}")
    return "\n\n".join(lines)


def build_single_turn_tts_text(entry, lang="zh", segment_position: str = "middle") -> str:
    """Build a single-speaker TTS prompt for one transcript turn."""
    segment_note = _tts_sample_context(segment_position)
    return f"""\
TTS this single Chinese podcast line using the configured voice for {speaker_name_for(entry['speaker_id'])}.

Delivery:
- Standard mainland Mandarin; calm, deadpan, staccato, light, fluent, slightly fast.
- No Taiwanese accent, Northeastern accent, heavy erhua, drama, heavy emphasis, over-articulation, or added words.
- Read the line exactly. Do not add speaker labels, transitions, summaries, or extra words.
- Segment note: {segment_note}

Line:
{entry["dialog"]}"""


def _entry_bytes(entry):
    """Byte size of a single transcript entry when rendered for TTS."""
    return len(f"{speaker_name_for(entry['speaker_id'])}: {entry['dialog']}".encode("utf-8")) + 4


def _find_best_split(entries, start, max_payload_bytes):
    """Find the best split point in entries[start:] that fits within byte budget.

    Prefers splitting at 'topic boundaries':
      1. Speaker 0 (Alice) turn after a Speaker 1 (Bob) turn — typically a new
         discussion segment, but only if the current segment already contains
         both speakers.
      2. Any speaker change as fallback, keeping both sides of the exchange in
         the same segment.
      3. Hard byte limit as last resort.
    """
    total = 0
    last_valid = start  # at least one entry
    best_boundary = None

    for i in range(start, len(entries)):
        total += _entry_bytes(entries[i])
        if total > max_payload_bytes and i > start:
            break
        last_valid = i + 1

        if i > start and total > max_payload_bytes * 0.4:
            prev_speaker = entries[i - 1]["speaker_id"]
            curr_speaker = entries[i]["speaker_id"]
            if curr_speaker == 0 and prev_speaker == 1:
                prior_speakers = {entry["speaker_id"] for entry in entries[start:i]}
                if len(prior_speakers) > 1:
                    best_boundary = i
            elif curr_speaker != prev_speaker and best_boundary is None:
                best_boundary = i + 1

    if best_boundary is not None and best_boundary <= last_valid:
        return best_boundary
    return last_valid


def split_transcript(entries, max_bytes=4000, lang="zh", tts_model: str | None = None):
    """Split transcript entries into multi-speaker TTS conversation segments."""
    if not entries:
        return []

    segments = []
    header_bytes = len(_build_tts_header("middle").encode("utf-8")) + 100
    payload_budget = max(1, max_bytes - header_bytes)
    start = 0
    while start < len(entries):
        end = _find_best_split(entries, start, payload_budget)
        if end <= start:
            end = start + 1
        segments.append(entries[start:end])
        start = end

    for i, seg in enumerate(segments):
        seg_bytes = sum(_entry_bytes(e) for e in seg) + header_bytes
        speakers = sorted({speaker_name_for(e["speaker_id"]) for e in seg})
        log_info(
            f"  📦 Segment {i+1}/{len(segments)}: "
            f"{seg_bytes} bytes, {len(seg)} turns [{', '.join(speakers)}]"
        )
    return segments


# ---------------------------------------------------------------------------
# Audio file helpers (paths, atomic writes, ffmpeg)
# ---------------------------------------------------------------------------

async def convert_pcm_to_mp3(pcm_file, mp3_file):
    """Convert raw PCM to MP3 with ffmpeg in a worker thread."""
    for attempt in range(1, 3):
        result = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
             "-i", pcm_file, "-b:a", "128k", mp3_file],
            capture_output=True,
            timeout=60,
            text=True,
        )
        if result.returncode == 0:
            ensure_file(mp3_file, "tts-audio-synthesis", "segment MP3 output")
            return
        if attempt < 2:
            log_warn(f"⚠️ ffmpeg convert failed on attempt {attempt}/2: {result.stderr.strip()[:300]}")
            await asyncio.sleep(attempt * 2)
            continue
        raise RuntimeError(result.stderr.strip()[:500] or "ffmpeg failed")


def _segment_metadata_path(mp3_file: str | Path) -> Path:
    return Path(f"{mp3_file}.meta.json")


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def _tts_output_paths(output_dir: str | Path, segment_idx: int) -> tuple[str, str, str]:
    stem = f"segment_{segment_idx:03d}"
    base_dir = Path(output_dir)
    return (
        str(base_dir / f"{stem}.pcm"),
        str(base_dir / f"{stem}.mp3"),
        str(base_dir / f".{stem}.tmp.{os.getpid()}.mp3"),
    )


# ---------------------------------------------------------------------------
# Per-segment metadata: makes TTS output uniquely tied to its inputs
# ---------------------------------------------------------------------------

def build_tts_segment_metadata(
    text: str,
    *,
    lang: str,
    voice_a: str,
    voice_b: str,
    tts_model: str,
    segment_idx: int,
    total_segments: int,
    segment_position: str,
    render_mode: str = "multi-speaker",
    speaker_id: int | None = None,
    voice_name: str | None = None,
) -> dict[str, Any]:
    """Build the exact identity for a generated TTS segment."""
    return {
        "schema_version": 1,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "lang": lang,
        "voice_a": voice_a,
        "voice_b": voice_b,
        "tts_model": tts_model,
        "segment_idx": segment_idx,
        "total_segments": total_segments,
        "segment_position": segment_position,
        "render_mode": render_mode,
        "speaker_id": speaker_id,
        "voice_name": voice_name,
    }


def is_reusable_tts_segment(mp3_file: str | Path, expected_metadata: dict[str, Any]) -> bool:
    """Return True only when an existing segment belongs to the current TTS request."""
    path = Path(mp3_file)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    metadata_path = _segment_metadata_path(path)
    if not metadata_path.exists():
        return False
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual_metadata == expected_metadata


# ---------------------------------------------------------------------------
# Gemini TTS request / response handling
# ---------------------------------------------------------------------------

async def _wait_for_tts_rate_limit(err: dict[str, Any], label: str, attempt: int, max_retries: int) -> bool:
    if not is_rate_limited_error(err):
        return False
    wait = (attempt + 1) * 30
    log_warn(f"  ⏳ {label} rate limited, wait {wait}s (attempt {attempt + 1}/{max_retries})")
    await asyncio.sleep(wait)
    return True


def _extract_tts_audio_b64(resp_data: dict[str, Any], label: str) -> str | None:
    candidates = resp_data.get("candidates")
    if not candidates:
        log_error(f"  ❌ {label} returned no candidates: {json.dumps(resp_data, ensure_ascii=False)[:500]}")
        return None
    finish_reason = str(candidates[0].get("finishReason", "") or "")
    if finish_reason and finish_reason not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        log_error(f"  ❌ {label} returned partial/blocked audio (finishReason={finish_reason})")
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        log_error(f"  ❌ {label} missing content parts")
        return None
    audio_b64 = parts[0].get("inlineData", {}).get("data")
    if not audio_b64:
        log_error(f"  ❌ {label} missing inline audio data")
        return None
    return audio_b64


async def _write_tts_audio_files(
    audio_b64: str,
    *,
    pcm_file: str,
    mp3_tmp: str,
    mp3_file: str,
    expected_metadata: dict[str, Any],
    output_label: str,
) -> bytes:
    audio_data = base64.b64decode(audio_b64)
    if not audio_data:
        raise RuntimeError("decoded audio payload is empty")
    Path(pcm_file).write_bytes(audio_data)
    await convert_pcm_to_mp3(pcm_file, mp3_tmp)
    ensure_file(mp3_tmp, "tts-audio-synthesis", f"{output_label} audio")
    Path(mp3_tmp).replace(mp3_file)
    _write_json_atomic(_segment_metadata_path(mp3_file), expected_metadata)
    ensure_file(mp3_file, "tts-audio-synthesis", f"{output_label} audio")
    return audio_data


def _build_tts_body(
    text: str,
    *,
    mode: str,
    voice_a: str,
    voice_b: str,
    voice_name: str | None,
) -> dict[str, Any]:
    """Build the generateContent body for a TTS request."""
    if mode == "per-turn":
        speech_config = {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}},
        }
    else:
        speech_config = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {"speaker": "Alice", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_a}}},
                    {"speaker": "Bob", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_b}}},
                ],
            },
        }
    return {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": speech_config,
        },
    }


# ---------------------------------------------------------------------------
# Rendering: one segment / one turn, plus the parallel/serial driver
# ---------------------------------------------------------------------------

async def tts_render_async(
    session,
    api_key,
    segment,
    segment_idx,
    total_segments,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    *,
    mode: str = "multi-speaker",
    segment_position: str = "middle",
):
    """Convert one transcript segment (or single turn) to audio.

    `segment` is always a list of transcript entries. For per-turn mode it
    contains exactly one entry; multi-speaker accepts the full mini-conversation.
    """
    pcm_file, mp3_file, mp3_tmp = _tts_output_paths(output_dir, segment_idx)
    is_per_turn = mode == "per-turn"
    kind = "Turn" if is_per_turn else "Segment"
    label = f"{kind} {segment_idx + 1}/{total_segments}"

    if is_per_turn:
        entry = segment[0]
        speaker_id = entry["speaker_id"]
        voice_name = voice_a if speaker_id == 0 else voice_b
        text = build_single_turn_tts_text(entry, lang, segment_position)
        meta_extras = {"render_mode": "per-turn", "speaker_id": speaker_id, "voice_name": voice_name}
        start_detail = f"{len(text.encode('utf-8'))} bytes, {speaker_name_for(speaker_id)}, voice={voice_name}"
    else:
        voice_name = None
        text = build_tts_text(segment, lang, segment_position, tts_model)
        meta_extras = {}
        start_detail = f"{len(text.encode('utf-8'))} bytes, {len(segment)} turns"

    ensure_non_empty_text("tts-audio-synthesis", text, f"TTS {kind.lower()} {segment_idx + 1} prompt")
    expected_metadata = build_tts_segment_metadata(
        text,
        lang=lang,
        voice_a=voice_a,
        voice_b=voice_b,
        tts_model=tts_model,
        segment_idx=segment_idx,
        total_segments=total_segments,
        segment_position=segment_position,
        **meta_extras,
    )

    if is_reusable_tts_segment(mp3_file, expected_metadata):
        log_info(f"  ⏩ {label} exists with matching metadata, skipping")
        return mp3_file
    if os.path.exists(mp3_file):
        log_warn(f"  ♻️ {label} exists but metadata does not match; regenerating")
        Path(mp3_file).unlink(missing_ok=True)
        _segment_metadata_path(mp3_file).unlink(missing_ok=True)

    log_info(f"  🎙️ {label} start ({start_detail})")

    body = _build_tts_body(text, mode=mode, voice_a=voice_a, voice_b=voice_b, voice_name=voice_name)
    output_label = f"{kind.lower()} {segment_idx + 1}"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            resp_data = await call_gemini_async(
                session, api_key, tts_model, body,
                timeout=300,
                request_label=f"TTS {output_label}/{total_segments}",
            )
            if "error" in resp_data:
                err = resp_data["error"]
                if await _wait_for_tts_rate_limit(err, label, attempt, max_retries):
                    continue
                log_error(f"  ❌ {label} API error: {err}")
                return None

            audio_b64 = _extract_tts_audio_b64(resp_data, label)
            if not audio_b64:
                return None

            audio_data = await _write_tts_audio_files(
                audio_b64,
                pcm_file=pcm_file,
                mp3_tmp=mp3_tmp,
                mp3_file=mp3_file,
                expected_metadata=expected_metadata,
                output_label=output_label,
            )

            duration = len(audio_data) / (24000 * 2)
            log_info(f"  ✅ {label}: {duration:.1f}s")

            Path(pcm_file).unlink(missing_ok=True)
            return mp3_file

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                log_warn(f"  ⚠️ {label} attempt {attempt + 1} failed: {type(e).__name__}: {e}. Retry in {wait}s.")
                await asyncio.sleep(wait)
            else:
                log_error(f"  ❌ TTS failed for {label.lower()}: {type(e).__name__}: {e}")
                return None
        finally:
            Path(pcm_file).unlink(missing_ok=True)
            Path(mp3_tmp).unlink(missing_ok=True)

    return None


def _infer_segment_position(idx: int, total: int) -> str:
    """Infer an acoustic scene position label from segment index."""
    if idx == 0:
        return "intro"
    if idx == total - 1:
        return "conclusion"
    if idx == 1 and total >= 4:
        return "technical"
    return "middle"


async def run_tts_async(
    api_key,
    segments,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    *,
    indexes=None,
    workers=1,
    render_mode="multi-speaker",
):
    """Render TTS for `segments`. Parallel over all by default, or serial over `indexes`.

    Returns a list (len == len(segments), None for not-run / failed) when `indexes` is None;
    returns a dict keyed by index when `indexes` is provided.
    """
    if gemini_mod.aiohttp is None:
        abort("tts-audio-synthesis", "Missing dependency: aiohttp. Install with: pip install aiohttp")
    aiohttp = gemini_mod.aiohttp

    total = len(segments)
    serial = indexes is not None
    target_indexes = list(indexes) if serial else list(range(total))

    async def render(session, idx):
        return await tts_render_async(
            session, api_key, segments[idx], idx, total, output_dir,
            lang, voice_a, voice_b, tts_model,
            mode=render_mode,
            segment_position=_infer_segment_position(idx, total),
        )

    async with aiohttp.ClientSession() as session:
        if serial:
            results: dict[int, str | None] = {}
            for idx in target_indexes:
                log_info(f"🔁 Retrying segment {idx + 1}/{total} serially")
                results[idx] = await render(session, idx)
            return results

        list_results: list[str | None] = [None] * total
        semaphore = asyncio.Semaphore(max(1, workers))

        async def run_one(idx):
            async with semaphore:
                list_results[idx] = await render(session, idx)

        tasks = [asyncio.create_task(run_one(i)) for i in target_indexes]
        for task in asyncio.as_completed(tasks):
            await task
        return list_results


# ---------------------------------------------------------------------------
# Backwards-compat shims (resume_tts.py still imports these by name)
# ---------------------------------------------------------------------------

async def tts_segment_async(
    session, api_key, entries, segment_idx, total_segments, output_dir,
    lang, voice_a, voice_b, tts_model, segment_position: str = "middle",
):
    """Compat wrapper around tts_render_async(mode='multi-speaker')."""
    return await tts_render_async(
        session, api_key, entries, segment_idx, total_segments, output_dir,
        lang, voice_a, voice_b, tts_model,
        mode="multi-speaker", segment_position=segment_position,
    )


async def tts_turn_async(
    session, api_key, entry, segment_idx, total_segments, output_dir,
    lang, voice_a, voice_b, tts_model, segment_position: str = "middle",
):
    """Compat wrapper around tts_render_async(mode='per-turn')."""
    return await tts_render_async(
        session, api_key, [entry], segment_idx, total_segments, output_dir,
        lang, voice_a, voice_b, tts_model,
        mode="per-turn", segment_position=segment_position,
    )
