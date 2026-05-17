"""Shared text / JSON / file-shape validators used by multiple pipeline stages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .runtime import abort, log_info, log_warn


def ensure_non_empty_text(stage: str, text: str, label: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        abort(stage, f"{label} is empty")
    return cleaned


def parse_json_payload(raw: str, stage: str, label: str) -> Any:
    raw = ensure_non_empty_text(stage, raw, label)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            abort(stage, f"Failed to parse {label} JSON. Raw response snippet: {raw[:1200]}")
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            abort(stage, f"Failed to parse extracted {label} JSON: {type(exc).__name__}: {exc}. Raw snippet: {raw[:1200]}", cause=exc)
    return None


def extract_text_from_gemini_result(result: dict[str, Any], stage: str, label: str) -> str:
    if not isinstance(result, dict):
        abort(stage, f"{label} returned non-object payload: {type(result).__name__}")
    candidates = result.get("candidates")
    if not candidates:
        prompt_feedback = result.get("promptFeedback")
        feedback = json.dumps(prompt_feedback, ensure_ascii=False)[:500] if prompt_feedback else "none"
        abort(stage, f"{label} returned no candidates. promptFeedback={feedback}")

    candidate = candidates[0]
    finish_reason = str(candidate.get("finishReason", "") or "")
    if finish_reason and finish_reason not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        abort(stage, f"{label} returned partial or blocked response (finishReason={finish_reason})")

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not parts:
        abort(stage, f"{label} response is missing content parts")

    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            texts.append(str(part["text"]))
    return ensure_non_empty_text(stage, "\n".join(texts), f"{label} text")


def validate_outline_segments(outline: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    segments = outline.get("segments", [])
    if not isinstance(segments, list) or not segments:
        abort(stage, "Outline JSON does not contain any segments")

    validated = []
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            abort(stage, f"Outline segment {idx + 1} is not an object")
        title = ensure_non_empty_text(stage, segment.get("title", ""), f"outline segment {idx + 1} title")
        key_points = segment.get("key_points", [])
        if not isinstance(key_points, list) or not key_points:
            abort(stage, f"Outline segment {idx + 1} has no key_points")
        cleaned_points = [str(point).strip() for point in key_points if str(point).strip()]
        if not cleaned_points:
            abort(stage, f"Outline segment {idx + 1} key_points are empty")
        try:
            word_budget = int(segment.get("word_budget", 0))
        except (TypeError, ValueError) as exc:
            abort(stage, f"Outline segment {idx + 1} has invalid word_budget: {segment.get('word_budget')}", cause=exc)
        if word_budget <= 0:
            abort(stage, f"Outline segment {idx + 1} has non-positive word_budget: {word_budget}")
        tone = ensure_non_empty_text(stage, segment.get("tone", ""), f"outline segment {idx + 1} tone")
        validated.append({
            "title": title,
            "key_points": cleaned_points,
            "word_budget": word_budget,
            "tone": tone,
        })
    return validated


def validate_transcript_entries(entries: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        abort(stage, "Transcript payload is not a list")

    validated = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            log_warn(f"⚠️ [{stage}] Skip invalid entry at index {idx}: not a dict")
            continue
        if "speaker_id" not in entry or "dialog" not in entry:
            log_warn(f"⚠️ [{stage}] Skip invalid entry at index {idx}: missing speaker_id/dialog")
            continue
        try:
            speaker_id = int(entry["speaker_id"])
        except (TypeError, ValueError):
            log_warn(f"⚠️ [{stage}] speaker_id at index {idx} cannot be parsed: {entry['speaker_id']} -> use 0")
            speaker_id = 0
        if speaker_id not in (0, 1):
            log_warn(f"⚠️ [{stage}] Normalize unexpected speaker_id at index {idx}: {speaker_id} -> 0")
            speaker_id = 0

        dialog = str(entry["dialog"]).strip()
        if not dialog:
            log_warn(f"⚠️ [{stage}] Skip empty dialog at index {idx}")
            continue
        validated_entry = {"speaker_id": speaker_id, "dialog": dialog}
        if entry.get("style"):
            style = str(entry["style"]).strip()
            if style:
                validated_entry["style"] = style
        validated.append(validated_entry)

    if not validated:
        abort(stage, "No valid transcript entries were produced")
    return validated


def ensure_file(path: str, stage: str, label: str, *, min_size: int = 1) -> str:
    target = Path(path)
    if not target.exists():
        abort(stage, f"{label} does not exist: {target}")
    size = target.stat().st_size
    if size < min_size:
        abort(stage, f"{label} is too small ({size} bytes): {target}")
    return str(target)


def write_json_file(path: str, payload: Any, stage: str, label: str) -> str:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(target)
    except Exception as exc:
        abort(stage, f"Failed to write {label} to {target}: {type(exc).__name__}: {exc}", cause=exc)
    ensure_file(str(target), stage, label)
    log_info(f"💾 {label} saved: {target}")
    return str(target)
