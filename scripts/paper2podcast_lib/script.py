"""Podcast-script generation: single-stage and multi-stage (Outline → Write → Review)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from . import gemini as gemini_mod
from .gemini import _gemini_json_body, call_gemini, search_paper_context
from .prompts import (
    OUTLINE_PROMPT_ZH,
    PROMPT_ZH,
    REVIEW_PROMPT_ZH,
    SEGMENT_PROMPT_ZH,
    _STYLE_GENTLE_ADDENDUM,
    _is_flash_tts_model,
    _segment_position_rules_zh,
    speaker_name_for,
)
from .runtime import abort, begin_stage, log_error, log_info, log_warn, record_degradation
from .validation import (
    extract_text_from_gemini_result,
    parse_json_payload,
    validate_outline_segments,
    validate_transcript_entries,
)

_PAPER_MAX_CHARS = 120000
_NO_CONTEXT_BLOCK = "（未提供背景信息，请根据论文内容本身进行讨论。）"


def _prepare_script_inputs(api_key, paper_text, model, skip_search):
    """Truncate paper, fetch context, and stamp today's date for the script prompt."""
    if len(paper_text) > _PAPER_MAX_CHARS:
        paper_text = paper_text[:_PAPER_MAX_CHARS] + "\n\n[... truncated for length ...]"
        log_info(f"✂️ Paper text truncated to {_PAPER_MAX_CHARS} chars")
    context_block = _NO_CONTEXT_BLOCK if skip_search else search_paper_context(api_key, paper_text, model)
    return paper_text, context_block, datetime.now().strftime("%Y-%m-%d")


def _validate_transcript_payload(payload: Any, stage: str) -> list[dict[str, Any]]:
    entries = payload if isinstance(payload, list) else payload.get("podcast_transcripts", [])
    return validate_transcript_entries(entries, stage)


def generate_script(api_key, paper_text, lang="zh", duration=10, model="gemini-3.1-pro-preview", skip_search=False, tts_model=None):
    """Generate structured podcast script JSON from paper text (single-stage)."""
    begin_stage("segment-generation", f"single-stage script generation lang={lang} duration={duration}m")
    log_info(f"📝 Generating podcast script ({lang}, ~{duration}min) with {model}...")

    paper_text, context_block, current_date = _prepare_script_inputs(api_key, paper_text, model, skip_search)
    word_count = duration * 250
    prompt = PROMPT_ZH.format(duration=duration, word_count=word_count, context_block=context_block)
    if _is_flash_tts_model(tts_model):
        prompt += _STYLE_GENTLE_ADDENDUM
    prompt = (
        f'【当前日期：{current_date}】请根据当前日期判断论文/文章的时间线，不要把过去的文章说成"未来"。\n\n'
        + prompt
        + f"\n\n<source_content>\n{paper_text}\n</source_content>"
    )

    try:
        result = call_gemini(
            api_key, model,
            _gemini_json_body(prompt, max_tokens=16384, temperature=0.9),
            timeout=420, retries=3,
            request_label="single-stage script generation",
        )
        raw = extract_text_from_gemini_result(result, "segment-generation", "Single-stage script generation")
        log_info(f"🧩 Received script draft: {len(raw)} chars")
    except Exception as exc:
        abort("segment-generation", f"Script generation failed: {type(exc).__name__}: {exc}", cause=exc)

    validated = _validate_transcript_payload(
        parse_json_payload(raw, "segment-generation", "generated script"),
        "segment-generation",
    )
    total_chars = sum(len(e.get("dialog", "")) for e in validated)
    log_info(f"✅ Script generated: {len(validated)} turns, {total_chars} chars")
    return {"podcast_transcripts": validated}


def generate_script_multistage(api_key, paper_text, lang="zh", duration=10, model="gemini-3.1-pro-preview", skip_search=False, tts_model=None):
    """Multi-stage podcast script generation: Outline → Write → Review."""
    log_info(f"📝 [Multi-stage] Generating podcast script ({lang}, ~{duration}min) with {model}...")

    paper_text, context_block, current_date = _prepare_script_inputs(api_key, paper_text, model, skip_search)
    word_count = duration * 450  # higher density target for the multi-stage path
    date_prefix = f'【当前日期：{current_date}】\n\n'
    source_block = f"\n\n<source_content>\n{paper_text}\n</source_content>"

    # ========== Stage 1: Outline ==========
    begin_stage("outline-generation", f"outline lang={lang} duration={duration}m")
    log_info("📋 Stage 1/3: Generating outline...")

    outline_prompt = OUTLINE_PROMPT_ZH.format(
        duration=duration, word_count=word_count, context_block=context_block,
    )

    try:
        result = call_gemini(
            api_key, model,
            _gemini_json_body(date_prefix + outline_prompt + source_block, max_tokens=4096, temperature=0.4),
            timeout=120, retries=3,
            request_label="outline generation",
        )
        outline_raw = extract_text_from_gemini_result(result, "outline-generation", "Outline generation")
        outline = parse_json_payload(outline_raw, "outline-generation", "outline")
        segments = validate_outline_segments(outline, "outline-generation")
        log_info(f"✅ Outline: {len(segments)} segments")
        for i, seg in enumerate(segments):
            log_info(f"   [{i+1}] {seg.get('title', '?')} ({seg.get('word_budget', '?')} 字, {seg.get('tone', '?')})")
    except Exception as exc:
        record_degradation("outline-generation", f"outline generation failed: {type(exc).__name__}: {exc}", "single-stage script generation")
        return generate_script(api_key, paper_text, lang, duration, model, skip_search, tts_model)

    # ========== Stage 2: Write segments ==========
    begin_stage("segment-generation", f"outline segments={len(segments)}")
    log_info("✍️ Stage 2/3: Writing segments...")

    all_transcripts: list[dict[str, Any]] = []
    prev_context = "（这是播客的开头）"
    segment_failures: list[str] = []

    for i, seg in enumerate(segments):
        segment_title = seg.get("title", f"Segment {i+1}")
        word_budget = seg.get("word_budget", word_count // len(segments))
        log_info(f"   ✍️ Writing segment {i+1}/{len(segments)}: {segment_title} ({word_budget} 字)...")

        seg_prompt = SEGMENT_PROMPT_ZH.format(
            segment_title=segment_title,
            segment_tone=seg.get("tone", "neutral"),
            key_points=json.dumps(seg.get("key_points", []), ensure_ascii=False),
            word_budget=word_budget,
            segment_position_rules=_segment_position_rules_zh(i, len(segments)),
            prev_context=prev_context,
        )
        if _is_flash_tts_model(tts_model):
            seg_prompt += _STYLE_GENTLE_ADDENDUM

        try:
            result = call_gemini(
                api_key, model,
                _gemini_json_body(date_prefix + seg_prompt + source_block, max_tokens=8192, temperature=0.9),
                timeout=300, retries=3,
                request_label=f"segment {i + 1} generation",
            )
            seg_raw = extract_text_from_gemini_result(result, "segment-generation", f"Segment {i + 1} generation")
            validated = _validate_transcript_payload(
                parse_json_payload(seg_raw, "segment-generation", f"segment {i + 1} script"),
                f"segment-generation segment {i + 1}",
            )

            seg_chars = sum(len(e["dialog"]) for e in validated)
            log_info(f"   ✅ Segment {i+1}: {len(validated)} turns, {seg_chars} chars")
            all_transcripts.extend(validated)

            if validated:
                last_turns = validated[-2:] if len(validated) >= 2 else validated
                prev_context = "\n".join(
                    f"{speaker_name_for(e['speaker_id'])}: {e['dialog'][:100]}..."
                    for e in last_turns
                )
        except Exception as exc:
            segment_failures.append(f"segment {i + 1} '{segment_title}': {type(exc).__name__}: {exc}")
            log_error(f"❌ Segment {i+1} failed: {type(exc).__name__}: {exc}")
            break

    if segment_failures or not all_transcripts:
        reason = "; ".join(segment_failures) if segment_failures else "outline segments produced no transcript entries"
        record_degradation("segment-generation", reason, "single-stage script generation")
        return generate_script(api_key, paper_text, lang, duration, model, skip_search, tts_model)

    # ========== Stage 3: Review ==========
    log_info("🔍 Stage 3/3: Quality review...")

    total_chars = sum(len(e["dialog"]) for e in all_transcripts)
    ratio = total_chars / word_count if word_count > 0 else 1.0
    if ratio < 0.7 or ratio > 1.4:
        log_warn(f"⚠️ Length deviation: {total_chars} chars vs target {word_count} (ratio: {ratio:.2f})")
    else:
        log_info(f"✅ Length check passed: {total_chars} chars (ratio: {ratio:.2f})")

    for check_label, check_entries in [("Opening", all_transcripts[:6]), ("Closing", all_transcripts[-6:])]:
        script_text = "\n".join(
            f"{speaker_name_for(e['speaker_id'])}: {e['dialog']}" for e in check_entries
        )
        review_prompt = REVIEW_PROMPT_ZH.format(
            key_points="开场/收尾质量",
            word_budget=len(script_text),
            script_text=script_text,
        )
        try:
            result = call_gemini(
                api_key, model,
                _gemini_json_body(review_prompt, max_tokens=2048, temperature=0.2),
                timeout=60, retries=2,
            )
            review = json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
            if review.get("pass", True):
                log_info(f"✅ {check_label} review passed")
            else:
                log_warn(f"⚠️ {check_label} review flagged issues: {review.get('issues', [])}")
        except Exception as exc:
            log_warn(f"⚠️ {check_label} review skipped: {exc}")

    log_info(f"✅ Multi-stage script complete: {len(all_transcripts)} turns, {total_chars} chars")
    return {"podcast_transcripts": all_transcripts}
