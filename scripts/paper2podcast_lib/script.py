"""Podcast-script generation: single-stage and multi-stage (Outline → Write → Review)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .gemini import _gemini_json_body, call_gemini
from .search import search_context
from .prompts import (
    OUTLINE_PROMPT_ZH,
    PROMPT_ZH,
    SEGMENT_PROMPT_ZH,
    SCRIPT_REVIEW_PROMPT_ZH,
    _STYLE_GENTLE_ADDENDUM,
    _is_flash_tts_model,
    _segment_position_rules_zh,
    speaker_name_for,
)
from .runtime import abort, begin_stage, create_run_work_dir, current_work_dir, log_error, log_info, log_warn, record_degradation
from .provenance import text_hash
from .validation import (
    extract_text_from_gemini_result,
    parse_json_payload,
    validate_outline_segments,
    validate_transcript_entries,
    write_json_file,
)

try:
    from agent.auxiliary_client import call_llm
except Exception:  # pragma: no cover - depends on the Hermes checkout being importable
    call_llm = None

_PAPER_MAX_CHARS = 250000
_NO_CONTEXT_BLOCK = "（未提供背景信息，请根据论文内容本身进行讨论。）"
_DEFAULT_REVIEW_PROVIDER = "deepseek"
_DEFAULT_REVIEW_MODEL = "deepseek-v4-pro"
_REVIEW_MAX_TOKENS = 24000
_REVIEW_MIN_TOKENS = 4096
_GEMINI_REVIEW_PROVIDERS = {
    "gemini",
    "vertex",
    "google",
    "google-ai",
    "googleai",
    "generativelanguage",
    "generative-language",
}


def _prepare_script_inputs(
    api_runtime,
    paper_text,
    model,
    skip_search,
    *,
    search_backend="tavily-agent",
    search_model=None,
    search_provider=None,
    search_model_api_key=None,
    search_model_api_key_file=None,
    search_model_base_url=None,
    search_use_hermes=False,
):
    """Truncate paper, fetch context, and stamp today's date for the script prompt."""
    original_text = paper_text
    if len(paper_text) > _PAPER_MAX_CHARS:
        paper_text = paper_text[:_PAPER_MAX_CHARS] + "\n\n[... truncated for length ...]"
        log_info(f"✂️ Paper text truncated to {_PAPER_MAX_CHARS} chars")
    context_block = _NO_CONTEXT_BLOCK if skip_search else search_context(
        original_text,
        backend=search_backend,
        model=search_model,
        provider=search_provider,
        api_runtime=api_runtime,
        model_api_key=search_model_api_key,
        model_api_key_file=search_model_api_key_file,
        model_base_url=search_model_base_url,
        use_hermes=search_use_hermes,
    )
    return paper_text, context_block, datetime.now().strftime("%Y-%m-%d")


def _validate_transcript_payload(payload: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(payload, (dict, list)):
        abort(stage, "Transcript JSON must be an object or list")
    entries = payload if isinstance(payload, list) else payload.get("podcast_transcripts", [])
    return validate_transcript_entries(entries, stage)


def _extract_llm_response_text(response: Any, stage: str) -> str:
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception as exc:
        abort(stage, f"Review model returned an unexpected response shape: {type(exc).__name__}: {exc}", cause=exc)


def _review_max_tokens(entries: list[dict[str, Any]]) -> int:
    total_chars = sum(len(entry.get("dialog", "")) for entry in entries)
    return max(_REVIEW_MIN_TOKENS, min(_REVIEW_MAX_TOKENS, total_chars * 2))


def _read_explicit_key(value: str | None, key_file: str | None, stage: str) -> str:
    explicit = str(value or "").strip()
    if explicit:
        return explicit
    path = str(key_file or "").strip()
    if not path:
        return ""
    try:
        explicit = Path(path).read_text(encoding="utf-8").strip()
    except Exception as exc:
        abort(stage, f"Failed to read API key file {path}: {type(exc).__name__}: {exc}", cause=exc)
    if explicit:
        return explicit
    abort(stage, f"API key file is empty: {path}")


def _is_gemini_review_route(provider: str, model: str) -> bool:
    provider_norm = str(provider or "").strip().lower()
    model_norm = str(model or "").strip().lower().removeprefix("models/")
    model_last_part = model_norm.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return (
        provider_norm in _GEMINI_REVIEW_PROVIDERS
        or model_norm.startswith("gemini")
        or model_last_part.startswith("gemini")
    )


def validate_review_route(provider, model, api_key=None, api_key_file=None, base_url=None):
    """Resolve explicit endpoints without ever falling through to SDK defaults."""
    provider = str(provider or "").strip().lower()
    model_name = str(model or "").strip().lower().rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    url = str(base_url or os.getenv("REVIEW_MODEL_BASE_URL", "")).strip()
    has_key = bool(api_key or api_key_file or os.getenv("REVIEW_MODEL_API_KEY")
                   or os.getenv("REVIEW_MODEL_API_KEY_FILE"))
    if _is_gemini_review_route(provider, model):
        abort("config", "Script review must use a non-Gemini provider/model")
    if model_name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")) and provider != "openai-codex":
        abort("config", "GPT review models must use --review-provider openai-codex")
    if provider == "openai-codex":
        if url or has_key:
            abort("config", "openai-codex review uses Hermes OAuth; explicit API keys/base URLs are not supported")
        return ""
    if has_key and not url:
        if provider == "deepseek":
            url = "https://api.deepseek.com/v1"
        else:
            abort("config", "Explicit review credentials require --review-base-url for this provider")
    if url:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            abort("config", "Invalid review base URL")
    return url


def _call_review_model(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    api_key: str | None,
    api_key_file: str | None,
    base_url: str | None,
) -> str:
    explicit_base_url = validate_review_route(provider, model, api_key, api_key_file, base_url)
    explicit_key = _read_explicit_key(
        api_key or os.getenv("REVIEW_MODEL_API_KEY"),
        api_key_file or os.getenv("REVIEW_MODEL_API_KEY_FILE"),
        "script-review",
    )
    if explicit_base_url or explicit_key:
        try:
            from openai import OpenAI
        except Exception as exc:
            abort("script-review", f"OpenAI-compatible review requires openai: {type(exc).__name__}: {exc}", cause=exc)
        client = OpenAI(
            api_key=explicit_key or ("not-needed" if explicit_base_url else None),
            base_url=explicit_base_url or None,
            max_retries=0,
            timeout=timeout,
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return _extract_llm_response_text(response, "script-review")
        finally:
            client.close()

    if call_llm is None:
        abort(
            "script-review",
            "Script review needs Hermes call_llm or explicit --review-base-url/--review-api-key; use --no-script-review to bypass it",
        )
    response = call_llm(
        provider=provider,
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return _extract_llm_response_text(response, "script-review")


def _log_length_check(entries: list[dict[str, Any]], word_count: int, label: str) -> None:
    total_chars = sum(len(entry["dialog"]) for entry in entries)
    ratio = total_chars / word_count if word_count > 0 else 1.0
    if ratio < 0.7 or ratio > 1.4:
        log_warn(f"⚠️ {label} length deviation: {total_chars} chars vs target {word_count} (ratio: {ratio:.2f})")
    else:
        log_info(f"✅ {label} length check passed: {total_chars} chars (ratio: {ratio:.2f})")


def _review_transcript_with_external_model(
    *,
    paper_text: str,
    context_block: str,
    entries: list[dict[str, Any]],
    review_provider: str,
    review_model: str,
    review_enabled: bool,
    review_api_key: str | None = None,
    review_api_key_file: str | None = None,
    review_base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Ask a non-Gemini model to revise the full Gemini-generated transcript."""
    # Save draft + exact review source before invoking the quality gate.
    import uuid
    import tempfile
    checkpoint_dir = current_work_dir()
    if checkpoint_dir is None:
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="paper2podcast-draft-"))
    checkpoint = checkpoint_dir / f"draft_{uuid.uuid4().hex}.json"
    write_json_file(str(checkpoint), {
        "podcast_transcripts": entries,
        "review": {"status": "draft", "reviewed": False},
        "source_sha256": text_hash(paper_text),
        "review_source_text": paper_text,
    }, "file-write", "unreviewed draft checkpoint")
    if not review_enabled:
        log_info("⏭️ Script review disabled")
        return entries
    if _is_gemini_review_route(review_provider, review_model):
        abort("script-review", "Script review must use a non-Gemini provider/model; use --no-script-review to bypass it")

    begin_stage("script-review", f"external review provider={review_provider} model={review_model}")
    draft_chars = sum(len(entry["dialog"]) for entry in entries)
    log_info(f"🔎 Reviewing full script with {review_provider}/{review_model} ({len(entries)} turns, {draft_chars} chars)")

    script_json = json.dumps({"podcast_transcripts": entries}, ensure_ascii=False, indent=2)
    prompt = SCRIPT_REVIEW_PROMPT_ZH.format(
        context_block=context_block,
        source_content=paper_text,
        script_json=script_json,
    )
    try:
        raw = _call_review_model(
            provider=review_provider,
            model=review_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_review_max_tokens(entries),
            timeout=420,
            api_key=review_api_key,
            api_key_file=review_api_key_file,
            base_url=review_base_url,
        )
        reviewed = _validate_transcript_payload(
            parse_json_payload(raw, "script-review", "reviewed script"),
            "script-review",
        )
    except Exception as exc:
        abort("script-review", f"Script review failed: {type(exc).__name__}: {exc}", cause=exc)

    reviewed_chars = sum(len(entry["dialog"]) for entry in reviewed)
    if reviewed_chars < draft_chars * 0.6:
        abort(
            "script-review",
            f"Reviewed script is unexpectedly short: {reviewed_chars} chars vs draft {draft_chars}",
        )
    if reviewed_chars > draft_chars * 1.5:
        abort(
            "script-review",
            f"Reviewed script is unexpectedly long: {reviewed_chars} chars vs draft {draft_chars}",
        )
    log_info(f"✅ Script review applied: {len(reviewed)} turns, {reviewed_chars} chars")
    return reviewed


def generate_script(
    api_runtime,
    paper_text,
    lang="zh",
    duration=10,
    model="gemini-3.1-pro-preview",
    skip_search=False,
    tts_model=None,
    *,
    review_provider=_DEFAULT_REVIEW_PROVIDER,
    review_model=_DEFAULT_REVIEW_MODEL,
    review_enabled=True,
    review_api_key=None,
    review_api_key_file=None,
    review_base_url=None,
    search_backend="tavily-agent",
    search_model=None,
    search_provider=None,
    search_model_api_key=None,
    search_model_api_key_file=None,
    search_model_base_url=None,
    search_use_hermes=False,
    _prepared_inputs=None,
):
    """Generate structured podcast script JSON from paper text (single-stage)."""
    begin_stage("segment-generation", f"single-stage script generation lang={lang} duration={duration}m")
    log_info(f"📝 Generating podcast script ({lang}, ~{duration}min) with {model}...")

    paper_text, context_block, current_date = (
        _prepared_inputs if _prepared_inputs is not None else _prepare_script_inputs(
            api_runtime, paper_text, model, skip_search, search_backend=search_backend,
            search_model=search_model, search_provider=search_provider,
            search_model_api_key=search_model_api_key,
            search_model_api_key_file=search_model_api_key_file,
            search_model_base_url=search_model_base_url,
            search_use_hermes=search_use_hermes,
        )
    )
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
            api_runtime, model,
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
    validated = _review_transcript_with_external_model(
        paper_text=paper_text,
        context_block=context_block,
        entries=validated,
        review_provider=review_provider,
        review_model=review_model,
        review_enabled=review_enabled,
        review_api_key=review_api_key,
        review_api_key_file=review_api_key_file,
        review_base_url=review_base_url,
    )
    total_chars = sum(len(e.get("dialog", "")) for e in validated)
    log_info(f"✅ Script generated: {len(validated)} turns, {total_chars} chars")
    return {"podcast_transcripts": validated}


def generate_script_multistage(
    api_runtime,
    paper_text,
    lang="zh",
    duration=10,
    model="gemini-3.1-pro-preview",
    skip_search=False,
    tts_model=None,
    *,
    review_provider=_DEFAULT_REVIEW_PROVIDER,
    review_model=_DEFAULT_REVIEW_MODEL,
    review_enabled=True,
    review_api_key=None,
    review_api_key_file=None,
    review_base_url=None,
    search_backend="tavily-agent",
    search_model=None,
    search_provider=None,
    search_model_api_key=None,
    search_model_api_key_file=None,
    search_model_base_url=None,
    search_use_hermes=False,
):
    """Multi-stage podcast script generation: Outline → Write → Review."""
    log_info(f"📝 [Multi-stage] Generating podcast script ({lang}, ~{duration}min) with {model}...")

    paper_text, context_block, current_date = _prepare_script_inputs(
        api_runtime, paper_text, model, skip_search, search_backend=search_backend,
        search_model=search_model, search_provider=search_provider,
        search_model_api_key=search_model_api_key,
        search_model_api_key_file=search_model_api_key_file,
        search_model_base_url=search_model_base_url,
        search_use_hermes=search_use_hermes,
    )
    word_count = duration * 250
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
            api_runtime, model,
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
        return generate_script(
            api_runtime,
            paper_text,
            lang,
            duration,
            model,
            skip_search,
            tts_model,
            review_provider=review_provider,
            review_model=review_model,
            review_enabled=review_enabled,
            review_api_key=review_api_key,
            review_api_key_file=review_api_key_file,
            review_base_url=review_base_url,
            _prepared_inputs=(paper_text, context_block, current_date),
        )

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
                api_runtime, model,
                _gemini_json_body(date_prefix + seg_prompt + "\n\n" + context_block + source_block, max_tokens=32768, temperature=0.9),
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
        return generate_script(
            api_runtime,
            paper_text,
            lang,
            duration,
            model,
            skip_search,
            tts_model,
            review_provider=review_provider,
            review_model=review_model,
            review_enabled=review_enabled,
            review_api_key=review_api_key,
            review_api_key_file=review_api_key_file,
            review_base_url=review_base_url,
            _prepared_inputs=(paper_text, context_block, current_date),
        )

    # ========== Stage 3: Review ==========
    log_info("🔍 Stage 3/3: Full-script external review...")
    _log_length_check(all_transcripts, word_count, "Draft")
    all_transcripts = _review_transcript_with_external_model(
        paper_text=paper_text,
        context_block=context_block,
        entries=all_transcripts,
        review_provider=review_provider,
        review_model=review_model,
        review_enabled=review_enabled,
        review_api_key=review_api_key,
        review_api_key_file=review_api_key_file,
        review_base_url=review_base_url,
    )
    _log_length_check(all_transcripts, word_count, "Reviewed")

    total_chars = sum(len(e["dialog"]) for e in all_transcripts)
    log_info(f"✅ Multi-stage script complete: {len(all_transcripts)} turns, {total_chars} chars")
    return {"podcast_transcripts": all_transcripts}
