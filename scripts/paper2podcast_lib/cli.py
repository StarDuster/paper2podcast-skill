"""CLI entry-point: argparse + the run_pipeline orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from .audio import concat_segments
from .config import get_api_runtime, normalize_gemini_model
from .input_parse import load_input
from .provenance import review_receipt, review_matches
from .prompts import _build_tts_header, speaker_name_for
from .runtime import (
    LOGGER,
    PipelineError,
    abort,
    begin_stage,
    configure_logging,
    create_run_work_dir,
    emit_final_summary,
    get_run_context,
    log_error,
    log_info,
    record_degradation,
    reset_run_context,
)
from .script import (
    _NO_CONTEXT_BLOCK,
    validate_review_route,
    _review_transcript_with_external_model,
    generate_script,
    generate_script_multistage,
)
from .tts import (
    _entry_bytes,
    build_single_turn_tts_text,
    build_tts_text,
    run_tts_async,
    split_transcript,
)
from .validation import ensure_file, ensure_non_empty_text, validate_transcript_entries, write_json_file


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper → Podcast Pipeline")
    p.add_argument("input", help="PDF file, text file, URL, or '-' for stdin")
    p.add_argument("--lang", default="zh", choices=["zh"], help="Language (Chinese only; default: zh)")
    p.add_argument("--duration", type=int, default=10, help="Target duration in minutes (default: 10)")
    p.add_argument("--voice-a", default="Kore", help="Voice for speaker 0/Alice (default: Kore)")
    p.add_argument("--voice-b", default="Charon", help="Voice for speaker 1/Bob (default: Charon)")
    p.add_argument("--script-model", default="gemini-3.1-pro-preview", help="Model for script generation (default: gemini-3.1-pro-preview)")
    p.add_argument("--tts-model", default="gemini-3.8-flash-tts", help="TTS model (default: gemini-3.8-flash-tts)")
    p.add_argument("--output", help="Output MP3 path")
    p.add_argument("--script-only", action="store_true", help="Only generate script")
    p.add_argument("--script", help="Use existing script JSON file")
    p.add_argument(
        "--review-provider",
        default="deepseek",
        help="Non-Gemini provider used to review and revise generated scripts (default: deepseek)",
    )
    p.add_argument(
        "--review-model",
        default="deepseek-v4-pro",
        help="Non-Gemini model used to review and revise generated scripts (default: deepseek-v4-pro)",
    )
    p.add_argument(
        "--no-script-review",
        action="store_false",
        dest="script_review",
        help="Skip external review/revision of generated scripts",
    )
    p.add_argument("--review-api-key", default=None, help="Explicit review model API key; prefer --review-api-key-file for shells")
    p.add_argument("--review-api-key-file", default=None, help="File containing the explicit review model API key")
    p.add_argument("--review-base-url", default=None, help="Explicit OpenAI-compatible base URL for the review model")
    p.add_argument("--review-source", default=None, help="File/URL the reviewer compares the script against (default: the positional input). Use with --script when the input is a URL whose page text is poor background")
    p.add_argument("--max-segment-bytes", type=int, default=2800, help="Max bytes per TTS segment (default: 2800)")
    p.add_argument("--workers", type=int, default=2, help="Parallel TTS workers")
    p.add_argument(
        "--turn-gap-ms",
        type=int,
        default=350,
        help="Silence inserted between per-turn TTS clips before final concat (default: 350ms)",
    )
    p.add_argument(
        "--tts-render-mode",
        choices=["per-turn", "multi-speaker"],
        default="per-turn",
        help="TTS rendering mode: per-turn forces one voiceConfig per transcript turn (default)",
    )
    p.add_argument(
        "--work-dir",
        default="",
        help="Directory for this run's temporary files (default: /tmp/paper2podcast_runs/<run_id>)",
    )
    p.add_argument("--skip-search", action="store_true", help="Skip background context search")
    p.add_argument("--search-backend", choices=["tavily-agent", "gemini"], default="tavily-agent", help="Background research backend (default: tavily-agent)")
    p.add_argument("--search-model", default=None, help="Independent research model (or SEARCH_MODEL); default gpt-6-astra with independent Codex")
    p.add_argument("--search-provider", default=None, help="Independent research provider (or SEARCH_PROVIDER); never inherits the global model route")
    p.add_argument("--search-model-api-key", default=None, help="Explicit search model API key; prefer --search-model-api-key-file for shells")
    p.add_argument("--search-model-api-key-file", default=None, help="File containing the explicit search model API key")
    p.add_argument("--search-model-base-url", default=None, help="Explicit OpenAI-compatible base URL for the search model")
    p.add_argument("--search-use-hermes", action="store_true", help="Resolve the search model through Hermes instead of standalone config")
    p.add_argument("--no-multistage", action="store_false", dest="multistage", help="Disable multi-stage pipeline (Outline -> Write -> Review)")
    p.set_defaults(multistage=True, script_review=True)
    p.add_argument(
        "--provider",
        choices=["gemini", "vertex"],
        default="gemini",
        help="Generation/TTS provider: gemini uses the Google Generative Language API; vertex is an explicit Vertex AI path (default: gemini)",
    )
    p.add_argument("--api-key", help="Gemini API key (only used with --provider gemini)")
    p.add_argument("--api-key-file", help="File containing Gemini API key (only used with --provider gemini)")
    p.add_argument(
        "--vertex-credentials-file",
        default="",
        help="Override Vertex service-account JSON file (default: Hermes vertex.credentials rotation)",
    )
    p.add_argument("--log-file", default="", help="Write detailed debug logs (default: <work-dir>/paper2podcast.log)")
    return p


def _extract_input_slug(input_path: str) -> str:
    """Extract a meaningful filename slug from the input path or URL."""
    if input_path == "-":
        return "stdin"

    # ArXiv detection
    # https://arxiv.org/pdf/2405.12305.pdf -> 2405.12305
    # https://arxiv.org/abs/2405.12305 -> 2405.12305
    arxiv_match = re.search(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?", input_path)
    if arxiv_match:
        return f"arxiv_{arxiv_match.group(1)}"

    if input_path.startswith(("http://", "https://")):
        # General URL: take the last part of the path, or hostname
        try:
            parsed = urlparse(input_path)
            path_parts = [p for p in parsed.path.split("/") if p]
            if path_parts:
                slug = path_parts[-1].split(".")[0]
                if slug:
                    return slug
            return parsed.netloc.replace(".", "_")
        except Exception:
            return "url_podcast"

    # Local file
    return Path(input_path).stem


def _resolve_output_path(args, work_dir: Path) -> str:
    if args.output:
        return args.output
    base_name = _extract_input_slug(args.input)
    return str(work_dir / f"{base_name}.mp3")


def _validate_cli_args(args) -> None:
    if args.duration <= 0:
        abort("config", f"--duration must be > 0, got {args.duration}")
    if args.script_review and not str(args.review_provider or "").strip():
        abort("config", "--review-provider must be non-empty when script review is enabled")
    if args.script_review and not str(args.review_model or "").strip():
        abort("config", "--review-model must be non-empty when script review is enabled")
    if args.script_review:
        validate_review_route(args.review_provider, args.review_model,
                              args.review_api_key, args.review_api_key_file,
                              args.review_base_url)
    if args.max_segment_bytes <= 0:
        abort("config", f"--max-segment-bytes must be > 0, got {args.max_segment_bytes}")
    if args.workers <= 0:
        abort("config", f"--workers must be > 0, got {args.workers}")
    if args.turn_gap_ms < 0:
        abort("config", f"--turn-gap-ms must be >= 0, got {args.turn_gap_ms}")


def _load_review_source(args) -> str:
    """Load the background text used to judge a `--script` draft.

    Prefers `--review-source` (e.g. the PDF-derived text or a local dump);
    falls back to the positional input. URL inputs are re-fetched only when
    no explicit source is given.
    """
    explicit = str(getattr(args, "review_source", "") or "").strip()
    candidate = explicit or args.input
    try:
        paper_text = load_input(candidate)
    except Exception as exc:
        abort("script-review", f"Failed to load review source {candidate}: {type(exc).__name__}: {exc}", cause=exc)
    paper_text = ensure_non_empty_text("script-review", paper_text, "review source text")
    log_info(f"📚 Review source: {candidate} ({len(paper_text)} chars)")
    return paper_text


def _review_supplied_script(args, script: dict) -> None:
    """Fail closed; reuse only an exact content/source/route-bound receipt."""
    paper_text = _load_review_source(args)
    base_url = args.review_base_url or os.getenv("REVIEW_MODEL_BASE_URL", "")
    if review_matches(script.get("review"), script["podcast_transcripts"], paper_text,
                      args.review_provider, args.review_model, base_url):
        log_info("✅ Script review reused: content, source and reviewer match")
        return
    begin_stage("script-review", "reviewing supplied script")
    script["podcast_transcripts"] = _review_transcript_with_external_model(
        paper_text=paper_text, context_block=_NO_CONTEXT_BLOCK,
        entries=script["podcast_transcripts"],
        review_provider=args.review_provider, review_model=args.review_model,
        review_enabled=True, review_api_key=args.review_api_key,
        review_api_key_file=args.review_api_key_file, review_base_url=args.review_base_url,
    )
    script["review"] = review_receipt(script["podcast_transcripts"], paper_text,
                                      args.review_provider, args.review_model, base_url)

def _load_or_generate_script(args, api_runtime, script_path: str):
    """Either load an existing script JSON, or run the generation pipeline."""
    if args.script:
        log_info(f"📄 Loading existing script: {args.script}")
        get_run_context().script_path = args.script
        try:
            script = json.loads(Path(args.script).read_text(encoding="utf-8"))
        except Exception as exc:
            abort("input-parse", f"Failed to load script file {args.script}: {type(exc).__name__}: {exc}", cause=exc)
        if isinstance(script, list):
            script = {"podcast_transcripts": script}
        if not isinstance(script, dict):
            abort("input-parse", f"Script file did not contain a JSON object: {args.script}")
        script["podcast_transcripts"] = validate_transcript_entries(
            script.get("podcast_transcripts", []),
            "input-parse existing script",
        )
        # An externally supplied script still gets reviewed when review is on:
        # `--script` exists to bypass brittle *generation*, not to skip the
        # non-Gemini quality gate. Background comes from --review-source/input.
        if args.script_review and args.review_provider.strip():
            _review_supplied_script(args, script)
        else:
            script["review"] = {"status": "skipped", "reviewed": False,
                                "reason": "explicit --no-script-review"}
        return script

    paper_text = load_input(args.input)
    paper_text = ensure_non_empty_text("input-parse", paper_text, "parsed input text")
    log_info(f"📄 Input: {len(paper_text)} chars")
    generator = generate_script_multistage if args.multistage else generate_script
    script = generator(
        api_runtime,
        paper_text,
        args.lang,
        args.duration,
        args.script_model,
        args.skip_search,
        args.tts_model,
        review_provider=args.review_provider,
        review_model=args.review_model,
        review_enabled=args.script_review,
        review_api_key=args.review_api_key,
        review_api_key_file=args.review_api_key_file,
        review_base_url=args.review_base_url,
        search_backend=args.search_backend,
        search_model=args.search_model,
        search_provider=args.search_provider,
        search_model_api_key=args.search_model_api_key,
        search_model_api_key_file=args.search_model_api_key_file,
        search_model_base_url=args.search_model_base_url,
        search_use_hermes=args.search_use_hermes,
    )

    script["review"] = (
        review_receipt(script["podcast_transcripts"], paper_text, args.review_provider,
                       args.review_model, args.review_base_url or os.getenv("REVIEW_MODEL_BASE_URL", ""))
        if args.script_review else {"status": "skipped", "reviewed": False,
                                   "reason": "explicit --no-script-review"}
    )
    return script


def _build_segments(args, entries) -> list[list[dict]]:
    if args.tts_render_mode == "per-turn":
        segments = [[entry] for entry in entries]
        for i, seg in enumerate(segments):
            speaker = speaker_name_for(seg[0]["speaker_id"])
            seg_bytes = len(build_single_turn_tts_text(seg[0], args.lang).encode("utf-8"))
            log_info(f"  📦 Turn {i+1}/{len(segments)}: {seg_bytes} bytes [{speaker}]")
    else:
        segments = split_transcript(entries, args.max_segment_bytes, args.lang, args.tts_model)

    # If multi-speaker collapsed into a single oversized segment, force a resplit.
    # Large single-shot TTS calls can hang or time out due to oversized audio payloads.
    if args.tts_render_mode == "multi-speaker" and len(segments) == 1:
        single_bytes = len(build_tts_text(segments[0], args.lang, tts_model=args.tts_model).encode("utf-8"))
        if single_bytes >= 3500:
            base_bytes = len(_build_tts_header("middle").encode("utf-8")) + 100  # padding
            target_segments = max(2, args.duration)
            total_dialog_bytes = sum(_entry_bytes(e) for e in entries)
            target_payload = max(900, (total_dialog_bytes + target_segments - 1) // target_segments)
            new_max_bytes = base_bytes + target_payload
            if new_max_bytes < args.max_segment_bytes:
                record_degradation(
                    "tts-audio-synthesis",
                    f"single TTS segment too large ({single_bytes} bytes)",
                    f"re-split into ~{target_segments} segments with max {new_max_bytes} bytes",
                )
                segments = split_transcript(entries, new_max_bytes, args.lang, args.tts_model)

    if not segments:
        abort("tts-audio-synthesis", "Transcript split produced no TTS segments")
    return segments


def _render_segments(args, api_runtime, segments, output_dir: str) -> list[str | None]:
    """Run parallel TTS, then retry failed indexes serially."""
    log_info(f"⚙️ Running async TTS with {max(1, args.workers)} workers...")
    segment_files = asyncio.run(
        run_tts_async(
            api_runtime, segments, output_dir,
            args.lang, args.voice_a, args.voice_b, args.tts_model,
            workers=args.workers,
            render_mode=args.tts_render_mode,
        )
    )

    failed = [i for i, segment_file in enumerate(segment_files) if not segment_file]
    if failed and args.workers > 1:
        record_degradation(
            "tts-audio-synthesis",
            f"segments failed with parallel workers={args.workers}: {[i + 1 for i in failed]}",
            "retry failed segments serially with workers=1",
        )
        retried = asyncio.run(
            run_tts_async(
                api_runtime, segments, output_dir,
                args.lang, args.voice_a, args.voice_b, args.tts_model,
                indexes=failed,
                render_mode=args.tts_render_mode,
            )
        )
        for idx, segment_file in retried.items():
            segment_files[idx] = segment_file
        failed = [i for i, segment_file in enumerate(segment_files) if not segment_file]

    if failed:
        abort(
            "tts-audio-synthesis",
            f"TTS failed for segments {[i + 1 for i in failed]}; no partial podcast will be emitted",
        )
    return segment_files


def main() -> int:
    reset_run_context()
    exit_code = 0
    args = _build_argparser().parse_args()

    try:
        work_dir = create_run_work_dir(args.work_dir or None)
        resolved_log = args.log_file or str(work_dir / "paper2podcast.log")
        configure_logging(resolved_log)
        get_run_context().log_path = resolved_log
        log_info(f"Log file: {resolved_log}")
        log_info(f"🗂️ Run workspace: {work_dir}")

        begin_stage("config", "validating CLI arguments and model config")
        _validate_cli_args(args)
        args.script_model = normalize_gemini_model(args.script_model)
        args.tts_model = normalize_gemini_model(args.tts_model)
        if args.tts_model == "gemini-3.1-flash-tts-preview":
            args.tts_model = "gemini-3.8-flash-tts"
            log_info("🔁 Using gemini-3.8-flash-tts for legacy Flash TTS selection")

        needs_api = not (args.script and args.script_only)
        api_runtime = get_api_runtime(args) if needs_api else None

        output_path = _resolve_output_path(args, work_dir)
        script_path = str(work_dir / "final_script.json")
        if args.script and Path(args.script).resolve() == Path(script_path).resolve():
            import uuid
            script_path = str(work_dir / f"final_script_{uuid.uuid4().hex}.json")
        get_run_context().output_path = output_path
        get_run_context().script_path = script_path

        begin_stage("input-parse", f"source={args.input}")
        script = _load_or_generate_script(args, api_runtime, script_path)

        # One atomic checkpoint for every path, before any TTS request.
        begin_stage("file-write", "persisting exact final transcript before TTS")
        written = write_json_file(script_path, script, "file-write", "final script JSON")
        get_run_context().script_path = written
        if args.script_only:
            log_info(f"📝 Script-only mode, done: {written}")
            return exit_code

        entries = validate_transcript_entries(script.get("podcast_transcripts", []), "tts-audio-synthesis")

        begin_stage("tts-audio-synthesis", f"preparing TTS workers={args.workers} render_mode={args.tts_render_mode}")
        segments = _build_segments(args, entries)
        log_info(f"📦 Split into {len(segments)} TTS segments ({args.tts_render_mode})")

        tmpdir = work_dir / "segments"
        tmpdir.mkdir(parents=True, exist_ok=True)
        log_info(f"🧹 Segment workspace: {tmpdir}")
        try:
            segment_files = _render_segments(args, api_runtime, segments, str(tmpdir))
            begin_stage("file-write", "writing final podcast MP3")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            gap_ms = args.turn_gap_ms if args.tts_render_mode == "per-turn" else 0
            concat_segments(segment_files, output_path, temp_dir=work_dir, gap_ms=gap_ms)
            ensure_file(output_path, "file-write", "final podcast MP3")
        finally:
            begin_stage("cleanup", f"keeping run workspace {work_dir}")

        log_info(f"📁 Output: {output_path}")
        return exit_code
    except PipelineError as exc:
        exit_code = exc.exit_code
        get_run_context().failed_stage = exc.stage
        return exit_code
    except KeyboardInterrupt:
        exit_code = 130
        ctx = get_run_context()
        ctx.failed_stage = ctx.failed_stage or ctx.current_stage or "interrupted"
        log_error("❌ [interrupt] Interrupted by user")
        return exit_code
    except Exception as exc:
        exit_code = 1
        ctx = get_run_context()
        stage = ctx.current_stage or "unknown"
        ctx.failed_stage = ctx.failed_stage or stage
        LOGGER.exception("Unhandled exception in stage %s", stage)
        log_error(f"❌ [{stage}] Unhandled exception: {type(exc).__name__}: {exc}")
        return exit_code
    finally:
        emit_final_summary(exit_code == 0, exit_code)


if __name__ == "__main__":
    sys.exit(main())
