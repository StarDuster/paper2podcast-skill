"""CLI entry-point: argparse + the run_pipeline orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .audio import concat_segments
from .config import get_api_key, normalize_gemini_model
from .input_parse import load_input
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
from .script import generate_script, generate_script_multistage
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
    p.add_argument("--tts-model", default="gemini-2.5-pro-preview-tts", help="TTS model (default: gemini-2.5-pro-preview-tts)")
    p.add_argument("--output", help="Output MP3 path")
    p.add_argument("--script-only", action="store_true", help="Only generate script")
    p.add_argument("--script", help="Use existing script JSON file")
    p.add_argument("--max-segment-bytes", type=int, default=2800, help="Max bytes per TTS segment (default: 2800)")
    p.add_argument("--workers", type=int, default=2, help="Parallel TTS workers")
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
    p.add_argument("--no-multistage", action="store_false", dest="multistage", help="Disable multi-stage pipeline (Outline -> Write -> Review)")
    p.set_defaults(multistage=True)
    p.add_argument("--api-key", help="Gemini API key")
    p.add_argument("--api-key-file", help="File containing Gemini API key")
    p.add_argument("--log-file", default="", help="Write detailed debug logs (default: <work-dir>/paper2podcast.log)")
    return p


def _resolve_output_path(args, work_dir: Path) -> str:
    if args.output:
        return args.output
    if args.input == "-":
        base_name = "stdin_podcast"
    elif args.input.startswith("http"):
        base_name = "url_podcast"
    else:
        base_name = Path(args.input).stem + "_podcast"
    return str(work_dir / f"{base_name}.mp3")


def _validate_cli_args(args) -> None:
    if args.duration <= 0:
        abort("config", f"--duration must be > 0, got {args.duration}")
    if args.max_segment_bytes <= 0:
        abort("config", f"--max-segment-bytes must be > 0, got {args.max_segment_bytes}")
    if args.workers <= 0:
        abort("config", f"--workers must be > 0, got {args.workers}")


def _load_or_generate_script(args, api_key, script_path: str):
    """Either load an existing script JSON, or run the generation pipeline."""
    if args.script:
        log_info(f"📄 Loading existing script: {args.script}")
        get_run_context().script_path = args.script
        try:
            script = json.loads(Path(args.script).read_text(encoding="utf-8"))
        except Exception as exc:
            abort("input-parse", f"Failed to load script file {args.script}: {type(exc).__name__}: {exc}", cause=exc)
        if not isinstance(script, dict):
            abort("input-parse", f"Script file did not contain a JSON object: {args.script}")
        script["podcast_transcripts"] = validate_transcript_entries(
            script.get("podcast_transcripts", []),
            "input-parse existing script",
        )
        return script

    paper_text = load_input(args.input)
    paper_text = ensure_non_empty_text("input-parse", paper_text, "parsed input text")
    log_info(f"📄 Input: {len(paper_text)} chars")
    generator = generate_script_multistage if args.multistage else generate_script
    script = generator(api_key, paper_text, args.lang, args.duration, args.script_model, args.skip_search, args.tts_model)

    begin_stage("file-write", "writing generated script JSON")
    script_path = write_json_file(script_path, script, "file-write", "script JSON")
    get_run_context().script_path = script_path
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


def _render_segments(args, api_key, segments, output_dir: str) -> list[str | None]:
    """Run parallel TTS, then retry failed indexes serially."""
    log_info(f"⚙️ Running async TTS with {max(1, args.workers)} workers...")
    segment_files = asyncio.run(
        run_tts_async(
            api_key, segments, output_dir,
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
                api_key, segments, output_dir,
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

        needs_api = not (args.script and args.script_only)
        api_key = get_api_key(args) if needs_api else None

        output_path = _resolve_output_path(args, work_dir)
        script_path = output_path.rsplit(".", 1)[0] + "_script.json"
        get_run_context().output_path = output_path
        get_run_context().script_path = script_path

        begin_stage("input-parse", f"source={args.input}")
        script = _load_or_generate_script(args, api_key, script_path)

        if args.script_only:
            log_info("📝 Script-only mode, done.")
            return exit_code

        entries = validate_transcript_entries(script.get("podcast_transcripts", []), "tts-audio-synthesis")

        begin_stage("tts-audio-synthesis", f"preparing TTS workers={args.workers} render_mode={args.tts_render_mode}")
        segments = _build_segments(args, entries)
        log_info(f"📦 Split into {len(segments)} TTS segments ({args.tts_render_mode})")

        tmpdir = work_dir / "segments"
        tmpdir.mkdir(parents=True, exist_ok=True)
        log_info(f"🧹 Segment workspace: {tmpdir}")
        try:
            segment_files = _render_segments(args, api_key, segments, str(tmpdir))
            begin_stage("file-write", "writing final podcast MP3")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            concat_segments(segment_files, output_path, temp_dir=work_dir)
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
