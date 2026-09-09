"""Concatenate per-segment MP3s into the final podcast MP3 via ffmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .runtime import abort, log_info, log_warn
from .validation import ensure_file


def _replace_or_copy(source: Path, target: Path) -> None:
    """Move source to target, falling back to copy across filesystems."""
    try:
        source.replace(target)
    except OSError:
        shutil.copyfile(source, target)
        source.unlink(missing_ok=True)


def _make_silence_mp3(path: Path, duration_ms: int) -> None:
    """Create a short mono MP3 silence clip for safe turn spacing."""
    duration_s = max(0.001, duration_ms / 1000)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{duration_s:.3f}",
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            str(path),
        ],
        capture_output=True,
        timeout=30,
        text=True,
    )
    if result.returncode != 0:
        abort("file-write", f"ffmpeg silence generation failed: {result.stderr[:500]}")
    ensure_file(str(path), "file-write", "inter-turn silence MP3")


def concat_segments(segment_files, output_file, temp_dir: str | Path | None = None, gap_ms: int = 0):
    """Concatenate MP3 segments into the final output file, optionally inserting fixed gaps."""
    valid: list[Path] = []
    invalid: list[str] = []
    for idx, segment_file in enumerate(segment_files):
        if not segment_file:
            invalid.append(f"segment {idx + 1}: missing path")
            continue
        path = Path(segment_file)
        if not path.exists():
            invalid.append(f"segment {idx + 1}: file not found ({segment_file})")
            continue
        if path.stat().st_size <= 0:
            invalid.append(f"segment {idx + 1}: file size is 0 ({segment_file})")
            continue
        valid.append(path.resolve())

    if invalid:
        abort("file-write", f"Cannot concatenate audio because some segments are invalid: {'; '.join(invalid)}")
    if not valid:
        abort("file-write", "No audio segments to concatenate")

    segment_dirs = {path.parent for path in valid}
    if len(segment_dirs) != 1:
        abort(
            "file-write",
            "Refusing to concatenate segments from multiple directories: "
            + ", ".join(str(path) for path in sorted(segment_dirs)),
        )

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(temp_dir) if temp_dir else target.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    target_tmp = temp_parent / f".{target.name}.tmp.{os.getpid()}.mp3"

    if len(valid) == 1:
        target_tmp.write_bytes(valid[0].read_bytes())
        _replace_or_copy(target_tmp, target)
    else:
        segment_dir = next(iter(segment_dirs))
        filelist_path = segment_dir / f"concat_{os.getpid()}_{int(time.time() * 1000)}.txt"
        silence_path: Path | None = None
        concat_result = None
        try:
            if gap_ms > 0:
                silence_path = segment_dir / f"silence_{gap_ms}ms_{os.getpid()}_{int(time.time() * 1000)}.mp3"
                _make_silence_mp3(silence_path, gap_ms)
                log_info(f"🔇 Inserting {gap_ms}ms silence between podcast turns")

            with filelist_path.open("w", encoding="utf-8") as f:
                for idx, segment_path in enumerate(valid):
                    escaped_path = str(segment_path).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped_path}'\n")
                    if silence_path is not None and idx < len(valid) - 1:
                        escaped_silence = str(silence_path).replace("\\", "\\\\").replace("'", "\\'")
                        f.write(f"file '{escaped_silence}'\n")

            for attempt in range(1, 3):
                concat_result = subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(filelist_path), "-c:a", "libmp3lame", "-b:a", "128k", str(target_tmp)],
                    capture_output=True,
                    timeout=300,
                    text=True,
                )
                if concat_result.returncode == 0:
                    break
                if attempt < 2:
                    log_warn(f"⚠️ ffmpeg concat failed on attempt {attempt}/2: {concat_result.stderr[:300]}")
                    time.sleep(attempt * 2)
        finally:
            filelist_path.unlink(missing_ok=True)
            if silence_path is not None:
                silence_path.unlink(missing_ok=True)

        if concat_result is None or concat_result.returncode != 0:
            target_tmp.unlink(missing_ok=True)
            abort("file-write", f"ffmpeg concat failed: {(concat_result.stderr if concat_result else '')[:500]}")
        _replace_or_copy(target_tmp, target)

    ensure_file(output_file, "file-write", "final podcast MP3")

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", output_file],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(result.stdout.strip().split("=")[1])
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        log_info(f"🎉 Final podcast: {duration:.1f}s ({duration / 60:.1f}min), {size_mb:.1f}MB")
    except (IndexError, ValueError):
        log_info(f"🎉 Final podcast saved to: {output_file}")

    return output_file
