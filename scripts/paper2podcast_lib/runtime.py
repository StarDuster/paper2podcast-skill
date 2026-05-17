"""Run context, staging, logging and fatal-error helpers.

Owns the singleton `RUN_CONTEXT` that every other module reads via
`get_run_context()`. The CLI resets it once per invocation via
`reset_run_context()`.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger("paper2podcast")


class PipelineError(RuntimeError):
    """Fatal pipeline error with stage metadata and exit code."""

    def __init__(self, stage: str, message: str, exit_code: int = 1):
        super().__init__(message)
        self.stage = stage
        self.exit_code = exit_code


@dataclass
class RunContext:
    started_at: float = field(default_factory=time.monotonic)
    current_stage: str | None = None
    stage_started_at: float = field(default_factory=time.monotonic)
    stage_durations: dict[str, float] = field(default_factory=dict)
    failed_stage: str | None = None
    output_path: str | None = None
    script_path: str | None = None
    log_path: str | None = None
    run_id: str | None = None
    work_dir: str | None = None
    degradations: list[str] = field(default_factory=list)


RUN_CONTEXT = RunContext()


def get_run_context() -> RunContext:
    return RUN_CONTEXT


def reset_run_context() -> RunContext:
    """Replace the module-level RUN_CONTEXT with a fresh one (CLI entry-point uses this)."""
    global RUN_CONTEXT
    RUN_CONTEXT = RunContext()
    return RUN_CONTEXT


def log_info(message: str) -> None:
    LOGGER.info(message)


def log_warn(message: str) -> None:
    LOGGER.warning(message)


def log_error(message: str) -> None:
    LOGGER.error(message)


def finalize_current_stage() -> None:
    ctx = get_run_context()
    if not ctx.current_stage:
        return
    elapsed = max(0.0, time.monotonic() - ctx.stage_started_at)
    ctx.stage_durations[ctx.current_stage] = ctx.stage_durations.get(ctx.current_stage, 0.0) + elapsed
    ctx.stage_started_at = time.monotonic()


def begin_stage(stage: str, detail: str = "") -> None:
    finalize_current_stage()
    ctx = get_run_context()
    ctx.current_stage = stage
    ctx.stage_started_at = time.monotonic()
    suffix = f" ({detail})" if detail else ""
    log_info(f"▶️ Stage: {stage}{suffix}")


def record_degradation(stage: str, reason: str, fallback: str) -> None:
    ctx = get_run_context()
    note = f"{stage}: {reason} -> {fallback}"
    ctx.degradations.append(note)
    log_warn(f"⚠️ [{stage}] Degrading because {reason}. Fallback: {fallback}")


def abort(stage: str, message: str, *, exit_code: int = 1, cause: Exception | None = None) -> None:
    ctx = get_run_context()
    ctx.failed_stage = stage
    if cause is not None:
        LOGGER.exception("[%s] %s", stage, message)
    log_error(f"❌ [{stage}] {message}")
    raise PipelineError(stage, message, exit_code=exit_code) from cause


def configure_logging(log_file: str | None = None, verbose: bool = True) -> str:
    """Configure root logging to stdout + optional file. Returns resolved log path or ''."""
    if LOGGER.handlers:
        return log_file or ""

    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.info("Log file: %s", log_path)
        return str(log_path)
    return ""


def make_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def default_runs_root() -> Path:
    return Path(tempfile.gettempdir()) / "paper2podcast_runs"


def create_run_work_dir(work_dir_arg: str | None = None) -> Path:
    ctx = get_run_context()
    ctx.run_id = make_run_id()
    if work_dir_arg:
        base = Path(work_dir_arg).expanduser()
        work_dir = base / ctx.run_id if base.exists() and base.is_dir() else base
    else:
        work_dir = default_runs_root() / ctx.run_id
    work_dir.mkdir(parents=True, exist_ok=False)
    ctx.work_dir = str(work_dir)
    return work_dir


def current_work_dir() -> Path | None:
    work_dir = get_run_context().work_dir
    return Path(work_dir) if work_dir else None


_STAGE_ORDER = (
    "config",
    "input-parse",
    "context-search",
    "outline-generation",
    "segment-generation",
    "tts-audio-synthesis",
    "file-write",
    "cleanup",
)


def emit_final_summary(success: bool, exit_code: int) -> None:
    finalize_current_stage()
    ctx = get_run_context()
    total_elapsed = max(0.0, time.monotonic() - ctx.started_at)

    seen = [
        f"{stage}={ctx.stage_durations[stage]:.1f}s"
        for stage in _STAGE_ORDER
        if stage in ctx.stage_durations
    ]
    for stage, elapsed in ctx.stage_durations.items():
        if stage not in _STAGE_ORDER:
            seen.append(f"{stage}={elapsed:.1f}s")

    logger = log_info if success else log_error
    logger("=== Final Status Summary ===")
    logger(f"status={'success' if success else 'failure'} exit_code={exit_code}")
    logger(f"failed_stage={ctx.failed_stage or '-'} total_elapsed={total_elapsed:.1f}s")
    logger(f"output_path={ctx.output_path or '-'}")
    logger(f"script_path={ctx.script_path or '-'}")
    logger(f"log_path={ctx.log_path or '-'}")
    logger(f"work_dir={ctx.work_dir or '-'}")
    logger(f"stage_timings={', '.join(seen) if seen else '-'}")
    if ctx.degradations:
        logger(f"degradations={'; '.join(ctx.degradations)}")
