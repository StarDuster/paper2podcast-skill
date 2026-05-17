"""API-key resolution and Gemini model-name normalization."""

from __future__ import annotations

import os
from pathlib import Path

from .runtime import abort, log_info


def get_api_key(args):
    """Resolve API key from args, env, or default file."""
    if args.api_key:
        api_key = args.api_key.strip()
        if api_key:
            return api_key
        abort("config", "--api-key was provided but is empty")
    if args.api_key_file:
        try:
            api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        except Exception as exc:
            abort("config", f"Failed to read API key file {args.api_key_file}: {type(exc).__name__}: {exc}", cause=exc)
        if api_key:
            return api_key
        abort("config", f"API key file is empty: {args.api_key_file}")
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and env_key.startswith("AIza"):
        return env_key
    default_path = Path.home() / ".hermes" / "secrets" / "gemini_api_key.txt"
    if default_path.exists():
        try:
            api_key = default_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            abort("config", f"Failed to read default API key file {default_path}: {type(exc).__name__}: {exc}", cause=exc)
        if api_key:
            return api_key
        abort("config", f"Default API key file is empty: {default_path}")
    abort("config", "No API key found. Set GEMINI_API_KEY or use --api-key / --api-key-file")


GEMINI_MODEL_ALIASES = {
    # Human-friendly shorthands
    "gemini": "gemini-3.1-pro-preview",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    # Common shortened raw Gemini names that the v1beta API does not accept directly
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini 3.1 pro preview": "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
}


_KNOWN_PROVIDER_PREFIXES = {
    "gemini",
    "google",
    "google-ai",
    "googleai",
    "generativelanguage",
    "generative-language",
}


def normalize_gemini_model(model_name: str) -> str:
    """Map friendly/alias Gemini model names to concrete API model IDs."""
    if model_name is None:
        abort("config", "Model name is missing")

    raw = str(model_name).strip().strip("'").strip('"')
    if not raw:
        abort("config", "Model name is empty after trimming")

    normalized = raw
    lowered = normalized.lower()
    if lowered.startswith("models/"):
        normalized = normalized.split("/", 1)[1].strip()
        log_info(f"🔁 Normalize model path prefix: {raw} -> {normalized}")

    for separator in (":", "/"):
        if separator in normalized:
            prefix, candidate = normalized.split(separator, 1)
            if prefix.strip().lower() in _KNOWN_PROVIDER_PREFIXES and candidate.strip():
                before = normalized
                normalized = candidate.strip()
                log_info(f"🔁 Normalize provider-prefixed model: {before} -> {normalized}")
                break

    alias_key = normalized.lower()
    normalized = GEMINI_MODEL_ALIASES.get(alias_key, GEMINI_MODEL_ALIASES.get(normalized, normalized))
    if normalized != raw:
        log_info(f"🔁 Normalize model alias: {raw} -> {normalized}")
    return normalized
