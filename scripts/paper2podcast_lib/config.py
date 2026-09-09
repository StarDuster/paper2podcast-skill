"""Provider/runtime resolution and Gemini model-name normalization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .runtime import abort, log_info

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:  # pragma: no cover - dependency check at runtime
    google = None
    service_account = None

try:
    from hermes_cli.runtime_provider import resolve_runtime_provider
except Exception:  # pragma: no cover - checked when Vertex is requested
    resolve_runtime_provider = None


@dataclass
class ApiRuntime:
    provider: str
    api_key: str = ""
    base_url: str = ""
    credentials_file: str = ""
    access_token: str = ""


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


def _mint_vertex_access_token(credentials_file: str) -> str:
    if google is None or service_account is None:
        abort("config", "Vertex provider requires google-auth. Install it in the Hermes venv.")
    try:
        if credentials_file:
            creds = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        return str(creds.token or "")
    except Exception as exc:
        abort("config", f"Failed to initialize Vertex credentials: {type(exc).__name__}: {exc}", cause=exc)


def get_api_runtime(args) -> ApiRuntime:
    provider = str(getattr(args, "provider", "") or "gemini").strip().lower()
    if provider == "gemini":
        explicit_key = str(getattr(args, "api_key", "") or "").strip()
        explicit_key_file = str(getattr(args, "api_key_file", "") or "").strip()
        if explicit_key or explicit_key_file:
            return ApiRuntime(provider="gemini", api_key=get_api_key(args))

        # Prefer the Hermes Gemini credential pool so AQ.* AI Studio keys and
        # future pool entries are used by the standalone podcast pipeline too.
        if resolve_runtime_provider is not None:
            try:
                runtime = resolve_runtime_provider(
                    requested="gemini",
                    target_model=getattr(args, "script_model", "") or None,
                )
                api_key = str(runtime.get("api_key") or "").strip()
                if api_key:
                    return ApiRuntime(provider="gemini", api_key=api_key)
            except Exception:
                pass

        # Standalone fallback when the Hermes runtime/pool is unavailable.
        return ApiRuntime(provider="gemini", api_key=get_api_key(args))
    if provider != "vertex":
        abort("config", f"Unsupported provider: {provider}. Expected 'vertex' or 'gemini'.")

    if resolve_runtime_provider is None:
        abort("config", "Failed to import Hermes Vertex runtime resolver")

    try:
        runtime = resolve_runtime_provider(
            requested="vertex",
            target_model=getattr(args, "script_model", "") or None,
        )
    except Exception as exc:
        abort("config", f"Failed to resolve Vertex runtime: {type(exc).__name__}: {exc}", cause=exc)

    base_url = str(runtime.get("base_url") or "").rstrip("/")
    credentials_file = str(getattr(args, "vertex_credentials_file", "") or runtime.get("credentials_file") or "")
    if not base_url:
        abort("config", "Vertex runtime did not return a base_url")
    access_token = _mint_vertex_access_token(credentials_file)
    if not access_token:
        abort("config", "Vertex access token is empty")
    log_info(f"🔐 Using Vertex provider: base_url={base_url} credentials_file={credentials_file or 'ADC'}")
    return ApiRuntime(
        provider="vertex",
        base_url=base_url,
        credentials_file=credentials_file,
        access_token=access_token,
    )


GEMINI_MODEL_ALIASES = {
    # Human-friendly shorthands
    "gemini": "gemini-3.1-pro-preview",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-3.5-flash",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    # Common shortened raw Gemini names that the v1beta API does not accept directly
    "gemini-3-flash": "gemini-3.5-flash",
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
