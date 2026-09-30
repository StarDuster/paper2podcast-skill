"""JSON-only model adapter. Credentials and synchronous clients stay in a child."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _env_truthy(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _usage_dict(usage):
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "__dict__"):
        return vars(usage)
    return {"available": False}


class Model:
    def __init__(self, model, provider, *, api_key=None, base_url=None, use_hermes=None):
        from tavily_agent import ResearchError

        if not model:
            raise ResearchError("model_required")
        if use_hermes is None:
            use_hermes = _env_truthy("SEARCH_MODEL_USE_HERMES")
        api_keys = json.loads(os.getenv("SEARCH_MODEL_API_KEYS", "[]"))
        if not isinstance(api_keys, list) or any(not isinstance(k, str) or not k.strip() for k in api_keys):
            raise ResearchError("model_invalid_key_pool")
        self.config = {
            "api_keys": api_keys,
            "model": model,
            "provider": provider,
            "api_key": api_key or os.getenv("SEARCH_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": base_url or os.getenv("SEARCH_MODEL_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            "use_hermes": bool(use_hermes),
        }

    async def complete(self, messages, timeout):
        from tavily_agent import ResearchError

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            payload = {"config": self.config, "messages": messages, "timeout": max(1.0, timeout)}
            async with asyncio.timeout(timeout):
                stdout, _ = await process.communicate(json.dumps(payload).encode())
            data = json.loads(stdout)
            if "error" in data:
                code = data["error"]
                if data.get("error_type"):
                    code += " [" + data["error_type"] + "]"
                if data.get("error_detail"):
                    code += ": " + data["error_detail"]
                raise ResearchError(code)
            return data["result"], data["usage"]
        except (json.JSONDecodeError, KeyError):
            raise ResearchError("model_invalid_response") from None
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


def _codex_exec(model, messages, timeout):
    """Run the Codex CLI directly; bypass Hermes' auxiliary-client total timeout."""
    parts = [m.get("content", "") or "" for m in messages if (m.get("content") or "").strip()]
    prompt = "\n\n".join(parts)
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "result.json")
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-m",
            model,
            "-s",
            "read-only",
            "-c",
            "model_reasoning_effort=low",
            "-o",
            out,
            "-",
        ]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            return {
                "error": "model_request_failed",
                "error_type": "TimeoutExpired",
                "error_detail": "codex exec exceeded deadline",
            }
        try:
            text = Path(out).read_text(encoding="utf-8").strip()
        except OSError:
            return {
                "error": "model_request_failed",
                "error_type": "MissingOutput",
                "error_detail": "codex produced no last message",
            }
        if not text:
            return {
                "error": "model_request_failed",
                "error_type": "EmptyOutput",
                "error_detail": "codex last message empty (exit=%s)" % proc.returncode,
            }
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        if not isinstance(result, dict):
            return {"error": "model_invalid_json"}
        return {"result": result, "usage": {"available": False}}


def _openai_compatible_complete(config, payload):
    from urllib.parse import urlsplit
    endpoint = urlsplit(config.get("base_url") or "")
    keys = list(dict.fromkeys(k for k in [config.get("api_key"), *config.get("api_keys", [])] if k))
    if (config["model"].startswith("gemini-")
            and endpoint.scheme == "https"
            and endpoint.netloc == "generativelanguage.googleapis.com"
            and endpoint.path.rstrip("/") == "/v1beta/openai" and keys):
        import time
        deadline = time.monotonic() + payload["timeout"]
        attempts = []
        for index, key in enumerate(keys, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Gemini key rotation exceeded deadline")
            try:
                result = _openai_single_complete({**config, "api_key": key}, {**payload, "timeout": remaining})
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                attempts.append({"key_index": index, "status": status or type(exc).__name__})
                if status != 503:
                    raise
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Gemini key rotation exceeded deadline")
            attempts.append({"key_index": index, "status": 200})
            result.setdefault("usage", {})["key_attempts"] = attempts
            return result
        return {"error": "model_request_failed", "error_type": "InternalServerError",
                "error_detail": "Gemini key pool exhausted: " + json.dumps(attempts), "key_attempts": attempts}
    return _openai_single_complete(config, payload)


def _openai_single_complete(config, payload):
    from openai import OpenAI

    client = OpenAI(
        api_key=config.get("api_key") or ("not-needed" if config.get("base_url") else None),
        base_url=config.get("base_url") or None,
        max_retries=0,
        timeout=payload["timeout"],
    )
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=payload["messages"],
            timeout=payload["timeout"],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        result = json.loads(text)
        if not isinstance(result, dict):
            return {"error": "model_invalid_json"}
        return {"result": result, "usage": _usage_dict(response.usage)}
    finally:
        client.close()


def _hermes_complete(config, payload):
    from agent.auxiliary_client import resolve_provider_client

    client, model = resolve_provider_client(config["provider"], model=config["model"])
    if client is None:
        return {"error": "model_credentials_unavailable"}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=payload["messages"],
            timeout=payload["timeout"],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        result = json.loads(text)
        if not isinstance(result, dict):
            return {"error": "model_invalid_json"}
        return {"result": result, "usage": _usage_dict(response.usage)}
    finally:
        client.close()


def worker(payload):
    config = payload["config"]
    provider = config["provider"]
    model = config["model"]
    try:
        if provider == "openai-codex":
            return _codex_exec(model, payload["messages"], payload["timeout"])
        if provider in ("portable", "openai-compatible"):
            if not config.get("api_key") and not config.get("base_url"):
                return {
                    "error": "model_credentials_unavailable",
                    "error_type": "ConfigurationError",
                    "error_detail": "set SEARCH_MODEL_API_KEY or SEARCH_MODEL_BASE_URL for OpenAI-compatible mode",
                }
            return _openai_compatible_complete(config, payload)
        if config.get("base_url"):
            return _openai_compatible_complete(config, payload)
        if config.get("use_hermes"):
            return _hermes_complete(config, payload)
        return {
            "error": "model_credentials_unavailable",
            "error_type": "ConfigurationError",
            "error_detail": "set SEARCH_MODEL_BASE_URL, use --provider openai-codex, or pass --use-hermes",
        }
    except json.JSONDecodeError:
        return {"error": "model_invalid_json"}
    except Exception as exc:
        detail = str(exc)
        for secret in (config.get("api_key"), config.get("base_url"), *config.get("api_keys", [])):
            if secret:
                detail = detail.replace(str(secret), "<redacted>")
        return {
            "error": "model_request_failed",
            "error_type": type(exc).__name__,
            "error_detail": detail[:500],
        }


if __name__ == "__main__":
    # Keep resolver diagnostics out of the JSON wire protocol and away from logs.
    payload = json.load(sys.stdin)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = worker(payload)
    print(json.dumps(result))
