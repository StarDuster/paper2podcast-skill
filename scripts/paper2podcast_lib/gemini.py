"""Gemini API clients (sync + async) and the grounded paper-context search."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import ApiRuntime
from .runtime import abort, begin_stage, log_error, log_info, log_warn, record_degradation
from .validation import extract_text_from_gemini_result

try:
    from hermes_cli.runtime_provider import resolve_runtime_provider
except Exception:
    resolve_runtime_provider = None

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None
    service_account = None

try:
    import aiohttp
except ImportError:  # pragma: no cover - dependency check at runtime
    aiohttp = None


def _gemini_json_body(prompt: str, *, max_tokens: int, temperature: float) -> dict[str, Any]:
    """Standard JSON-response body used by every script-generation call."""
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }


def _get_vertex_fallback_runtime() -> ApiRuntime | None:
    """Lazy-resolve a Vertex runtime when Gemini API keys are exhausted."""
    if resolve_runtime_provider is None or google is None:
        log_warn("⚠️ Vertex fallback unavailable: missing hermes_cli or google-auth")
        return None
    try:
        runtime = resolve_runtime_provider(requested="vertex", target_model=None)
        base_url = str(runtime.get("base_url") or "").rstrip("/")
        if not base_url:
            return None
        credentials_file = str(runtime.get("credentials_file") or "")
        if credentials_file:
            creds = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        access_token = str(creds.token or "")
        if not access_token:
            return None
        log_info("🔁 Fallback: Gemini key exhausted → resolving Vertex runtime")
        return ApiRuntime(
            provider="vertex",
            base_url=base_url,
            access_token=access_token,
            credentials_file=credentials_file,
        )
    except Exception as exc:
        log_warn(f"⚠️ Vertex fallback failed: {type(exc).__name__}: {exc}")
        return None


def _is_billing_exhausted_error(exc) -> bool:
    """Check if an exception indicates Gemini billing cap exhaustion."""
    status = getattr(exc, "code", None)
    if status == 429:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "billing cap" in text.lower()


def _coerce_runtime(runtime_or_key) -> ApiRuntime:
    if isinstance(runtime_or_key, ApiRuntime):
        return runtime_or_key
    return ApiRuntime(provider="gemini", api_key=str(runtime_or_key or ""))


def _generate_content_url(runtime: ApiRuntime, model: str) -> str:
    if runtime.provider == "vertex":
        return f"{runtime.base_url.rstrip('/')}/models/{model}:generateContent"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={runtime.api_key}"


def _request_headers(runtime: ApiRuntime) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if runtime.provider == "vertex":
        headers["Authorization"] = f"Bearer {runtime.access_token}"
    return headers


def call_gemini(api_runtime, model, body, timeout=300, retries=2, request_label="Gemini request"):
    """Call Gemini/Vertex generateContent and return parsed response."""
    runtime = _coerce_runtime(api_runtime)
    url = _generate_content_url(runtime, model)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _request_headers(runtime)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            call_started = time.monotonic()
            log_info(f"🤖 {request_label}: provider={runtime.provider} model={model} attempt {attempt}/{retries} timeout={timeout}s")
            if attempt > 1:
                backoff = attempt * 5
                log_info(f"⏳ Retrying {request_label} ({attempt}/{retries}) after {backoff}s")
                time.sleep(backoff)
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                response = json.loads(payload)
                log_info(f"✅ {request_label}: received response in {time.monotonic() - call_started:.1f}s")
                return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_err = exc
            status = getattr(exc, "code", "")
            text = getattr(exc, "read", lambda: b"")()
            if isinstance(text, (bytes, bytearray)):
                text_snip = text[:200].decode("utf-8", errors="replace")
            else:
                text_snip = str(text)[:200]
            if status == 429:
                wait = 10 * attempt
                log_warn(f"⚠️ {request_label} rate limited (429), wait {wait}s before retry ({attempt}/{retries})")
                time.sleep(wait)
                continue

            if attempt < retries:
                log_warn(f"⚠️ {request_label} failed (attempt {attempt}/{retries}): {type(exc).__name__}: {exc} {text_snip}")
                continue
            log_error(f"❌ {request_label} failed after {retries} attempts: {type(exc).__name__}: {exc} {text_snip}")
    # All Gemini retries exhausted — try Vertex fallback if billing exhausted
    if last_err is not None and runtime.provider == "gemini" and _is_billing_exhausted_error(last_err):
        vertex_runtime = _get_vertex_fallback_runtime()
        if vertex_runtime is not None:
            record_degradation(
                request_label,
                f"Gemini key exhausted after {retries} retries",
                "Vertex fallback",
            )
            log_info(f"🔁 Retrying {request_label} with Vertex provider...")
            return call_gemini(vertex_runtime, model, body, timeout, retries, f"{request_label} (Vertex fallback)")
    if last_err:
        raise last_err
    raise RuntimeError(f"{request_label} failed")


def _is_async_billing_error(err: dict) -> bool:
    """Check if an API error response dict indicates Gemini billing exhaustion."""
    code = err.get("code")
    status = str(err.get("status", ""))
    message = str(err.get("message", ""))
    return code == 429 or "RESOURCE_EXHAUSTED" in status or "billing cap" in message.lower()


async def _async_vertex_fallback(session, api_runtime, model, body, timeout, request_label):
    """Retry a failed async Gemini request with Vertex fallback."""
    vertex_runtime = _get_vertex_fallback_runtime()
    if vertex_runtime is None:
        return {"error": {"code": 429, "message": "Gemini key exhausted and Vertex fallback unavailable"}}
    record_degradation(
        request_label,
        "Gemini key exhausted",
        "Vertex fallback",
    )
    log_info(f"🔁 Retrying {request_label} with Vertex provider...")
    return await call_gemini_async(
        session, vertex_runtime, model, body, timeout,
        f"{request_label} (Vertex fallback)",
    )


async def call_gemini_async(session, api_runtime, model, body, timeout=600, request_label="Gemini async request"):
    """Call Gemini/Vertex generateContent asynchronously and return parsed JSON."""
    runtime = _coerce_runtime(api_runtime)
    url = _generate_content_url(runtime, model)
    headers = _request_headers(runtime)
    try:
        call_started = time.monotonic()
        log_info(f"🤖 {request_label}: provider={runtime.provider} model={model} timeout={timeout}s")
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            text = await resp.text()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Non-JSON response (HTTP {resp.status}): {text[:300]}") from exc
            if resp.status >= 400:
                if "error" not in payload:
                    payload = {"error": {"code": resp.status, "message": text[:500]}}
                # Check for billing exhaustion — auto-fallback to Vertex
                if runtime.provider == "gemini" and _is_async_billing_error(payload.get("error", {})):
                    return await _async_vertex_fallback(session, api_runtime, model, body, timeout, request_label)
                return payload
            log_info(f"✅ {request_label}: received response in {time.monotonic() - call_started:.1f}s")
            return payload
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{request_label} timed out") from exc
    except Exception as exc:
        raise RuntimeError(f"{request_label} failed: {type(exc).__name__}: {exc}") from exc


def is_rate_limited_error(err):
    code = err.get("code")
    status = str(err.get("status", ""))
    message = str(err.get("message", ""))
    return code == 429 or "RESOURCE_EXHAUSTED" in status or "RESOURCE_EXHAUSTED" in message


def search_paper_context(api_runtime, paper_text, model="gemini-3.1-pro-preview"):
    """Use Gemini with Google Search grounding to fetch publication/citation context."""
    begin_stage("context-search", "searching for publication context")
    log_info("🔍 Searching for paper background and publication context...")

    header = paper_text[:3000]

    body = {
        "contents": [{"role": "user", "parts": [{"text": f"""Based on the following paper header, search for this paper's:
1. Full title and authors
2. Publication venue and date (conference/journal, year)
3. Number of citations (approximate)
4. Key related/competing works published around the same time or after
5. Current status: has this work been superseded by newer methods? What is the current state-of-the-art in this area?

Return a concise factual summary (no opinions, just facts). If you cannot find the paper, state that clearly.

Paper header:
{header}"""}]}],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.1,
        },
    }

    for attempt in range(1, 4):
        try:
            result = call_gemini(
                api_runtime, model, body,
                timeout=120, retries=3,
                request_label="context search",
            )
            context = extract_text_from_gemini_result(result, "context-search", "Context search")
            log_info(f"✅ Context retrieved: {len(context)} chars")
            return context
        except Exception as e:
            log_warn(f"⚠️ Context search attempt {attempt}/3 failed: {type(e).__name__}: {e}")
            if attempt < 3:
                time.sleep(attempt * 8)
                continue
            record_degradation(
                "context-search",
                f"context search failed after retries: {type(e).__name__}: {e}",
                "proceed without external context",
            )
            return "（未能检索到论文背景信息，请根据论文内容本身进行讨论。）"
