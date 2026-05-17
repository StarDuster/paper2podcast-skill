"""Source input loaders: PDF / URL / file / stdin → plain text."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .runtime import abort, current_work_dir, log_info, log_warn
from .validation import ensure_non_empty_text

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional PDF fallback
    fitz = None


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdftotext, fallback to PyMuPDF."""
    for attempt in range(1, 4):
        try:
            log_info(f"📄 Extracting PDF (attempt {attempt}/3): {pdf_path}")
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                log_info(f"✅ PDF extracted with pdftotext: {len(result.stdout)} chars")
                return result.stdout
            if result.returncode != 0:
                log_warn(f"⚠️ pdftotext failed (exit {result.returncode}): {result.stderr[:200]}")
        except FileNotFoundError:
            log_warn("⚠️ pdftotext not found, skipping")
            break
        except Exception as exc:
            log_warn(f"⚠️ pdftotext exception on attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(attempt)

    if fitz is None:
        log_warn("⚠️ PyMuPDF not installed")
        abort(
            "input-parse",
            "Cannot extract PDF. Install poppler-utils (apt install poppler-utils) or pymupdf (pip install pymupdf)",
        )

    for attempt in range(1, 3):
        try:
            log_info(f"📄 Trying PyMuPDF fallback (attempt {attempt}/2)")
            doc = fitz.open(pdf_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                log_info(f"✅ PDF extracted via PyMuPDF: {len(text)} chars")
                return text
            raise RuntimeError("empty text output")
        except Exception as exc:
            log_warn(f"⚠️ PyMuPDF attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < 2:
                time.sleep(attempt)

    abort(
        "input-parse",
        "Cannot extract PDF. Install poppler-utils (apt install poppler-utils) or pymupdf (pip install pymupdf)",
    )


def _extract_main_text_from_html(html: str) -> str:
    """Extract clean main text from raw HTML using stdlib html.parser.

    Strips script/style/nav/header/footer tags and collapses whitespace.
    """
    class _Extractor(HTMLParser):
        SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
        BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "article", "section", "li", "blockquote", "td", "th"}

        def __init__(self):
            super().__init__()
            self._skip_depth = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP_TAGS:
                self._skip_depth += 1
            if tag in self.BLOCK_TAGS:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data):
            if self._skip_depth == 0:
                stripped = data.strip()
                if stripped:
                    self.parts.append(stripped)

    parser = _Extractor()
    try:
        parser.feed(html)
    except Exception as exc:
        log_warn(f"⚠️ HTML parsing fallback failed: {type(exc).__name__}: {exc}")

    text = " ".join(parser.parts)
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_text_from_url(url):
    """Fetch text from URL, extracting clean main body content from HTML."""
    headers = {"User-Agent": "Mozilla/5.0 paper2podcast/1.0"}
    last_err = None

    for attempt in range(1, 4):
        try:
            log_info(f"🌐 Fetching URL (attempt {attempt}/3): {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()

            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                log_info("📄 URL returned PDF-like content, extracting text")
                temp_dir = current_work_dir()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=temp_dir) as f:
                    f.write(data)
                    tmp_path = f.name
                try:
                    return extract_text_from_pdf(tmp_path)
                finally:
                    os.unlink(tmp_path)

            raw = data.decode("utf-8", errors="replace")

            if "<html" in raw[:2000].lower() or "<!doctype" in raw[:200].lower():
                text = _extract_main_text_from_html(raw)
                if len(text) > 500:
                    log_info(f"✅ URL fetch + HTML extraction: {len(raw)} → {len(text)} chars")
                    return text
                log_warn(f"⚠️ HTML extraction yielded only {len(text)} chars, falling back to raw text")

            ensure_non_empty_text("input-parse", raw, f"URL response from {url}")
            log_info(f"✅ URL fetch success: {len(raw)} chars")
            return raw

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            log_warn(f"⚠️ URL fetch failed (attempt {attempt}/3): {type(exc).__name__}: {exc}")
            if attempt < 3:
                sleep_s = attempt * 5
                log_info(f"⏳ Retry in {sleep_s}s")
                time.sleep(sleep_s)

    abort("input-parse", f"Failed to fetch URL after retries: {url}: {last_err}")


def load_input(input_path):
    """Load input text from file, URL, or stdin."""
    if input_path == "-":
        text = sys.stdin.read()
        ensure_non_empty_text("input-parse", text, "stdin input")
        log_info(f"📝 Loaded stdin input: {len(text)} chars")
        return text
    if input_path.startswith("http://") or input_path.startswith("https://"):
        return extract_text_from_url(input_path)
    p = Path(input_path)
    if not p.exists():
        abort("input-parse", f"File not found: {input_path}")
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(p))
    try:
        text = p.read_text(encoding="utf-8")
        ensure_non_empty_text("input-parse", text, f"input file {input_path}")
        log_info(f"📄 Loaded text file: {input_path} ({len(text)} chars)")
        return text
    except Exception as exc:
        abort("input-parse", f"Failed to read file {input_path}: {type(exc).__name__}: {exc}", cause=exc)
