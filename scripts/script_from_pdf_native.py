#!/usr/bin/env python3
"""Native-PDF → podcast script JSON.

把 PDF 原文以 Gemini 原生 inline_data 直接喂给模型，产出 paper2podcast CLI 能吃的
`podcast_transcripts` JSON，然后配合 `--script` 渲染 TTS，不经过 pdftotext。

用法:
    python3 script_from_pdf_native.py <input.pdf> <out_script.json> \
        [--duration 15] [--model gemini-3.1-pro-preview] [--pdf-pages N]

设计要点 (2026-09-10 实测):
- PDF 以 base64 inline_data 传，页按 IMAGE 模态入库 (~1065 token/页)；
  inline 上限 20MB，超过时提示改用 Files API。
- responseMimeType=application/json 强制模型只吐 JSON，省掉解析容错。
- 单次调用直出整份脚本；中文简体、speaker_id 0/1、无主持人花名。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import hashlib
from pathlib import Path
from paper2podcast_lib.runtime import PipelineError
from paper2podcast_lib.validation import extract_text_from_gemini_result, parse_json_payload, write_json_file
from paper2podcast_lib.script import _validate_transcript_payload

DENSITY = 245  # 中文 TTS 约 240-250 字/分钟


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    raise SystemExit("set GEMINI_API_KEY or GOOGLE_API_KEY")


PROMPT = """你是技术播客的脚本作者。下面附上的是一篇论文/技术报告的原始 PDF。

请为它写一期中文双人技术播客脚本，直接输出 JSON，不要任何解说文字。

硬性要求：
- 输出格式严格为：{{"podcast_transcripts": [{{"speaker_id": 0, "dialog": "..."}}, ...]}}
- speaker_id=0 是主持人 Alice（负责引导、建立直觉、总结），speaker_id=1 是技术专家 Bob（负责追问细节、边界条件、质疑）。
- 对话里不要出现主持人花名（不要写「我是 Alice/Bob/小明/大白」这类自我介绍），直接进入内容。
- 全程简体中文，禁止繁体字、台湾腔、港澳台书面表达。
- 总轮数 30-40 轮，总字数约 {words} 字（目标时长 {duration} 分钟，语速按 {density} 字/分钟计）。
- 单轮不要超过 200 字，保持一问一答的对话感，不要变成单向讲座。
- 第 1 轮可以自然开场；中间轮次直接承接上文，禁止「感谢收听」「下期再见」「本期就到这里」这类收尾话术；
  最后 1-2 轮才允许自然收尾，不要升华煽情。
- 忠实于 PDF 内容：技术名词、数字、指标要准确；可以解释直觉，但不要编造论文里没有的结论。
- 不要出现「本文/这篇论文里提到」这类复读式表达，把它讲成两个人的讨论。
"""


def build_request(pdf_b64: str, duration: int) -> dict:
    words = int(duration * DENSITY * 0.95)
    prompt = PROMPT.format(words=words, duration=duration, density=DENSITY)
    return {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 32768,
            "temperature": 0.6,
            "responseMimeType": "application/json",
        },
    }


def call(key: str, model: str, body: dict, timeout: int = 600) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def resolve_pdf_source(src: str) -> str:
    """本地路径直接用；arXiv / http(s) 链接先下载成临时 PDF。

    - `arxiv.org/abs/<id>` / `arxiv.org/html/<id>` → `arxiv.org/pdf/<id>`
    - 已经是 .pdf 的 URL 直接下
    """
    import re
    import tempfile

    if not re.match(r"^https?://", src):
        return src

    url = src
    m = re.search(r"arxiv\.org/(?:abs|html)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", src)
    if m:
        url = f"https://arxiv.org/pdf/{m.group(1)}{m.group(2) or ''}"
    elif "arxiv.org" in src and "/pdf/" not in src:
        raise SystemExit(f"unrecognised arXiv URL: {src}")

    print(f"[pdf-native] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper2podcast)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if not data.startswith(b"%PDF-"):
        raise SystemExit(f"downloaded content is not a PDF (first bytes: {data[:16]!r})")
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="pdfnative_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    print(f"[pdf-native] saved {len(data)/1e6:.2f} MB -> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="本地 PDF 路径，或 arXiv / 任意 PDF URL")
    ap.add_argument("out")
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    args = ap.parse_args()

    pdf_path = resolve_pdf_source(args.pdf)
    pdf = Path(pdf_path).read_bytes()
    if not pdf.startswith(b"%PDF-"):
        raise SystemExit("input is not a PDF")
    if args.duration <= 0:
        raise SystemExit("duration must be positive")
    b64 = base64.b64encode(pdf).decode()
    print(f"[pdf-native] {args.pdf}: {len(pdf)/1e6:.2f} MB -> base64 {len(b64)/1e6:.2f} MB")
    if len(b64) > 20 * 1024 * 1024:
        raise SystemExit("base64 > 20MB inline limit; use Files API instead")

    key = _api_key()
    body = build_request(b64, args.duration)
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            data = call(key, args.model, body)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            print(f"[pdf-native] attempt {attempt}/3 HTTP {e.code}: {detail}")
            if e.code in (429, 500, 503):
                time.sleep(10 * attempt)
                continue
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"[pdf-native] attempt {attempt}/3 {type(e).__name__}: {e}")
            time.sleep(10 * attempt)
    else:
        print("[pdf-native] all attempts failed")
        return 1

    try:
        text = extract_text_from_gemini_result(data, "native-script", "Native PDF draft")
        if data["candidates"][0].get("finishReason") != "STOP":
            print("[pdf-native] rejected: finishReason must be STOP")
            return 1
        turns = _validate_transcript_payload(parse_json_payload(text, "native-script", "draft"), "native-script")
        script = {"podcast_transcripts": turns,
                  "review": {"status": "draft", "reviewed": False},
                  "generation": {"model": args.model, "source": args.pdf,
                                 "pdf_sha256": hashlib.sha256(pdf).hexdigest()}}
        write_json_file(args.out, script, "file-write", "native PDF draft JSON")
    except PipelineError as exc:
        print(f"[pdf-native] rejected: {exc}")
        return exc.exit_code
    total = sum(len(t["dialog"]) for t in turns)
    print(f"[pdf-native] draft only: turns={len(turns)} chars={total}; external review required")
    print(f"[pdf-native] draft written atomically to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
