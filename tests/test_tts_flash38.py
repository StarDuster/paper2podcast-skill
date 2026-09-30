"""Offline regressions for Flash 3.8 payloads and container decoding."""
import asyncio
import base64
import io
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper2podcast_lib import cli, tts


def test_default_model():
    assert cli._build_argparser().parse_args(["article.md"]).tts_model == "gemini-3.8-flash-tts"


@pytest.mark.parametrize("model", ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"])
@pytest.mark.parametrize("mode", ["per-turn", "multi-speaker"])
def test_flash_payload(tmp_path, monkeypatch, model, mode):
    call = AsyncMock(return_value={"candidates": []})
    monkeypatch.setattr(tts, "call_gemini_async", call)
    monkeypatch.setattr(tts, "is_reusable_tts_segment", lambda *args: False)
    turns = [{"speaker_id": 0, "dialog": "标准普通话。"}]
    if mode == "multi-speaker":
        turns.append({"speaker_id": 1, "dialog": "平翘舌区分准确。"})
    asyncio.run(tts.tts_render_async(None, None, turns, 0, 1, str(tmp_path),
                                   "zh", "Kore", "Charon", model, mode=mode))
    parts = call.call_args.args[3]["contents"][0]["parts"]
    assert [p["text"] for p in parts] == [t["dialog"] for t in turns]
    for p in parts:
        assert "平翘舌区分准确" in p["speech_metadata"]["style"]
        assert "断奏" not in p["speech_metadata"]["style"]
    if mode == "multi-speaker":
        assert [p["speech_metadata"]["speaker"] for p in parts] == ["Alice", "Bob"]
    else:
        assert call.call_args.args[3]["generationConfig"]["speechConfig"] == {"voiceConfig": {"voice": "Kore"}}


def test_pro_payload_unchanged(tmp_path, monkeypatch):
    call = AsyncMock(return_value={"candidates": []})
    monkeypatch.setattr(tts, "call_gemini_async", call)
    monkeypatch.setattr(tts, "is_reusable_tts_segment", lambda *args: False)
    asyncio.run(tts.tts_render_async(None, None, [{"speaker_id": 0, "dialog": "测试。"}],
                                   0, 1, str(tmp_path), "zh", "Kore", "Charon",
                                   "gemini-2.5-pro-preview-tts", mode="per-turn"))
    body = call.call_args.args[3]
    assert "speech_metadata" not in body["contents"][0]["parts"][0]
    assert "slightly fast" in body["contents"][0]["parts"][0]["text"]
    assert body["generationConfig"]["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}


def test_wav_trailing_metadata_excluded(tmp_path, monkeypatch):
    pcm = b"\x00\x00" * 2400
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    payload = buf.getvalue() + b"C2PA trailing metadata must not become audio"
    async def convert(src, dst):
        assert Path(src).read_bytes() == pcm
        Path(dst).write_bytes(b"mock mp3")
    monkeypatch.setattr(tts, "convert_pcm_to_mp3", convert)
    actual = asyncio.run(tts._write_tts_audio_files(base64.b64encode(payload).decode(),
        pcm_file=str(tmp_path / "audio.pcm"), mp3_tmp=str(tmp_path / "temp.mp3"),
        mp3_file=str(tmp_path / "audio.mp3"), expected_metadata={}, output_label="test"))
    assert actual == pcm
