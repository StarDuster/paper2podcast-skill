# API Debugging Reference (May 2026 Incident)

## Incident: Gemini 3.1 Flash TTS 500 Errors
In early May 2026, `gemini-3.1-flash-tts-preview` started returning `503 Service Unavailable` or `500 Internal Error` for all requests, even minimal ones.

### Verification Recipe
If a model starts failing, use this minimal Python snippet to verify if it's an upstream issue or a client-side config/parameter issue.

```python
import json, urllib.request
api_key = "AIza..." # Get from secrets
model = "gemini-3.1-flash-tts-preview"
body = {
    "contents": [{"parts": [{"text": "Test"}]}],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "kore"}}
        }
    }
}
req = urllib.request.Request(
    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
# If this returns 500 for multiple valid voiceNames, it is an upstream outage.
```

### Known Voice Names (as of May 2026)

**gemini-2.5-pro-preview-tts** (Discovered via 400 Bad Request error message):
- `achernar`, `achird`, `algenib`, `algieba`, `alnilam`, `aoede`, `autonoe`, `callirrhoe`, `charon`, `despina`, `enceladus`, `erinome`, `fenrir`, `gacrux`, `iapetus`, `kore`, `laomedeia`, `leda`, `orus`, `puck`, `pulcherrima`, `rasalgethi`, `sadachbia`, `sadaltager`, `schedar`, `sulafat`, `umbriel`, `vindemiatrix`, `zephyr`, `zubenelgenubi`.

**gemini-3.1-flash-tts-preview**:
- Supports `zart`, `aqua` in addition to the standard set above.

### Recovery Strategy
1. **Switch Model**: If Flash fails, downgrade to `gemini-2.5-pro-preview-tts`.
2. **Switch Voices**: Ensure voice names are compatible with the new model.
3. **Slow Down**: Pro models have lower RPM; reduce `--workers` to 1 or 2.
4. **Prompt Check**: Remove high-energy/pause instructions from Pro prompts if output is too slow.

## Pitfall: Text Models vs TTS Models

**Do NOT confuse text models with TTS models.** They are completely different endpoints.

- `gemini-3-flash-preview` → **text model**. Even if you set `responseModalities: ["AUDIO"]`, it will silently ignore it and return text. It will return HTTP 200 but the response has no `inlineData` — only `text` parts. This is NOT a working TTS call.
- `gemini-2.5-flash-preview-tts` → **TTS model** (note the `-tts` suffix)
- `gemini-2.5-pro-preview-tts` → **TTS model**
- `gemini-3.1-flash-tts-preview` → **TTS model**

**How to verify**: After a successful 200 response, check `candidates[0].content.parts[0]` — TTS models return `inlineData` (base64 audio), text models return `text`.

## Outage Timeline: May 2, 2026

| Time (JST) | Model | Status |
|:---|:---|:---|
| 2026-05-02 03:07 | `gemini-3.1-flash-tts-preview` | ✅ Normal (77/77 segments succeeded in NCCL Gin v5) |
| 2026-05-02 ~04:00 | `gemini-3.1-flash-tts-preview` | ❌ 500 Internal Error (all segments, even minimal "Hello") |
| 2026-05-02 04:06+ | `gemini-2.5-pro-preview-tts` | ✅ Normal (88/88 segments eventually succeeded) |

**Conclusion**: The 500 was a service-side outage affecting only the Flash TTS model. The Pro TTS model remained available as a fallback.
