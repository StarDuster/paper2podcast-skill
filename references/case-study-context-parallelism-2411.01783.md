# Case Study: Context Parallelism (arXiv 2411.01783)

This paper is a "stress test" case for the `paper2podcast` pipeline due to its length and technical depth.

## Historical Context (OpenClaw Era)
- **Problem**: Initial attempts failed due to TTS timeouts. The script segments were too large (6700+ characters), causing the Gemini TTS API to hang or return 500/503.
- **Attempt 1**: Used `gemini-2.5-flash` for script. Failed with 503 during outline generation.
- **Attempt 2**: Used `gemini-3-flash-preview`. Encountered JSON parsing error (`AttributeError: 'list' object has no attribute 'get'`) in multi-stage generation.
- **Attempt 3**: Used `gemini-2.0-flash` for script and `gemini-2.5-pro-preview-tts` for TTS. 
- **Successful Rescue**:
    - **Command**: `python3 scripts/paper2podcast.py ... --max-segment-bytes 4000`
    - **Result**: 16.2 minutes, 14.8 MB. 
    - **Key Lesson**: Lowering `max-segment-bytes` (defaulting to 2800 now) is critical for long-context papers to ensure TTS reliability.

## 2026-05-16 Update
- **Request**: "Redo this podcast".
- **Context**: The paper has a v3 update (2025-04-21).
- **Strategy**: 
    - Use `gemini-3.1-pro-preview` for script (per user preference for quality/duration).
    - Use `gemini-2.5-pro-preview-tts` (per user preference for pro voice quality).
    - Use `terminal(background=true)` to bypass the 10-minute `delegate_task` timeout.

## Key Parameters for "Heavy" Papers
- `--duration 10` or higher.
- `--script-model gemini-3.1-pro-preview`.
- `--tts-model gemini-2.5-pro-preview-tts`.
- `--max-segment-bytes 2800` (safe default).
