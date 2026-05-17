# Session Reference: 2026-05-14 Podcast Degradation

## Event Summary
- **Target**: `arXiv:2604.13016` (Rethinking On-Policy Distillation)
- **Requested Duration**: 10 minutes
- **Actual Duration**: 5.4 minutes
- **Failure Cause**: `gemini-3-flash-preview` failed to produce valid JSON for the multi-stage outline, triggering a fallback to `single-stage` generation.
- **Context Search**: Failed (`failed_stage=context-search`), likely due to the paper being very recent (2026).

## Error Details
```python
degradations=outline-generation: outline generation failed: AttributeError: 'list' object has no attribute 'get' -> single-stage script generation
```
The script generator expected a dictionary `{"outline": [...]}` or similar, but the Flash model returned a raw list or an invalid structure.

## Lessons Learned
1. **Model Choice**: Do not preemptively downgrade to Flash to avoid 503 errors if the task requires complex JSON logic (like the multi-stage podcast pipeline). The user prefers the risk of 503/Retry over a guaranteed degraded output.
2. **Single-Stage Limit**: Single-stage generation is a "best effort" fallback. It cannot reach 10+ minute targets because the model's output window (approx 2k-4k tokens for conversational content) is insufficient for a 10-minute script (usually requires ~10k+ tokens).
3. **Recovery**: To fix a "short" podcast caused by degradation, the parsing bug must be addressed, or a more capable model (Pro) must be used.
