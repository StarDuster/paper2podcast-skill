# Session 2026-05-16: GitHub Raw PDF & Pro Model Degradation

## GitHub PDF Extraction
When user provides a GitHub URL like:
`https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf`

Direct `pdftotext` or `web_fetch` might get the HTML wrapper instead of the binary PDF. 
**Fix**: Append `?raw=true` to the URL.
`https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf?raw=true`

## Gemini 3.1 Pro MAX_TOKENS Degradation
Even with `gemini-3.1-pro-preview`, segment generation can hit `MAX_TOKENS` during the multi-stage script generation process.

**Case Study (arXiv 2411.01783)**:
- **Input**: 115k chars.
- **Goal**: 10 min podcast.
- **Error**: `PipelineError: Segment 1 generation returned partial or blocked response (finishReason=MAX_TOKENS) -> single-stage script generation`.
- **Result**: System fallback to single-stage generation.
- **Verification**: Despite the fallback, the final script produced a ~10.5 min podcast (634s), which met the user's requirement.
- **Lesson**: Don't panic on `MAX_TOKENS` fallback for Pro models if the output length is still sufficient. Single-stage Pro generation is sometimes "denser" but still effective for 10-minute targets.
