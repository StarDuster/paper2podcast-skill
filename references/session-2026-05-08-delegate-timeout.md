# Session Reference: 2026-05-08 - Delegate Task Timeout

## Context
User requested a 10-minute podcast for an arXiv paper.

## Problem
Using `delegate_task` to handle the entire pipeline (script generation + background search + per-turn TTS + ffmpeg) resulted in a 600s timeout. Even though the background process in the subagent might have continued (depending on how it was launched), the main agent lost the direct handle on the task and marked it as a failure.

## Resolution
1. **Fallback to Terminal Background**: Launched the task using `terminal(background=true)` in the main agent session.
2. **Manual Monitoring**: Used `ls -dt /tmp/paper2podcast_runs/*/ | head -n 1` to locate the `work_dir` of the background run.
3. **Log Inspection**: Monitored `paper2podcast.log` to confirm the stage (it was at segment 2/4 after ~10 minutes).

## Lessons Learned
- For tasks involving heavy TTS (Gemini TTS per-turn can be slow), skip `delegate_task` for the execution phase. Use it for planning/script-writing if needed, but the rendering should be a background terminal task.
- `ls -dt` is the standard "I lost my folder" recovery tool for this skill.
