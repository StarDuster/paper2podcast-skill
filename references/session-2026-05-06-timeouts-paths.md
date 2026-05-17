# Session Detail: 2026-05-06 Subagent Timeouts & Pathing Issues

## Context
User requested a podcast for ArXiv paper 2605.01597 (10 min).
Used `delegate_task` to handle the long run.

## Issue 1: Timeout
The subagent timed out after 600s.
**Analysis**: 10 minutes of podcast requires ~30 TTS segments. Each takes 15-20s. Sequential generation exceeds the 10-minute timeout.
**Fix**: Manual resumption via background process or increase subagent timeout if possible.

## Issue 2: FileNotFoundError for MP3
After resumption, the agent tried to send `/tmp/arxiv_2605_01597.mp3`.
**Error**: `FileNotFoundError: [Errno 2] No such file or directory`.
**Cause**: The script defaults to `/tmp/paper2podcast_runs/<run_id>/url_podcast.mp3` unless `--output` is explicitly provided as an absolute path.
**Solution**: Always read the script stdout or log to find the specific `output_path`.

## Issue 3: Missing Caption File
`tg_send_audio.py` failed when trying to read `@/tmp/caption.txt`.
**Cause**: The caption file is also located in the dynamic `work_dir`.
**Solution**: Use the full path from the run directory.
