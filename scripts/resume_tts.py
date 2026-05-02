#!/usr/bin/env python3
"""
Template to resume an interrupted paper2podcast TTS run.
Copy this to /tmp/resume_job.py and update the paths.
"""
import asyncio
import json
import os
import sys

# === CONFIGURATION ===
SCRIPT_JSON = "/tmp/your_script.json"
SEGMENTS_DIR = "/tmp/existing_segments_dir"
OUTPUT_MP3 = "/tmp/final_output.mp3"
API_KEY_FILE = "/root/.hermes/secrets/gemini_api_key.txt"

VOICE_A = "Kore"
VOICE_B = "Charon"
TTS_MODEL = "gemini-3.1-flash-tts-preview"
LANG = "zh"
MAX_SEGMENT_BYTES = 2800
TTS_RENDER_MODE = "per-turn"  # "per-turn" is production-safe; "multi-speaker" is kept for experiments.
WORKERS = 1  # Keep low for resume to avoid rate limits
# =====================

# Add skill script dir to path
sys.path.insert(0, "/root/.hermes/skills/devops/paper2podcast/scripts")
import paper2podcast as p2p

aiohttp = p2p.aiohttp

def main():
    if not os.path.exists(API_KEY_FILE):
        print(f"Error: API key file not found at {API_KEY_FILE}")
        sys.exit(1)
        
    api_key = open(API_KEY_FILE).read().strip()
    data = json.load(open(SCRIPT_JSON))
    entries = data["podcast_transcripts"]

    # Re-split transcript exactly as the original run did.
    # Existing files are only reused when their metadata matches the current
    # prompt/model/voice fingerprint; stale segment_000.mp3 files regenerate.
    if TTS_RENDER_MODE == "per-turn":
        segments = [[entry] for entry in entries]
    else:
        segments = p2p.split_transcript(entries, MAX_SEGMENT_BYTES, LANG, TTS_MODEL)
    print(f"Transcript contains {len(segments)} segments")

    if not os.path.exists(SEGMENTS_DIR):
        os.makedirs(SEGMENTS_DIR, exist_ok=True)
        print(f"Created segments directory: {SEGMENTS_DIR}")

    segment_files = [None] * len(segments)
    print("Synthesizing missing or stale segments; matching metadata will be reused...")

    async def run_segments():
        if aiohttp is None:
            raise RuntimeError("Missing dependency: aiohttp. Install with: pip install aiohttp")
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(WORKERS)

            async def worker(idx):
                async with semaphore:
                    pos = p2p._infer_segment_position(idx, len(segments))
                    if TTS_RENDER_MODE == "per-turn":
                        path = await p2p.tts_turn_async(
                            session, api_key, segments[idx][0], idx, len(segments),
                            SEGMENTS_DIR, LANG, VOICE_A, VOICE_B, TTS_MODEL, pos
                        )
                    else:
                        path = await p2p.tts_segment_async(
                            session, api_key, segments[idx], idx, len(segments),
                            SEGMENTS_DIR, LANG, VOICE_A, VOICE_B, TTS_MODEL, pos
                        )
                    return idx, path

            results = await asyncio.gather(*[worker(i) for i in range(len(segments))])
            for idx, path in results:
                segment_files[idx] = path

    asyncio.run(run_segments())

    # Verify all segments done
    failed = [i for i, f in enumerate(segment_files) if not f]
    if failed:
        print(f"Failed segments after resume: {failed}")
        sys.exit(1)

    print("Concatenating segments...")
    p2p.concat_segments(segment_files, OUTPUT_MP3, temp_dir=SEGMENTS_DIR)
    print(f"SUCCESS: {OUTPUT_MP3}")

if __name__ == "__main__":
    main()
