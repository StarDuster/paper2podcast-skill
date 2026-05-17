#!/usr/bin/env python3
"""Send an audio file to Telegram via Bot API's sendAudio endpoint.

Handles MP3/M4A natively with caption in a single message.
This is a reusable utility for podcast delivery when the gateway
extract_media() splits text and audio into separate messages.

Usage:
    python3 tg_send_audio.py <chat_id> <audio_path> [@caption_file|caption_text] [--thread-id N]

Examples:
    python3 tg_send_audio.py <CHAT_ID> /tmp/podcast.mp3 "@/tmp/caption.txt"
    python3 tg_send_audio.py <CHAT_ID> /tmp/podcast.mp3 "@/tmp/caption.txt" --thread-id <THREAD_ID>
"""

import argparse
import os
import sys

import requests


def get_bot_token() -> str:
    env_path = os.path.expanduser("~/.hermes/.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise FileNotFoundError(f"TELEGRAM_BOT_TOKEN not found in {env_path}")


def send_audio(chat_id: str, audio_path: str, caption: str = "", thread_id: int = None) -> dict:
    token = get_bot_token()
    url = f"https://api.telegram.org/bot{token}/sendAudio"

    audio_path = os.path.expanduser(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data: dict[str, str | int] = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    if thread_id:
        data["message_thread_id"] = thread_id

    basename = os.path.basename(audio_path)
    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            data=data,
            files={"audio": (basename, f, "audio/mpeg")},
        )

    result = resp.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result.get('description', result)}")
    return result["result"]


def main():
    parser = argparse.ArgumentParser(description="Send audio to Telegram with caption in one message")
    parser.add_argument("chat_id", help="Telegram chat_id")
    parser.add_argument("audio_path", help="Path to audio file (.mp3/.m4a)")
    parser.add_argument("caption", nargs="?", default="", help="Caption text or @path/to/file")
    parser.add_argument("--thread-id", "-t", type=int, default=None, help="Forum topic thread ID")
    args = parser.parse_args()

    caption = args.caption
    if caption.startswith("@"):
        with open(caption[1:]) as f:
            caption = f.read().strip()

    size_mb = os.path.getsize(os.path.expanduser(args.audio_path)) / (1024 * 1024)
    print(f"Sending {args.audio_path} ({size_mb:.1f} MB) -> {args.chat_id}")

    result = send_audio(args.chat_id, args.audio_path, caption, args.thread_id)
    print(f"OK: message_id={result.get('message_id')} chat={result.get('chat', {}).get('id')}")


if __name__ == "__main__":
    main()
