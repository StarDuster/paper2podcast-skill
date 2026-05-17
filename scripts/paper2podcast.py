#!/usr/bin/env python3
"""Paper → Podcast Pipeline — CLI entry point.

The implementation lives in the ``paper2podcast_lib`` package next to this
file. This shim star-imports the package so existing callers that do
``import paper2podcast as p2p`` (e.g. resume_tts.py) keep working unchanged.

Usage:
  python3 paper2podcast.py <input> [options]

Run with ``-h`` for the full option list.
"""

from __future__ import annotations

import sys

from paper2podcast_lib import *  # noqa: F401,F403 — re-export public API
from paper2podcast_lib import aiohttp, main  # noqa: F401 — explicit names for callers
from paper2podcast_lib.tts import _infer_segment_position  # noqa: F401


if __name__ == "__main__":
    sys.exit(main())
