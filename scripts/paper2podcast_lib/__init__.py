"""paper2podcast_lib — internal package backing scripts/paper2podcast.py.

Module layout:
  runtime      — RunContext, stage tracking, logging, fatal-error helpers
  config       — API-key resolution, Gemini model-name normalization
  prompts      — ZH prompt templates + speaker bindings (Alice/Bob)
  validation   — JSON / transcript / file-shape validators
  input_parse  — PDF / URL / file / stdin → text
  gemini       — sync + async Gemini API clients, paper-context search
  script       — single- and multi-stage script generation
  tts          — TTS rendering (per-turn / multi-speaker), splitting, cache
  audio        — ffmpeg concat into final MP3
  cli          — argparse + main()

The top-level `paper2podcast.py` shim star-imports from this package so
existing callers (`import paper2podcast as p2p`) keep working.
"""

from .runtime import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .prompts import *  # noqa: F401,F403
from .validation import *  # noqa: F401,F403
from .input_parse import *  # noqa: F401,F403
from .gemini import *  # noqa: F401,F403
from .script import *  # noqa: F401,F403
from .tts import *  # noqa: F401,F403
from .audio import *  # noqa: F401,F403

# Names that import * skips (underscore prefix or imported-name) but external
# callers reference. Re-export them explicitly.
from .gemini import aiohttp  # noqa: F401  (None when aiohttp is not installed)
from .tts import _infer_segment_position  # noqa: F401  (resume_tts.py uses this)

from .cli import main  # noqa: F401
