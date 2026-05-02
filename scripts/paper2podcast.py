#!/usr/bin/env python3
"""
Paper → Podcast Pipeline (Skill version)

Flow:
  1. Extract text from PDF / text file / URL
  2. Generate structured podcast script (JSON) via Gemini
  3. TTS the full script via Gemini 3.1 Flash TTS multi-speaker
  4. Output MP3

Usage:
  python3 paper2podcast.py <input> [options]

  <input> can be:
    - A local PDF file
    - A local text file
    - A URL (https://...)
    - "-" to read from stdin

Options:
  --lang zh             Language (Chinese only; default: zh)
  --duration 10         Target duration in minutes (default: 10)
  --voice-a Kore        Voice for speaker 0 (default: Kore)
  --voice-b Charon      Voice for speaker 1 (default: Charon)
  --script-model MODEL  Model for script generation (default: gemini-3.1-pro-preview)
  --tts-model MODEL     Model for TTS (default: gemini-3.1-flash-tts-preview)
  --output FILE         Output MP3 path (default: auto-generated)
  --script-only         Only generate script, skip TTS
  --script FILE         Use existing script JSON, skip generation
  --max-segment-bytes N Max bytes per TTS segment (default: 2800)
  --workers N           Parallel TTS workers (default: 2)
  --api-key KEY         Gemini API key (or set GEMINI_API_KEY env)
  --api-key-file FILE   Read API key from file
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - dependency check at runtime
    aiohttp = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional PDF fallback
    fitz = None

# Ensure progress logs remain visible when stdout/stderr are piped (e.g. via `tee`).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Runtime logging
LOGGER = logging.getLogger("paper2podcast")


class PipelineError(RuntimeError):
    """Fatal pipeline error with stage metadata and exit code."""

    def __init__(self, stage: str, message: str, exit_code: int = 1):
        super().__init__(message)
        self.stage = stage
        self.exit_code = exit_code


@dataclass
class RunContext:
    started_at: float = field(default_factory=time.monotonic)
    current_stage: str | None = None
    stage_started_at: float = field(default_factory=time.monotonic)
    stage_durations: dict[str, float] = field(default_factory=dict)
    failed_stage: str | None = None
    output_path: str | None = None
    script_path: str | None = None
    log_path: str | None = None
    run_id: str | None = None
    work_dir: str | None = None
    degradations: list[str] = field(default_factory=list)


RUN_CONTEXT = RunContext()


def get_run_context() -> RunContext:
    return RUN_CONTEXT


def finalize_current_stage() -> None:
    ctx = get_run_context()
    if not ctx.current_stage:
        return
    elapsed = max(0.0, time.monotonic() - ctx.stage_started_at)
    ctx.stage_durations[ctx.current_stage] = ctx.stage_durations.get(ctx.current_stage, 0.0) + elapsed
    ctx.stage_started_at = time.monotonic()


def begin_stage(stage: str, detail: str = "") -> None:
    finalize_current_stage()
    ctx = get_run_context()
    ctx.current_stage = stage
    ctx.stage_started_at = time.monotonic()
    suffix = f" ({detail})" if detail else ""
    log_info(f"▶️ Stage: {stage}{suffix}")


def record_degradation(stage: str, reason: str, fallback: str) -> None:
    ctx = get_run_context()
    note = f"{stage}: {reason} -> {fallback}"
    ctx.degradations.append(note)
    log_warn(f"⚠️ [{stage}] Degrading because {reason}. Fallback: {fallback}", file_err=True)


def abort(stage: str, message: str, *, exit_code: int = 1, cause: Exception | None = None) -> None:
    ctx = get_run_context()
    ctx.failed_stage = stage
    if cause is not None:
        LOGGER.exception("[%s] %s", stage, message)
    log_error(f"❌ [{stage}] {message}")
    raise PipelineError(stage, message, exit_code=exit_code) from cause


def configure_logging(log_file: str | None = None, verbose: bool = True) -> str:
    """Configure root logging to stdout + optional file.

    Returns resolved log file path (if enabled) for downstream reporting.
    """
    if LOGGER.handlers:
        # Avoid duplicate handlers when rerun in same process (e.g. tests).
        return log_file or ""

    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.info("Log file: %s", log_path)
        return str(log_path)
    return ""


def log_info(message: str):
    """Emit a UTF-8-safe info log and print for immediate visibility."""
    LOGGER.info(message)
    print(message, flush=True)


def log_warn(message: str, *, file_err: bool = False):
    LOGGER.warning(message)
    print(message, file=sys.stderr if file_err else sys.stdout, flush=True)


def log_error(message: str):
    LOGGER.error(message)
    print(message, file=sys.stderr, flush=True)


def make_run_id() -> str:
    """Create a readable unique id for one paper2podcast invocation."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def default_runs_root() -> Path:
    """Base directory for per-run temporary workspaces."""
    return Path(tempfile.gettempdir()) / "paper2podcast_runs"


def create_run_work_dir(work_dir_arg: str | None = None) -> Path:
    """Create an isolated directory for all temporary files of this run."""
    ctx = get_run_context()
    ctx.run_id = make_run_id()
    if work_dir_arg:
        base = Path(work_dir_arg).expanduser()
        work_dir = base / ctx.run_id if base.exists() and base.is_dir() else base
    else:
        work_dir = default_runs_root() / ctx.run_id
    work_dir.mkdir(parents=True, exist_ok=False)
    ctx.work_dir = str(work_dir)
    return work_dir


def current_work_dir() -> Path | None:
    work_dir = get_run_context().work_dir
    return Path(work_dir) if work_dir else None


def emit_final_summary(success: bool, exit_code: int) -> None:
    finalize_current_stage()
    ctx = get_run_context()
    total_elapsed = max(0.0, time.monotonic() - ctx.started_at)
    stage_order = [
        "config",
        "input-parse",
        "context-search",
        "outline-generation",
        "segment-generation",
        "tts-audio-synthesis",
        "file-write",
        "cleanup",
    ]
    seen = []
    for stage in stage_order:
        if stage in ctx.stage_durations:
            seen.append(f"{stage}={ctx.stage_durations[stage]:.1f}s")
    for stage, elapsed in ctx.stage_durations.items():
        if stage not in stage_order:
            seen.append(f"{stage}={elapsed:.1f}s")

    logger = log_info if success else log_error
    logger("=== Final Status Summary ===")
    logger(f"status={'success' if success else 'failure'} exit_code={exit_code}")
    logger(f"failed_stage={ctx.failed_stage or '-'} total_elapsed={total_elapsed:.1f}s")
    logger(f"output_path={ctx.output_path or '-'}")
    logger(f"script_path={ctx.script_path or '-'}")
    logger(f"log_path={ctx.log_path or '-'}")
    logger(f"work_dir={ctx.work_dir or '-'}")
    logger(f"stage_timings={', '.join(seen) if seen else '-'}")
    if ctx.degradations:
        logger(f"degradations={'; '.join(ctx.degradations)}")


# ---------------------------------------------------------------------------
# Prompts (inspired by SurfSense podcaster)
# ---------------------------------------------------------------------------

PROMPT_ZH = """\
你是一位资深 AI 学术播客脚本撰写者。你的任务是将学术论文转化为一期专业、深入但引人入胜的双人学术播客（类似 Latent Space 或 Lex Fridman 的风格）。

<podcast_generation_system>
## 角色设定
- **Speaker 0（Alice）**：主持人/资深研究者。负责引导讨论流程，善于将复杂概念转化为通俗易懂的总结，控制节奏。语气知性、好奇、包容。
- **Speaker 1（Bob）**：技术专家/审稿人视角。对技术细节极其敏感，喜欢刨根问底，对论文的创新点持审慎态度，经常提出犀利的反问。语气极客、直率、犀利。

## 背景信息
{context_block}

## 讨论结构

### 第一部分：宏观概述与直觉构建（约占 40%）
- **核心目标**：这是播客的"钩子"。必须在前 5 分钟内建立对论文核心思想的直觉模型。必须包含合适的播客开场词。
- **专业化与通俗化平衡**：可以适度高层抽象，但禁止过度使用比喻和夸张的修辞，保持技术严肃性。不要一上来就陷入数学细节，先讲清楚 "What" 和 "Why"。
- 讨论内容：
  - 这个研究是为了解决什么现实世界的问题？（Why it matters?）
  - 核心思想是什么？（用一句话直觉性地描述）
  - 它的主要贡献和效果如何？
  - 论文的发表背景和在领域内的位置。

### 第二部分：硬核技术深挖（约占 30%）
- **过渡**：Alice 引导话题进入深水区，Bob 开始展示技术肌肉。
- **内容要求**：
  - 深入剖析模型架构、算法细节或数学原理。
  - 讨论实验设置的巧妙之处或漏洞。
  - 关注"魔鬼在细节中"的部分（如数据处理、超参数、训练技巧）。
  - Bob 应该在这里多质疑，Alice 负责确认理解。
- **节奏要求**：这部分使用更短、更密集的对话轮次（每轮1-3句），模拟两位研究者快速交流细节的感觉。

### 第三部分：批判性评价（约占 30%）
- **结尾要求**：必须包含合适的播客结束词。
- **优点**：论文的真实贡献是什么？方法是否优雅？实验是否充分？
- **缺点**：方法有哪些局限性？实验是否有明显遗漏？假设是否合理？
- **新颖性**：相比同期和之前的工作，创新程度如何？是增量改进还是范式突破？
- **领域价值**：对该研究方向的实际推动作用如何？
- **相关工作**：论文是否合理地引用和对比了相关工作？是否存在明显遗漏？
- 批判性解读文中声称的优点，谨慎给出赞美。论文可能不是最新发表的，需要结合当前最新进展来重新审视其贡献。

## 风格与口语化要求
1.  **对话感（Crucial）**：
    - 严禁使用书面语（如"综上所述"、"显而易见"、"笔者认为"）。
    - 使用自然的口语连接词（如"说实话"、"这就很有意思了"、"你是说...？"、"打个比方"）。
    - 允许自然的打断、追问和确认。
2.  **听众定位**：
    - 听众设定为 **AI 领域从业者**。他们懂基础术语（如 Transformer, Loss Function），但可能不熟悉该具体细分领域。
    - Part 1 的通俗化是为了**降低认知负荷**，快速建立直觉，而非科普基础知识。
    - Part 2 & 3 保持硬核，直接讨论技术细节。
3.  **禁止事项**：
    - 禁止机械地罗列章节（不要说"接下来我们讨论第二部分"）。
    - 禁止互相吹捧（保持客观冷静）。
    - 脚本必须使用中文表达；除论文标题、模型名、方法名和必要技术术语外，禁止输出整句英文。
    - 禁止过度使用比喻和夸张的修辞，保持技术严肃性。
4.  **时长控制**：约 {duration} 分钟（约 {word_count} 字）。请确保内容足够充实，不要注水。
5.  **轮数控制**：减少碎片化问答。整期总对话轮数建议控制在 {duration} 分钟 × 2 到 {duration} 分钟 × 3 之间，除非内容确实需要，不要超过 30 turns。每轮可以包含 2-4 个短句，把同一逻辑点讲完整后再切给另一位 speaker。

## 输出格式
严格输出 JSON，不要包含任何其他文字：
{{
  "podcast_transcripts": [
    {{"speaker_id": 0, "dialog": "...", "style": "（可选）该句的表达风格，用自然语言描述语气、语速或情绪"}},
    {{"speaker_id": 1, "dialog": "..."}},
    ...
  ]
}}

其中 `style` 字段为可选，表示该句的表达风格（自然语言描述语气、语速或情绪），例如：`"语气略带疑问，句末上扬"` 或 `"放缓语速，语气沉稳"`。
</podcast_generation_system>
"""

TTS_AUDIO_PROFILE_ZH = """\
这是双说话人中文播客音频。speaker 名称只用于绑定 API voice，不要朗读。
API speaker `Alice` 对应主持人声线；API speaker `Bob` 对应技术评论者声线。
必须严格使用 `multiSpeakerVoiceConfig` 中为每个 speaker 配置的 voice，不要根据句子内容、角色性格或性别重新推断声线。"""

TTS_SCENE_ZH = """\
两位 AI 研究者在安静、近距离收音的录音空间里讨论一篇论文。环境稳定，没有舞台感，没有广播新闻感。
这是一段真实的专业对话，不是朗读稿，也不是演讲。"""

TTS_DIRECTORS_NOTES_ZH = """\
风格：冷面克制。整体干燥、冷静、克制，不要戏剧化。
节奏：断奏式表达。使用短促、干净的短语停顿；整体速度略快，但要轻而流畅。
口音：标准大陆普通话。不要台湾腔、不要东北腔、不要重儿化音、不要新闻播报式发音。
表达：说话要轻，不要重读关键词，不要过度咬字，不要把反问句演得很夸张，不要像正式广播或演讲。
英文术语：论文标题、模型名、方法名和技术术语可以保留英文原文；朗读英文术语时吐字清楚、发音准确，不要中式英语发音。
禁止添加转写内容里没有的填充词、拟声词或额外句子。"""

TTS_READING_RULES_ZH = """\
上面的音频配置、场景、导演备注和当前片段说明只用于控制声音表现，绝对不要朗读。
下面的转写内容严格采用 `Alice:` / `Bob:` 行首标签。行首标签只用于匹配 API speaker，不要朗读标签本身。
遇到 `Alice:` 行，必须使用 API 中 speaker=`Alice` 配置的 voice。
遇到 `Bob:` 行，必须使用 API 中 speaker=`Bob` 配置的 voice。
不得交换、合并或重新解释两个 speaker 的 voice。
必须从第一行台词开始，按顺序逐行朗读每一行台词。
不得跳过任何一行，不得改写、合并、总结或补充台词。
音频开头必须直接进入第一行台词，不得先说“说到这”“接下来”“我们来聊”“好”等开场白、承接语或过渡句。"""

_TTS_SAMPLE_CONTEXT_ZH = {
    "intro": "这是播客开头的宏观概述段。保持克制，不要刻意制造悬念或热场。",
    "technical": "这是技术深挖段。保持冷静、紧凑、清楚，像两位研究者在快速核对细节。",
    "conclusion": "这是结尾的批判评价与展望段。保持客观、收束，不要升华或煽情。",
    "middle": "这是播客中段的连续讨论。保持自然衔接，以及稳定的冷面克制和断奏式表达。",
}


def _is_flash_tts_model(model_name: str | None) -> bool:
    """Return True if the TTS model is a 'flash' variant that needs restrained prompts."""
    if not model_name:
        return False
    return "flash" in str(model_name).lower()


def _tts_sample_context(segment_position: str) -> str:
    """Return a non-spoken segment note for a TTS segment."""
    return _TTS_SAMPLE_CONTEXT_ZH.get(segment_position, _TTS_SAMPLE_CONTEXT_ZH["middle"])


def _build_tts_header(segment_position: str = "middle") -> str:
    """Build a compact TTS prompt header that keeps speaker labels unambiguous."""
    segment_note = _tts_sample_context(segment_position)
    return f"""\
TTS the following conversation between Alice and Bob.

Voice binding:
- `Alice:` lines MUST use the API voice configured for speaker `Alice`.
- `Bob:` lines MUST use the API voice configured for speaker `Bob`.
- Do not swap voices or infer voices from content. Do not speak speaker labels.

Delivery:
- Standard mainland Mandarin; calm, deadpan, staccato, light, fluent, slightly fast.
- No Taiwanese accent, Northeastern accent, heavy erhua, drama, heavy emphasis, over-articulation, or added words.
- Read every line exactly in order. Do not skip, rewrite, merge, summarize, or add transitions.
- Segment note: {segment_note}

Conversation:"""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

OUTLINE_PROMPT_ZH = """\
你是一位资深 AI 学术播客策划人。请根据以下论文内容，生成一份结构化的播客大纲。

## 要求
- 总时长目标：{duration} 分钟（约 {word_count} 字）
- 将内容分为 3-5 个讨论段落（Segment）
- 所有段落标题、核心要点和对话基调必须使用中文，必要技术术语可以保留英文原文
- 每个 Segment 必须包含：
  - `title`: 段落标题（简短）
  - `key_points`: 该段必须讨论的 2-3 个核心要点
  - `word_budget`: 该段的目标字数
  - `tone`: 该段的对话基调（如 "好奇探索"、"技术深挖"、"犀利质疑"）
- 第一个 Segment 必须是"宏观概述与直觉构建"，用于吸引听众
- 最后一个 Segment 必须是"批判性评价与展望"
- 所有 Segment 的 word_budget 之和必须接近 {word_count}

## 背景信息
{context_block}

## 输出格式
严格输出 JSON，不要包含任何其他文字：
{{
  "segments": [
    {{
      "title": "...",
      "key_points": ["...", "..."],
      "word_budget": 800,
      "tone": "..."
    }}
  ]
}}
"""

SEGMENT_PROMPT_ZH = """\
你是一位资深 AI 学术播客脚本撰写者。请根据大纲要求，为以下段落生成双人对话脚本。

## 角色设定
- **Speaker 0（Alice）**：主持人/资深研究者。引导讨论，善于通俗化总结。语气知性、好奇。
- **Speaker 1（Bob）**：技术专家/审稿人视角。刨根问底，对创新点持审慎态度。语气极客、犀利。

## 当前段落要求
- 标题：{segment_title}
- 基调：{segment_tone}
- 核心要点（必须全部覆盖）：{key_points}
- **严格字数限制：{word_budget} 字（允许 ±10% 浮动）**

## 风格要求
1. 严禁书面语（"综上所述"、"显而易见"等）
2. 使用自然口语连接词（"说实话"、"这就很有意思了"、"你是说...？"）
3. 允许自然打断和追问
4. 听众定位：AI 领域从业者
5. 禁止机械地宣布章节转换
6. 禁止互相吹捧
7. **技术术语必须保留英文原文**（如 KV cache、attention、transformer、MoE 等），不要翻译成中文（禁止"键值缓存"、"注意力机制"等译法）
8. 对话主体必须是中文；除必要技术术语、模型名、方法名和论文标题外，禁止输出整句英文；必须使用中国大陆简体中文，禁止繁体字、台湾腔用词和港澳台书面表达
9. 禁止过度使用比喻和夸张的修辞，保持技术严肃性
10. 禁止使用夸张表达，例如“降维打击”“暴力美学”“效率狂魔”“断崖式下跌”“炸裂”“封神”“天花板”等
11. 减少对话轮数：本段建议 4-6 turns，复杂段落最多 8 turns。每轮 2-4 个短句，讲完一个完整技术点再交给另一位 speaker。不要一问一答拆得太碎。
12. 段落位置规则：
{segment_position_rules}

## 上下文
{prev_context}

## 输出格式
严格输出 JSON：
{{
  "podcast_transcripts": [
    {{"speaker_id": 0, "dialog": "...", "style": "（可选）该句的表达风格，用自然语言描述语气、语速或情绪，如：语气略带疑问、恍然大悟等"}},
    {{"speaker_id": 1, "dialog": "..."}},
    ...
  ]
}}

其中 `style` 为可选字段，只在有明确表达意图时填写（如问句、强调、转折处）。
"""


def _segment_position_rules_zh(index: int, total: int) -> str:
    """Return script-generation rules for a segment's global podcast position."""
    if total <= 1:
        return (
            "- 这是整期播客的唯一段落：开头必须包含自然开场词，结尾最后 1-2 轮必须包含自然结束词。\n"
            "- 结束词只能出现在整段最后，禁止在中间提前说“感谢收听”“下期再见”“就到这里”。"
        )
    if index == 0:
        return (
            "- 这是整期播客的第一个段落：开头必须包含自然开场词。\n"
            "- 本段不是整期结尾，禁止出现任何结束词或收尾话术，包括“感谢收听”“下期再见”“本期就到这里”“今天就聊到这里”。"
        )
    if index == total - 1:
        return (
            "- 这是整期播客的最后一个段落：开头直接承接上文，不要重新欢迎听众。\n"
            "- 只有本段最后 1-2 轮可以包含自然结束词；结束词要克制，不要升华或煽情。"
        )
    return (
        "- 这是整期播客的中间段落：开头直接承接上文，不要重新欢迎听众。\n"
        "- 本段绝对不是结尾，禁止出现任何结束词或收尾话术，包括“感谢收听”“下期再见”“本期就到这里”“今天就聊到这里”。\n"
        "- 不要把当前段落写成独立节目，也不要说“这一期关于某主题就到这里”。"
    )

REVIEW_PROMPT_ZH = """\
你是一位播客主编。请审查以下播客脚本段落，检查是否存在以下问题：

1. **偏题**：是否偏离了要求讨论的核心要点？
2. **注水**：是否有废话、重复、或无意义的客套？
3. **语气不自然**：是否有书面语混入？是否像真实对话？
4. **字数**：实际字数是否在目标字数 ±15% 范围内？

## 段落要求
- 核心要点：{key_points}
- 目标字数：{word_budget}

## 待审查脚本
{script_text}

## 输出格式
严格输出 JSON：
{{
  "pass": true/false,
  "issues": ["问题1", "问题2"],
  "suggestion": "修改建议（如果 pass 为 true 则留空）"
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key(args):
    """Resolve API key from args, env, or default file."""
    if args.api_key:
        api_key = args.api_key.strip()
        if api_key:
            return api_key
        abort("config", "--api-key was provided but is empty")
    if args.api_key_file:
        try:
            api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        except Exception as exc:
            abort("config", f"Failed to read API key file {args.api_key_file}: {type(exc).__name__}: {exc}", cause=exc)
        if api_key:
            return api_key
        abort("config", f"API key file is empty: {args.api_key_file}")
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and env_key.startswith("AIza"):
        return env_key
    # Default location
    default_path = Path.home() / ".hermes" / "secrets" / "gemini_api_key.txt"
    if default_path.exists():
        try:
            api_key = default_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            abort("config", f"Failed to read default API key file {default_path}: {type(exc).__name__}: {exc}", cause=exc)
        if api_key:
            return api_key
        abort("config", f"Default API key file is empty: {default_path}")
    abort("config", "No API key found. Set GEMINI_API_KEY or use --api-key / --api-key-file")


GEMINI_MODEL_ALIASES = {
    # OpenClaw-style aliases / human-friendly shorthands
    "gemini": "gemini-3.1-pro-preview",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
    # Common shortened raw Gemini names that the v1beta API does not accept directly
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini 3.1 pro preview": "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
}


def normalize_gemini_model(model_name: str) -> str:
    """Map friendly/alias Gemini model names to concrete API model IDs."""
    if model_name is None:
        abort("config", "Model name is missing")

    raw = str(model_name).strip().strip("'").strip('"')
    if not raw:
        abort("config", "Model name is empty after trimming")

    normalized = raw
    lowered = normalized.lower()
    if lowered.startswith("models/"):
        normalized = normalized.split("/", 1)[1].strip()
        log_info(f"🔁 Normalize model path prefix: {raw} -> {normalized}")
        lowered = normalized.lower()

    known_prefixes = {
        "gemini",
        "google",
        "google-ai",
        "googleai",
        "generativelanguage",
        "generative-language",
    }
    for separator in (":", "/"):
        if separator in normalized:
            prefix, candidate = normalized.split(separator, 1)
            if prefix.strip().lower() in known_prefixes and candidate.strip():
                before = normalized
                normalized = candidate.strip()
                log_info(f"🔁 Normalize provider-prefixed model: {before} -> {normalized}")
                break

    alias_key = normalized.lower()
    normalized = GEMINI_MODEL_ALIASES.get(alias_key, GEMINI_MODEL_ALIASES.get(normalized, normalized))
    if normalized != raw:
        log_info(f"🔁 Normalize model alias: {raw} -> {normalized}")
    return normalized


def ensure_non_empty_text(stage: str, text: str, label: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        abort(stage, f"{label} is empty")
    return cleaned


def parse_json_payload(raw: str, stage: str, label: str) -> Any:
    raw = ensure_non_empty_text(stage, raw, label)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            abort(stage, f"Failed to parse {label} JSON. Raw response snippet: {raw[:1200]}")
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            abort(stage, f"Failed to parse extracted {label} JSON: {type(exc).__name__}: {exc}. Raw snippet: {raw[:1200]}", cause=exc)
    return None


def extract_text_from_gemini_result(result: dict[str, Any], stage: str, label: str) -> str:
    if not isinstance(result, dict):
        abort(stage, f"{label} returned non-object payload: {type(result).__name__}")
    candidates = result.get("candidates")
    if not candidates:
        prompt_feedback = result.get("promptFeedback")
        feedback = json.dumps(prompt_feedback, ensure_ascii=False)[:500] if prompt_feedback else "none"
        abort(stage, f"{label} returned no candidates. promptFeedback={feedback}")

    candidate = candidates[0]
    finish_reason = str(candidate.get("finishReason", "") or "")
    if finish_reason and finish_reason not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        abort(stage, f"{label} returned partial or blocked response (finishReason={finish_reason})")

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not parts:
        abort(stage, f"{label} response is missing content parts")

    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            texts.append(str(part["text"]))
    return ensure_non_empty_text(stage, "\n".join(texts), f"{label} text")


def validate_outline_segments(outline: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    segments = outline.get("segments", [])
    if not isinstance(segments, list) or not segments:
        abort(stage, "Outline JSON does not contain any segments")

    validated = []
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            abort(stage, f"Outline segment {idx + 1} is not an object")
        title = ensure_non_empty_text(stage, segment.get("title", ""), f"outline segment {idx + 1} title")
        key_points = segment.get("key_points", [])
        if not isinstance(key_points, list) or not key_points:
            abort(stage, f"Outline segment {idx + 1} has no key_points")
        cleaned_points = [str(point).strip() for point in key_points if str(point).strip()]
        if not cleaned_points:
            abort(stage, f"Outline segment {idx + 1} key_points are empty")
        try:
            word_budget = int(segment.get("word_budget", 0))
        except (TypeError, ValueError) as exc:
            abort(stage, f"Outline segment {idx + 1} has invalid word_budget: {segment.get('word_budget')}", cause=exc)
        if word_budget <= 0:
            abort(stage, f"Outline segment {idx + 1} has non-positive word_budget: {word_budget}")
        tone = ensure_non_empty_text(stage, segment.get("tone", ""), f"outline segment {idx + 1} tone")
        validated.append(
            {
                "title": title,
                "key_points": cleaned_points,
                "word_budget": word_budget,
                "tone": tone,
            }
        )
    return validated


def validate_transcript_entries(entries: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        abort(stage, "Transcript payload is not a list")

    validated = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            log_warn(f"⚠️ [{stage}] Skip invalid entry at index {idx}: not a dict")
            continue
        if "speaker_id" not in entry or "dialog" not in entry:
            log_warn(f"⚠️ [{stage}] Skip invalid entry at index {idx}: missing speaker_id/dialog")
            continue
        try:
            speaker_id = int(entry["speaker_id"])
        except (TypeError, ValueError):
            log_warn(f"⚠️ [{stage}] speaker_id at index {idx} cannot be parsed: {entry['speaker_id']} -> use 0")
            speaker_id = 0
        if speaker_id not in (0, 1):
            log_warn(f"⚠️ [{stage}] Normalize unexpected speaker_id at index {idx}: {speaker_id} -> 0")
            speaker_id = 0

        dialog = str(entry["dialog"]).strip()
        if not dialog:
            log_warn(f"⚠️ [{stage}] Skip empty dialog at index {idx}")
            continue
        validated_entry = {"speaker_id": speaker_id, "dialog": dialog}
        if entry.get("style"):
            style = str(entry["style"]).strip()
            if style:
                validated_entry["style"] = style
        validated.append(validated_entry)

    if not validated:
        abort(stage, "No valid transcript entries were produced")
    return validated


def ensure_file(path: str, stage: str, label: str, *, min_size: int = 1) -> str:
    target = Path(path)
    if not target.exists():
        abort(stage, f"{label} does not exist: {target}")
    size = target.stat().st_size
    if size < min_size:
        abort(stage, f"{label} is too small ({size} bytes): {target}")
    return str(target)


def write_json_file(path: str, payload: Any, stage: str, label: str) -> str:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(target)
    except Exception as exc:
        abort(stage, f"Failed to write {label} to {target}: {type(exc).__name__}: {exc}", cause=exc)
    ensure_file(str(target), stage, label)
    log_info(f"💾 {label} saved: {target}")
    return str(target)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdftotext or fallback to Python."""
    for attempt in range(1, 4):
        try:
            log_info(f"📄 Extracting PDF (attempt {attempt}/3): {pdf_path}")
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                log_info(f"✅ PDF extracted with pdftotext: {len(result.stdout)} chars")
                return result.stdout
            if result.returncode != 0:
                log_warn(f"⚠️ pdftotext failed (exit {result.returncode}): {result.stderr[:200]}")
        except FileNotFoundError:
            log_warn("⚠️ pdftotext not found, skipping")
            break
        except Exception as exc:
            log_warn(f"⚠️ pdftotext exception on attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(attempt)

    if fitz is None:
        log_warn("⚠️ PyMuPDF not installed")
        abort(
            "input-parse",
            "Cannot extract PDF. Install poppler-utils (apt install poppler-utils) or pymupdf (pip install pymupdf)",
        )

    for attempt in range(1, 3):
        try:
            log_info(f"📄 Trying PyMuPDF fallback (attempt {attempt}/2)")
            doc = fitz.open(pdf_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                log_info(f"✅ PDF extracted via PyMuPDF: {len(text)} chars")
                return text
            raise RuntimeError("empty text output")
        except Exception as exc:
            log_warn(f"⚠️ PyMuPDF attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < 2:
                time.sleep(attempt)

    abort(
        "input-parse",
        "Cannot extract PDF. Install poppler-utils (apt install poppler-utils) or pymupdf (pip install pymupdf)",
    )


def _extract_main_text_from_html(html: str) -> str:
    """Extract clean main text from raw HTML using stdlib html.parser.

    Strips script/style/nav/header/footer tags and collapses whitespace.
    Falls back to raw text if extraction yields too little content.
    """
    class _Extractor(HTMLParser):
        SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}
        BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "article", "section", "li", "blockquote", "td", "th"}

        def __init__(self):
            super().__init__()
            self._skip_depth = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP_TAGS:
                self._skip_depth += 1
            if tag in self.BLOCK_TAGS:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data):
            if self._skip_depth == 0:
                stripped = data.strip()
                if stripped:
                    self.parts.append(stripped)

    parser = _Extractor()
    try:
        parser.feed(html)
    except Exception as exc:
        log_warn(f"⚠️ HTML parsing fallback failed: {type(exc).__name__}: {exc}")

    text = " ".join(parser.parts)
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_text_from_url(url):
    """Fetch text from URL, extracting clean main body content from HTML."""
    headers = {"User-Agent": "Mozilla/5.0 paper2podcast/1.0"}
    last_err = None

    for attempt in range(1, 4):
        try:
            log_info(f"🌐 Fetching URL (attempt {attempt}/3): {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()

            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                log_info("📄 URL returned PDF-like content, extracting text")
                temp_dir = current_work_dir()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=temp_dir) as f:
                    f.write(data)
                    tmp_path = f.name
                try:
                    return extract_text_from_pdf(tmp_path)
                finally:
                    os.unlink(tmp_path)

            raw = data.decode("utf-8", errors="replace")

            # Detect HTML and extract main text to strip JS/CSS noise
            if "<html" in raw[:2000].lower() or "<!doctype" in raw[:200].lower():
                text = _extract_main_text_from_html(raw)
                if len(text) > 500:
                    log_info(f"✅ URL fetch + HTML extraction: {len(raw)} → {len(text)} chars")
                    return text
                # Extraction yielded too little; fall back to raw (truncated) text
                log_warn(f"⚠️ HTML extraction yielded only {len(text)} chars, falling back to raw text")

            ensure_non_empty_text("input-parse", raw, f"URL response from {url}")
            log_info(f"✅ URL fetch success: {len(raw)} chars")
            return raw

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            log_warn(f"⚠️ URL fetch failed (attempt {attempt}/3): {type(exc).__name__}: {exc}")
            if attempt < 3:
                sleep_s = attempt * 5
                log_info(f"⏳ Retry in {sleep_s}s")
                time.sleep(sleep_s)

    abort("input-parse", f"Failed to fetch URL after retries: {url}: {last_err}")


def load_input(input_path):
    """Load input text from file, URL, or stdin."""
    if input_path == "-":
        text = sys.stdin.read()
        ensure_non_empty_text("input-parse", text, "stdin input")
        log_info(f"📝 Loaded stdin input: {len(text)} chars")
        return text
    if input_path.startswith("http://") or input_path.startswith("https://"):
        return extract_text_from_url(input_path)
    p = Path(input_path)
    if not p.exists():
        abort("input-parse", f"File not found: {input_path}")
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(p))
    try:
        text = p.read_text(encoding="utf-8")
        ensure_non_empty_text("input-parse", text, f"input file {input_path}")
        log_info(f"📄 Loaded text file: {input_path} ({len(text)} chars)")
        return text
    except Exception as exc:
        abort("input-parse", f"Failed to read file {input_path}: {type(exc).__name__}: {exc}", cause=exc)


def call_gemini(api_key, model, body, timeout=300, retries=2, request_label="Gemini request"):
    """Call Gemini API and return parsed response with retry + diagnostics."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            call_started = time.monotonic()
            log_info(f"🤖 {request_label}: model={model} attempt {attempt}/{retries} timeout={timeout}s")
            if attempt > 1:
                backoff = attempt * 5
                log_info(f"⏳ Retrying {request_label} ({attempt}/{retries}) after {backoff}s")
                time.sleep(backoff)
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                response = json.loads(payload)
                log_info(f"✅ {request_label}: received response in {time.monotonic() - call_started:.1f}s")
                return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_err = exc
            status = getattr(exc, "code", "")
            text = getattr(exc, "read", lambda: b"")()
            if isinstance(text, (bytes, bytearray)):
                text_snip = text[:200].decode("utf-8", errors="replace")
            else:
                text_snip = str(text)[:200]
            if status == 429:
                wait = 10 * attempt
                log_warn(f"⚠️ {request_label} rate limited (429), wait {wait}s before retry ({attempt}/{retries})")
                time.sleep(wait)
                continue

            if attempt < retries:
                log_warn(f"⚠️ {request_label} failed (attempt {attempt}/{retries}): {type(exc).__name__}: {exc} {text_snip}")
                continue
            log_error(
                f"❌ {request_label} failed after {retries} attempts: {type(exc).__name__}: {exc} {text_snip}"
            )
    if last_err:
        raise last_err
    raise RuntimeError(f"{request_label} failed")


def search_paper_context(api_key, paper_text, model="gemini-3.1-pro-preview"):
    """Use Gemini with grounding (Google Search) to find publication info and context for the paper.

    Returns a context string with publication date, venue, citations, and related work info.
    """
    begin_stage("context-search", "searching for publication context")
    log_info("🔍 Searching for paper background and publication context...")

    # Extract title and key info from first ~2000 chars of paper
    header = paper_text[:3000]

    body = {
        "contents": [
            {"role": "user", "parts": [
                {"text": f"""Based on the following paper header, search for this paper's:
1. Full title and authors
2. Publication venue and date (conference/journal, year)
3. Number of citations (approximate)
4. Key related/competing works published around the same time or after
5. Current status: has this work been superseded by newer methods? What is the current state-of-the-art in this area?

Return a concise factual summary (no opinions, just facts). If you cannot find the paper, state that clearly.

Paper header:
{header}"""}
            ]}
        ],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.1,
        }
    }

    for attempt in range(1, 4):
        try:
            result = call_gemini(
                api_key,
                model,
                body,
                timeout=120,
                retries=3,
                request_label="context search",
            )
            context = extract_text_from_gemini_result(result, "context-search", "Context search")
            log_info(f"✅ Context retrieved: {len(context)} chars")
            return context
        except Exception as e:
            log_warn(f"⚠️ Context search attempt {attempt}/3 failed: {type(e).__name__}: {e}")
            if attempt < 3:
                time.sleep(attempt * 8)
                continue
            record_degradation("context-search", f"context search failed after retries: {type(e).__name__}: {e}", "proceed without external context")
            return "（未能检索到论文背景信息，请根据论文内容本身进行讨论。）"


# ---------------------------------------------------------------------------
# Step 1: Generate podcast script
# ---------------------------------------------------------------------------

_STYLE_GENTLE_ADDENDUM = (
    '\n\n## style 字段额外约束（重要）\n'
    'style 字段的情绪描述必须克制、轻微。禁止使用强烈的形容词（如\u201c兴奋\u201d、\u201c震撼\u201d、\u201c振奋\u201d、\u201c犀利\u201d、\u201c惊讶\u201d、\u201c不可思议\u201d、\u201c感叹\u201d、\u201c钦佩\u201d）。'
    '只使用温和的描述（如\u201c语气平稳\u201d、\u201c略带好奇\u201d、\u201c冷静分析\u201d、\u201c微微疑问\u201d）。'
    '宁可不填 style，也不要写夸张的情绪标注。'
)


def generate_script(api_key, paper_text, lang="zh", duration=10, model="gemini-3.1-pro-preview", skip_search=False, tts_model=None):
    """Generate structured podcast script JSON from paper text."""
    if lang != "zh":
        abort("config", f"只支持中文播客生成，已移除英文 prompt：lang={lang!r}")
    begin_stage("segment-generation", f"single-stage script generation lang={lang} duration={duration}m")
    log_info(f"📝 Generating podcast script ({lang}, ~{duration}min) with {model}...")

    # Truncate very long papers
    max_chars = 120000
    if len(paper_text) > max_chars:
        paper_text = paper_text[:max_chars] + "\n\n[... truncated for length ...]"
        log_info(f"✂️ Paper text truncated to {max_chars} chars")

    # Step 0: Search for paper context
    if skip_search:
        context_block = "（未提供背景信息，请根据论文内容本身进行讨论。）"
    else:
        context_block = search_paper_context(api_key, paper_text, model)

    # Inject current date so the model has correct temporal awareness
    current_date = datetime.now().strftime("%Y-%m-%d")

    word_count = duration * 250
    prompt = PROMPT_ZH.format(duration=duration, word_count=word_count, context_block=context_block)
    if _is_flash_tts_model(tts_model):
        prompt += _STYLE_GENTLE_ADDENDUM
    prompt = f'【当前日期：{current_date}】请根据当前日期判断论文/文章的时间线，不要把过去的文章说成"未来"。\n\n' + prompt

    body = {
        "contents": [
            {"role": "user", "parts": [
                {"text": prompt + f"\n\n<source_content>\n{paper_text}\n</source_content>"}
            ]}
        ],
        "generationConfig": {
            "maxOutputTokens": 16384,
            "temperature": 0.9,
            "responseMimeType": "application/json",
        }
    }

    try:
        result = call_gemini(
            api_key,
            model,
            body,
            timeout=420,
            retries=3,
            request_label="single-stage script generation",
        )
        raw = extract_text_from_gemini_result(result, "segment-generation", "Single-stage script generation")
        log_info(f"🧩 Received script draft: {len(raw)} chars")
    except Exception as exc:
        abort("segment-generation", f"Script generation failed: {type(exc).__name__}: {exc}", cause=exc)

    # Parse JSON
    script = parse_json_payload(raw, "segment-generation", "generated script")
    if isinstance(script, list):
        validated = validate_transcript_entries(script, "segment-generation")
    else:
        validated = validate_transcript_entries(script.get("podcast_transcripts", []), "segment-generation")

    total_chars = sum(len(e.get("dialog", "")) for e in validated)
    log_info(f"✅ Script generated: {len(validated)} turns, {total_chars} chars")
    return {"podcast_transcripts": validated}



def generate_script_multistage(api_key, paper_text, lang="zh", duration=10, model="gemini-3.1-pro-preview", skip_search=False, tts_model=None):
    """Multi-stage podcast script generation: Outline → Write → Review."""
    if lang != "zh":
        abort("config", f"只支持中文播客生成，已移除英文 prompt：lang={lang!r}")
    log_info(f"📝 [Multi-stage] Generating podcast script ({lang}, ~{duration}min) with {model}...")

    # Truncate very long papers
    max_chars = 120000
    if len(paper_text) > max_chars:
        paper_text = paper_text[:max_chars] + "\n\n[... truncated for length ...]"
        log_info(f"✂️ Paper text truncated to {max_chars} chars")

    # Step 0: Search for paper context
    if skip_search:
        context_block = "（未提供背景信息，请根据论文内容本身进行讨论。）"
    else:
        context_block = search_paper_context(api_key, paper_text, model)

    current_date = datetime.now().strftime("%Y-%m-%d")
    word_count = duration * 450  # 提高目标字数，确保内容充实

    # ========== Stage 1: Outline ==========
    begin_stage("outline-generation", f"outline lang={lang} duration={duration}m")
    log_info("📋 Stage 1/3: Generating outline...")

    outline_prompt = OUTLINE_PROMPT_ZH.format(
        duration=duration, word_count=word_count, context_block=context_block
    )

    date_prefix = f'【当前日期：{current_date}】\n\n'

    outline_body = {
        "contents": [
            {"role": "user", "parts": [
                {"text": date_prefix + outline_prompt + f"\n\n<source_content>\n{paper_text}\n</source_content>"}
            ]}
        ],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.4,
            "responseMimeType": "application/json",
        }
    }

    try:
        result = call_gemini(
            api_key,
            model,
            outline_body,
            timeout=120,
            retries=3,
            request_label="outline generation",
        )
        outline_raw = extract_text_from_gemini_result(result, "outline-generation", "Outline generation")
        outline = parse_json_payload(outline_raw, "outline-generation", "outline")
        segments = validate_outline_segments(outline, "outline-generation")
        log_info(f"✅ Outline: {len(segments)} segments")
        for i, seg in enumerate(segments):
            log_info(f"   [{i+1}] {seg.get('title', '?')} ({seg.get('word_budget', '?')} 字, {seg.get('tone', '?')})")
    except Exception as exc:
        record_degradation("outline-generation", f"outline generation failed: {type(exc).__name__}: {exc}", "single-stage script generation")
        return generate_script(api_key, paper_text, lang, duration, model, skip_search, tts_model)

    # ========== Stage 2: Write segments ==========
    begin_stage("segment-generation", f"outline segments={len(segments)}")
    log_info("✍️ Stage 2/3: Writing segments...")

    all_transcripts = []
    prev_context = "（这是播客的开头）"
    segment_failures = []

    for i, seg in enumerate(segments):
        segment_title = seg.get("title", f"Segment {i+1}")
        segment_tone = seg.get("tone", "neutral")
        key_points = seg.get("key_points", [])
        word_budget = seg.get("word_budget", word_count // len(segments))

        log_info(f"   ✍️ Writing segment {i+1}/{len(segments)}: {segment_title} ({word_budget} 字)...")

        seg_prompt = SEGMENT_PROMPT_ZH.format(
            segment_title=segment_title,
            segment_tone=segment_tone,
            key_points=json.dumps(key_points, ensure_ascii=False),
            word_budget=word_budget,
            segment_position_rules=_segment_position_rules_zh(i, len(segments)),
            prev_context=prev_context,
        )
        if _is_flash_tts_model(tts_model):
            seg_prompt += _STYLE_GENTLE_ADDENDUM

        seg_body = {
            "contents": [
                {"role": "user", "parts": [
                    {"text": date_prefix + seg_prompt + f"\n\n<source_content>\n{paper_text}\n</source_content>"}
                ]}
            ],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.9,
                "responseMimeType": "application/json",
            }
        }

        try:
            result = call_gemini(
                api_key,
                model,
                seg_body,
                timeout=300,
                retries=3,
                request_label=f"segment {i + 1} generation",
            )
            seg_raw = extract_text_from_gemini_result(result, "segment-generation", f"Segment {i + 1} generation")
            seg_script = parse_json_payload(seg_raw, "segment-generation", f"segment {i + 1} script")
            if isinstance(seg_script, list):
                validated = validate_transcript_entries(
                    seg_script,
                    f"segment-generation segment {i + 1}",
                )
            else:
                validated = validate_transcript_entries(
                    seg_script.get("podcast_transcripts", []),
                    f"segment-generation segment {i + 1}",
                )

            seg_chars = sum(len(e["dialog"]) for e in validated)
            log_info(f"   ✅ Segment {i+1}: {len(validated)} turns, {seg_chars} chars")

            all_transcripts.extend(validated)

            # Build context for next segment (last 2 turns)
            if validated:
                last_turns = validated[-2:] if len(validated) >= 2 else validated
                prev_context = "\n".join(
                    f"{speaker_name_for(e['speaker_id'])}: {e['dialog'][:100]}..."
                    for e in last_turns
                )

        except Exception as exc:
            segment_failures.append(f"segment {i + 1} '{segment_title}': {type(exc).__name__}: {exc}")
            log_error(f"❌ Segment {i+1} failed: {type(exc).__name__}: {exc}")
            break

    if segment_failures or not all_transcripts:
        reason = "; ".join(segment_failures) if segment_failures else "outline segments produced no transcript entries"
        record_degradation("segment-generation", reason, "single-stage script generation")
        return generate_script(api_key, paper_text, lang, duration, model, skip_search, tts_model)

    # ========== Stage 3: Review ==========
    log_info("🔍 Stage 3/3: Quality review...")

    total_chars = sum(len(e["dialog"]) for e in all_transcripts)
    target_chars = word_count
    ratio = total_chars / target_chars if target_chars > 0 else 1.0

    if ratio < 0.7 or ratio > 1.4:
        log_warn(f"⚠️ Length deviation: {total_chars} chars vs target {target_chars} (ratio: {ratio:.2f})")
        # Could trigger rewrite here in future iterations
    else:
        log_info(f"✅ Length check passed: {total_chars} chars (ratio: {ratio:.2f})")

    # Quick review of first and last segments
    for check_label, check_entries in [("Opening", all_transcripts[:6]), ("Closing", all_transcripts[-6:])]:
        script_text = "\n".join(
            f"{speaker_name_for(e['speaker_id'])}: {e['dialog']}"
            for e in check_entries
        )

        review_prompt = REVIEW_PROMPT_ZH.format(
            key_points="开场/收尾质量",
            word_budget=len(script_text),
            script_text=script_text,
        )

        try:
            review_body = {
                "contents": [{"role": "user", "parts": [{"text": review_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 2048,
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                }
            }
            result = call_gemini(api_key, model, review_body, timeout=60, retries=2)
            review_raw = result["candidates"][0]["content"]["parts"][0]["text"]
            review = json.loads(review_raw)
            passed = review.get("pass", True)
            issues = review.get("issues", [])
            if not passed:
                log_warn(f"⚠️ {check_label} review flagged issues: {issues}")
            else:
                log_info(f"✅ {check_label} review passed")
        except Exception as exc:
            log_warn(f"⚠️ {check_label} review skipped: {exc}")

    log_info(f"✅ Multi-stage script complete: {len(all_transcripts)} turns, {total_chars} chars")
    return {"podcast_transcripts": all_transcripts}


# ---------------------------------------------------------------------------
# Step 2: TTS with Gemini multi-speaker
# ---------------------------------------------------------------------------

_STYLE_SOFTEN_MAP = {
    "兴奋": "轻快",
    "犀利": "冷静",
    "震撼": "平静",
    "振奋": "平稳",
    "惊讶": "微微好奇",
    "钦佩": "客观",
    "不可思议": "淡然",
    "感叹": "平和",
    "疑虑": "平静质疑",
    "充满好奇": "略带好奇",
    "充满疑虑": "略带疑问",
    "充满惊讶": "略感意外",
    "恍然大悟": "理解了",
    "赞赏": "认可",
}

_STYLE_SOFTEN_COMPILED = None

_STYLE_AUDIO_TAGS = (
    ("疑问", "略带疑问"),
    ("好奇", "略带好奇"),
    ("严肃", "严肃"),
    ("冷静", "冷静"),
    ("沉稳", "沉稳"),
    ("平稳", "平稳"),
    ("平和", "平和"),
    ("转折", "短暂停顿"),
    ("停顿", "短暂停顿"),
    ("强调", "轻微强调"),
    ("恍然", "轻微恍然"),
    ("理解", "轻微恍然"),
    ("认可", "认可"),
    ("质疑", "冷静质疑"),
    ("犀利", "干燥克制"),
)

SPEAKER_NAMES = ("Alice", "Bob")


def speaker_name_for(speaker_id: int) -> str:
    return SPEAKER_NAMES[0] if speaker_id == 0 else SPEAKER_NAMES[1]


def _soften_style(style: str) -> str:
    """Replace intense emotion words in style annotations with milder variants."""
    global _STYLE_SOFTEN_COMPILED
    if _STYLE_SOFTEN_COMPILED is None:
        _STYLE_SOFTEN_COMPILED = [(re.compile(re.escape(k)), v) for k, v in _STYLE_SOFTEN_MAP.items()]
    result = style
    for pat, repl in _STYLE_SOFTEN_COMPILED:
        result = pat.sub(repl, result)
    return result


def _style_to_audio_tag(style: str) -> str:
    """Convert generated Chinese style notes into concise audio tags."""
    softened = _soften_style(style.strip())
    for keyword, tag in _STYLE_AUDIO_TAGS:
        if keyword in softened:
            return tag
    return softened


def build_tts_text(entries, lang="zh", segment_position: str = "middle", tts_model: str | None = None):
    """Build the text input for a Gemini TTS multi-speaker conversation segment."""
    if lang != "zh":
        abort("config", f"只支持中文播客生成，已移除英文 prompt：lang={lang!r}")
    if not entries:
        return ""

    lines = [_build_tts_header(segment_position)]
    for entry in entries:
        lines.append(f"{speaker_name_for(entry['speaker_id'])}: {entry['dialog']}")

    return "\n\n".join(lines)


def build_single_turn_tts_text(entry, lang="zh", segment_position: str = "middle") -> str:
    """Build a single-speaker TTS prompt for one transcript turn."""
    if lang != "zh":
        abort("config", f"只支持中文播客生成，已移除英文 prompt：lang={lang!r}")
    segment_note = _tts_sample_context(segment_position)
    return f"""\
TTS this single Chinese podcast line using the configured voice for {speaker_name_for(entry['speaker_id'])}.

Delivery:
- Standard mainland Mandarin; calm, deadpan, staccato, light, fluent, slightly fast.
- No Taiwanese accent, Northeastern accent, heavy erhua, drama, heavy emphasis, over-articulation, or added words.
- Read the line exactly. Do not add speaker labels, transitions, summaries, or extra words.
- Segment note: {segment_note}

Line:
{entry["dialog"]}"""


def _entry_bytes(entry):
    """Byte size of a single transcript entry when rendered for TTS."""
    return len(f"{speaker_name_for(entry['speaker_id'])}: {entry['dialog']}".encode("utf-8")) + 4


def _find_best_split(entries, start, max_payload_bytes):
    """Find the best split point in entries[start:] that fits within byte budget.

    Prefers splitting at 'topic boundaries':
      1. Speaker 0 (Alice) turn after a Speaker 1 (Bob) turn - typically a new
         discussion segment (Alice introduces next topic after Bob finishes), but
         only if the current segment already contains both speakers.
      2. Any speaker change as fallback, keeping both sides of the exchange in
         the same segment.
      3. Hard byte limit as last resort.

    Returns the exclusive end index of the segment.
    """
    total = 0
    last_valid = start  # at least one entry
    best_boundary = None

    for i in range(start, len(entries)):
        total += _entry_bytes(entries[i])
        if total > max_payload_bytes and i > start:
            break
        last_valid = i + 1

        # Look for natural boundaries (only after accumulating some content)
        if i > start and total > max_payload_bytes * 0.4:
            prev_speaker = entries[i - 1]["speaker_id"]
            curr_speaker = entries[i]["speaker_id"]
            # Best: Alice starts speaking after Bob - likely a new topic. Split
            # before Alice, but only after at least one Alice/Bob exchange.
            if curr_speaker == 0 and prev_speaker == 1:
                prior_speakers = {entry["speaker_id"] for entry in entries[start:i]}
                if len(prior_speakers) > 1:
                    best_boundary = i
            # Decent: any speaker change. Include the new speaker so the TTS
            # request remains a real multi-speaker exchange.
            elif curr_speaker != prev_speaker and best_boundary is None:
                best_boundary = i + 1

    # If we found a natural boundary within range, prefer it
    if best_boundary is not None and best_boundary <= last_valid:
        return best_boundary

    return last_valid


def split_transcript(entries, max_bytes=4000, lang="zh", tts_model: str | None = None):
    """Split transcript entries into multi-speaker TTS conversation segments."""
    if lang != "zh":
        abort("config", f"只支持中文播客生成，已移除英文 prompt：lang={lang!r}")
    if not entries:
        return []

    segments = []
    header_bytes = len(_build_tts_header("middle").encode("utf-8")) + 100
    payload_budget = max(1, max_bytes - header_bytes)
    start = 0

    while start < len(entries):
        end = _find_best_split(entries, start, payload_budget)
        if end <= start:
            end = start + 1
        segments.append(entries[start:end])
        start = end

    # Log segment info
    for i, seg in enumerate(segments):
        seg_bytes = sum(_entry_bytes(e) for e in seg) + header_bytes
        turns = len(seg)
        speakers = sorted({speaker_name_for(e["speaker_id"]) for e in seg})
        log_info(
            f"  📦 Segment {i+1}/{len(segments)}: "
            f"{seg_bytes} bytes, {turns} turns [{', '.join(speakers)}]"
        )

    return segments


async def call_gemini_async(session, api_key, model, body, timeout=600, request_label="Gemini async request"):
    """Call Gemini API asynchronously and return parsed JSON."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    try:
        call_started = time.monotonic()
        log_info(f"🤖 {request_label}: model={model} timeout={timeout}s")
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            text = await resp.text()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Non-JSON response (HTTP {resp.status}): {text[:300]}") from exc
            if resp.status >= 400:
                # Preserve upstream shape but ensure error field exists.
                if "error" not in payload:
                    payload = {"error": {"code": resp.status, "message": text[:500]}}
                return payload
            log_info(f"✅ {request_label}: received response in {time.monotonic() - call_started:.1f}s")
            return payload
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{request_label} timed out") from exc
    except Exception as exc:
        raise RuntimeError(f"{request_label} failed: {type(exc).__name__}: {exc}") from exc


def is_rate_limited_error(err):
    code = err.get("code")
    status = str(err.get("status", ""))
    message = str(err.get("message", ""))
    return code == 429 or "RESOURCE_EXHAUSTED" in status or "RESOURCE_EXHAUSTED" in message


async def convert_pcm_to_mp3(pcm_file, mp3_file):
    """Convert raw PCM to MP3 with ffmpeg in a worker thread."""
    for attempt in range(1, 3):
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                "-i", pcm_file, "-b:a", "128k", mp3_file
            ],
            capture_output=True,
            timeout=60,
            text=True,
        )
        if result.returncode == 0:
            ensure_file(mp3_file, "tts-audio-synthesis", "segment MP3 output")
            return
        if attempt < 2:
            log_warn(f"⚠️ ffmpeg convert failed on attempt {attempt}/2: {result.stderr.strip()[:300]}")
            await asyncio.sleep(attempt * 2)
            continue
        raise RuntimeError(result.stderr.strip()[:500] or "ffmpeg failed")


def _segment_metadata_path(mp3_file: str | Path) -> Path:
    return Path(f"{mp3_file}.meta.json")


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def _tts_output_paths(output_dir: str | Path, segment_idx: int) -> tuple[str, str, str]:
    stem = f"segment_{segment_idx:03d}"
    base_dir = Path(output_dir)
    return (
        str(base_dir / f"{stem}.pcm"),
        str(base_dir / f"{stem}.mp3"),
        str(base_dir / f".{stem}.tmp.{os.getpid()}.mp3"),
    )


async def _wait_for_tts_rate_limit(err: dict[str, Any], label: str, attempt: int, max_retries: int) -> bool:
    if not is_rate_limited_error(err):
        return False
    wait = (attempt + 1) * 30
    log_warn(
        f"  ⏳ {label} rate limited, wait {wait}s "
        f"(attempt {attempt + 1}/{max_retries})"
    )
    await asyncio.sleep(wait)
    return True


def _extract_tts_audio_b64(resp_data: dict[str, Any], label: str) -> str | None:
    candidates = resp_data.get("candidates")
    if not candidates:
        log_error(
            f"  ❌ {label} returned no candidates: "
            f"{json.dumps(resp_data, ensure_ascii=False)[:500]}"
        )
        return None
    finish_reason = str(candidates[0].get("finishReason", "") or "")
    if finish_reason and finish_reason not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        log_error(f"  ❌ {label} returned partial/blocked audio (finishReason={finish_reason})")
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        log_error(f"  ❌ {label} missing content parts")
        return None
    audio_b64 = parts[0].get("inlineData", {}).get("data")
    if not audio_b64:
        log_error(f"  ❌ {label} missing inline audio data")
        return None
    return audio_b64


async def _write_tts_audio_files(
    audio_b64: str,
    *,
    pcm_file: str,
    mp3_tmp: str,
    mp3_file: str,
    expected_metadata: dict[str, Any],
    output_label: str,
) -> bytes:
    audio_data = base64.b64decode(audio_b64)
    if not audio_data:
        raise RuntimeError("decoded audio payload is empty")
    Path(pcm_file).write_bytes(audio_data)
    await convert_pcm_to_mp3(pcm_file, mp3_tmp)
    ensure_file(mp3_tmp, "tts-audio-synthesis", f"{output_label} audio")
    Path(mp3_tmp).replace(mp3_file)
    _write_json_atomic(_segment_metadata_path(mp3_file), expected_metadata)
    ensure_file(mp3_file, "tts-audio-synthesis", f"{output_label} audio")
    return audio_data


def build_tts_segment_metadata(
    text: str,
    *,
    lang: str,
    voice_a: str,
    voice_b: str,
    tts_model: str,
    segment_idx: int,
    total_segments: int,
    segment_position: str,
    render_mode: str = "multi-speaker",
    speaker_id: int | None = None,
    voice_name: str | None = None,
) -> dict[str, Any]:
    """Build the exact identity for a generated TTS segment."""
    return {
        "schema_version": 1,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "lang": lang,
        "voice_a": voice_a,
        "voice_b": voice_b,
        "tts_model": tts_model,
        "segment_idx": segment_idx,
        "total_segments": total_segments,
        "segment_position": segment_position,
        "render_mode": render_mode,
        "speaker_id": speaker_id,
        "voice_name": voice_name,
    }


def is_reusable_tts_segment(mp3_file: str | Path, expected_metadata: dict[str, Any]) -> bool:
    """Return True only when an existing segment belongs to the current TTS request."""
    path = Path(mp3_file)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    metadata_path = _segment_metadata_path(path)
    if not metadata_path.exists():
        return False
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual_metadata == expected_metadata


async def tts_segment_async(
    session,
    api_key,
    entries,
    segment_idx,
    total_segments,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    segment_position: str = "middle",
):
    """Convert one segment of transcript entries to audio.

    Args:
        segment_position: Acoustic scene hint ('intro'/'technical'/'conclusion'/'middle').
            Passed through to build_tts_text() to append positional context to the TTS prompt.
    """
    pcm_file, mp3_file, mp3_tmp = _tts_output_paths(output_dir, segment_idx)
    label = f"Segment {segment_idx + 1}/{total_segments}"

    text = build_tts_text(entries, lang, segment_position, tts_model)
    ensure_non_empty_text("tts-audio-synthesis", text, f"TTS segment {segment_idx + 1} prompt")
    expected_metadata = build_tts_segment_metadata(
        text,
        lang=lang,
        voice_a=voice_a,
        voice_b=voice_b,
        tts_model=tts_model,
        segment_idx=segment_idx,
        total_segments=total_segments,
        segment_position=segment_position,
    )

    if is_reusable_tts_segment(mp3_file, expected_metadata):
        log_info(f"  ⏩ {label} exists with matching metadata, skipping")
        return mp3_file
    if os.path.exists(mp3_file):
        log_warn(f"  ♻️ {label} exists but metadata does not match; regenerating")
        Path(mp3_file).unlink(missing_ok=True)
        _segment_metadata_path(mp3_file).unlink(missing_ok=True)

    text_bytes = len(text.encode("utf-8"))
    log_info(f"  🎙️ {label} start ({text_bytes} bytes, {len(entries)} turns)")

    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "multiSpeakerVoiceConfig": {
                    "speakerVoiceConfigs": [
                        {
                            "speaker": "Alice",
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice_a}
                            },
                        },
                        {
                            "speaker": "Bob",
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice_b}
                            },
                        },
                    ]
                }
            }
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp_data = await call_gemini_async(
                session,
                api_key,
                tts_model,
                body,
                timeout=300,
                request_label=f"TTS segment {segment_idx + 1}/{total_segments}",
            )
            if "error" in resp_data:
                err = resp_data["error"]
                if await _wait_for_tts_rate_limit(err, label, attempt, max_retries):
                    continue
                log_error(f"  ❌ {label} API error: {err}")
                return None

            audio_b64 = _extract_tts_audio_b64(resp_data, label)
            if not audio_b64:
                return None

            audio_data = await _write_tts_audio_files(
                audio_b64,
                pcm_file=pcm_file,
                mp3_tmp=mp3_tmp,
                mp3_file=mp3_file,
                expected_metadata=expected_metadata,
                output_label=f"segment {segment_idx + 1}",
            )

            duration = len(audio_data) / (24000 * 2)
            log_info(f"  ✅ {label}: {duration:.1f}s")

            Path(pcm_file).unlink(missing_ok=True)
            return mp3_file

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                log_warn(
                    f"  ⚠️ {label} attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {e}. "
                    f"Retry in {wait}s."
                )
                await asyncio.sleep(wait)
            else:
                log_error(
                    f"  ❌ TTS failed for {label.lower()}: "
                    f"{type(e).__name__}: {e}"
                )
                return None
        finally:
            Path(pcm_file).unlink(missing_ok=True)
            Path(mp3_tmp).unlink(missing_ok=True)

    return None


async def tts_turn_async(
    session,
    api_key,
    entry,
    segment_idx,
    total_segments,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    segment_position: str = "middle",
):
    """Convert one transcript turn to audio with a single forced voiceConfig."""
    pcm_file, mp3_file, mp3_tmp = _tts_output_paths(output_dir, segment_idx)
    label = f"Turn {segment_idx + 1}/{total_segments}"

    speaker_id = entry["speaker_id"]
    voice_name = voice_a if speaker_id == 0 else voice_b
    text = build_single_turn_tts_text(entry, lang, segment_position)
    ensure_non_empty_text("tts-audio-synthesis", text, f"TTS turn {segment_idx + 1} prompt")
    expected_metadata = build_tts_segment_metadata(
        text,
        lang=lang,
        voice_a=voice_a,
        voice_b=voice_b,
        tts_model=tts_model,
        segment_idx=segment_idx,
        total_segments=total_segments,
        segment_position=segment_position,
        render_mode="per-turn",
        speaker_id=speaker_id,
        voice_name=voice_name,
    )

    if is_reusable_tts_segment(mp3_file, expected_metadata):
        log_info(f"  ⏩ {label} exists with matching metadata, skipping")
        return mp3_file
    if os.path.exists(mp3_file):
        log_warn(f"  ♻️ {label} exists but metadata does not match; regenerating")
        Path(mp3_file).unlink(missing_ok=True)
        _segment_metadata_path(mp3_file).unlink(missing_ok=True)

    text_bytes = len(text.encode("utf-8"))
    log_info(
        f"  🎙️ {label} start "
        f"({text_bytes} bytes, {speaker_name_for(speaker_id)}, voice={voice_name})"
    )

    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            },
        },
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp_data = await call_gemini_async(
                session,
                api_key,
                tts_model,
                body,
                timeout=300,
                request_label=f"TTS turn {segment_idx + 1}/{total_segments}",
            )
            if "error" in resp_data:
                err = resp_data["error"]
                if await _wait_for_tts_rate_limit(err, label, attempt, max_retries):
                    continue
                log_error(f"  ❌ {label} API error: {err}")
                return None

            audio_b64 = _extract_tts_audio_b64(resp_data, label)
            if not audio_b64:
                return None

            audio_data = await _write_tts_audio_files(
                audio_b64,
                pcm_file=pcm_file,
                mp3_tmp=mp3_tmp,
                mp3_file=mp3_file,
                expected_metadata=expected_metadata,
                output_label=f"turn {segment_idx + 1}",
            )

            duration = len(audio_data) / (24000 * 2)
            log_info(f"  ✅ {label}: {duration:.1f}s")

            Path(pcm_file).unlink(missing_ok=True)
            return mp3_file

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                log_warn(
                    f"  ⚠️ {label} attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {e}. Retry in {wait}s."
                )
                await asyncio.sleep(wait)
            else:
                log_error(
                    f"  ❌ TTS failed for {label.lower()}: "
                    f"{type(e).__name__}: {e}"
                )
                return None
        finally:
            Path(pcm_file).unlink(missing_ok=True)
            Path(mp3_tmp).unlink(missing_ok=True)

    return None


def _infer_segment_position(idx: int, total: int) -> str:
    """Infer an acoustic scene position label from segment index.

    Rules (matching task spec):
      - idx == 0               → 'intro'
      - idx == last            → 'conclusion'
      - idx == 1 and total>=4  → 'technical'
      - otherwise              → 'middle'
    """
    if idx == 0:
        return "intro"
    if idx == total - 1:
        return "conclusion"
    if idx == 1 and total >= 4:
        return "technical"
    return "middle"


async def run_tts_segments_async(
    api_key,
    segments,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    workers,
    render_mode="multi-speaker",
):
    """Run TTS generation with bounded async concurrency."""
    if aiohttp is None:
        abort("tts-audio-synthesis", "Missing dependency: aiohttp. Install with: pip install aiohttp")

    total = len(segments)
    results = [None] * total
    semaphore = asyncio.Semaphore(max(1, workers))

    async with aiohttp.ClientSession() as session:
        async def run_one(idx, seg_entries):
            async with semaphore:
                position = _infer_segment_position(idx, total)
                if render_mode == "per-turn":
                    results[idx] = await tts_turn_async(
                        session,
                        api_key,
                        seg_entries[0],
                        idx,
                        total,
                        output_dir,
                        lang,
                        voice_a,
                        voice_b,
                        tts_model,
                        segment_position=position,
                    )
                else:
                    results[idx] = await tts_segment_async(
                        session,
                        api_key,
                        seg_entries,
                        idx,
                        total,
                        output_dir,
                        lang,
                        voice_a,
                        voice_b,
                        tts_model,
                        segment_position=position,
                    )

        tasks = [asyncio.create_task(run_one(i, seg)) for i, seg in enumerate(segments)]
        for task in asyncio.as_completed(tasks):
            await task

    return results


async def retry_tts_segments_serially(
    api_key,
    segments,
    failed_indexes,
    output_dir,
    lang,
    voice_a,
    voice_b,
    tts_model,
    render_mode="multi-speaker",
):
    """Retry failed TTS segments one by one as a degraded fallback path."""
    if aiohttp is None:
        abort("tts-audio-synthesis", "Missing dependency: aiohttp. Install with: pip install aiohttp")

    total = len(segments)
    results = {}
    async with aiohttp.ClientSession() as session:
        for idx in failed_indexes:
            log_info(f"🔁 Retrying segment {idx + 1}/{total} serially")
            position = _infer_segment_position(idx, total)
            if render_mode == "per-turn":
                results[idx] = await tts_turn_async(
                    session,
                    api_key,
                    segments[idx][0],
                    idx,
                    total,
                    output_dir,
                    lang,
                    voice_a,
                    voice_b,
                    tts_model,
                    segment_position=position,
                )
            else:
                results[idx] = await tts_segment_async(
                    session,
                    api_key,
                    segments[idx],
                    idx,
                    total,
                    output_dir,
                    lang,
                    voice_a,
                    voice_b,
                    tts_model,
                    segment_position=position,
                )
    return results


def _replace_or_copy(source: Path, target: Path) -> None:
    """Move source to target, falling back to copy across filesystems."""
    try:
        source.replace(target)
    except OSError:
        shutil.copyfile(source, target)
        source.unlink(missing_ok=True)


def concat_segments(segment_files, output_file, temp_dir: str | Path | None = None):
    """Concatenate MP3 segments into final output."""
    valid = []
    invalid = []
    for idx, segment_file in enumerate(segment_files):
        if not segment_file:
            invalid.append(f"segment {idx + 1}: missing path")
            continue
        path = Path(segment_file)
        if not path.exists():
            invalid.append(f"segment {idx + 1}: file not found ({segment_file})")
            continue
        if path.stat().st_size <= 0:
            invalid.append(f"segment {idx + 1}: file size is 0 ({segment_file})")
            continue
        valid.append(path.resolve())

    if invalid:
        abort("file-write", f"Cannot concatenate audio because some segments are invalid: {'; '.join(invalid)}")
    if not valid:
        abort("file-write", "No audio segments to concatenate")

    segment_dirs = {path.parent for path in valid}
    if len(segment_dirs) != 1:
        abort(
            "file-write",
            "Refusing to concatenate segments from multiple directories: "
            + ", ".join(str(path) for path in sorted(segment_dirs)),
        )

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(temp_dir) if temp_dir else target.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    target_tmp = temp_parent / f".{target.name}.tmp.{os.getpid()}.mp3"

    if len(valid) == 1:
        target_tmp.write_bytes(valid[0].read_bytes())
        _replace_or_copy(target_tmp, target)
    else:
        segment_dir = next(iter(segment_dirs))
        filelist_path = segment_dir / f"concat_{os.getpid()}_{int(time.time() * 1000)}.txt"
        concat_result = None
        try:
            with filelist_path.open("w", encoding="utf-8") as f:
                for segment_path in valid:
                    escaped_path = str(segment_path).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped_path}'\n")

            for attempt in range(1, 3):
                concat_result = subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(filelist_path), "-c:a", "libmp3lame", "-b:a", "128k", str(target_tmp)],
                    capture_output=True,
                    timeout=300,
                    text=True,
                )
                if concat_result.returncode == 0:
                    break
                if attempt < 2:
                    log_warn(f"⚠️ ffmpeg concat failed on attempt {attempt}/2: {concat_result.stderr[:300]}")
                    time.sleep(attempt * 2)
        finally:
            filelist_path.unlink(missing_ok=True)

        if concat_result is None or concat_result.returncode != 0:
            target_tmp.unlink(missing_ok=True)
            abort("file-write", f"ffmpeg concat failed: {(concat_result.stderr if concat_result else '')[:500]}")
        _replace_or_copy(target_tmp, target)

    ensure_file(output_file, "file-write", "final podcast MP3")

    # Get duration info
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", output_file],
        capture_output=True,
        text=True
    )
    try:
        duration = float(result.stdout.strip().split("=")[1])
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        log_info(f"🎉 Final podcast: {duration:.1f}s ({duration / 60:.1f}min), {size_mb:.1f}MB")
    except (IndexError, ValueError):
        log_info(f"🎉 Final podcast saved to: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RUN_CONTEXT
    RUN_CONTEXT = RunContext()
    exit_code = 0

    parser = argparse.ArgumentParser(description="Paper → Podcast Pipeline")
    parser.add_argument("input", help="PDF file, text file, URL, or '-' for stdin")
    parser.add_argument("--lang", default="zh", choices=["zh"], help="Language (Chinese only; default: zh)")
    parser.add_argument("--duration", type=int, default=10, help="Target duration in minutes (default: 10)")
    parser.add_argument("--voice-a", default="Kore", help="Voice for speaker 0/Alice (default: Kore)")
    parser.add_argument("--voice-b", default="Charon", help="Voice for speaker 1/Bob (default: Charon)")
    parser.add_argument("--script-model", default="gemini-3.1-pro-preview", help="Model for script generation (default: gemini-3.1-pro-preview)")
    parser.add_argument("--tts-model", default="gemini-3.1-flash-tts-preview", help="TTS model")
    parser.add_argument("--output", help="Output MP3 path")
    parser.add_argument("--script-only", action="store_true", help="Only generate script")
    parser.add_argument("--script", help="Use existing script JSON file")
    parser.add_argument("--max-segment-bytes", type=int, default=2800, help="Max bytes per TTS segment (default: 2800)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel TTS workers")
    parser.add_argument(
        "--tts-render-mode",
        choices=["per-turn", "multi-speaker"],
        default="per-turn",
        help="TTS rendering mode: per-turn forces one voiceConfig per transcript turn (default)",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Directory for this run's temporary files (default: /tmp/paper2podcast_runs/<run_id>)",
    )
    parser.add_argument("--skip-search", action="store_true", help="Skip background context search")
    parser.add_argument("--no-multistage", action="store_false", dest="multistage", help="Disable multi-stage pipeline (Outline -> Write -> Review)")
    parser.set_defaults(multistage=True)
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--api-key-file", help="File containing Gemini API key")
    parser.add_argument(
        "--log-file",
        default="",
        help="Write detailed debug logs to this path (default: <work-dir>/paper2podcast.log)",
    )
    args = parser.parse_args()

    try:
        work_dir = create_run_work_dir(args.work_dir or None)
        # Configure logging first so every later step is captured.
        resolved_log = args.log_file
        if not resolved_log:
            resolved_log = str(work_dir / "paper2podcast.log")
        configure_logging(resolved_log)
        get_run_context().log_path = resolved_log
        log_info(f"Log file: {resolved_log}")
        log_info(f"🗂️ Run workspace: {work_dir}")

        begin_stage("config", "validating CLI arguments and model config")
        if args.duration <= 0:
            abort("config", f"--duration must be > 0, got {args.duration}")
        if args.max_segment_bytes <= 0:
            abort("config", f"--max-segment-bytes must be > 0, got {args.max_segment_bytes}")
        if args.workers <= 0:
            abort("config", f"--workers must be > 0, got {args.workers}")
        args.script_model = normalize_gemini_model(args.script_model)
        args.tts_model = normalize_gemini_model(args.tts_model)

        needs_api = not (args.script and args.script_only)
        api_key = get_api_key(args) if needs_api else None

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            if args.input == "-":
                base_name = "stdin_podcast"
            elif args.input.startswith("http"):
                base_name = "url_podcast"
            else:
                base_name = Path(args.input).stem + "_podcast"
            output_path = str(work_dir / f"{base_name}.mp3")

        script_path = output_path.rsplit(".", 1)[0] + "_script.json"
        get_run_context().output_path = output_path
        get_run_context().script_path = script_path

        begin_stage("input-parse", f"source={args.input}")

        # Step 1: Get script
        if args.script:
            log_info(f"📄 Loading existing script: {args.script}")
            get_run_context().script_path = args.script
            try:
                script = json.loads(Path(args.script).read_text(encoding="utf-8"))
            except Exception as exc:
                abort("input-parse", f"Failed to load script file {args.script}: {type(exc).__name__}: {exc}", cause=exc)
            if not isinstance(script, dict):
                abort("input-parse", f"Script file did not contain a JSON object: {args.script}")
            script["podcast_transcripts"] = validate_transcript_entries(
                script.get("podcast_transcripts", []),
                "input-parse existing script",
            )
        else:
            paper_text = load_input(args.input)
            paper_text = ensure_non_empty_text("input-parse", paper_text, "parsed input text")
            log_info(f"📄 Input: {len(paper_text)} chars")
            if args.multistage:
                script = generate_script_multistage(api_key, paper_text, args.lang, args.duration, args.script_model, args.skip_search, args.tts_model)
            else:
                script = generate_script(api_key, paper_text, args.lang, args.duration, args.script_model, args.skip_search, args.tts_model)

            begin_stage("file-write", "writing generated script JSON")
            script_path = write_json_file(script_path, script, "file-write", "script JSON")
            get_run_context().script_path = script_path

        if args.script_only:
            log_info("📝 Script-only mode, done.")
            return exit_code

        entries = validate_transcript_entries(script.get("podcast_transcripts", []), "tts-audio-synthesis")

        # Step 2: Split into TTS segments
        begin_stage("tts-audio-synthesis", f"preparing TTS workers={args.workers} render_mode={args.tts_render_mode}")
        if args.tts_render_mode == "per-turn":
            segments = [[entry] for entry in entries]
            for i, seg in enumerate(segments):
                speaker = speaker_name_for(seg[0]["speaker_id"])
                seg_bytes = len(build_single_turn_tts_text(seg[0], args.lang).encode("utf-8"))
                log_info(f"  📦 Turn {i+1}/{len(segments)}: {seg_bytes} bytes [{speaker}]")
        else:
            segments = split_transcript(entries, args.max_segment_bytes, args.lang, args.tts_model)
        if args.tts_render_mode == "multi-speaker" and len(segments) == 1:
            # Large single-shot TTS calls can hang or time out due to oversized audio payloads.
            single_bytes = len(build_tts_text(segments[0], args.lang, tts_model=args.tts_model).encode("utf-8"))
            if single_bytes >= 3500:
                base_bytes = len(_build_tts_header("middle").encode("utf-8")) + 100  # padding
                target_segments = max(2, args.duration)  # ~1 minute per segment
                total_dialog_bytes = sum(_entry_bytes(e) for e in entries)
                target_payload = max(
                    900,
                    (total_dialog_bytes + target_segments - 1) // target_segments,
                )
                new_max_bytes = base_bytes + target_payload
                if new_max_bytes < args.max_segment_bytes:
                    record_degradation(
                        "tts-audio-synthesis",
                        f"single TTS segment too large ({single_bytes} bytes)",
                        f"re-split into ~{target_segments} segments with max {new_max_bytes} bytes",
                    )
                    segments = split_transcript(entries, new_max_bytes, args.lang, args.tts_model)
        if not segments:
            abort("tts-audio-synthesis", "Transcript split produced no TTS segments")
        log_info(f"📦 Split into {len(segments)} TTS segments ({args.tts_render_mode})")

        # Step 3: TTS each segment
        tmpdir = work_dir / "segments"
        tmpdir.mkdir(parents=True, exist_ok=True)
        log_info(f"🧹 Segment workspace: {tmpdir}")
        try:
            log_info(f"⚙️ Running async TTS with {max(1, args.workers)} workers...")
            segment_files = asyncio.run(
                run_tts_segments_async(
                    api_key,
                    segments,
                    str(tmpdir),
                    args.lang,
                    args.voice_a,
                    args.voice_b,
                    args.tts_model,
                    args.workers,
                    render_mode=args.tts_render_mode,
                )
            )

            failed = [i for i, segment_file in enumerate(segment_files) if not segment_file]
            if failed and args.workers > 1:
                record_degradation(
                    "tts-audio-synthesis",
                    f"segments failed with parallel workers={args.workers}: {[i + 1 for i in failed]}",
                    "retry failed segments serially with workers=1",
                )
                retried = asyncio.run(
                    retry_tts_segments_serially(
                        api_key,
                        segments,
                        failed,
                        str(tmpdir),
                        args.lang,
                        args.voice_a,
                        args.voice_b,
                        args.tts_model,
                        render_mode=args.tts_render_mode,
                    )
                )
                for idx, segment_file in retried.items():
                    segment_files[idx] = segment_file
                failed = [i for i, segment_file in enumerate(segment_files) if not segment_file]

            if failed:
                abort(
                    "tts-audio-synthesis",
                    f"TTS failed for segments {[i + 1 for i in failed]}; no partial podcast will be emitted",
                )

            # Step 4: Concatenate
            begin_stage("file-write", "writing final podcast MP3")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            concat_segments(segment_files, output_path, temp_dir=work_dir)
            ensure_file(output_path, "file-write", "final podcast MP3")
        finally:
            begin_stage("cleanup", f"keeping run workspace {work_dir}")

        log_info(f"📁 Output: {output_path}")
        return exit_code
    except PipelineError as exc:
        exit_code = exc.exit_code
        get_run_context().failed_stage = exc.stage
        return exit_code
    except KeyboardInterrupt:
        exit_code = 130
        get_run_context().failed_stage = get_run_context().failed_stage or get_run_context().current_stage or "interrupted"
        log_error("❌ [interrupt] Interrupted by user")
        return exit_code
    except Exception as exc:
        exit_code = 1
        stage = get_run_context().current_stage or "unknown"
        get_run_context().failed_stage = get_run_context().failed_stage or stage
        LOGGER.exception("Unhandled exception in stage %s", stage)
        log_error(f"❌ [{stage}] Unhandled exception: {type(exc).__name__}: {exc}")
        return exit_code
    finally:
        emit_final_summary(exit_code == 0, exit_code)


if __name__ == "__main__":
    sys.exit(main())
