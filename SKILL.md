---
name: paper2podcast
description: "Convert papers, articles, PDFs, local text, or URLs into a Chinese dual-host technical podcast using Gemini script generation and per-turn Gemini TTS. Use when the user wants a document turned into a serious seminar-style podcast audio file with stable speaker voices."
---

# Paper → Podcast

把论文、技术文章、网页或本地文本转成中文双人研讨式播客音频。

当前实现只支持中文播客生成。不要再尝试英文 prompt、英文脚本模式，`--lang` 只能是 `zh`。

## 当前流程

1. **提取文本**：从 PDF、文本文件、URL，或 stdin 读取正文。
2. **背景搜索**：默认启用 Gemini + Google Search，补充论文背景、时间点和相关上下文；可用 `--skip-search` 跳过。
3. **生成脚本**：优先使用多阶段流程：outline → segment generation → review。输出 JSON：`{"podcast_transcripts": [...]}`。
4. **TTS 渲染**：默认 `per-turn`。每个 JSON turn 单独发起一个 TTS 请求，用单说话人的 `voiceConfig` 强制绑定音色。
5. **可选 multi-speaker**：`--tts-render-mode multi-speaker` 仍可用于实验，但不作为生产默认路径。
6. **拼接输出**：用 ffmpeg 重新编码拼接所有本次分片，生成最终 MP3。

## 默认声音与风格

### 角色

- **Alice / speaker_id=0 / voice-a=Kore**：主持人、资深研究者，负责引导讨论和总结直觉。声音坚定、冷静、克制。**Starduster 偏好 pro TTS，生成时默认使用 `--tts-model gemini-2.5-pro-preview-tts`。**
- **Bob / speaker_id=1 / voice-b=Charon**：技术专家、审稿人视角，负责追问细节和边界条件。声音信息密集、稳重，不要再用 Puck 作为默认值。

Puck 在 Google 文档中是 `Upbeat / 欢快`，不适合当前 deadpan + staccato 的技术播客风格。除非用户明确要求更轻快，否则 Bob 默认保持 `Charon`。

- **GitHub PDF 链接**：直接从 GitHub `blob` 页面提取会失败。必须使用 Raw 链接，通常是在 URL 末尾添加 `?raw=true`。
- **Flash 逻辑瓶颈与时长缩水**：在 2026-05-14 的任务中，使用 `gemini-3-flash-preview` 作为脚本模型时，在大纲生成阶段出现了 `AttributeError: 'list' object has no attribute 'get'`（模型返回了非预期 JSON 格式）。这触发了自动降级到“单阶段生成”模式。单阶段模式受限于模型单次输出 token 限制，通常只能维持 5-6 分钟的时长。**Starduster 明确表示即使在 503 高峰期，也应优先尝试 Pro 模型以保证时长和逻辑质量，而不是为了躲避 503 而主动选择导致严重降级的 Flash 模型。**
- **Pro 模型降级**：即使使用 `gemini-3.1-pro-preview`，在生成极其密集或长篇幅的 Segment 时仍可能触发 `finishReason=MAX_TOKENS`。此时系统会降级到 single-stage 模式。如果最终 `ffprobe` 时长仍符合预期（如 10 分钟），则该降级是可接受的，无需重跑。
- **背景搜索失效**：对于尚未被广泛讨论的新论文（如 2026 年发表），`context-search` 可能会失败。在这种情况下，脚本生成更依赖模型对 PDF 本身的理解。

## ⚠️ 关键坑点 (Critical Pitfalls)

- **JSON 格式不一致**：Flash 模型在处理多阶段 `outline -> segment` 流程时，返回的 JSON 可能与 `AIAgent` 期望的结构不符，导致解析崩溃并降级。
- **降级后的时长预期**：如果日志显示 `-> single-stage script generation`，产出的播客时长将远低于 10 分钟。如果用户对时长有硬要求，应在修复 Bug 后用 Pro 模型重跑。
- **Google Gemini API 503 / 高负载降级**：`gemini-3.1-pro-preview` 在高峰期（尤其日本夜间/US 白天交界）可能持续返回 HTTP 503（model experiencing high demand）。这是模型级别不可用，不是 rate limit，`--skip-search` 无效。如果用户未明确许可降级到 Flash，应优先重试 Pro。

### Subagent 超时
对于目标时长 10 分钟以上的播客，TTS 渲染非常耗时。使用 `delegate_task` 时可能会因为 10 分钟的默认限制而超时。**对于时长 > 5 分钟或多任务并行的请求，强烈建议直接使用 `terminal(background=true)`。**

### 源文章确认
生成播客前先确认用户给的 URL 就是他们想做成播客的内容。如果用户表述模糊（如"nccl gin 那期"），追问明确链接再执行。避免做错文章重来浪费时间和 API 成本。

### 处理“重做 (Redo)”请求

当用户要求“重做”某篇论文或文章的播客时，应遵循以下流程：

1. **历史溯源**：使用 `session_search` 搜索该文章的 URL 或 arXiv ID。
2. **分析失败/优化点**：检查之前的运行记录。是时长不够？TTS 质量差？还是生成中途崩了？
3. **确认版本更新**：对于 arXiv 论文，检查是否有新的版本（如 v1 -> v3）。
4. **参数优化**：
   - 如果之前因为时长短，确保本次使用 Pro 模型且 `--duration` 足够。
   - 如果之前 TTS 质量不佳，确保使用 `gemini-2.5-pro-preview-tts`。
   - 如果之前超时，检查 `--max-segment-bytes` 设置。
5. **执行策略**：对于这种已知复杂的“重做”任务，优先使用 `terminal(background=true)` 运行，并在回复中告知用户你基于历史记录做了哪些针对性改进。

### 监控后台任务
- **脚本生成降级 (MAX_TOKENS / Schema Error)**：如果某段脚本生成触发了 `MAX_TOKENS` 限制，或者模型输出的 JSON 格式不符合预期（例如 Flash 模型返回 list 而非 dict 导致 Python `AttributeError`），系统会自动降级到单阶段脚本生成。这种降级是“保命”机制，但会导致最终时长由于模型单次输出长度限制而大幅缩水（通常只能撑 5-6 分钟）。**为了保证 10 分钟以上的足额时长，优先使用 Pro 模型进行脚本生成。**
- **并行任务处理**：当用户同时请求多个播客时，应为每个 URL 启动独立的后台进程，不要在同一个 `delegate_task` 中串行执行。
- **文件名规范**：如果输入包含 arXiv ID（如 `2411.01783`），优先使用该 ID 作为输出文件名（例如 `arxiv_2411_01783.mp3`），而不是默认的 `url_podcast.mp3`。这有助于用户通过文件名快速识别内容。
- **动态输出路径**：`scripts/paper2podcast.py` 会将结果放在 `/tmp/paper2podcast_runs/<timestamp>/` 下。**不要猜测**输出路径为 `/tmp/arxiv_*.mp3`。必须从脚本输出或 `paper2podcast.log` 中获取 `output_path` 的真实值。
- **发送 Caption**：`tg_send_audio.py` 的 caption 参数如果以 `@` 开头，会读取文件内容。确保路径正确（通常在 work_dir 下）。

## TTS 口播要求

per-turn TTS prompt 只服务单个 turn。不要在正文里插入 `[冷静]`、`[疑问]` 这类 inline style tag，避免干扰 voice 绑定。语气通过全局 delivery 指令控制。

必须保持：

- 标准大陆普通话。
- 不要台湾腔、东北腔、重儿化音。
- 不要重读关键词，不要咬文嚼字，不要戏剧化反问。
- 英文术语可以保留原文，朗读时清楚准确。

## 脚本生成要求

脚本生成 prompt 已加入以下硬约束：

- 必须使用中国大陆简体中文，禁止繁体字、台湾腔用词和港澳台书面表达。
- 允许必要技术术语、论文标题、模型名和方法名保留英文。
- 禁止过度比喻和夸张修辞，保持技术严肃性。
- 禁止词包括但不限于：“降维打击”“暴力美学”“效率狂魔”“断崖式下跌”“炸裂”“封神”“天花板”。
- 第一个 segment 允许自然开场，但禁止提前结束。
- 中间 segment 必须直接承接上文，禁止“感谢收听”“下期再见”“本期就到这里”“今天就聊到这里”等收尾话术。
- 最后一个 segment 只有最后 1-2 轮允许自然结束词，且不要升华或煽情。

如果生成脚本仍出现中段收尾、繁体、夸张词，先修 `SEGMENT_PROMPT_ZH` 和 `_segment_position_rules_zh()`，不要靠手工剪音频掩盖问题。

## 命令

```bash
SKILL_DIR=/root/.hermes/skills/devops/paper2podcast
cd "${SKILL_DIR}"
python3 scripts/paper2podcast.py <input> [options]
```

`<input>` 是 positional argument，可以是：

- 本地 PDF 文件
- 本地文本/Markdown 文件
- URL
- `-`，从 stdin 读取

没有 `--url` 选项。

### 常用参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `input` | 必填 | PDF、文本文件、URL，或 `-` |
| `--lang` | `zh` | 只支持中文 |
| `--duration` | `10` | 目标时长，单位分钟 |
| `--voice-a` | `Kore` | Alice / speaker_id=0 |
| `--voice-b` | `Charon` | Bob / speaker_id=1 |
| `--script-model` | `gemini-3.1-pro-preview` | 脚本生成模型 |
| `--tts-model` | `gemini-2.5-pro-preview-tts` | TTS 模型（pro 音质，Starduster 确认偏好） |
| `--output` | 自动 | 输出 MP3 路径 |
| `--script-only` | 关闭 | 只生成脚本 JSON，不合成音频 |
| `--script` | 无 | 使用已有脚本 JSON，跳过脚本生成 |
| `--skip-search` | 关闭 | 跳过背景搜索 |
| `--max-segment-bytes` | `2800` | 每个 TTS 请求的最大 prompt 字节数 |
| `--workers` | `2` | TTS 并行数 |
| `--tts-render-mode` | `per-turn` | `per-turn` 强制每轮单 voice；`multi-speaker` 使用 Gemini 多说话人 API |
| `--work-dir` | 自动 | 本次运行的独立工作目录 |
| `--log-file` | 自动 | 日志文件路径 |

### 示例

```bash
# 从 URL 生成中文播客
python3 scripts/paper2podcast.py "https://example.com/article" --duration 10 --output /tmp/article_podcast.mp3

# 从本地 Markdown 生成
python3 scripts/paper2podcast.py /tmp/article.md --duration 5 --skip-search --output /tmp/article_podcast.mp3

# 先看脚本，再用同一份脚本合成音频
python3 scripts/paper2podcast.py /tmp/article.md --duration 5 --script-only --output /tmp/article_podcast.mp3
python3 scripts/paper2podcast.py /tmp/article.md --script /tmp/article_podcast_script.json --output /tmp/article_podcast.mp3

# 更短的 TTS 请求，生成更多分片
python3 scripts/paper2podcast.py /tmp/article.md --max-segment-bytes 2400 --output /tmp/article_podcast.mp3
```

## TTS 分段与音频安全

### 渲染策略

默认 `--tts-render-mode=per-turn`。每个 `podcast_transcripts` entry 单独合成：

- `speaker_id=0` → `voice-a` → 默认 `Kore`
- `speaker_id=1` → `voice-b` → 默认 `Charon`

⚠️ **TTS 质量反馈**：2026-05-03 NCCL Gin 播客使用了 gemini-3.1-flash-tts-preview，用户反馈不满意。重做时应先确认用户偏好或使用默认 pro TTS（gemini-2.5-pro-preview-tts）。

`--max-segment-bytes` 只影响 `multi-speaker` 实验模式。生产路径不要再依赖 Gemini 在同一个请求里区分 Alice/Bob。

### 防旧分片混入

当前实现会为每个 `segment_000.mp3` 写入旁路 metadata：`segment_000.mp3.meta.json`。

metadata 包含：

- `prompt_sha256`
- `voice_a`
- `voice_b`
- `tts_model`
- `segment_idx`
- `total_segments`
- `segment_position`
- `lang`
- `render_mode`
- `speaker_id`
- `voice_name`

只有 mp3 和 metadata 完全匹配当前请求时才会复用。没有 metadata、metadata 不匹配、voice/model/prompt 改过的旧分片都会被删除并重生成。

### 临时目录

主流程使用：

```text
/tmp/paper2podcast_runs/<run_id>/
```

每次运行独立目录。默认日志、segments、metadata、concat filelist 和最终临时输出都在这个目录内。不要把分片、日志、filelist 平铺到 `/tmp` 根目录。若指定 `--output /tmp/foo.mp3`，最终成品可以在该路径，但临时文件仍应留在 `--work-dir`。

### 拼接规则

`concat_segments()` 只接受来自同一个分片目录的文件。跨目录混拼会直接失败。

最终输出 MP3 使用临时 `.mp3` 文件原子替换，避免半成品污染目标输出。

## Resume / 断点续传

`scripts/resume_tts.py` 仍是手工模板。使用前必须编辑配置区：

```python
SCRIPT_JSON = "/tmp/your_script.json"
SEGMENTS_DIR = "/tmp/existing_segments_dir"
OUTPUT_MP3 = "/tmp/final_output.mp3"
VOICE_A = "Kore"
VOICE_B = "Charon"
TTS_MODEL = "gemini-2.5-pro-preview-tts"
LANG = "zh"
MAX_SEGMENT_BYTES = 2800
TTS_RENDER_MODE = "per-turn"
WORKERS = 1
```

新的 resume 行为：

- 不再按文件名相信 `segment_000.mp3`。
- 每个 segment 都走当前 metadata 校验。
- 匹配 metadata 的分片复用。
- 旧分片、缺 metadata 的分片、voice/model/prompt 不匹配的分片重生成。

如果你改了 `--max-segment-bytes`，resume 模板里的 `MAX_SEGMENT_BYTES` 必须同步，否则切分数量和 metadata 会变。

## 验证清单

代码修改后至少运行：

```bash
python3 -m py_compile scripts/paper2podcast.py scripts/resume_tts.py
python3 scripts/paper2podcast.py --help
```

真实 TTS 运行后检查：

```bash
ffprobe -v quiet -show_entries format=duration,size -of default=noprint_wrappers=1 /tmp/output.mp3
```

必要时抽取开头转写，确认第一句没有丢：

```bash
ffmpeg -y -i /tmp/output.mp3 -t 12 -ar 16000 -ac 1 -acodec pcm_s16le /tmp/output_start.wav
```

检查日志时注意：

- 是否出现 `Split into N TTS segments`。
- 是否每段都有 `Segment i/N start`。
- 是否出现 `exists with matching metadata, skipping`。新跑通常不应出现；resume 时可以出现。
- 是否出现 `metadata does not match; regenerating`。这说明旧分片被正确识别并重生成。

## 常见问题

### 中间段突然“感谢收听”

根因是 segment prompt 没有区分全局段落位置，模型把当前 segment 当成独立节目收尾。

修复点：

- `_segment_position_rules_zh(index, total)`
- `SEGMENT_PROMPT_ZH`

中间段必须禁止：

- “感谢收听”
- “下期再见”
- “本期就到这里”
- “今天就聊到这里”

### 繁体中文混入

脚本 prompt 必须显式要求中国大陆简体中文，禁止繁体字、台湾腔用词和港澳台书面表达。生成后如果还混入繁体，重新生成脚本，不要直接拿去 TTS。

### Bob 声线不对

检查 `--voice-b`。当前默认是 `Charon`。`Puck` 是欢快音色，不适合冷面技术审稿人。

### 音频像混入旧段落

优先检查日志和分片 metadata。当前主流程已防旧分片复用，但手工拼接、旧版 resume 脚本、固定分片目录仍可能造成污染。

正确处理：

1. 不要手工从多个目录拼接 `segment_*.mp3`。
2. 使用新版 `scripts/resume_tts.py`。
3. 确认 `segment_*.mp3.meta.json` 存在且匹配当前脚本、voice、model 和分段数量。

### Review 阶段误报“偏题”

目前 review 仍可能把“开场/收尾质量”误当成技术内容标准，导致误报。这不一定影响最终脚本文本，但它会污染日志判断。若要继续修，应让 review 使用实际大纲 key_points，而不是抽象的开场/收尾标签。

### JSON 返回是 list

Gemini 有时返回：

```json
[
  {"speaker_id": 0, "dialog": "..."}
]
```

而不是：

```json
{"podcast_transcripts": [...]}
```

代码必须同时支持 list 和 dict。不要移除 `isinstance(script, list)` 分支。

## API Key

优先级：

```text
--api-key > --api-key-file > GEMINI_API_KEY > ~/.hermes/secrets/gemini_api_key.txt
```

## 依赖

- `ffmpeg`
- `pdftotext` / poppler-utils
- `aiohttp`
- Gemini API key

## 发送音频

## 发送音频

**发送策略：播客生成后，只发送到固定的播客 topic（用 `<CHAT_ID>` / `<THREAD_ID>` 占位，按你自己的群组替换），不再额外发一份到私聊，避免消息重复占空间。**

优先用同目录脚本发送 Telegram 音频，caption 和音频在同一条消息里：
```bash
python3 scripts/tg_send_audio.py <CHAT_ID> <mp3_path> "@/tmp/caption.txt" --thread-id <THREAD_ID>
```

**Caption 格式规范：**
```text
🎙️ 播客：[标题]
🔗 [URL]
👤 [来源]
🗣️ zh · 双人研讨式 · [时长] min · [大小] MB

[内容简介]
```

**⚠️ 严格禁令：** Caption 必须严格遵守上述模版，禁止在下方或其中添加任何个人评论、推销语或“本期为满血版”之类的额外注释。Starduster 极其反感冗余的文案干扰。

**发送大文件坑点：** 播客音频通常 5–15 MB，走 Bot API `sendAudio` 上传较慢，`tg_send_audio.py` 在前台容易超时。建议用 `background: true` 或 `systemd-run` 放到后台跑，给足 300 秒超时。

**包装脚本坑点：** 如果 `paper2podcast.py` 已经输出 `status=success` 和 `📁 Output: ...`，但外层 bash 脚本随后因为 caption/postprocess 报错退出 1，不要重跑整条播客生成流程。先用 `ffprobe` 验证 MP3，再只修 caption/发送步骤。写 caption 时不要在 Python heredoc 里直接引用未注入的 shell 变量（如 `{TITLE}`）；应通过环境变量 `TITLE=$TITLE python3 ...` 传入，否则会触发 `NameError`。

**手动恢复技巧：** 若 TTS 在最后 1-2 段因 `finishReason=OTHER` 失败导致主程序不合成最终 MP3，可手动进入 `segments/` 目录，使用 `ls -v *.mp3 | sed "s/^/file '/;s/$/'/" > list.txt && ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp3` 进行物理拼接，挽救已生成的 99% 的进度。

**Telegram approval 坑点：** 从 agent terminal 发送带 emoji caption 时，Unicode variation selectors 可能触发安全审批；用户审批或新消息可能中断当前 sendAudio。遇到这种情况不要重做音频，改用纯文本 caption 或后台重试发送即可。

## 相关资源 (Resources)

- `references/session-2026-05-16-github-raw-pdf.md`: 关于 GitHub Raw PDF 链接处理与 Pro 模型 MAX_TOKENS 降级的记录。
- `references/gemini-503-2026-05-12.md`: Gemini API 503 高峰期 outage 记录与降级路径。
- `references/session-2026-05-08-delegate-timeout.md`: 记录了使用 delegate_task 导致超时的具体案例与恢复方法。
- `references/session-2026-05-06-timeouts-paths.md`: 关于 Subagent 超时与动态路径的教训。
- `references/session-2026-05-02-bugfixes.md`: 脚本解析与 TTS 错误处理。
- `references/tts-feedback-nccl-gin-2026-05-03.md`: TTS 质量反馈案例（用户不满意 flash TTS）。
