---
name: paper2podcast
description: "Convert PDFs, articles, text files or URLs into a Chinese dual-host technical podcast with Gemini generation, external script review and per-turn Gemini TTS."
---

# Paper → Podcast

把论文、技术文章、网页或本地文本转成中文双人技术播客。当前只支持 `--lang zh`。

## 默认配置

- 主命令的默认目标时长为 10 分钟，原生 PDF 草稿工具默认为 15 分钟；可通过 `--duration 15` 指定目标。脚本生成模型为 `gemini-3.1-pro-preview`。
- 默认语音模型为 `gemini-3.8-flash-tts`。旧的 `gemini-3.1-flash-tts-preview` 选项会映射到 3.8；Pro 可通过 `--tts-model gemini-2.5-pro-preview-tts` 显式选择。
- Alice 使用 Kore，Bob 使用 Charon。默认逐轮渲染，每轮只绑定一个声音，轮间静音为 350 毫秒。
- 3.8 的 `parts[].text` 只包含原始台词；朗读要求写入 `speech_metadata.style`，多说话人模式按 part 设置 `speech_metadata.speaker`。
- 3.8 使用自然的中国大陆标准普通话，平翘舌区分准确，翘舌音发音适度；吐字清楚、语流连贯，按语意自然停顿，避免逐字强调和刻意拉开音节。Pro 保留原有朗读指令。
- 3.8 返回 WAV 容器时只读取音频帧，避免尾部元数据被当成原始 PCM 音频。分片缓存身份包含台词、朗读元数据、模型和声音。

## 安装与凭据

需要 Python 3.11+、`aiohttp`、`openai`、`python-dotenv`、ffmpeg 和 poppler-utils（提供 pdftotext）。建议在虚拟环境中安装 Python 依赖。

生成侧通过 `GEMINI_API_KEY` 或 `--api-key-file` 配置凭据。显式 `--provider vertex` 需要可用的 Vertex 配置。只有选择 Hermes 凭据解析或 Hermes 模型适配时才需要 Hermes 源码位于 Python 导入路径；独立运行请使用显式模型地址和凭据配置。

不要把凭据、环境文件、音频、PDF、日志或工作目录提交到仓库。

## 输入与生成流程

主命令的 positional `input` 接受本地 PDF、文本、Markdown、URL 或 `-`（标准输入），没有 `--url` 参数。主命令遇到 PDF 会提取文本。

保留原始 PDF 的图表、公式和版式时，先运行原生 PDF 草稿工具：

```bash
python scripts/script_from_pdf_native.py input.pdf draft.json --duration 15
python scripts/paper2podcast.py input.pdf --script draft.json --output podcast.mp3
```

原生 PDF 工具也接受 arXiv 和 PDF URL，检查 PDF magic、模型完成状态和 speaker_id，原子保存明确标记为未审阅的草稿。它从 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` 读取凭据。

默认背景研究使用独立 Codex CLI（`gpt-6-astra`）和原生网页搜索。主研究失败后，只回退一次到 DeepSeek（`deepseek-v4-pro`）与 Tavily。回退需要 `DEEPSEEK_API_KEY`、可选 `DEEPSEEK_BASE_URL` 和 `TAVILY_API_KEY`。研究过程保存真实参考资料、来源、失败原因和用量；两条路径都失败时明确记录资料缺口。`--search-provider`、`--search-model` 和对应独立模型配置可覆盖默认路由。`--skip-search` 跳过研究；导入已有脚本也跳过研究和生成。

脚本生成采用大纲、分段写作和外部审阅流程。最终台词应有技术密度，忠实于来源，避免夸张判断、中段提前收尾和重复内容。

## 审阅与脚本复用

默认审阅器为 `deepseek` / `deepseek-v4-pro`。显式凭据使用 `--review-api-key-file`；可通过 `--review-base-url` 指定兼容地址。DeepSeek 的显式密钥未指定地址时使用其官方地址，其他 provider 的显式密钥必须配地址。GPT 审阅只允许 Hermes `openai-codex` 路由。

`--script` 仅跳过生成和背景研究，仍经过审阅。审阅需要真实 positional 来源或 `--review-source`；失败、超时、无效 JSON 或异常长度会停止流程，不进入语音合成。仅显式 `--no-script-review` 跳过审阅，输出标记为 `reviewed=false/status=skipped`。

审阅凭据绑定规范化台词、来源文本和审阅路由。三者未变才复用，修改台词、替换来源或更换审阅器会重新审阅。哈希用于完整性记录，不提供防恶意伪造的签名保证。

每条成功路径都在合成前原子保存 `<work-dir>/final_script.json`，实际路径由日志和摘要的 `script_path` 给出。导入同路径脚本时使用独立文件名保护原稿。审阅前另存草稿检查点。

```json
{"podcast_transcripts": [{"speaker_id": 0, "dialog": "我们先来看这个方法的基本假设。"}, {"speaker_id": 1, "dialog": "关键约束是模型需要访问完整的上下文。"}]}
```

## 使用示例

```bash
python scripts/paper2podcast.py "https://example.com/article" --duration 15 --output podcast.mp3
python scripts/paper2podcast.py article.md --script-only --output podcast.mp3
python scripts/paper2podcast.py article.md --script run/final_script.json --output podcast.mp3
python scripts/paper2podcast.py article.md --skip-search --output podcast.mp3
```

`--workers` 默认为 2。`--tts-render-mode multi-speaker` 为实验模式，`--max-segment-bytes` 只影响该模式。默认生产模式为 `per-turn`。

## 断点恢复与音频检查

`scripts/resume_tts.py` 是需编辑配置区的手工模板。设置脚本、分片目录、输出、provider 和凭据配置；默认声音为 Kore / Charon，默认模型为 `gemini-3.8-flash-tts`。它只复用身份完全匹配的分片；更改台词、模型、声音或朗读元数据后会重新生成。

拼接只接受同一目录的分片，并以临时文件原子替换最终 MP3。完整运行后用 ffprobe 检查时长和大小，核对每轮是否成功；需要诊断结尾时检查末段音频与最终文件，避免只凭播放器听感判断。

## 离线验证

```bash
python -B -m pytest tests -q -p no:cacheprovider
python scripts/paper2podcast.py --help
```

测试使用受控响应及本地服务，不代表真实付费模型调用成功。
