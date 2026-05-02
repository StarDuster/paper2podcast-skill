# Session Bug Fixes — 2026-05-02

## Context

DeepSeek Attention 文章播客重跑过程中，暴露了脚本生成、Gemini multi-speaker TTS、分片拼接和断点续传的一组问题。本记录只写实际确认过的行为和修复，不把未完成项写成已修复。

测试材料：

- 本地正文：`/tmp/deepseek_attention_weixin_markdown.md`
- 生成脚本：`/tmp/deepseek_attention_v8_podcast_script.json`
- 隔离 TTS 输出：`/tmp/deepseek_attention_v8_isolated_podcast.mp3`

## Bug 1: Gemini JSON 返回 list 导致 `.get()` 崩溃

### 现象

多阶段生成时，segment generation 成功返回 JSON，但后续代码调用：

```python
seg_script.get("podcast_transcripts", [])
```

在 `seg_script` 为 list 时抛出：

```text
AttributeError: 'list' object has no attribute 'get'
```

### 根因

Gemini 有时直接返回：

```json
[
  {"speaker_id": 0, "dialog": "..."}
]
```

而不是：

```json
{"podcast_transcripts": [...]}
```

### 修复

`generate_script()` 和 `generate_script_multistage()` 都必须支持 list/dict 两种形态：

```python
script = parse_json_payload(raw, stage, label)
if isinstance(script, list):
    validated = validate_transcript_entries(script, stage)
else:
    validated = validate_transcript_entries(script.get("podcast_transcripts", []), stage)
```

## Bug 2: AI Studio Composer 字段误当成 API 参数

### 现象

一开始误把 Style、Pace、Accent、Scene、Sample Context 等 UI 字段当成独立 API 参数。

### 根因

Gemini TTS API 的稳定控制面主要是：

- `responseModalities: ["AUDIO"]`
- `speechConfig`
- `voiceConfig.prebuiltVoiceConfig.voiceName`
- `multiSpeakerVoiceConfig.speakerVoiceConfigs`

AI Studio Composer 里的风格、场景、示例上下文、导演备注，本质上是 prompt 结构，不是单独 JSON 参数。

### 修复

TTS prompt 改为 Composer 风格文本结构：

- Audio Profile
- Scene
- Director's Notes
- Segment Context
- Reading Rules
- Transcript

脚本中的 `TTS_AUDIO_PROFILE_ZH`、`TTS_SCENE_ZH`、`TTS_DIRECTORS_NOTES_ZH`、`TTS_READING_RULES_ZH` 和 `_build_tts_header()` 是对应实现。

## Bug 3: 单说话人片段导致每段不是 Alice/Bob 混合对话

### 现象

旧实现按说话人切分，TTS 段落容易变成单声道片段，不符合 Gemini multi-speaker TTS 的使用方式。

### 修复

每个 TTS segment 内保留 Alice/Bob 混合对话，API 请求使用：

```json
"multiSpeakerVoiceConfig": {
  "speakerVoiceConfigs": [
    {"speaker": "Alice", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
    {"speaker": "Bob", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}}
  ]
}
```

prompt 中每行台词必须以 `Alice:` 或 `Bob:` 开头，并且 speaker 名称必须和 `speakerVoiceConfigs[].speaker` 完全一致。

## Bug 4: Bob 默认 voice 选错

### 现象

Bob 默认使用 `Puck`，实际听感偏欢快。

### 根因

Google 文档里 `Puck` 是 `Upbeat / 欢快`，不符合用户要求的 deadpan + staccato、冷面克制技术审稿人风格。

### 修复

默认值改为：

```text
Alice: Kore
Bob: Charon
```

`Charon` 更接近信息密集、稳重的技术讲解语气。

## Bug 5: TTS 长输入导致开头丢失或跳读

### 现象

音频开头没有从脚本第一行开始，甚至模型自行插入了“说到这儿”等脚本中不存在的过渡语。

### 根因

单个 TTS 请求文本过长，并且 prompt 中场景说明和转写内容边界不够硬，模型把开头内容当成上下文进行改写或总结。

### 修复

1. `TTS_READING_RULES_ZH` 明确要求：
   - 只朗读【转写内容】里的 Alice/Bob 台词
   - 从第一行开始逐行朗读
   - 禁止跳过、改写、合并、总结或补充台词
   - 音频开头不得自行加承接语
2. `--max-segment-bytes` 默认值降为 `2800`。
3. 自动重切阈值降到更小，目标约 1 分钟以内一段。

## Bug 6: 中间 segment 提前“感谢收听”

### 现象

脚本第 2、3 段中间出现：

```text
这期关于 CSA 的深度拆解就到这里，感谢大家收听。
```

### 根因

`SEGMENT_PROMPT_ZH` 要求“开头段有开场词、结尾段有结束词”，但没有把当前 segment 的全局位置传进去。模型把每个 segment 当成独立小节目，于是中间段也自行收尾。

### 修复

新增：

```python
_segment_position_rules_zh(index, total)
```

并在 `SEGMENT_PROMPT_ZH.format(...)` 中传入 `segment_position_rules`。

规则：

- 第一个 segment：允许开场，禁止结束词。
- 中间 segment：禁止“感谢收听”“下期再见”“本期就到这里”“今天就聊到这里”。
- 最后一个 segment：只允许最后 1-2 轮自然收尾。

同时加入简体中文硬约束，禁止繁体字、台湾腔用词和港澳台书面表达。

## Bug 7: 旧音频分片可能混入新结果

### 现象

重跑后音频内容疑似混入旧片段，尤其是固定分片目录或 resume 场景。

### 根因

旧代码只要发现 `segment_000.mp3` 已存在且大于 50KB，就直接跳过，不校验该文件是否属于当前脚本、当前 voice、当前模型、当前 prompt。`resume_tts.py` 也按文件名相信旧分片。

### 修复

为每个 TTS 分片写入：

```text
segment_000.mp3.meta.json
```

metadata 包含：

- `prompt_sha256`
- `voice_a`
- `voice_b`
- `tts_model`
- `segment_idx`
- `total_segments`
- `segment_position`
- `lang`

只有 mp3 和 metadata 完全匹配当前请求时才复用。没有 metadata 或不匹配的旧分片会被删除并重生成。

相关函数：

- `build_tts_segment_metadata()`
- `is_reusable_tts_segment()`
- `_segment_metadata_path()`
- `_write_json_atomic()`

## Bug 8: concat 文件隔离不足

### 现象

旧拼接逻辑把 filelist 写在输出路径旁边，且没有拒绝跨目录分片。手工拼接或旧 resume 可能把不同 run 的分片混到一起。

### 修复

`concat_segments()` 现在：

- 只接受来自同一个分片目录的文件。
- 跨目录分片直接报错。
- filelist 写入当前分片目录。
- 最终 MP3 写入临时 `.mp3` 后原子替换目标输出。

## Bug 9: ffmpeg 临时输出文件没有 `.mp3` 后缀

### 现象

离线回归测试中，ffmpeg 转码失败。

### 根因

临时输出文件名类似：

```text
.segment_000.mp3.tmp.<pid>
```

ffmpeg 无法从扩展名推断输出格式。

### 修复

临时输出文件改为 `.mp3` 结尾：

```text
.segment_000.tmp.<pid>.mp3
.<output>.tmp.<pid>.mp3
```

## 验证记录

### 静态检查

```bash
python3 -m py_compile scripts/paper2podcast.py scripts/resume_tts.py
python3 scripts/paper2podcast.py --help
```

### 离线回归

1. 手动放入假的旧 `segment_000.mp3`。
2. 第一次运行必须检测 metadata 不匹配并重生成。
3. 第二次相同 prompt/voice/model 才允许跳过。

结果：

```text
PASS: stale segment regenerated; matching metadata reused
```

### concat 回归

1. 同目录分片可以拼接。
2. 跨目录分片必须报错。

结果：

```text
PASS: concat uses isolated filelist/temp output and rejects mixed directories
```

### 真实 TTS

使用 v8 脚本重跑：

```text
Split into 20 TTS segments
Run workspace: /tmp/paper2podcast_runs/<run_id>
Final podcast: 562.0s (9.4min), 8.6MB
Output: /tmp/deepseek_attention_v8_isolated_podcast.mp3
```

日志中没有出现旧分片 `skipping`，说明本次分片全部重新生成。

## 已知待修

### Review Prompt 仍可能误报

Quality Review 阶段仍可能把“开场/收尾质量”当成技术内容标准，导致日志里误报“偏题”。这不一定污染脚本文本，但会误导排查。

后续应改为：

- review 开头时只检查开场是否存在、是否简体、是否过度修辞；
- review 结尾时只检查结束是否在最后 1-2 轮；
- 技术内容审查使用 outline 的真实 key_points，而不是抽象标签。

## 工作纪律

- URL 是 positional argument，没有 `--url`。
- 不要手工跨目录拼接 `segment_*.mp3`。
- 不要相信无 metadata 的旧分片。
- 修改三引号 prompt 后必须跑 `py_compile`。
- TTS 相关改动必须做离线回归，必要时再做真实 TTS。
