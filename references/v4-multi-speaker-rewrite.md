# paper2podcast per-turn TTS 架构说明

## 当前结论

Gemini `multiSpeakerVoiceConfig` 在中文长播客场景里无法稳定保证可听辨 speaker identity。当前生产默认路径改为 `per-turn`：每个 JSON turn 单独发起一个 TTS 请求，并用普通 `voiceConfig` 强制绑定该 turn 的音色。

核心目标是让 `speaker_id -> voiceName` 由代码决定，而不是让 TTS 模型在 multi-speaker prompt 中自行区分 Alice/Bob。

## API 结构

per-turn TTS 请求体必须包含：

```json
{
  "contents": [
    {
      "parts": [
        {"text": "..."}
      ]
    }
  ],
  "generationConfig": {
    "responseModalities": ["AUDIO"],
    "speechConfig": {
      "voiceConfig": {
        "prebuiltVoiceConfig": {"voiceName": "Charon"}
      }
    }
  }
}
```

要求：

- `speaker_id=0` 映射 Alice。
- `speaker_id=1` 映射 Bob。
- Alice 默认 `voiceName=Kore`。
- Bob 默认 `voiceName=Charon`。
- 单个请求只包含一个 turn，不要让模型自行切换 speaker。

## Prompt 结构

`build_single_turn_tts_text()` 生成单 turn prompt：

```text
TTS this single Chinese podcast line using the configured voice for Bob.

Delivery:
- Standard mainland Mandarin; calm, deadpan, staccato, light, fluent, slightly fast.
- No Taiwanese accent, Northeastern accent, heavy erhua, drama, heavy emphasis, over-articulation, or added words.
- Read the line exactly. Do not add speaker labels, transitions, summaries, or extra words.

Line:
...
```

关键点：

- 不朗读 speaker label。
- 不插入 `[冷静]`、`[疑问]` 等 inline style tag。
- 只读当前 turn 的台词。
- 禁止跳过、改写、合并、总结、补充台词。

## 分段策略

默认 `per-turn` 不使用 `split_transcript()`。主流程直接将每个 transcript entry 包成一个 TTS segment：

```python
segments = [[entry] for entry in entries]
```

`--max-segment-bytes` 只影响 `--tts-render-mode multi-speaker` 实验路径。

如果用户明确要试 multi-speaker，可以运行：

```bash
python3 scripts/paper2podcast.py /tmp/article.md --tts-render-mode multi-speaker --max-segment-bytes 1800
```

但若出现 Bob/Alice 声线漂移，应回到默认 per-turn。

## Voice 默认值

| 角色 | speaker_id | 默认 voice | 原因 |
| :--- | :--- | :--- | :--- |
| Alice | `0` | `Kore` | Firm，适合冷静主持人 |
| Bob | `1` | `Charon` | Informative，适合技术专家/审稿人 |

不要把 Bob 默认改回 `Puck`。Puck 是 Upbeat / 欢快，和 deadpan + staccato 不匹配。

## 分片复用与防污染

每个生成出的 mp3 必须带 metadata sidecar：

```text
segment_000.mp3
segment_000.mp3.meta.json
```

metadata 包含当前 TTS 请求身份：

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

只有 metadata 完全匹配时才允许跳过生成。旧 mp3、没有 metadata 的 mp3、prompt 或 voice 改过的 mp3 都必须重生成。

相关函数：

- `build_tts_segment_metadata()`
- `is_reusable_tts_segment()`
- `tts_turn_async()`
- `tts_segment_async()`

## 拼接

必须使用 ffmpeg 重新编码：

```bash
ffmpeg -y -f concat -safe 0 -i <filelist> -c:a libmp3lame -b:a 128k output.mp3
```

`concat_segments()` 还必须保证：

- 所有分片来自同一个目录。
- filelist 在分片目录中生成。
- 最终输出先写临时 `.mp3`，再原子替换目标文件。

不要手工拼接多个旧目录里的 `segment_*.mp3`。

## 验证方法

### 代码检查

```bash
python3 -m py_compile scripts/paper2podcast.py scripts/resume_tts.py
python3 scripts/paper2podcast.py --help
```

### 分片污染回归

测试逻辑：

1. 在临时目录里手动放入假的旧 `segment_000.mp3`。
2. 调用 `tts_turn_async()` 或 `tts_segment_async()`。
3. 预期第一次 metadata 不匹配，旧文件被删掉并重生成。
4. 第二次相同 prompt/voice/model 才允许复用。

期望输出：

```text
PASS: stale segment regenerated; matching metadata reused
```

### concat 回归

测试逻辑：

1. 同目录两个分片可以拼接。
2. 不同目录的分片必须拒绝。

期望输出：

```text
PASS: concat uses isolated filelist/temp output and rejects mixed directories
```

### 真实 TTS 检查

跑完后检查：

```bash
ffprobe -v quiet -show_entries format=duration,size -of default=noprint_wrappers=1 /tmp/output.mp3
```

必要时抽取开头：

```bash
ffmpeg -y -i /tmp/output.mp3 -t 12 -ar 16000 -ac 1 -acodec pcm_s16le /tmp/output_start.wav
```

开头必须从脚本第一行开始，不得出现脚本外承接语。
