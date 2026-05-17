# paper2podcast 失败模式记录 (2026-05-15)

### 1. TTS 拦截: `finishReason=OTHER`
**现象**：TTS 请求成功发送，但返回的音频被标记为 `OTHER` 且内容不完整。
**复现场景**：通常发生在文本极短（如“下期见”）或段落结尾时。
**根因**：模型在处理极短音频流时可能由于服务端分片逻辑触发边界异常，或被误认为某种不合规输出。
**对策**：
- 如果只有 1-2 段失败，使用 `ffmpeg concat` 手动拼接已有的 `.mp3` 分片。
- 在脚本生成时，尽量避免出现少于 5 个字的单轮对话。

### 2. 脚本缩水: `MAX_TOKENS` 降级
**现象**：目标 10 分钟的播客，实际只产出 5-7 分钟，且日志显示 `single-stage script generation`。
**根因**：在 `segment-generation` 阶段，Pro 模型在扩充某一分段（通常是中间部分）时，由于生成内容过多触发了 `MAX_TOKENS` 限制，导致该分段管线失败。系统为了“保命”会抛弃多阶段生成，转而让模型一次性生成全文。由于模型单次输出上限通常只有几千 token，无法承载 10 分钟以上的对话。
**对策**：
- 确认 `MAX_TOKENS` 设置。
- 如果发生缩水，可考虑手动使用上一次运行留下的 `_script.json`（如果大纲生成成功了）进行 resume，或者缩短分段目标字数。

### 3. Shell 变量注入失败: `NameError`
**现象**：Python 脚本报错 `NameError: name 'TITLE' is not defined`。
**复现场景**：在 Bash 脚本的 heredoc 中混用 Shell 变量和 Python 逻辑。
**根因**：
```bash
python3 <<EOF
# 如果 $TITLE 没被 shell 展开成字符串，Python 会把它当成变量名
print($TITLE) 
EOF
```
**对策**：
- 在调用 Python 前 `export TITLE`，然后在 Python 内部使用 `os.environ['TITLE']` 获取。
- 或者在 heredoc 中确保展开后的内容带引号：`print("$TITLE")`。
