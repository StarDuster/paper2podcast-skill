# TTS Feedback: NCCL Gin / DeepEPv2 Podcast (2026-05-03)

## 背景

2026-05-03 为用户生成了 DeepEPv2 相关微信公众号文章的中文播客。

- 来源 URL: `https://mp.weixin.qq.com/s/31NUL8v5M9w7ZbykXZowRg`
- 脚本模型: `gemini-3.1-pro-preview`
- TTS 模型: `gemini-3.1-flash-tts-preview`
- 音色: Kore (Alice) + Charon (Bob)
- 输出: 7.2min, 6.6MB
- Script 降级: context-search 全部失败（MAX_TOKENS），segment 5 也 MAX_TOKENS，最终单阶段生成

## 用户反馈

> "帮我重做nccl gin那期播客吧，我不是很喜欢那次的tts"

用户明确表示不满意 TTS 质量。具体不满方向尚未确认（可能是 flash TTS 音质不够好，也可能是音色选择问题）。

## 处理要点

- 重新生成前必须先询问用户偏好：换 TTS 模型？换音色？脚本也重做？
- 若换模型可以考虑 `gemini-2.5-pro-preview-tts`（质量更好但更慢）
- 若换音色需要在生成前确认用户期望的风格