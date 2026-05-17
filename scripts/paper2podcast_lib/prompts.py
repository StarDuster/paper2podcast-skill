"""ZH podcast-generation and TTS prompt templates + speaker bindings.

Also owns small helpers that touch the same identifiers (segment position rules,
TTS sample-context blurbs, the conversation-header builder).
"""

from __future__ import annotations

SPEAKER_NAMES = ("Alice", "Bob")


def speaker_name_for(speaker_id: int) -> str:
    return SPEAKER_NAMES[0] if speaker_id == 0 else SPEAKER_NAMES[1]


def _is_flash_tts_model(model_name: str | None) -> bool:
    """Return True if the TTS model is a 'flash' variant that needs restrained prompts."""
    if not model_name:
        return False
    return "flash" in str(model_name).lower()


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

# style 字段额外约束——只在 flash 类 TTS 模型上启用，避免它把强情绪词演得太夸张。
_STYLE_GENTLE_ADDENDUM = (
    '\n\n## style 字段额外约束（重要）\n'
    'style 字段的情绪描述必须克制、轻微。禁止使用强烈的形容词（如“兴奋”、“震撼”、“振奋”、“犀利”、“惊讶”、“不可思议”、“感叹”、“钦佩”）。'
    '只使用温和的描述（如“语气平稳”、“略带好奇”、“冷静分析”、“微微疑问”）。'
    '宁可不填 style，也不要写夸张的情绪标注。'
)


# Long-form scene/director/reading-rule blurbs. Kept in case downstream tooling
# wants to compose them; the active TTS path uses the compact header below.
TTS_AUDIO_PROFILE_ZH = """\
这是双说话人中文播客音频。speaker 名称只用于绑定 API voice，不要朗读。
API speaker `Alice` 对应主持人声线；API speaker `Bob` 对应技术评论者声线。
必须严格使用 `multiSpeakerVoiceConfig` 中为每个 speaker 配置的 voice，不要根据句子内容、角色性格或性别重新推断声线。"""

TTS_SCENE_ZH = """\
两位 AI 研究者在安静、近距离收音的录音空间里讨论一篇论文。环境稳定，没有舞台感，没有广播新闻感。
这是一段真实的专业对话，不是朗读稿，也不是演讲。"""

TTS_DIRECTORS_NOTES_ZH = """\
风格：平铺直叙的自然说话。整体冷静、克制，不要戏剧化。
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


def _tts_sample_context(segment_position: str) -> str:
    return _TTS_SAMPLE_CONTEXT_ZH.get(segment_position, _TTS_SAMPLE_CONTEXT_ZH["middle"])


def _build_tts_header(segment_position: str = "middle") -> str:
    """Compact TTS prompt header that keeps speaker labels unambiguous."""
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


def _segment_position_rules_zh(index: int, total: int) -> str:
    """Script-generation rules for a segment's global podcast position."""
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
