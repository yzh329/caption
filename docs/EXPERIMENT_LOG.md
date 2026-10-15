# Experiment Log

本文件记录本地会议转录、翻译、总结和模型蒸馏/微调相关实验。记录重点是想法、假设、实验设置、结果和后续判断。

## 记录规则

- 每次实验追加一个日期条目。
- 先写清楚假设，再写实验设置和结果。
- 保留失败结果，尤其是 OOM、速度慢、质量差、格式不稳定等问题。
- 结论要能指导下一步配置，不只写主观感受。

## 2026-06-16

### 背景

实时链路里同时使用 VAD 和 `gpt-oss:20b` 做实时翻译/段落修复，导致延迟偏高。总结会议记录可以接受更长耗时，但实时翻译需要优先降低延迟。

### 已确定方向

- 实时翻译/段落修复保留，但默认模型改为 `qwen3:8b`。
- 总结继续使用 `gpt-oss:20b`，包括 summary chunk model 和 final summary model。
- 实时 Whisper 转录关闭 Silero VAD，避免每个短 chunk 额外经过模型级静音过滤。
- 后续蒸馏/微调优先面向会议总结，不优先训练实时翻译模型。

### 蒸馏/微调想法

Teacher 候选：

- DeepSeek API：适合生成高质量结构化 teacher labels，支持长上下文和 JSON 输出。
- `gpt-oss:20b`：适合作为本地 fallback 和 benchmark，也可参与 teacher 对照。

Student 候选：

- `Qwen3-8B`：第一版 baseline，主要用于验证数据集、训练脚本和评估流程。
- `Qwen3-14B`：主力目标，更能利用 RTX 5080 16GB VRAM，但需要 QLoRA、短序列、micro batch 1 和 gradient checkpointing。
- `gpt-oss:20b`：暂不作为第一阶段 student。虽然适合 16GB 推理，但训练长上下文 summary 风险较高，可能 OOM 或极慢。

### 当前假设

- 直接训练“整场 transcript -> deep summary”不适合 16GB VRAM。
- 更稳的训练任务拆分是：
  1. chunk -> facts/actions/decisions/risks JSON
  2. chunk JSONs -> coverage map
  3. coverage map -> short/deep/Chinese summary
- 14B QLoRA 可能是本机硬件利用率和可训练性的较好平衡点。

### 待验证

- `Qwen3-8B` baseline 相比原始 `qwen3:8b` 是否明显改善结构化总结。
- `Qwen3-14B` QLoRA 是否能在 RTX 5080 16GB 上稳定训练 4k、6k、8k 序列。
- DeepSeek teacher labels 是否比当前 `gpt-oss:20b` pipeline 更稳定、更完整。
- Teacher 输出是否需要 redaction，避免把敏感会议内容发到云端。

### 尚无结果

还没有开始生成 teacher labels，也没有开始训练。当前只是完成实时链路模型拆分和蒸馏路线判断。

## 2026-10-14：补齐 Qwen3.8 3-bit 正式接入

- 找回 2026-09-02 的试验产物：`output/live/2026-09-02T16-59-48-507Z_dataops9_2/meeting_notes_*_qwen38.md`，包含 partials、coverage 和中英文 short/deep 六份文件。
- 本机模型 `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S` 仍在，27.3B、IQ3_S，磁盘约 12GB。此前产物已生成，但应用下拉框及默认配置未更新。
- 正式接入：summary chunk/final、CLI、backfill、PowerShell pipeline 默认使用该模型；界面默认值直接读取后端常量，保留旧模型选项。
- 验证：7 项测试通过；使用真实 Ollama 请求生成三条英文摘要，负责人、截止日期和未决事项断言及来源校验通过，单次调用含加载耗时 78.3 秒。字幕服务已重启载入新配置。

### ASR 模型及 3 秒切片对照

- 使用当天 `propvu-10_14` 会议，预先等距选取六份约 30 秒音频，总计 179.94 秒，比较 distil-large-v3、large-v3、distil-large-v3.5 的 3/6/15/30 秒窗口，共 72 组输出；均使用现有 faster-whisper、CUDA float16、beam size 3、英语、无 VAD/提示词。
- 多个模型在 3 秒窗口下均输出 `make it configure`，15 秒改为 `make it configurable`；其他片段也出现长窗口输出更完整词句的现象。没有人工参考稿，不计算 WER，不宣称所有改写都正确；新版 distil 在这六段上没有一致的明显优势。
- 代码确认每段停止后等待 150 毫秒再重建录音器，且独立 ASR 请求没有共享上下文。应优先测试较长窗口和连续采集，保留模型替换选项。
- 新版 distil 权重已缓存，没有增加 Python 依赖，没有改实时采集设置或原始会议文件。
- 详细报告：`output/asr_comparison_2026-10-14/report.md`；完整输出及音频链接：`transcripts.md`；机器结果：`results.json`；复现脚本：`compare.py`。

### 滚动字幕正式接入

- 使用浏览器 AudioWorklet 连续采集 PCM；按实际 AudioContext 采样率编码独立 WAV，每 3 秒发送新音频，删除原先停录后等待 150 毫秒再重启的逻辑，没有新增依赖。
- 服务端保留约 15 秒上下文；不稳定的音频允许超出目标长度，避免强行滑窗丢掉尚未确认的语音。根据两轮结果的共同前缀确认文字，词组对齐及时间戳辅助消除窗口重叠。
- 前端替换末尾 Draft，确认文字才写入 JSONL/Markdown 和翻译缓冲；Stop 排空采样器、上传队列及待确认尾段后才触发 summary。
- 顺序上传、相同序号请求去重、失败重试、原音频归档及原子状态检查点支持连接恢复和服务重启后的继续处理。若较早手动生成过 paragraph transcript，summary 会根据原始转录更新时间重新生成。
- 验证：10 项 Python 测试；Node 连续采样、双声道混音、末尾 flush、Draft 替换、实际采样率 WAV 编码和上传重试测试；六段真实会议音频回放均通过。回放中发现并修复了时间戳偏移导致缩写重复的问题。
- 回放结果：`output/rolling_caption_check/final/replay.json`。当前浏览器控制接口没有可用浏览器，未做真实麦克风界面实录测试。
- 字幕服务已在确认空闲后重启；真实 HTTP 检查通过，包括页面默认模型、Worklet 文件、CUDA 识别、重试去重、状态检查点及 Stop 尾段。测试产物移至 `output/rolling_caption_check/http-v1`，没有混入真实会议目录。

### 2026-10-14: Qwen3.5-9B translation trial

- Compared the latest meeting's six fixed rolling-ASR samples as 15 real translation buffers. Ran 60 original model calls plus 30 Qwen3.5 calls with presence_penalty=0. Both models fit on GPU alongside active distil-large-v3.
- Active-ASR median paragraph repair + translation: Qwen2.5-7B 2.603s; Qwen3.5-9B Q4 (penalty=0) 2.718s. Total observed GPU maximum 11,134 MiB (includes idle app ASR residency).
- No consistent quality advantage: default Qwen3.5 omitted a negative priority statement; zero penalty retained negation but translated low priority as unimportance. Other fragments also showed unsupported completion, while some Chinese phrasing improved. Keep Qwen2.5-7B as default.
- Downloaded qwen3.5:9b-q4_K_M and added it as an optional UI trial. Disable thinking for Qwen3 requests and set presence_penalty=0 for translation.
- Artifacts: output/translation_comparison_2026-10-14/{report.md,translations.md,results.json,results_presence0.json,compare.py,app_smoke.json}. Latest Python checks: 11 passed; Node capture checks passed; real application translation smoke passed. Idle server restarted and its HTTP UI checked.
