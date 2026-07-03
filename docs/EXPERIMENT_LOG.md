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
