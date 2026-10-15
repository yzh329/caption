# Current Work

本文件记录当前正在做什么、下一步做什么，以及哪些决策已经确定。它不是实验结果归档，实验细节和结果放在 `docs/EXPERIMENT_LOG.md`。

## 当前目标

构建一条可复现的本地会议总结蒸馏/微调流程，让本地小模型在 RTX 5080 上尽量接近高质量 teacher summary，同时保持可接受的速度和显存占用。

## 当前状态

已完成：

- 实时翻译/段落修复默认模型改为 `qwen3:8b`。
- Summary chunk model 和 final summary model 保留独立选择；2026-10-14 正式接入 `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S`，两阶段默认均使用该模型，CLI 和 backfill 同步。
- 实时 Whisper chunk 转录关闭 VAD。
- 2026-10-14 实装连续 PCM 录音、3 秒刷新、约 15 秒滚动上下文和可修订 Draft；两轮一致后确认，停止时收尾。保留 distil-large-v3 为默认 ASR。

尚未开始：

- DeepSeek API teacher label 生成脚本。
- Summary SFT 数据集构建脚本。
- Qwen3 8B/14B QLoRA 训练配置。
- 自动评估脚本。

## 决策记录

- 保留实时翻译，但不用 `gpt-oss:20b` 做实时翻译。
- `gpt-oss:20b` 保留为总结备选和 benchmark；当前默认总结模型为 Qwen3.8 27B 3-bit。
- DeepSeek API 可以作为高质量 teacher 候选。
- 第一阶段不训练 `gpt-oss:20b`，优先训练 Qwen3 系列。
- 同时维护 `Qwen3-8B` baseline 和 `Qwen3-14B` 主力配置。

## 下一步任务

1. 新建 `scripts/generate_teacher_labels_deepseek.py`
   - 扫描 `output/live`
   - 读取 `paragraph_transcript.md`，没有则回退到 `live_transcript.md`
   - 调用 DeepSeek API 生成结构化 JSON teacher labels
   - 支持断点续跑和费用/ token 统计

2. 新建 `scripts/build_summary_sft_dataset.py`
   - 过滤坏样本和空样本
   - 生成 train/val JSONL
   - 输出数据质量报告

3. 新建训练配置
   - `configs/qwen3_8b_summary_qlora.yaml`
   - `configs/qwen3_14b_summary_qlora.yaml`

4. 新建 `scripts/evaluate_summary_student.py`
   - 对比 baseline、student、`gpt-oss:20b` 和 DeepSeek teacher
   - 记录速度、显存、结构完整性、行动项覆盖、幻觉风险和中文质量

## 开放问题

- 是否允许把真实会议 transcript 发给 DeepSeek API，还是先做 redaction。
- DeepSeek teacher 使用 `deepseek-v4-pro` 还是先用 `deepseek-v4-flash` 批量生成。
- 14B QLoRA 的最大可承受 sequence length 是 4k、6k 还是 8k。
- 最终部署是 Ollama Modelfile、GGUF，还是保留 Transformers/adapter 形式。

### 2026-10-14: Qwen3.5-9B translation trial

- Compared the latest meeting's six fixed rolling-ASR samples as 15 real translation buffers. Ran 60 original model calls plus 30 Qwen3.5 calls with presence_penalty=0. Both models fit on GPU alongside active distil-large-v3.
- Active-ASR median paragraph repair + translation: Qwen2.5-7B 2.603s; Qwen3.5-9B Q4 (penalty=0) 2.718s. Total observed GPU maximum 11,134 MiB (includes idle app ASR residency).
- No consistent quality advantage: default Qwen3.5 omitted a negative priority statement; zero penalty retained negation but translated low priority as unimportance. Other fragments also showed unsupported completion, while some Chinese phrasing improved. Keep Qwen2.5-7B as default.
- Downloaded qwen3.5:9b-q4_K_M and added it as an optional UI trial. Disable thinking for Qwen3 requests and set presence_penalty=0 for translation.
- Artifacts: output/translation_comparison_2026-10-14/{report.md,translations.md,results.json,results_presence0.json,compare.py,app_smoke.json}. Latest Python checks: 11 passed; Node capture checks passed; real application translation smoke passed. Idle server restarted and its HTTP UI checked.
