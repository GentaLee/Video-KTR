# 集群 2 / H200 KTR 全量训练归档

## 协作记录格式

对外交接使用 `KT-HANDOFF/v1`：注明 profile、运行身份、证据时间、已验证事实、未验证事项、下一步及操作授权。区分历史运行源码与归档提交；不要用新分支的 HEAD 替代原运行版本。机器仅用集群编号，路径使用持久根占位符。

PROFILE: h200  
STATUS: KTR_COMPLETED  
RUN: ktr-20260923T054132Z-2458824

## 结论

本轮正常完成全部 **2,115 步 / 1 epoch，exit=0**。完成记录、外层 PASS、launcher exit 文件和最终 checkpoint 相互一致。没有发生 B200 的末批 shape 错误，没有续跑或补算。本次归档不改训练算法，也不启动 baseline。

北京时间 2026-09-23 13:41:32 提交；2026-09-24 13:37:27 写入训练完成记录，13:37:52 外层完成。训练后日志确认先释放训练 GPU 进程，再恢复唯一保活。

2026-09-24 09:45 UTC 再检查时，训练已无进程，保活也已不在运行。不能仅凭旧退出日志宣称当前保活存在；已使用原有身份核验控制器重新启动，并确认 GPU compute。保活为何在两次检查间消失未确定，不能归因于训练失败。

## 实验与资源

| 项目 | 本轮结果 / 配置 |
| --- | --- |
| 硬件 | 4×H200 |
| 初始模型 | Video-R1/Qwen2.5-VL-7B-COT-SFT，revision f71f0f1e22c015007fccd080eef87824fe292a10 |
| 数据 | Holmes 16,916 条：8,765 图片 + 8,151 视频；sampler 补齐 4 条，实际发出 16,920 条 |
| 批量 | 每卡 1，累积 2，G=8；每更新名义 8 prompts / 64 completions |
| 生成与视觉 | prompt 16384 / completion 768；8 帧；FA2；视频有效像素 cap 105369 |
| KTR | E/V/T 各 top20%，per-completion，绝对 delta，三类并集；时序置换 1 次，无 reverse |
| 外层训练窗口 | 86,065.144 秒，约 23 小时 54 分 25 秒；不是含全部前置检查的端到端耗时 |
| Trainer 耗时 | 86,002.1368 秒，约 23 小时 53 分 22 秒 |
| 平均 train_loss | 0.005958747757303179 |
| 保存 | 每 100 步，保留最近 2 个；最终保留 checkpoint-2100、checkpoint-2115 |

| GPU | 显存峰值 MiB | 平均显存 MiB | 平均利用率 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 100745 | 99681.07 | 47.77% | 16788 |
| 1 | 96001 | 94820.89 | 48.86% | 16788 |
| 2 | 108473 | 106698.47 | 52.86% | 16788 |
| 3 | 111489 | 104746.22 | 52.13% | 16788 |

这是采样窗口统计，不是 GPU kernel profiler 数据；显存为设备占用，不等于 PyTorch allocated。监控未报告错误。峰值最高约 108.88 GiB。

## 产物和验证

- 最终模型 4 个 safetensors shard，逐文件 SHA256 见 [final_model_manifest.json](final_model_manifest.json)。模型 index 的所有 shard 都存在且非空。
- checkpoint-2115 的 Trainer step、4 卡 optimizer/model 分片、4 卡 RNG 和 scheduler 文件存在。**仅验证文件结构，未进行重新加载或 checkpoint 恢复测试。**
- Trainer 日志包含连续 step 1–2115 的 2,115 条 loss 记录，所记录数值均有限，见 [validation.json](validation.json)。这不等同于检查所有权重张量是否有限。
- [resource_summary.json](resource_summary.json) 是去除具体机器路径后的资源记录；其中 latest_step_metrics 是完成记录的最后一次 compute_loss，不应混同最终 optimizer-step 的聚合日志。
- [launch_config.txt](launch_config.txt) 与 [source_provenance.txt](source_provenance.txt) 保存脱敏运行参数及原始源码身份；原运行是 `8a2e3cb48a34b7a4a8b9d975efe73d2d8dca0b61` 加 dirty/untracked 工作区，不是本次归档分支的干净源码。
- 原始证据文件哈希见 [evidence_manifest.json](evidence_manifest.json)。原始日志和大文件保留在原位，未移动、删除或提交 Git。

存储位置：`<CLUSTER_2_ROOT>/Video-KTR/artifacts/h200-aligned-runtime/artifacts/grpo-full-h200/ktr-20260923T054132Z-2458824/`。最终模型在 `training/`，完整状态在 `training/checkpoint-2115/`。已有短跑 token 示例在 `artifacts/h200-aligned-runtime/artifacts/validation/paired-20260923T0521Z/ktr/token_examples.md`；它们是 smoke 示例，不是本次 full 的逐 token 记录。

## B200 尾批问题与结论边界

本轮仍使用旧的 completion 统计聚合代码，但实际成功退出。4-rank 与 8-rank 的尾批条件不同；不能把 B200 必然失败的结论机械套到 H200，也不能因 H200 成功就认定旧统计实现没有风险。后续可单独验证并移植 B200 的统计修复，本次不修改已完成实验的源码。

训练成功只证明全量流水线完成；训练 reward 不是独立评测准确率。baseline 尚无本协议下的完整结果，不能给出 KTR 优于 baseline 的结论。后续 baseline 应从同一原始 SFT 开始，而不是 KTR checkpoint，并使用同一独立评测集比较。若以后更改统计实现，需明确标记版本与指标口径，不能覆盖本轮原始证据。

本次提交范围是结果归档、归档校验工具和 H200 状态；此前未提交的源码/文档改动保留，不自动一并收录。复现运行源码应核对原始 provenance 与此前保存的源码快照，不能仅 clone 本次报告提交就假定获得完整 dirty 运行环境。
