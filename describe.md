# Video-KTR：当前可运行的验证流程

```text
视频 / 图像 + 问题
        ↓
Qwen2.5-VL 为每个 prompt 采样 G=4 个 CoT completion
        ↓
对同一 completion 做原始 teacher-forcing
        ├── token entropy                         → 高熵 E
        ├── 视觉 token 屏蔽（共享原始 mRoPE）     → 视觉敏感 V
        └── 多个非恒等帧置换（共享原始 mRoPE）    → 时序敏感 T
        ↓
每条 completion 分别取 E/V/T 的 top 20%
        ↓
U = E ∪ V ∪ T
        ↓
baseline：所有有效 completion token 进入 GRPO + KL
KTR：只有 U 中 token 进入 GRPO policy + KL
        ↓
direct GRPO backward / optimizer，并记录耗时、显存和利用率
```

目前实现采用 `paper/per_completion`：视觉与时序分数使用原始与 counterfactual token log-probability 的绝对差；时序分数是多个帧置换的平均值。`U` 是三类 mask 的并集，而不是三类 token 的人工语义标签。

## 已完成的 smoke

历史 4×H200 smoke 已完成 baseline 与 KTR 各一个优化步（`global_step=1`）：

| 指标 | baseline | KTR |
| --- | ---: | ---: |
| 外层时长 | 66.633 s | 70.598 s |
| 单卡峰值显存（四卡最大） | 89,839 MiB | 92,197 MiB |
| GPU 利用率峰值 | 100% | 100% |
| KTR 最后一步 token 平均 | — | E/V/T=45.5/45.5/45.5，union=80.188，update ratio=35.7% |

这些 smoke 与同形状容量 smoke 都发生在 length-control 修复和启动时 provenance 契约加入之前：它们证明链路与资源量级，但**不是** KTR/baseline 的公平数值对比。修复后的 paired smoke 必须重跑。KTR 的实际 CoT 中已保存 E/V/T token 与局部上下文；例如 E 中有 `for`、`,`、`I`，V 中有 `question`、`letters`、`image`，T 中有 `about`、`at`。这些 example 的判断依据是分数变化，功能词也可能被选中，不能只凭 token 字面含义判断对错。完整证据位于本地、不随 Git 提交的 artifact；其交接方式与内联对比表见 [REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md)。

随后又以 full 相同的 completion=512、5 次时序置换运行 KTR-only 容量 smoke：真实优化步通过，外层 79.013 s、训练内部 40.6615 s、单卡峰值仍为 92,197 MiB、四卡峰值利用率 100%。因此 4×H200 已经实测足够启动该 full profile，不切换到 8×B200；完整 epoch 的稳定性仍以后续 full artifact 为准。

为规避当前 FlashAttention2/mRoPE 的 dtype 兼容问题，验证过的 launcher 默认使用 `ATTN_IMPLEMENTATION=sdpa`。

## 已结束的 full 故障与当前修复

旧探索性 full 已在约 `global_step=529` 结束：rank 1/2/3 等待下一次 ZeRO collective，rank 0 停在前一 collective，且显存未到 OOM 范围。最高置信推断是 rank 0 在 generation 前的 whole-file 视频预处理停住；这仍需新的短恢复实际回归验证，不能写成已证明的唯一根因。

首次 checkpoint-500 诊断没有复现该问题：它在任何 batch 之前就因 PyTorch 2.10 默认 `weights_only=True` 拒绝 DeepSpeed 0.15.4 的旧 ZeRO state 而退出，所需类型仅为 `ZeroStageEnum` 和 `LossScaler`。修复保持 `weights_only=True`，只在恢复的 `trainer.train(...)` 作用域内 allowlist 这两个已核验类型；4×H200 已完成 step 500→501 的真实恢复 smoke，包含完整 ZeRO load、生成、E/V/T/union、反向和优化，且结束后 GPU 无残留。step 529 仍待后续受控诊断跨越。

修复后的 launcher 默认先以训练相同的 Qwen/torchvision 视频预处理做 CPU-only、可终止子进程 decode verification；它实时显示吞吐/ETA，写入绑定 source SHA、video root、backend、帧数与像素配置的 manifest/rejection JSON。可恢复的媒体错误才会被过滤；worker 异常退出或 0 条通过会 fail-fast 并保留诊断证据。训练期会保留 rank 0 实时进度、GPU heartbeat、四个 rank 日志；诊断模式还会写每 rank phase trace 和 NCCL timeout 证据。若 decoder 过滤任何视频，结果只能称为“数据质量过滤后的降级复现”。

## 完整验证

full 由操作者在 GPU 机器手动启动。脚本不含私有路径默认值，先显式设置 `MODEL_PATH=<MODEL_ROOT>/Video-R1/Qwen2.5-VL-7B-COT-SFT` 与 `DATA_ROOT=<DATA_ROOT>`；`RUN_ROOT` 必须是不存在的新目录，且 `NFRAMES` 必须为不少于 2 的偶数。当前离线 wheelhouse 依赖基础 GPU image 的显式 import gate。clean Python 在本版本不受支持：完整离线依赖 bundle 还必须配合后续扩展并验证的 full-overlay 安装模式，不能绕过 gate 或联网补包。若节点有外部保活，还必须同时设置 `KEEPALIVE_MAIN=<KEEPALIVE_MAIN>`、`KEEPALIVE_LAUNCHER=<KEEPALIVE_LAUNCHER>` 后才可传入暂停授权；授权后会在显存 gate 前精确暂停该保活：

```bash
cd <REPO_ROOT>
VARIANT=ktr RUN_ROOT=<ARTIFACT_ROOT>/ktr-decoder-verified-<UTC> ./run_full.sh
```

有已配置且获准暂停的外部保活时，才使用 `ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1 VARIANT=ktr ./run_full.sh`。

先用 checkpoint-500 做恢复诊断，才建议启动上述正式 full：

```bash
old_run=<ARTIFACT_ROOT>/ktr-20260921T131854Z
VARIANT=ktr \
RUN_ROOT=<ARTIFACT_ROOT>/ktr-checkpoint500-diag-<UTC> \
PREPARED_DATASET="${old_run}/Video-R1-260k-video-verified.json" \
USE_EXISTING_PREPARED_DATASET=1 \
RESUME_FROM_CHECKPOINT="${old_run}/training/checkpoint-500" \
VERIFY_VIDEO_DECODE=0 MAX_STEPS=540 SKIP_FINAL_MODEL_SAVE=true \
NCCL_DIAGNOSTICS=1 RANK_PHASE_TRACE=1 \
./run_full.sh
```

诊断 phase 2 会先核验所有 optimizer shard 的 restricted safe-global 契约，并经同一 DeepSpeed loader 读入一个小 probe；不要设置全局 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`。脚本会先筛选路径、随后默认做全量 decoder verification，并连续输出验证 ETA 与训练 heartbeat；写入训练时长/显存/利用率、源码和数据 provenance。历史 full 输入均为 video，诊断恢复会在终端确认使用父类 `RandomSampler`，以保持 checkpoint 的取样契约。预检与训练均被 launcher 跟踪；收到中断时会先停止它们，才恢复此前验证性暂停的保活。NCCL watchdog dump 应从 `training.log`/`torchrun-logs` 与 rank trace 分析，不应假定有可直接传给 `fr_trace.py` 的磁盘文件。未配置保活时会说明没有恢复命令：

```bash
bash <KEEPALIVE_LAUNCHER>
```

只在自动恢复校验失败、确认训练已停止且 `main.py` 不存在时执行该命令。完整参数、baseline 对照命令与产物说明见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。
