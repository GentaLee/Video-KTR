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

## 完整验证

full 由操作者在 GPU 机器手动启动。脚本不含私有路径默认值，先显式设置 `MODEL_PATH=<MODEL_ROOT>/Video-R1/Qwen2.5-VL-7B-COT-SFT` 与 `DATA_ROOT=<DATA_ROOT>`；当前离线 wheelhouse 依赖基础 GPU image 的显式 import gate。clean Python 在本版本不受支持：完整离线依赖 bundle 还必须配合后续扩展并验证的 full-overlay 安装模式，不能绕过 gate 或联网补包。若节点有外部保活，还必须同时设置 `KEEPALIVE_MAIN=<KEEPALIVE_MAIN>`、`KEEPALIVE_LAUNCHER=<KEEPALIVE_LAUNCHER>` 后才可传入暂停授权：

```bash
cd <REPO_ROOT>
VARIANT=ktr ./run_full.sh
```

有已配置且获准暂停的外部保活时，才使用 `ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1 VARIANT=ktr ./run_full.sh`。

脚本会筛选路径存在且非空的视频、连续输出进度与 GPU heartbeat、写入训练时长/显存/利用率和源码 provenance；它不会把该筛选误称为全量解码成功。配置了外部保活控制时，结束后会自动恢复此前验证性暂停的保活，并在末尾打印人工兜底命令；未配置时会说明没有恢复命令：

```bash
bash <KEEPALIVE_LAUNCHER>
```

只在自动恢复校验失败、确认训练已停止且 `main.py` 不存在时执行该命令。完整参数、baseline 对照命令与产物说明见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。
