# Video-KTR：集群 3 B200 当前复现状态

本文是当前可执行验证的简要快照。跨机器协作时，以 [REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md) 的固定对话格式、证据位置和交接规则为准；其中的历史 H200 记录不代表当前 B200 结论。

```text
图像 / 视频 + 问题
        ↓
Qwen2.5-VL 对每个 prompt 采样 G=8 个 CoT completion
        ↓
同一 completion 的 teacher-forcing 反事实打分
 ├─ 熵                           → 高熵 E
 ├─ 屏蔽视觉 token（共享原 mRoPE） → 视觉敏感 V
 └─ 1 个非恒等随机帧置换（共享原 mRoPE）
                                → 时序敏感 T
        ↓
每条 completion 内，E/V/T 各自精确取 top 20%
（视觉/时序分数均为 |Δ log p|）
        ↓
U = E ∪ V ∪ T
        ↓
baseline：全部有效 completion token 进入 GRPO + KL
KTR：仅 U 中 token 进入 GRPO policy + KL
```

`E`、`V`、`T` 是由分数与反事实变化选出的 mask，并非 token 的人工语义类别；一个 token 可以同时属于多个集合。E/V 对每条 completion 均精确 top 20%；视频 completion 的 T 也精确 top 20%，而图像 completion 的 T 因没有帧序而设计为全零 mask。实现已分别读取 `image_token_id` 和 `video_token_id`，不再把视频 token 当作图像 token 处理。

## 对齐配置

| 项目 | 当前 B200 配置 | 状态 |
| --- | --- | --- |
| 计算资源 | 集群 3，8×B200、8 ranks | 已通过 paired smoke |
| 模型 / 数据请求 | Qwen2.5-VL-7B-COT-SFT / Holmes-16k | 可信 exact-revision manifest + runtime hash gate 是 full 前置；本机重哈希只以落盘 gate 证据为准；数据见下文 |
| 最大 prompt / completion | 16,384 / 768 | 对齐原始 launcher |
| 生成数 | G=8 | 对齐 |
| Attention | FlashAttention-2（原生 B200 架构；Qwen rotary dtype 兼容 shim） | 已做真实前反向验证 |
| `nframes` | 8 | 对齐原始脚本未显式传参时的默认值 |
| `max_pixels` | CLI 请求 `401408` | 已传入；实际视频限制另见下文 |
| token 筛选 | `paper/per_completion`、精确 20%、绝对 delta | 已验证 |
| 时序扰动 | 每个 rank/step 1 个可复现的非恒等随机置换；不额外包含 reverse | 已验证 |

## 最终 B200 paired smoke

最终 smoke 使用 8 个固定视频样本，baseline 与 KTR 各完成一个真实优化步。它验证的是：8 rank 启动、FA2 实际解析、Qwen rotary 前反向、视频 V/T 路径、E/V/T/union、GRPO 反向与优化、资源采集，以及任务结束后无训练进程和保活恢复。image/video token ID 的分离由启动 preflight 验证；mixed-modality sampler 的真实训练覆盖留给 full。证据保存在 `<共享持久卷>/video-ktr-b200/artifacts/grpo-smoke-b200/<UTC>/`，不随 Git 提交。

| 指标 | baseline | KTR |
| --- | ---: | ---: |
| 外层耗时 | 94.840 s | 95.008 s |
| 跨已监控 GPU 最大显存 | 47,858 MiB | 45,870 MiB |
| GPU 利用率峰值 | 100% | 100% |
| KTR 末步 E / V / T 平均 token 数 | — | 44.172 / 44.172 / 44.172 |
| KTR union 平均 token 数 / 更新比例 | — | 75.859 / 35.0% |

该结果只是一对单步、小样本 smoke，不能据此宣称完整 epoch 的速度、显存或训练效果结论；但 8 卡单步峰值远低于单卡容量，证明当前 profile 可以进入后续全量稳定性验证。

KTR token report 共保存 768 个 union token，均位于 CoT：`E=341`、`V=592`、`T=472`（集合可重叠）。局部例子如下：

| 类别 | 选中的 token 示例 | CoT 局部语境示例 |
| --- | --- | --- |
| E | ` step`、` First`、` analyze`、` with` | `<think>Let me think about this step by step...` |
| V | ` step`、` First`、`'ll`、` analyze`、` what` | 视觉内容推理段 |
| T | ` step`、` by`、` First`、`'ll`、` analyze` | 帧间过程推理段 |

这些字面上包含功能词的 token 是正常现象：分类依据是对应反事实下的分数变化，不应凭 token 文本本身判断其“是否视觉/时序”。

## 数据与严格复现边界

集群 3 已补齐并路径核验 `16,916 / 16,916` 条 Holmes 记录，其中 `8,765` 条图像、`8,151` 条视频；补齐了 233 个训练视频媒体路径。独立的 120 秒 mixed decoder/data-class gate 曾通过 `16,916 / 16,916`。但首次 strict KTR full 在 CPU 预检重跑时，两条引用同一个长视频的记录超过 120 秒，只得到 `16,914 / 16,916`，严格 data-class gate exit=1。因此：

- 修复后的 strict full 对每条媒体设置 600 秒 CPU 预检上限；本次启动仍须完整 `16,916` 条都通过（`rejected=0`、`timed_out=0`）才能进入 GPU。600 秒是预检容错上限，不改变训练算法或媒体数据。
- 历史 `15,365` 条 run 必须永久保留 `reproduction_class=reduced-media-subset` 标记；媒体后来补齐不改变其冻结的数据契约。
- full 启动前会以训练相同的 Qwen/torchvision 路径检查混合图像和视频，并实时报告吞吐与 ETA；若 decode 有 rejection，默认拒绝，另需显式 `B200_ALLOW_DECODER_FILTERED=1` 才能以额外 filtered 降级继续。

`max_pixels=401408` 是原始 launcher 对 Qwen 工具链传入的 CLI 请求值，当前没有偷偷改小。需要区分的是，随代码携带的 Qwen 视频预处理还存在约 `105369` 的单帧有效像素上限；本次 smoke 实际视频帧约为 `94,080–98,784` 像素。因此报告会同时记录“请求值”和“实际有效限制”，而不会错误声称每帧都达到了 `401408`。

## 下一步

`somke-b200.sh` 已成功完成最终 smoke；首次 strict full 的 120 秒媒体门禁失败，尚无 B200 `torchrun` / GPU full epoch。修复后的启动脚本会在终端显示并记录 600 秒预检上限；操作者须以新 run root 重试 strict KTR，并在其成功后串行启动采用相同设置的 baseline。每次 full 会在训练期间暂停已核验的保活，且无论成功、失败或中断都会确认训练进程退出后恢复保活。环境契约、FA2 兼容处理、数据门禁与证据索引见 [REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md)。
