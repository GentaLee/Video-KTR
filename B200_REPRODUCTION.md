# Video-KTR：集群 3 B200 复现交付

> 远程协作消息必须先遵守 [REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md) 第 0 节的 `KT-HANDOFF/v1` 格式。本文只记录可复查事实；未运行的 full 不写成已完成。

## 当前结论

集群 3 的最终 8×B200 paired smoke 已通过：baseline 与 KTR 各完成一次真实生成、前向、反向和优化，FA2、Qwen rotary 兼容、E/V/T/union、资源遥测以及保活恢复均有 artifact 证据。尚无 B200 `torchrun` / GPU full epoch；一个历史 reduced wrapper 已在 CPU decoder data-class gate fail-closed，正式 full 仍仅由操作者手动运行 [run_full_b200.sh](run_full_b200.sh)。

| 项目 | B200 实现 / 已验证值 | 对齐状态 |
| --- | --- | --- |
| GPU / ranks | 集群 3，8×B200，8 ranks | 对齐原始 8 卡 launcher |
| 模型身份 | Qwen2.5-VL-7B-COT-SFT；以可信 exact-revision manifest 与 runtime hash gate 绑定 config、index、weight shard、tokenizer/processor 等全部顶层常规文件 | CPU-only runtime gate 已实测通过：19 文件、4 shard；full 仍会在占卡前再次复验 |
| 数据请求 | Holmes-16k | 严格性取决于媒体完整度，见“数据门禁” |
| prompt / completion / G | 16,384 / 768 / 8 | 对齐原始 launcher |
| attention | `flash_attention_2`；FA2 二进制含 `sm_100`，训练运行时强制解析为 FA2 | 已验证，不静默回退 |
| 视频帧数 | `nframes=8` | 对齐上游脚本未显式设置时的代码默认值 |
| token selector | `paper/per_completion`；E/V 对每 completion 精确 `ceil(20%)` top-k；视频 completion 的 T 同样精确 top-k；绝对 `|Δ log p|` | image completion 的 T 为设计上的全零 mask（无帧序），不伪造 temporal top-k |
| temporal probe | 每 rank/step 一个可复现的非恒等随机置换；不加入 reverse | 满足“单次乱序”；不是所有 rank 共用同一置换 |
| 视觉 token 定位 | 分别读取 `image_token_id` 与 `video_token_id` 后取并集 | 已修复旧 video-id/image-id 混用 |

上游命令本身可见 [原始 launcher](https://github.com/zywang0104/Video-KTR/blob/main/src/scripts/run_grpo_video_ktr.sh)：8 ranks、16k prompt、768 completion、G=8、`max_pixels=401408` 与 FA2。当前实现新增的是可审计的数据/保活门禁、mixed-modality sampler、严格 attention gate 和选择器语义修正，不是把 H200 配置直接搬到 B200。

full 的默认信任锚是随代码审阅、按精确 revision 固定的 [模型完整性 manifest](manifests/video-r1-qwen25vl-7b-cot-sft-f71f0f1e22c015007fccd080eef87824fe292a10.json)，而不是可变的本机输出。runtime gate 必须对本地 checkpoint 的全部顶层常规文件重新计算 SHA-256 并与该 JSON 对比，通过后才可进入 GPU 阶段，并应落盘 `runtime_gate.json`。`grpo_write_model_sha256_manifest.py` 生成的本地 verified-copy 仅可用于独立比较或恢复诊断，不能替代默认锚。2026-09-22 的 CPU-only 实测已通过 19 文件、4 shard 校验，且未暂停保活、未启动 full；full 自身仍会在开始与占卡前各复验一次。

## 已通过的最终 smoke

Artifact 根为 `<共享持久卷>/video-ktr-b200/artifacts/grpo-smoke-b200/20260922T044249Z/`；不随 Git 提交。

| 指标 | baseline | KTR | KTR − baseline |
| --- | ---: | ---: | ---: |
| 外层耗时 | 94.840 s | 95.008 s | +0.168 s |
| 跨已监控 GPU 最大显存 | 47,858 MiB | 45,870 MiB | −1,988 MiB |
| GPU 利用率峰值 | 100% | 100% | 0 pp |
| 真实优化步 | 1 | 1 | — |
| KTR E / V / T（每 completion） | — | 44.172 / 44.172 / 44.172 | — |
| KTR union / 更新比例 | — | 75.859 / 35.0% | — |

这只是容量和链路 smoke，不是完整 epoch 的速度、峰值显存或训练效果结论。单步的最大显存小于每张 B200 的可用容量，足以支持进入 full 稳定性验证；完整 epoch 是否稳定仍必须以 full artifact 判断。

最终 smoke 使用 8 条确定性**视频**记录，保证每个 rank 都执行视频 V/T 路径。它没有把自己宣称为 mixed-modality sampler 的训练覆盖；full 前新增的 mixed image/video CPU decode 门禁会保留两种模态和原始顺序，真正的分布式 mixed sampler 由 full 运行记录其计划。

### CoT 中的三类 token 证据

最终 token report 保存 768 个 union token，均在 CoT 中；集合可重叠，计数为 E=341、V=592、T=472。

| 类别 | 例子 | `<think>` 内短上下文 |
| --- | --- | --- |
| E | ` step`、` First`、` analyze`、` with` | `Let me think about this step by step...` |
| V | ` step`、` First`、`'ll`、` analyze`、` what` | 视觉内容推理段 |
| T | ` step`、` by`、` First`、`'ll`、` analyze` | 帧间过程推理段 |

这三类是 log-probability/entropy 的归因 mask，不是词性或人工语义标签；功能词被选中是预期现象。一个 token 可以同时落在 E、V、T 中，最终用的是并集 U。

## B200 环境、踩坑与修复

| 观察到的问题 | 已验证原因 | 处理方式 | 降级边界 |
| --- | --- | --- | --- |
| 旧 H200 profile 的 FA2/mRoPE 路径不能直接作为 B200 结论 | Qwen 旧 Transformers 路径使 BF16 Q/K 与 rotary cos/sin dtype 不匹配 | 仅在 `transformers=4.49.0.dev0` + FA2 时启用上游等价的 FP32 rotary 计算后 cast 回原 dtype；有单独前反向 gate | shim 或 attention 解析失败即停止，绝不退到 SDPA |
| B200 Triton 首次 JIT 找不到 CUDA 头/`ptxas` | 隔离 venv 不继承 image 的编译器搜索路径 | 显式验证 CUDA headers、`ptxas`，将 include/cache/tmp 指向持久卷 | `/tmp`、`/mnt` 直接拒绝 |
| 旧选择器不符合当前验收定义 | 旧路径包含 5 次置换、batch quantile、signed delta，且视频 id 曾误用图像 id | 设为单次非恒等乱序、per-completion 精确 top-20%、absolute delta、分离 image/video id | 改任一选择定义必须重跑 smoke |
| 文件型视频的像素数看似未到 401408 | vendored qwen-vl-utils 对单帧还有约 105369 的硬上限 | 保留上游 CLI `--max_pixels 401408`，并在 profile 同时记录 requested/effective/observed 值 | 若要求实际逐帧 401408，须建名为 patched-intent 的新 profile、改代码并重跑 smoke |
| shared-GPU 保活会污染显存/性能数据 | 保活与训练不能同时占用卡 | CPU-only 数据预检时保活运行；仅 GPU FA2/torchrun 前由官方 controller 停止；结束/失败/中断先清理任务再恢复并校验 | 无法证明训练已停时故意不恢复，防止冲突；终端给出人工恢复命令 |

Qwen rotary 修复对应 [Transformers 上游修正](https://github.com/huggingface/transformers/commit/8ee50537fe7613b87881cd043a85971c85e99519)。它是窄范围兼容 shim，不改变 selector 或训练目标。

## 数据门禁与降级路径

已补齐并传输 233 个 Video-Holmes 训练视频路径；独立 strict gate 已确认 `16,916 / 16,916`：图像 `8,765 / 8,765`，视频 `8,151 / 8,151`。在 CUDA-hidden、torchvision、`nframes=8`、CLI `max_pixels=401408` 的 canonical mixed decoder 中，`verified=16,916`、`rejected=0`、`timed_out=0`；data-class gate 结果为 `strict_holmes_16k=true`、`reproduction_class=strict-holmes-16k`。此前 `15,365` 条的 run 是冻结清单的历史 reduced run，不能因媒体后来补齐而重新标成 strict。

| 路径 | 触发条件 | 产物标记 | 是否能称严格 Holmes-16k |
| --- | --- | --- | --- |
| strict | 16,916 条路径均在，mixed Qwen decode 0 rejection | `reproduction_class=strict-holmes-16k` | 可以 |
| historical reduced | 显式 `B200_ALLOW_REDUCED_HOLMES=1`，且冻结为 15,365 条契约 | `reproduction_class=reduced-media-subset` | 不可以 |
| decoder filtered | decode 有 rejection，另需显式 `B200_ALLOW_DECODER_FILTERED=1` | 仍保留 reduced/filtered 证据与 rejection manifest | 不可以 |

历史上默认 strict 路径曾在 CPU path-preflight 报 `available=15365/16916` 并停止；当前 standalone strict path、decoder 与 data-class gate 均已通过。每次正式 full 仍会在占用 GPU 前重新执行同一 path/decode gate，因此 standalone 成功不等价于某次 full 已开始或已成功。

## 手动 full 启动（不由 AI 执行）

在集群 3 上，`b200.env` 已由机器本地配置提供模型、数据、持久根、保活主程序和官方 controller 身份。先从一个干净、已部署的仓库工作树执行。每个 `RUN_ROOT` 必须是新目录。

严格 KTR（仅在完整媒体且本次 mixed decoder `0 rejection` 后启动）：

```bash
cd <REPO_ROOT>
B200_ALLOW_PAUSE_KEEPALIVE=1 \
VARIANT=ktr \
RUN_ROOT=<共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-<UTC> \
./run_full_b200.sh
```

历史 reduced KTR 路径（仅用于定位；不可标为严格全量）：

```bash
cd <REPO_ROOT>
B200_ALLOW_PAUSE_KEEPALIVE=1 \
B200_ALLOW_REDUCED_HOLMES=1 \
VARIANT=ktr \
RUN_ROOT=<共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-reduced-<UTC> \
./run_full_b200.sh
```

严格 paired 对照必须使用同一 clean commit、相同 strict data gate，并且先 KTR、后 baseline 串行运行，不能两者并行抢占卡：

```bash
B200_ALLOW_PAUSE_KEEPALIVE=1 \
VARIANT=baseline \
RUN_ROOT=<共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/baseline-strict-<UTC> \
./run_full_b200.sh

<B200_PYTHON> src/grpo_compare_runs.py \
  --baseline-dir <BASELINE_RUN_ROOT> \
  --ktr-dir <KTR_RUN_ROOT> \
  --output-dir <COMPARISON_ROOT> \
  --require-pass
```

终端会连续显示 path/decode 进度与 ETA、rank 0 training output、GPU heartbeat。结束时自动恢复失败才在终端中使用泛化兜底命令：

```bash
KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start
```

只能在已确认 torchrun/rank 进程已退出、自动恢复校验失败时执行；不要手工 `pkill` 或另起重复保活。

## Full artifact 的最低验收

```text
<RUN_ROOT>/
├── terminal.log                         # 每阶段、ETA、heartbeat、保活 lifecycle
├── smoke_gate.json / smoke_gate_pre_gpu.json
│                                        # passed smoke、关键训练源码 hash，以及占卡前复验
├── runtime_gate.json / runtime_gate_pre_gpu.json
│                                        # pinned 模型/环境完整性，以及占卡前复验
├── data_path_gate.json / data_gate.json # strict/reduced 与 mixed decoder 契约
├── launch_config.txt / source_provenance.txt
├── gpu_metrics.jsonl / resource_monitor.log
├── torchrun-logs/ / training.log
├── training/training_complete.json
└── training_summary.md / training_summary.json
```

full 成功的最低条件是脚本零退出、`training_complete.json` 与 `training_summary.json` 标记成功、数据/源码 provenance 存在、保活恢复为唯一已验证实例。对于 baseline/KTR 比较，还必须确认二者 `data_gate.json`、关键源码 hash 与 profile 相同。
