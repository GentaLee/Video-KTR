# 集群 3：B200 数据流水线故障、修复与验证

> 最新补充（2026-09-24）：完整 epoch 末批统计故障和 checkpoint 恢复见 [B200_CHECKPOINT_RECOVERY.md](B200_CHECKPOINT_RECOVERY.md)，当前状态见 [handoff/STATUS.md](handoff/STATUS.md)。下面为历史验收，baseline 应使用新恢复报告中的 release 命令。

更新日期：2026-09-23（UTC）。环境一律匿名为“集群 3”，项目根目录写作 `<共享持久卷>/video-ktr-b200`；本文不记录真实账号、主机名、IP、认证材料或私有路径。

**启动前验收已通过，正式全量训练待用户手动启动。** 两种 8 卡 7-step 回归、完整数据缓存与门禁、模型和 optimizer 状态保存、真实保活故障恢复均已验证；本次没有代替用户运行完整 epoch。

远程交接遵循 [REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md) 的 `KT-HANDOFF/v1` 约定：已执行、已通过、仍在运行和推断必须区分。以下运行路径均相对项目根目录。本文是本次故障的独立补充记录，不能把旧日志中的通过状态当作新运行已经通过。

## 1. 结论与当前验收状态

本次 `ktr-20260922T134530Z` 在 CPU 数据门禁失败，尚未进入正式 GPU 训练。16 条媒体预检超时后，输出从 16,916 条变为 16,900 条；严格门禁正确阻止了静默改变训练集。

这 16 条随后在独立 CPU 实验中全部解码成功：16/16、0 拒绝、0 超时，解码阶段 67 秒。原日志中的两次长时间停顿更符合监督程序受终端输出阻塞的特征；该原因是有多项证据支持的高可信推断，故障发生时没有保存 `strace`，不能声称直接观测到了阻塞调用。

| 验收项 | 截至本次记录的状态 | 证据与边界 |
| --- | --- | --- |
| 失败的 16 条记录 CPU 复验 | 已通过 | `artifacts/decoder-diagnostics/retry16-threads1-20260923T022051Z/`；manifest 为 `passed`；0 reject / 0 timeout |
| 提交范围内的自动测试 | 已通过 | Git 跟踪的 9 个测试模块，共 89 项，8.137 秒；日志 `repo/artifacts/b200-committed-tests-20260923-module.log`。完整本地工作区另为 93 项，其中 4 项属于已有未跟踪 H200 测试，不随本次交付 |
| 终端阻塞与保活动态回归 | 已通过 | `repo/tests/test_b200_lifecycle.py`，12 个动态测试；包含保活中途消失后的恢复与身份刷新；模拟控制器，不触碰真实 GPU |
| 真实媒体缓存一致性 | 已通过 | `artifacts/media-cache-v1-20260923/parity.json`；1 张图像和 2 段视频，共 3 项；创建缓存及命中缓存均验证像素 / tensor 与 sampled FPS 完全相等 |
| 新缓存路径的 8 卡 paired smoke | 已通过 | `artifacts/grpo-smoke-b200/cache-prefetch-20260923/`；baseline / KTR 均完成 1 个 optimizer step；资源记录正常，结束后已恢复并确认唯一保活。此轮为 8 条视频小样本 |
| 全量精确媒体缓存 | 已通过 | `artifacts/media-cache-v1-20260923/verified.json.manifest.json`：16,916/16,916，14,286 个唯一媒体，0 拒绝 / 0 超时，输入与输出 SHA 相同；UTC 02:53:04 完成 |
| 全数据门禁后的 KTR 7-step bounded full | 已通过 | `artifacts/grpo-full-b200/cache-regression-ktr-7step-20260923/`；step 7、exit 0、8 卡资源记录完整，checkpoint 与最终模型已保存，UTC 02:59:23 wrapper 成功退出并恢复唯一保活 |
| 全数据门禁后的 baseline 7-step bounded full | 已通过 | `artifacts/grpo-full-b200/cache-regression-baseline-7step-20260923/`，生命周期版本 `99687e5`；step 7、exit 0、完整资源记录、checkpoint 和最终模型均已保存；UTC 03:07:27 wrapper 成功退出并恢复唯一保活 |
| 完整热缓存门禁 | 已通过 | baseline 7-step 的预检阶段：从 UTC 03:00:59 到 03:01:29，完整 30 秒；16,916/16,916、0 拒绝 / 0 超时，严格混合数据门禁通过 |
| CPU 保活中断故障注入 | 已通过 | 官方 controller 于 UTC 03:00:15 停止保活；guard 在 03:00:33 检测到缺失，03:00:50 验证唯一新进程，03:00:51 刷新身份；训练启动流程继续 |
| 正式 KTR / baseline full | 未由本次修复工作启动 | 用户在上述门禁通过后分别手动启动，两者串行运行 |

真实 paired smoke 的 `comparison.json` 已记录如下数字，`terminal.log` 和启动日志最后均为成功；保活恢复日志确认 1 个匹配主进程：

| 8 卡单步视频 smoke | variant 进程用时 | 监控到的单卡最大显存峰值 |
| --- | --- | --- |
| baseline | 126.395 秒 | 47,859 MiB |
| KTR | 111.417 秒 | 56,892 MiB |

这些时间含启动开销，只是可运行性及资源采样证据，不能由单步结果断言 KTR 更快。该轮跳过最终模型保存；混合全数据路径、最终保存与生命周期增量由后续 7-step bounded full 补验。

KTR 的 7-step bounded full 已通过：`training_summary.json` 的 `succeeded=true`、`exit_code=0`、`monitor_errors=[]`，8 卡均有采样。外层训练段 312.470 秒，Trainer 内部 223.667 秒，单卡监测峰值 74,827 MiB。`training/training_complete.json` 为 `global_step=7`、`skip_final_model_save=false`；`checkpoint-7` 已保存 8 份 model state、8 份 optimizer state、8 份 RNG state、scheduler 与 step-7 trainer state，checkpoint 及最终模型目录均含 4 个 safetensors shard 与 index。该证据验证了保存流程，尚未验证从这些状态恢复训练。

baseline 的同等 7-step 回归随后也通过：`succeeded=true`、`exit_code=0`、`monitor_errors=[]`，8 卡均有采样，`training_complete.global_step=7`、`skip_final_model_save=false`；checkpoint-7 同样包含 8 份 model / optimizer / RNG state，最终模型也有 4 个 shard。汇总产物为 `artifacts/paired-cache-regression-7step-20260923/comparison.json` 和 `comparison.md`。两种回归的资源记录如下；这仍是短程功能回归，不是完整 epoch 的速度或质量结论。

| 8 卡混合数据 7-step 回归 | 外层训练段用时 | Trainer 内部用时 | 单卡监测显存峰值 | 完成后保活 |
| --- | --- | --- | --- | --- |
| KTR（缓存 / 预取源码 `0b1bfb5`） | 312.470 秒 | 223.667 秒 | 74,827 MiB | 自动恢复，确认唯一进程 |
| baseline（生命周期增量 `99687e5`） | 290.077 秒 | 208.587 秒 | 51,021 MiB | 自动恢复，确认唯一进程 |

在仓库根目录复跑本次提交范围内的测试：

```bash
PYTHONPATH=tests python3 -m unittest -v \
  test_b200_lifecycle test_grpo_media_cache test_grpo_media_prefetch \
  test_grpo_validate_b200_dataset test_grpo_validate_b200_runtime \
  test_grpo_validate_b200_smoke test_grpo_verify_media_decode \
  test_key_token_attribution test_run_full_b200_contract
```

不要用标准输入临时 harness 运行含 multiprocessing spawn 的测试；以上 `python3 -m unittest` 命令是实际通过的可导入入口。

交付前还在集群 3 的实际隔离环境中以 CUDA 隐藏、CPU-only 方式复跑相同 89 项，全部通过（8.474 秒），保活保持运行；日志为 `artifacts/b200-committed-tests-on-target-20260923.log`。

## 2. 为什么这次失败

故障产物位于 `artifacts/grpo-full-b200/ktr-20260922T134530Z/`：

- `Holmes-16k-media-decoder-verified.json.manifest.json`：`source_records=16916`、`verified_records=16900`、`rejected_records=16`、`timed_out_records=16`。
- `Holmes-16k-media-decoder-verified.json.rejections.json`：16 条均为 600 秒 timeout，涉及 10 个唯一视频；末尾 8 条实际是两个视频重复 6 次与 2 次。
- `terminal.log`：随后 `grpo_validate_b200_dataset.py` 报 `decoder preflight filtered media; refuse a silent dataset change` 并退出 1。

原来的解码工具输出 `PASSED` 只表示“检查流程已跑完并发布了筛选结果”，不表示每条媒体都通过。上层严格门禁因此仍会失败。新的严格预检使用 `--require-all`，拒绝任意失败样本，避免把过滤后的输出误认为严格完整数据。

### 监督程序停顿的证据

| 原始日志事件 | 时间变化 | 同期独立保活日志 |
| --- | --- | --- |
| 完成 9,219 条后停顿，恢复后 8 个在途任务一起 timeout | `elapsed=2170 → 4306`，相差 2,136 秒 | 有 71 个保活采样点，最大间隔 31 秒 |
| 完成 16,310 条后停顿，恢复后另一组 8 个在途任务一起 timeout | `elapsed=6368 → 39186`，相差 32,818 秒，约 9.1 小时 | 有 1,089 个保活采样点，最大间隔 31 秒 |

解码监督程序应每 30 秒报告状态，却在上述区间没有心跳；整组任务在恢复后同时被判超时。保活进程的独立文件日志始终推进，说明不是整个节点暂停。检查时 cgroup 的 OOM 计数为 0，累计 CPU 限流约 0.25 秒；CPU / IO 压力计数也不足以解释 9 小时的停顿。

旧 launcher 把解码进程、监督程序和训练输出接到 `tee → IDE 终端`。终端停止读取时，管道回压可阻塞监督程序的 `print(..., flush=True)`，还会阻塞同管道上的 worker 警告输出；恢复后墙钟 deadline 已过，导致误判。该链路现已从生产者与终端的依赖关系中移除。

### 不是稳定的坏数据，也不是本次 GPU OOM

旧 run `artifacts/grpo-full-b200/ktr-strict-600-20260922T102554Z/` 已用相同 8 worker、600 秒 timeout、torchvision、8 帧和 401408 请求参数，在约 75 分钟内完成 16,916/16,916，0 拒绝、0 超时。两次 path-verified JSON 逐字节相等；旧 decoder 输出与 path-verified 输入逐字节相等。其 SHA256 为：

```text
19025cd0a08fb6b01b495668867f6888d79336caa23c2fe379652cb16af46a1f
```

本次 16 条失败样本复验保持原 Qwen / torchvision 预处理、8 帧、401408、CUDA 隐藏；将 OMP / MKL / OPENBLAS 各设为 1，使用 2 worker、单条 300 秒、整体 900 秒上限，日志直接落文件，保活保持开启。实际 67 秒解码完毕，整体约 71 秒，16/16 全通过，输入与输出 SHA 相同。实验同时改变了线程数和输出方式，不能单独量化各自贡献。

## 3. 新的数据与运行流水线

```text
固定模型 / 环境 / smoke 门禁
    → CPU 数据准备阶段维持保活
    → 路径与样本数量检查
    → 首次：构建精确预处理缓存；后续：验证并复用缓存
    → 严格确认全部 16,916 条按原序保留
    → 暂停保活，确认 GPU 无其他计算进程
    → GPU 训练 + CPU DataLoader 预取缓存并行
    → 停止本次拥有的进程，确认 GPU 释放
    → 自动恢复并确认唯一保活进程
```

首次完整缓存仍在训练前建立，以便在昂贵的正式训练前证明全部样本可用。改动把“每次全量解码、训练时再次解码”变成“一次精确预处理、之后复用”，而不是放宽数据门禁或边训练边静默丢样本。原有通过报告本身没有完整媒体快照，不能直接冒充已经构建好的新缓存。

### 缓存到底保存什么

`src/grpo_media_cache.py` 保存实际 Qwen `fetch_image` / `fetch_video` 的输出。图像使用无损 RGB PNG；视频保留原始预处理输出的 **CPU float32 tensor 和 sampled FPS**，不重新压缩视频、不转成 uint8，也不改变采帧、缩放或实际像素上限。每个唯一媒体的缓存按键加锁，完整写入后原子发布。

缓存键绑定源文件路径、size、mtime_ns、ctime_ns、device / inode、媒体参数、实际 Qwen 代码哈希、依赖版本及相关 reader 环境。创建时记录源媒体 SHA256；每次读取验证缓存 payload SHA256。训练采用只读缓存模式，缺失、损坏或参数 / 源身份变化时明确失败，不在训练 rank 内回退到可能无界的原始视频解码。

16,916 条标注含 8,765 个图像样本、8,151 个视频样本，对应 14,286 个唯一媒体。去重的是预处理工作，不是训练记录：同一视频对应多个问题的样本仍全部保留；dataset 顺序、问题内容和 sampler 语义保持原样。

UTC 2026-09-23 02:53:04 的全量 manifest 已确认 16,916 条全部通过，`supervisor_pause_seconds=[]`，0 超时 / 0 拒绝；source / output SHA 均为前述 `19025cd0…46a1f`。独立 KTR full wrapper 的 `data_gate.json` 同时确认 `strict-holmes-16k`、`decoder_filtered=false`，图像 / 视频计数均未改变。当时缓存目录占用约 51 GiB（`du -sh` 显示 `51G`）。

首次构建期间使用了共享缓存及分段预热，不能据此声称单任务冷启动吞吐。早期“约 4,200 项 / 7 秒”只是部分热点命中的局部进度，不代表全量热缓存门禁耗时。随后 baseline 7-step 的独立完整热门禁已实测 **30 秒**：UTC 03:00:59 开始、03:01:29 通过，验证全部 14,286 个唯一媒体，发布原序 16,916 条，0 拒绝 / 0 超时。该计时包含整个媒体缓存检查，不包含前后的模型 hash、import 和 GPU 预检；不能把 30 秒写成整个 launcher 启动时长。

**stat 的局限：**后续读取不重新散列整个原视频，只比较文件身份与时间戳等信息。它适合受控实验数据目录的一般变更检测，不防御刻意伪造文件时间 / 身份的修改；创建时的源 SHA 也不能证明后续每一步都重新核对了源内容。媒体或预处理环境改变时应重新验证并构建相应缓存，不得手工改 metadata 绕过检查。

### CPU 与 GPU 如何重叠

检查时 PyTorch 默认 intraop=144、interop=144，8 个解码进程可能产生大量线程竞争。新 CPU 解码 worker 限制为 1 线程；训练侧每 rank 默认 2 个 `spawn` DataLoader worker、`prefetch_factor=2`、persistent workers。worker 提前读取、校验并准备 CPU 缓存 payload，当前 batch 在 GPU 上生成 / 前后向时，CPU 可准备后续 batch。

多 worker 不重新创建或改变分布式 sampler；图像 / 视频混合规则保持原有实现。缓存 payload 在 collator 中保持 CPU 对象，进入模型前仍走原 processor 和输入搬运路径。这里重叠的是缓存读取及 CPU 准备，不能把它描述为“所有 CPU 工作与 GPU 工作完全重叠”。实际端到端吞吐改善需通过新的 smoke 和 full 资源记录测量。

### 输出、保活与并发

- 生产者直接写持久日志，终端由独立 observer 跟随。终端慢、流控或断开都不应反压解码监督与训练进程。1 MB 输出动态测试已覆盖无人消费终端管道的情况。
- CPU 数据预检开始时检测唯一已知保活；不存在就先确认 GPU 空闲，再通过官方 controller 启动，多个匹配进程则拒绝继续。生命周期增量 `99687e5` 还在 CPU 等待期间按 heartbeat 间隔检查，保活中途消失时恢复并刷新 PID / 启动时间身份；已请求暂停或已暂停阶段不运行这个 guard。此增量已通过本地动态测试，真实节点验证状态见上表。
- 只在占用 GPU 前暂停保活，验证停止及 GPU 空闲后才开训。CPU 任务结束时若受管理保活已消失，也会进入同一安全恢复流程。
- 成功、失败及可处理的退出信号都会清理本次拥有的 preflight / torchrun 进程组；确认没有 GPU 计算进程后再恢复保活，并验证唯一进程。若进程无法停止或 GPU 查询失败，拒绝叠加保活并记录恢复说明。
- smoke / full 使用同一个节点级 `flock`，防止 KTR 与 baseline 或另一份 smoke 同时操作 GPU 和保活；每次使用新 run 目录，拒绝覆盖旧结果。
- `launch_b200.sh` 分离后台任务与前台日志查看器。**Ctrl-C 只关闭当前查看器，训练继续**；SSH 断开同样不表示训练停止。不要用重复启动命令来“重新连接”已有任务，应打开该次打印的日志。
- `SIGKILL`、节点重启或掉电不能由 shell EXIT trap 保证恢复；此时需根据持久日志及实际进程状态恢复。该限制不影响正常成功 / 失败路径的自动保活管理。

真实 CPU 阶段故障注入已通过：在 baseline 的 CPU 预检中，通过官方 controller 主动停止唯一保活一次，随后不手工启动它。`artifacts/baseline-cpu-keepalive-fault-20260923.log` 记录注入 UTC 03:00:15；该 run 的 `terminal.log` 记录 guard 在 UTC 03:00:33 检测缺失、03:00:34 调用官方 controller、03:00:50 确认唯一新进程、03:00:51 更新 PID / 启动时间身份。约 35 秒内恢复保护，之后完整热缓存门禁通过，并能按计划暂停新保活进入 GPU 预检。日志终端时间为 UTC+8，本文以上时间已换算为 UTC。

## 4. 科学设置与 checkpoint 边界

| 项目 | 本次配置 / 行为 |
| --- | --- |
| 模型 | 已锁定并逐项校验的 `Video-R1/Qwen2.5-VL-7B-COT-SFT`；policy / reference 基座不变 |
| GPU / attention | 8×B200；FlashAttention-2；不静默降级 attention 后端 |
| 数据 | Holmes-16k 全部 16,916 条，图像与视频混合；不启用 reduced / decoder-filtered |
| prompt / completion / generations | 16384 / 768 / 8 |
| 视频 | 8 帧，CLI 请求 max_pixels=401408；Qwen 原有有效上限行为保持不变，已有约 105369 的上限警告不能写成每帧实际都使用 401408 |
| 时序反事实 | 单次随机乱序；不是 5 次置换平均 |
| token 选择 | 每 completion 精确 top-20%；视觉 / 时序使用绝对 delta；已区分 image / video token ID |
| baseline 对比 | 同模型、数据与共同训练设置；KTR 的 union token mask 与 baseline 按实验定义分别运行 |
| checkpoint | 每 100 optimizer steps 保存，保留最近 2 个；正常结束另保存最终结果 |
| 恢复训练 | **当前 full 入口不自动 resume**。重新启动不是自动从最近 checkpoint 继续；已有 checkpoint 不等于已有完整、验证过的恢复流程 |

无损媒体缓存和 CPU 预取属于执行实现变化，不能仅凭像素一致或单步 smoke 就宣称训练轨迹逐 bit 一致；公平 full 对照仍需要各自完整训练结果。

本轮保留了两类非致命日志，未为消除文字警告而修改科学设置：

- baseline 某个 regression completion 的答案文本不能转成浮点数；原 `accuracy_reward` 捕获异常并给该 completion 0 分，后续训练继续。它是原奖励函数的零奖励容错，不是解码失败；本次没有改奖励定义。
- 视频 `dDc9I0eFqT8.mp4` 首帧全 0，processor 检查首帧数值范围后提示 `already rescaled`。独立原始 `fetch_video` 重解码与缓存逐元素及 FPS 完全相同，原路径也触发相同提示，其他帧仍为 0–255。该提示不证明缓存误归一化，不能因此关闭 `do_rescale`。

## 5. 用户启动命令

以下正式入口已完成启动前验收，由用户手动执行。完整缓存、8 卡 paired smoke、两种 7-step 回归、保存及自动保活恢复均已通过。`<共享持久卷>` 需要替换为集群 3 的实际持久卷路径；模型、数据、环境与官方保活 controller 的机器私有配置保存在项目外的 `b200.env`，不放进 Git。

KTR：

```bash
cd <共享持久卷>/video-ktr-b200/repo
bash launch_b200.sh ktr
```

baseline（等 KTR 结束后执行，或反过来）：

```bash
cd <共享持久卷>/video-ktr-b200/repo
bash launch_b200.sh baseline
```

入口显式选择完整 epoch、严格全量数据和自动管理保活。两次运行各自从固定基座开始，生成独立目录；不要并发提交，也不要把 KTR 产出的权重作为 baseline 的初始权重。启动器会打印本次 run / log 路径。若只关闭了查看器，重新查看该日志即可。

## 6. 本次交接：已就绪，正式 full 待用户启动

```text
[KT-HANDOFF/v1]
ID: 20260923-AI-PIPELINE-001
TIME_UTC: 2026-09-23T03:07:59Z
FROM / TO: AI / 用户与远程协作者
TYPE: HANDOFF
STATUS: PASSED
STATE_CHANGE: READY：16条复验、8卡paired smoke、全量精确缓存、两种7-step回归及真实保活故障恢复全部通过；正式完整epoch未启动。
SCOPE: 集群3的B200 KTR与baseline数据/训练启动流水线
ENV: 集群3；branch=<owner>/video-ktr-repro-handoff；KTR回归缓存/预取源码=0b1bfb5；baseline回归生命周期版本=99687e5；文档版本=包含本文的交付提交（见Git历史）；failed_run=ktr-20260922T134530Z
CLAIM: 修复后的严格全量训练入口已通过有界真实验证，可由用户串行启动KTR和baseline；这不代表正式完整训练已完成。
EVIDENCE: 提交范围89测试通过；16条复验0拒绝/超时、67s；3媒体一致性及8卡paired smoke通过；全量缓存16916/16916、14286unique、0reject/timeout且SHA不变；KTR7-step exit0、312.470s/74827MiB；baseline7-step exit0、290.077s/51021MiB；两者checkpoint/最终模型及唯一保活恢复均通过；完整热门禁30秒，CPU故障注入后约35秒恢复保活并刷新身份。
CHANGES: run_full_b200.sh；launch_b200.sh；somke-b200.sh；src/grpo_verify_media_decode.py；src/grpo_media_cache.py；src/grpo_media_prefetch.py；src/grpo_probe_media_cache.py；trainer缓存预取接入；相关测试；B200_PIPELINE_VALIDATION.md。最终以git diff为准。
EXECUTED: 已完成只读诊断、16条CPU复验、本地回归、3媒体一致性、8卡paired smoke、全量精确缓存、串行KTR与baseline各7-step；已通过官方controller实施一次CPU保活停止故障注入并验证自动恢复；两种回归结束后均验证GPU释放与唯一保活恢复。未执行正式完整epoch。
ARTIFACTS: artifacts/grpo-full-b200/ktr-20260922T134530Z/；artifacts/decoder-diagnostics/retry16-threads1-20260923T022051Z/；artifacts/media-cache-v1-20260923/{parity.json,build.log,verified.json}；artifacts/grpo-smoke-b200/cache-prefetch-20260923/；artifacts/grpo-full-b200/cache-regression-{ktr,baseline}-7step-20260923/；artifacts/paired-cache-regression-7step-20260923/comparison.{json,md}；artifacts/baseline-cpu-keepalive-fault-20260923.log；repo/artifacts/b200-committed-tests-20260923-module.log。
RISK / ROLLBACK: stat不是每步源内容hash；缓存只读miss会失败；短程回归不保证完整epoch全程或收敛；checkpoint保存已验但自动resume未实现；异常硬杀不保证EXIT恢复；不得放宽完整数据门禁解决timeout。
LAST_VERIFIED: UTC03:07:27 baseline7-step wrapper exit0且唯一保活恢复；两种training_complete均step7、skip_final_model_save=false、资源monitor_errors=[]；全量严格门禁与CPU保活故障恢复通过。
RUNNING: NONE（无本次训练/解码任务；唯一保活已恢复运行）。
RECOVERY: 正常退出由官方controller自动恢复保活；若硬故障，先确认GPU计算进程与唯一controller状态，再按机器私有配置恢复。
FIRST_COMMAND: git status --short
NEXT: 用户按第5节串行执行bash launch_b200.sh ktr与bash launch_b200.sh baseline；各自从固定基座开始，完整epoch完成后再比较质量、耗时与显存。
ASK: NONE
```
