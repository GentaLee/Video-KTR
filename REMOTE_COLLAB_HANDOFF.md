# Video-KTR 远程协作交接、环境与复现记录

> 适用范围：本文件是远程同事与 AI 助手共同使用的单一交接入口。
> 匿名约定：不记录真实机器名、账号、IP、私有路径、密钥或保活 PID；环境统一称为“集群 1 / 2 / 3”，持久卷统一写作 `<共享持久卷>`。

## 0. 强制对话格式（人和 AI 都必须遵守）

远程协作者不能访问彼此的终端、机器和对话历史。因此每次有关本项目的消息都使用下列格式；没有证据时必须写“待验证”，不能用推测替代事实。人类和 AI 的消息一律适用，接收方也必须用同一格式回复 ACK、决策或下一次交接。

```text
[KT-HANDOFF/v1]
ID: <UTC日期>-<发送方>-<序号>
TIME_UTC: <ISO-8601>
FROM / TO: <角色>
TYPE: ACK | UPDATE | EXECUTION | DECISION | INCIDENT | REVIEW | VERIFICATION | HANDOFF
STATUS: PLANNED | RUNNING | PASSED | FAILED | BLOCKED | NEEDS_DECISION
STATE_CHANGE: <相对上一条消息的状态变化；无则写 NONE>
SCOPE: <本消息仅覆盖的对象>
ENV: 集群<n>；branch=<...>；commit=<...>；run=<相对标识>
CLAIM: <一句可检验的结论；计划不能写成已完成>
EVIDENCE: <退出码、相对路径、摘要数字或日志行；不得粘贴密钥>
CHANGES: <修改文件；只读则写 NONE>
EXECUTED: <已实际执行的命令；未执行则写 NONE>
ARTIFACTS: <相对路径；无则写 NONE>
RISK / ROLLBACK: <风险、偏差、恢复方式>
LAST_VERIFIED: <最后一次已通过的检查 + 证据；无则写 NONE>
RUNNING: <正在运行的 task/run + 最近进度 + 下次检查条件；无则写 NONE>
RECOVERY: <泛化恢复命令或 NONE；不得写密钥/私有路径>
FIRST_COMMAND: <下一位执行者第一条命令；只读优先>
NEXT: <负责人 + 下一步 + 验收条件>
ASK: NONE | <需要对方做出的明确决策>
```

### 对话规则

1. `EXECUTED` 和建议执行的命令必须分开；建议命令只放在 `NEXT`，不得伪装成已执行。
2. `PASSED` 必须同时给出退出码和可定位证据；`RUNNING` 必须写 run 标识、最近进度与下次检查条件；状态变化写入 `STATE_CHANGE`。
3. 启动、停止、恢复训练/保活、删除文件、创建或推送分支，都必须填入 `CHANGES` 或 `EXECUTED`。
4. 报告环境时只写“集群 1/2/3”和硬件/软件版本，不能泄露真实主机标识、私有目录、IP、令牌或 SSH 材料。
5. 任何人或 AI 在改变模型、数据、超参数、attention 后端、GPU 数或 token 选择定义前，必须在 `RISK / ROLLBACK` 中说明其与基线的差异。
6. 遇到不确定性，使用“已验证 / 待验证 / 推断”三个标签；不得把旧日志当作当前机器状态。
7. AI 助手同样受此格式约束：不隐式启动 full run、不隐式暂停外部保活、不直接向他人分支推送，也不将凭据写入代码、日志或文档。
8. 交接消息必须通过 `ENV`、`LAST_VERIFIED`、`RUNNING`、`RECOVERY` 和 `FIRST_COMMAND` 包含当前分支、最后一次成功验证、正在运行任务、恢复命令的泛化描述和下一位执行者第一条命令。

推荐的最小交接示例：

```text
[KT-HANDOFF/v1]
ID: <UTC日期>-<角色>-001
TIME_UTC: <ISO-8601>
FROM / TO: AI / 远程协作者
TYPE: HANDOFF
STATUS: RUNNING
STATE_CHANGE: NONE
SCOPE: KTR full run
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=<commit>；run=<run-root>；GPU 被训练独占，外部保活已由官方 controller 暂停
CLAIM: B200 KTR full run 正常推进；尚未可宣称训练收敛。
EVIDENCE: 最近 heartbeat、training.log 的 step、连续 gpu_metrics.jsonl；training_complete.json 仅在结束后存在。
CHANGES: NONE
EXECUTED: 只读检查进程、step、显存和错误日志。
ARTIFACTS: <run-root>/terminal.log；<run-root>/training.log；<run-root>/gpu_metrics.jsonl
RISK / ROLLBACK: 当前请求 G=8/FA2；若数据为 reduced-media-subset，不能称严格 Holmes-16k；不要重复启动训练。
LAST_VERIFIED: <last smoke>；exit=0；<相对 artifact>/training_complete.json
RUNNING: <run-root>；最近 step=<step>；下次检查=下一个 heartbeat 或错误关键词出现时。
RECOVERY: KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start（仅已确认训练停止且自动恢复失败时）
FIRST_COMMAND: tail -n 40 <run-root>/terminal.log
NEXT: 接收者只读检查上述三类证据，并回传新的 KT-HANDOFF/v1 记录。
ASK: 是否为了严格上游配置另行准备 8-GPU/FA2 profile。
```

## 1. 交付边界与分支规则

本次交付在独立分支 `<owner>/video-ktr-repro-handoff` 上完成，不直接修改或推送到协作者的功能分支。

| 规则 | 约定 |
| --- | --- |
| 分支命名 | `<owner>/video-ktr-<purpose>`；每个可审阅实验/交接单独建分支 |
| 提交前检查 | `bash -n`、`py_compile`、`git diff --check`，以及与改动相称的 smoke/解析检查 |
| 推送前审阅 | 检查 `git status`、`git diff --cached`、忽略规则；不得提交模型、数据、wheel、cache、artifact、凭据 |
| 训练中的 worktree | 不执行会覆盖运行源码的切换/重置；新建分支只移动 ref，不中断已加载的训练进程 |
| 推送后交接 | 报告分支名、commit hash、远端 tracking 状态、未提交项和仍在运行的任务 |

## 2. 匿名环境配置

| 环境 | 角色 | 已验证配置 | 使用约束 |
| --- | --- | --- | --- |
| 集群 1 | CPU / 联网准备端 | 有外网；与集群 2 共享持久卷 | 下载依赖、查询官方元数据、准备 wheel；不承担 GPU 训练 |
| 集群 2 | 历史 H200 诊断端 | 4×H200（约 143,771 MiB/卡）、Python 3.12、PyTorch 2.10 + CUDA 12.8 | 仅保留旧 full 故障定位；不再作为 active full profile |
| 集群 3 | active B200 训练端 | 8×B200、Python 3.12、CUDA 13.1、B200 原生 FA2、隔离 venv | `setup_b200_env.sh`、`somke-b200.sh` 已真实验证；full 只能通过 `run_full_b200.sh` 手动启动 |
| `<共享持久卷>` | CPU/GPU 共享资产 | 项目源码、模型、Video-R1 数据、wheel/overlay、运行 artifact | 只存非机密运行资产；不要存私钥、token 或认证文件 |

### 已锁定的软件与资产

| 类别 | 已验证版本/身份 | 用途 |
| --- | --- | --- |
| 模型 | `Video-R1/Qwen2.5-VL-7B-COT-SFT`，Qwen2.5-VL，BF16，8.29B 参数 | policy 与 reference 的基座 checkpoint |
| 模型身份（集群 3 target gate） | checked-in 的 exact-revision `manifests/video-r1-qwen25vl-7b-cot-sft-f71f0f1e22c015007fccd080eef87824fe292a10.json` 覆盖模型目录全部顶层常规文件（含 config、index、weight shard、tokenizer/processor），runtime gate 逐项重算 | CPU-only 实测已通过 19 文件/4 shard；full 必须继续 fail-closed 并在占卡前复验。本地 verified-copy 只能用于独立比较/恢复，不能替代该锚；旧“4 shard 匹配”仅为无机器可读证据的历史说法 |
| Transformers | `4.49.0.dev0` | 与 checkpoint 所需 Qwen2.5-VL 路径兼容 |
| tokenizers / TRL / DeepSpeed | `0.21.4` / `0.16.0` / `0.15.4` | project-local overlay，避免污染系统环境 |
| 离线安装资产 | GRPO runtime wheelhouse + checkpoint-compatible wheelhouse | 两者都不随 Git 提交；当前 bundle 对基础镜像仍有显式依赖，不能把它称为 clean-Python 完整闭包 |
| 数据源 | Video-R1 标注共 263,071 条；路径存在且非空的 video 为 116,248 条 | full 默认先做 path verification，再以训练同一 Qwen/torchvision 视频预处理做隔离 decode verification；是否有过滤必须以本次 manifest 为准 |
| 分布式 | 4 rank、ZeRO-3、BF16、gradient checkpointing | 当前集群 2 已验证 profile |

### 离线环境重建前置条件

源码仓库故意不提交模型、数据、wheelhouse、Python overlay 或运行 artifact。集群 2 重建环境前，交接者必须先确认以下非 Git 资产已经被放置到可读的共享位置：

| 前置资产 | 必需内容 | 可配置入口 |
| --- | --- | --- |
| GRPO runtime wheelhouse | TRL、DeepSpeed、datasets、accelerate、PEFT 及已暂存依赖 | `GRPO_WHEELHOUSE` |
| checkpoint-compatible wheelhouse | Transformers `4.49.0.dev0`、tokenizers `0.21.4`、视频解码依赖 | `COMPAT_WHEELHOUSE` |
| GPU image runtime | CUDA-coupled torch、torchvision、flash-attn、Python 3.12，以及 `numpy`/Pillow/huggingface-hub/filelock/packaging/PyYAML/requests/safetensors/tqdm/psutil/msgpack | `GRPO_BASE_PYTHON`（必要时） |
| 模型与数据 | 已验哈希的 checkpoint、标注与实际媒体文件 | `MODEL_PATH`、`DATA_ROOT`、`DATASET_SOURCE` |

在 GPU 节点上执行 `setup_grpo_env.sh` 前，应先按第 0 节格式报告这些资产的位置类别与版本；脚本会拒绝缺失的 wheelhouse、关键 wheel 或上述基础镜像 import，不会联网下载，也不会写入系统 Python。**已知妥协与降级边界：**当前 wheelhouses 并非 clean-Python 的完整 transitive closure，当前脚本也没有“完整辅助 bundle → full overlay”的安装路径。因此 clean-Python 重建在本版本**不受支持**；不能只准备一个 bundle 后绕过 gate，更不能让 GPU 节点联网补包。该降级路径需先补齐完整、版本锁定的 bundle，扩展并验证 `setup_grpo_env.sh` 的 full-overlay 安装模式后才能启用；目前应使用已验证的 GPU image。`smoke.sh`、`run_full.sh` 与非 GRPO selector 都没有私有路径默认值，必须显式提供 `MODEL_PATH` 和（适用时）`DATA_ROOT`；selector 可用 `SITE_PACKAGES` 覆盖默认的 GRPO overlay。

### 历史集群 2 / H200 最小可执行阶梯（归档；不得在集群 3 执行）

> **归档警告：以下仅是集群 2 / H200 故障定位记录，禁止复制到集群 3。** 集群 3 只能使用第 10 节的 `run_full_b200.sh`，并采用该节的保活语法与数据门禁。

以下命令只使用占位符，且每次 `RUN_ROOT` 都必须是尚不存在的新目录。旧探索性 full 已失败结束；checkpoint-500→501 的恢复 smoke 已通过，下一步是跨越旧 step-529 的诊断，再由其结果决定是否启动新的 full。

```bash
cd <REPO_ROOT>
export GRPO_WHEELHOUSE=<GRPO_RUNTIME_WHEELHOUSE>
export COMPAT_WHEELHOUSE=<CHECKPOINT_COMPAT_WHEELHOUSE>
export MODEL_PATH=<MODEL_ROOT>/Video-R1/Qwen2.5-VL-7B-COT-SFT
export DATA_ROOT=<DATA_ROOT>

./setup_grpo_env.sh
VARIANT=both ./smoke.sh

# [历史 H200 ONLY；不得在集群 3 执行]
# 仅在修复后 paired smoke 通过后；没有外部保活时不要设置 ALLOW_*。
# 仅定位/回归：保留旧 manifest 顺序，跨越旧故障点；不要将其当作正式结果。
# launcher 会 scoped allowlist 两个已核验 DeepSpeed 类；不要设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD。
VARIANT=ktr \
RUN_ROOT=<ARTIFACT_ROOT>/ktr-checkpoint500-diag-<UTC> \
PREPARED_DATASET=<OLD_RUN>/Video-R1-260k-video-verified.json \
USE_EXISTING_PREPARED_DATASET=1 \
RESUME_FROM_CHECKPOINT=<OLD_RUN>/training/checkpoint-500 \
VERIFY_VIDEO_DECODE=0 MAX_STEPS=540 SKIP_FINAL_MODEL_SAVE=true \
NCCL_DIAGNOSTICS=1 RANK_PHASE_TRACE=1 \
./run_full.sh

# 诊断通过后，正式 full 从 step 0 开始；默认 decode verification 会持续报告 ETA。
VARIANT=ktr RUN_ROOT=<ARTIFACT_ROOT>/ktr-decoder-verified-<UTC> ./run_full.sh
```

若节点确有外部保活，额外显式配置其两个身份并取得暂停授权后才运行对应命令（**历史 H200 模板；不适用于集群 3**）：

```bash
export KEEPALIVE_MAIN=<KEEPALIVE_MAIN>
export KEEPALIVE_LAUNCHER=<KEEPALIVE_LAUNCHER>
ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1 VARIANT=both ./smoke.sh
ALLOW_PAUSE_EXTERNAL_KEEPALIVE=1 VARIANT=ktr RUN_ROOT=<ARTIFACT_ROOT>/ktr-<UTC> ./run_full.sh
```

## 3. 历史 H200 方法与实现记录（归档）

目标流程保持如下语义：

```text
视频/问题 → 每个 prompt 生成 G 条 CoT
         → 对同一 completion 做原始、视觉反事实、时序反事实 teacher-forcing
         → E（高熵）∪ V（视觉敏感）∪ T（时序敏感）
         → 仅 union token 进入 GRPO policy + KL
         → backward / optimizer，并记录资源与证据
```

### 最终实现组件

| 组件 | 相对路径 | 职责 |
| --- | --- | --- |
| direct KTR trainer | `src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py` | 同一 completion 的 E/V/T 归因；union 同时掩码 policy 与 KL；输出最后一步指标 |
| packaged KTR helpers | `src/r1-v/src/open_r1/trainer/ktr_token_utils.py` | 与 standalone selector 同语义的纯 tensor/helper；不依赖仓库根 `PYTHONPATH` |
| GRPO entrypoint | `src/r1-v/src/open_r1/grpo.py` | 参数校验、direct/vLLM 路径选择、训练完成元数据 |
| 依赖安装 | `setup_grpo_env.sh` | 建立 project-local Python overlay，并做 import gate |
| 单步 smoke | `smoke.sh` | baseline/KTR 真正的 generate→forward→backward→optimizer；持续 heartbeat 和遥测 |
| full launcher | `run_full.sh` | 新 run-root 门禁、path/decode verification、容量门禁、rank 日志/NCCL 诊断、监控、checkpoint、退出清理、保活恢复与源码快照 |
| 数据准备 | `src/grpo_prepare_dataset.py` | 绝对路径安全校验、路径存在/非空筛选、manifest 与进度输出；不做全量解码 |
| 解码验证 | `src/grpo_verify_video_decode.py` | 在可杀死的 CPU 子进程中执行与训练相同的 Qwen 视频预处理；硬超时、完整 rejection manifest、30 秒进度/ETA |
| 资源遥测 | `src/grpo_resource_monitor.py`、`src/grpo_run_summary.py` | GPU 显存/利用率/温度/功耗 JSONL，运行时长与峰值摘要 |
| 证据与对比 | `src/grpo_token_report.py`、`src/grpo_compare_runs.py` | E/V/T CoT 样例、baseline 与 KTR 时长/资源对照 |

### 方法级对齐状态

| 原始高层要求 | 当前实现 | 状态 | 证据 |
| --- | --- | --- | --- |
| 视频/问题输入 | full 默认过滤为路径存在且非空的 video，再做 Qwen/torchvision decode verification | 已实现；若有 rejection，则是数据质量过滤后的降级复现，不是原始全量语料的严格复现 | 两层 manifest、rejections JSON、dataset SHA-256 |
| 每 prompt 生成 8 个回答 | 当前 `G=4` | 未严格对齐 | 见第 6 节妥协 |
| 正确性与格式奖励 | accuracy + format reward | 已对齐 | trainer metrics |
| 同一 completion 三类归因 | 原始、视觉屏蔽、多个帧置换复用 completion | 已对齐 | KTR trainer / smoke |
| 高熵 E | 原始前向 token entropy | 已对齐 | `ktr/entropy_tokens` |
| 视觉敏感 V | 视觉 placeholder 屏蔽后的绝对 logp delta | 已对齐并修复旧 token-id 问题 | `ktr/visual_tokens` |
| 时序敏感 T | 5 个非恒等帧置换的平均绝对 logp delta | 已实现；与旧代码的单次乱序不同 | `ktr/temporal_tokens` |
| E∪V∪T 选择 | 每 completion 精确 top-20%，取 union | 已实现；与旧 batch quantile 不同 | `ktr/union_tokens` |
| union 进入 policy + KL | `loss_mask = union_mask` 同时用于两项 | 已对齐 | direct trainer loss |
| 更新 Qwen2.5-VL | 真正 optimizer step 和 full run | 已实现 | `global_step`、checkpoint |

## 4. 历史 H200 已验证结果与对比表（归档）

### 历史 baseline 与 KTR smoke（4×H200，真实 1 optimizer step）

| 指标 | baseline | KTR | KTR − baseline |
| --- | ---: | ---: | ---: |
| 外层运行时长 | 66.633 s | 70.598 s | +3.965 s |
| Trainer 内部 runtime | 29.0875 s | 31.7611 s | +2.6736 s |
| 单卡峰值显存（四卡最大） | 89,839 MiB | 92,197 MiB | +2,358 MiB |
| 单卡 GPU 利用率峰值 | 100% | 100% | 0 pp |
| 优化步 | 1 | 1 | — |

该历史 smoke 证明了生成、三类 mask、union policy+KL、反传和资源遥测均可运行；KTR 最后一步的分布式每 completion 均值为 E/V/T=`45.5/45.5/45.5`、union=`80.188`，union 更新比例 `35.7%`。保存的 CoT 证据中 E、V、T 三类均非空，且记录位于 `<think>` 内；功能词也可能被选中，这是 log-probability 扰动归因的正常结果，而不是人工词性筛选。

**限制：**这组数值在长度奖励修复前产生。旧代码错误地用 union 长度而非真实 completion 长度决定 length-control bonus，因此不能作为 KTR 与 baseline 的公平训练效果/奖励对比。新代码已修复为 `valid_mask.sum(dim=1)`；在任何结论性比较之前，必须在修复后的同一 commit 上重新跑 baseline/KTR smoke。

### 历史 CoT token 形式证据（非公平训练效果结论）

下表来自长度奖励修复前、且没有启动时新 provenance 的历史 token report；它只证明三类筛选能在 CoT 中落到具体位置，不能用于比较训练效果。分数分别是对应类别的 entropy / visual delta / temporal delta。

| 展示的类别分数 | 选中 token | 分数 | `<think>` 内的短上下文 | 该 token 的全部标签 |
| --- | --- | ---: | --- | --- |
| E（高熵） | ` being` | 3.511903 | `...car crash test scene, the car is being hit by a barrier at 40 km/h...` | E / V / T / U |
| V（视觉敏感） | ` pool` | 7.127037 | `...two different scenes: one with a pool table...` | E / V / T / U |
| T（时序敏感） | ` crash` | 4.130900 | `...another with a car crash test...` | E / V / T / U |

这三个代表性 token 恰好同时落入 E、V、T 三个 mask；表中“展示的类别分数”只说明该行展示的是哪一种归因分数，不表示其只属于该类别。这正是最终使用并集 `U` 的原因。

### 历史的 full 同形状容量 smoke（KTR-only，真实 1 optimizer step）

| 指标 | 结果 |
| --- | ---: |
| profile | 4×H200，G=4，prompt=4096，completion=512，8 帧，5 次时序置换 |
| 外层运行时长 | 79.013 s |
| Trainer 内部 runtime | 40.6615 s |
| 单卡峰值显存（四卡最大） | 92,197 MiB |
| GPU 利用率峰值 | 100% |
| E / V / T / union（每 completion） | 48.25 / 48.25 / 48.25 / 84.0625 |
| union 更新比例 | 35.26% |

结论：该修复前 KTR-only 单步证据表明集群 2 的 4×H200 曾可启动相同容量量级；它不等价证明修复后 profile 的行为、完整 epoch 的长期稳定性或最终指标。修复后 full 前应先完成新的 paired smoke。

### 已结束的探索性 full 故障（待回归验证）

旧探索性 KTR run 已结束，不能再视为 active run。其 checkpoint-500 有完整的 4-rank ZeRO-3 model/optimizer shard，可用于**仅定位用**的短恢复；它不是修复后正式实验的结果。

| 观测 | 已验证证据 | 结论边界 |
| --- | --- | --- |
| 终止位置 | 在 `global_step=529` 附近失败；checkpoint-500 可机械读取 | 只说明前 500 step 已保存，不代表后续稳定 |
| NCCL 表现 | rank 1/2/3 在下一次 `_ALLGATHER_BASE` 等待约 30 分钟；rank 0 的最后 enqueue/complete 均停在前一序号 | 这是 rank 间 collective 顺序失步的证据，不单独证明根因 |
| GPU 形态 | 等待期间 rank 0 长时间近 0% 利用率，其他 rank 接近满载 | 最高置信推断：rank 0 在下一 collective 前的本地工作停住 |
| 显存 | 最大观测约 115.7 GiB，小于单卡约 143.8 GiB | 不是 OOM 证据 |
| 最可能本地阶段 | `qwen-vl-utils` 的 torchvision whole-file 视频 decode 在 generation 前执行；旧 manifest 只验证路径/非空 | **推断而非已证明的单一根因**；新代码以相同 Qwen 预处理做隔离验证并留下 rank trace |

一次新的 checkpoint-500 诊断先在训练开始前失败，但它是独立的恢复兼容性问题而不是上述 step-529 故障：PyTorch 2.10 的 `weights_only=True` 拒绝了 DeepSpeed 0.15.4 ZeRO state 中的 `ZeroStageEnum` 和 `LossScaler`。四 rank 都在 checkpoint load 退出，没有 rank trace、NCCL timeout 或数据读取。当前实现保持 `weights_only=True`，仅在 `trainer.train(resume_from_checkpoint=...)` 作用域内 allowlist 这两个已对 4 个 optimizer shard 静态核验的类；`PREFLIGHT_ONLY` 也会先经相同 DeepSpeed loader 验证。修复后的 4×H200 smoke 已从 step 500 完成实际 step 501、四份 rank trace 均到 `metrics_done`、GPU 已清空；它不代表已跨越 step 529。

为此新增的保护不是把问题归因于一个未验证的 NCCL 开关：

1. full 默认在 CPU-only、可终止子进程中跑训练同一 Qwen/torchvision decode、采样和 resize；可恢复的媒体错误/超时写入有序 rejections JSON，意外 worker crash 则 fail-fast。即使 0 条通过也会保留 failed manifest/rejections。若有过滤，run 必须标为数据质量过滤后的降级复现。
2. KTR trainer 在 decode/generate/policy/visual/temporal/ref/metrics 边界可选写入每 rank JSONL；短恢复设置 `RANK_PHASE_TRACE=1` 后，最后一条事件可定位卡住的样本/阶段。
3. 多 rank generation 显式传递 `synced_gpus=True`，并显式传递 ZeRO-3 generation gather 配置。该项是稳健性/可观测性保护，不能修复 generation 之前的 decode stall。
4. launcher 默认启用 PyTorch NCCL trace/dump 诊断、10 分钟 DDP timeout、`torchrun --tee` 分 rank 日志，并在失败时写 `training_failed.json`；watchdog dump 应从 `training.log`/`torchrun-logs` 读取，不能假定存在 `fr_trace.py` 所需的磁盘 trace 文件。
5. checkpoint 恢复强制使用**新** run root、旧 path-verified JSON，并强制关闭 decoder filtering。PyTorch/DeepSpeed legacy state 只在 train resume 的 scoped `safe_globals([ZeroStageEnum, LossScaler])` 中读取，不能用全局 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` 代替。历史数据全为 video，当前 trainer 对 homogeneous 数据委托父类 `RandomSampler`，保留旧 checkpoint 的 sampler contract。
6. 所有可能长时间运行的预检（import/path/decode）均为 launcher 所追踪的独立 process group；`TERM` 会先停它们再恢复已授权保活。`PREFLIGHT_ONLY=1` 可完整验证这些阶段且明确不启动 `torchrun`。

非训练验证已通过：4 条真实视频的 tracked `PREFLIGHT_ONLY=1` 为 4/4，明确写入 `torchrun_started=false` 且结束后无 worker/GPU 残留；16 worker、32 条真实视频抽样为 32/32，无 timeout。该抽样只验证实现和隔离机制，不能外推为 116,248 条全量数据的耗时或零 rejection。

## 5. 历史 H200 已踩坑、修复与优化（归档）

| 类别 | 现象 / 风险 | 已验证处理 | 结果 |
| --- | --- | --- | --- |
| 网络与依赖 | GPU 端应视为离线，直接安装会失败或版本漂移 | 集群 1 下载，写入共享卷；集群 2 使用 project-local overlay | import gate 通过 |
| Python 安装能力 | GPU Python 缺少可用 `ensurepip` 路径 | 使用目标目录 overlay，而不是破坏系统 Python | 可复现加载训练依赖 |
| 模型版本 | checkpoint 依赖特定 Transformers 行为 | 锁定 `4.49.0.dev0` 与对应 tokenizers | 模型/processor 本地加载成功 |
| FlashAttention2 | Qwen mRoPE 路径出现 float32/bfloat16 rotary dtype 断言 | 默认 `ATTN_IMPLEMENTATION=sdpa` | smoke 与 full 可运行 |
| vLLM 路径 | 不适合直接施加 E∪V∪T 的 policy+KL 掩码 | KTR 强制 direct trainer；拒绝 `use_vllm=true` + KTR | union loss 语义可验证 |
| 视频 token id | 旧实现把 video placeholder 当 image id 处理 | 显式读取独立 image/video token id | V 归因正确覆盖视频 placeholder |
| mRoPE 位置 | 视觉屏蔽可能重算位置，混入位置变化 | 三种前向共享原始 mRoPE position ids | V/T counterfactual 更可解释 |
| T 归因稳定性 | 单次随机乱序方差大 | 使用 5 个非恒等置换，包含 reverse，取平均绝对 delta | T mask 稳定且非空 |
| token 筛选 | 原代码跨 batch quantile、signed delta 且可能保留 ties | `paper/per_completion` 精确 top-20%、absolute delta | 每条 completion 的选中数量可审计 |
| 数据路径 | 标注不保证媒体文件存在 | 训练前筛选路径存在且非空的 video，写 manifest | 避免路径缺失；不等于视频可解码 |
| 全量媒体健康度 | path-verified 并不验证 codec、抽帧或 resize，且单 rank decoder stall 会拖住 ZeRO collective | `run_full.sh` 默认调用可杀死子进程的 Qwen decode preflight，训练端固定同一 torchvision backend；manifest 绑定 source SHA、video root、decoder/backend/nframes/max-pixels | 有 rejection 时必须保留 manifest/rejections，并标为数据质量过滤后的降级复现；worker crash 不静默过滤 |
| length-control 对齐 | 旧 direct KTR 用 union 长度判定长度奖励，KTR/baseline reward 不对等 | 改用真实 `valid_mask` completion 长度；历史 comparison 仅保留为操作/容量证据 | 后续比较必须重跑两变体 |
| 入口可移植性 | helper 位于仓库根会使标准 `src/r1-v` launcher 导入失败 | helper 放入 `open_r1` package；仅有 `--data_root` 的 baseline 与 KTR 共用 direct trainer，无该参数 baseline 保持原 trainer | 不能把直接路径默认强加给普通上游 baseline |
| 可观测性 | 原训练不完整记录时长/显存/token 示例 | heartbeat、JSONL 遥测、summary、comparison、bounded token report | 可远程复查 |
| rank 失步定位 | 旧日志无法定位 rank 0 停在 decode、generation 还是 teacher forcing | 可选 rank-phase JSONL、NCCL trace/dump、torchrun per-rank logs 和失败 marker | 短恢复先跨过旧故障步，再决定正式 full |
| 外部保活 | 保活占满 GPU 会污染资源数据；粗暴停止有风险 | 精确 PID + start-ticks 校验，需显式授权；在空闲显存 gate 前暂停、退出时原 launcher 恢复 | 不会因保活占卡而提前误判显存不足 |
| 中断安全 | 仅终止 shell 可能让 torchrun/decoder worker 继续、保活提前恢复 | launcher 跟踪 torchrun 与 import/path/decode process group；清理时先 TERM/wait，再恢复保活 | 降低 GPU 冲突风险 |
| 环境交接 | wheelhouse/overlay 被 `.gitignore` 排除，clone 后并不天然可运行 | `setup_grpo_env.sh` 明确检查两个 wheelhouse，支持 `GRPO_WHEELHOUSE` 与 `COMPAT_WHEELHOUSE` 覆盖 | 避免同事误以为源码仓库包含全部离线依赖 |

## 6. 历史 H200 妥协、降级路径与不可自动化选择（归档）

### 当前 profile 与上游 launcher 的对比

| 项目 | 上游 KTR launcher | 当前 full profile | 状态 / 原因 |
| --- | ---: | ---: | --- |
| rank 数 | 8 | 4 | 当前可用训练端只有 4×H200；全局 batch 随之从 8 变为 4 |
| 数据 | Holmes-16k | 116,248 条 path-verified video，经本次 decoder manifest 决定最终训练集 | 当前更大规模，不能作为同一数据分布的严格数值复现；若 decoder 有 rejection，另有数据质量过滤偏差 |
| prompt 上限 | 16,384 | 4,096 | 为当前容量与吞吐做的保守配置 |
| completion 上限 | 768 | 512 | 为当前容量与吞吐做的保守配置 |
| GRPO generation 数 `G` | 8 | 4 | 未满足原始高层描述的“8 个回答” |
| 最大像素 | 401,408 | 100,352 | 降低视觉 token 数、显存与吞吐压力 |
| attention 后端 | FlashAttention2 | SDPA | FA2 与当前 mRoPE/版本组合不兼容；SDPA 已实测稳定 |
| 时序 probe | 旧代码一次随机乱序 | 5 次非恒等置换平均 | 方法修正/增强，不是逐行回放 |
| V/T token 语义 | signed delta、batch quantile | absolute delta、per-completion exact top-k | 方法修正/增强，不是旧实现的逐位复现 |
| checkpoint | 每 100 step | 每 500 step、保留 2 个 | 降低长期存储压力 |

### 降级决策树

| 触发条件 | 允许动作 | 代价 / 必须记录 | 禁止动作 |
| --- | --- | --- | --- |
| 预检显存不足 | 退出，不启动训练；先保留日志和 GPU 快照 | 无训练结果；需另行做容量决策 | 静默抢占或停掉外部任务 |
| FA2 dtype 错误 | 切换到 SDPA（当前已采用） | 速度下降、数值实现不完全相同 | 伪装为 FA2 成功 |
| 接近 OOM | 先用 `MAX_STEPS` 的短 run；再评估降低 completion、时序 probe 或像素 | 改变协议，必须建新 run root 并标记为非严格对齐 | 覆盖原 run 或混合日志 |
| 数据解码失败 | 保留失败样本与日志；使用 `grpo_verify_video_decode.py` 的有界 Qwen preflight 重写 manifest | 新 manifest 是新数据契约，不能与旧 run 混合；checkpoint resume 必须继续使用旧原始 manifest | 把 path-verified 或 filtered manifest 宣称为原始全量 decode-verified |
| 速度不可接受 | 先测每 step、数据解码、保存开销；必要时降低 T probe 或使用更大已验证集群 | 结果与当前 full 不可直接混合 | 未经验证直接切集群 3 |
| 模型/数据/overlay 缺失 | 在集群 1 获取并经共享卷交付；集群 2 做离线 import/model gate | 延迟启动 | 在 GPU 端反复联网安装 |
| 需要严格上游 8-rank/FA2/G8 | 单独准备新 profile，先做环境、显存和一 step smoke | 不是当前 full run 的恢复参数 | 修改正在跑的 run |
| 保活恢复失败 | 仅在训练已确认停止后，使用原始保活 launcher 的人工恢复命令 | 需要在交接消息中记录 | 在 torchrun 仍运行时启动保活 |

## 7. 历史 H200 运行与远程核验方式（归档）

运行任务状态是动态信息，不能依赖聊天历史。下一位执行者首先进行只读核验：

```text
1. 查看 <run-root>/terminal.log 的最后 heartbeat。
2. 查看 <run-root>/training.log 的最近 step、ETA、错误关键词。
3. 检查 GPU 显存、利用率和是否仍有 torchrun 进程。
4. 若训练存在，不要重启 launcher、不要拉起保活、不要切换 worktree。
5. 若训练结束，检查 training_complete.json、training_summary.md/json、gpu_metrics.jsonl 和 checkpoint。
```

截至本次交接，旧探索性 full 已失败结束，4 张 GPU 没有本项目训练进程；不要对旧 run 重启、覆盖或在其目录内写入新文件。它没有启动时自动写入 `source_provenance.txt`，且不能与修复后的 baseline 混作结论性对比。checkpoint-500→501 的恢复 smoke 已通过；下一次操作应是新的 checkpoint-500 诊断 run 跨越旧 step-529，以验证原故障是否复发并采集 rank 证据，终端应确认 homogeneous 数据使用父类 `RandomSampler`、restricted safe-global resume 和 `synced_gpus=True`；诊断通过后才启动从 step 0 开始的 decoder-verified full。后续 launcher 会在训练前写入 `source_provenance.txt`（HEAD、dirty 状态/差异 hash、关键源码 SHA-256、运行时版本）。

## 8. 推送前最终核对表

| 检查 | 要求 |
| --- | --- |
| 文档匿名化 | 没有真实机器名、账号、IP、私有绝对路径、密钥、token、保活 PID |
| 分支隔离 | 当前分支不是协作者分支；名称与交付目的对应 |
| Git 内容 | 只包含源码、测试、脚本、文档；`artifacts/`、模型、数据、环境 overlay 被忽略 |
| 静态验证 | `bash -n`、Python `py_compile`、`git diff --check` 通过 |
| 实测证据 | baseline/KTR smoke 与 full 同形状容量 smoke 的摘要保留在本地 artifact；不把 artifact 大文件提交 |
| 对比有效性 | 修复 length-control 后重新跑同 commit 的 baseline/KTR；历史对比只作链路/资源证据 |
| 训练安全 | push 不影响运行中的训练；不触碰训练/保活进程 |
| 交接消息 | 按第 0 节模板包含分支、commit、运行状态、证据、风险和下一步 |

## 9. 历史 H200 复现结论（归档）

1. 历史记录曾称 4 个 weight shards 与官方 LFS SHA-256 元数据匹配，但缺少当前机器可读的 manifest/output，不能作为集群 3 的模型身份结论。active B200 full 必须使用可信 exact-revision manifest 覆盖所有顶层常规文件，并由 runtime gate 重新核验后才可启动。
2. 当前实现已真实验证 E/V/T 选择、union policy+KL、optimizer 更新、资源遥测与 CoT token 证据；修复后需重跑 paired smoke 才能给出公平的 baseline/KTR 数值比较。
3. 旧探索性 full 已在约 step 529 因 rank 间 collective timeout 失败；最高置信推断是 rank 0 的训练前视频预处理 stall，而不是 OOM。checkpoint-500→501 的恢复已在 scoped safe-global 修复后通过，新增 decoder isolation、rank trace 和 NCCL flight recorder 后，仍须由新的短恢复跨越 step 529 实际回归验证。
4. 后续正式 full 会有源码/数据 manifest provenance，且它**不是**上游 8-rank、G=8、16k/768、401k pixels、FlashAttention2 launcher 的严格逐参数复现。要求严格上游对齐时，必须另做 8-rank/FA2/G8 profile 的容量和兼容性 smoke。

## 10. 集群 3 B200 active handoff（覆盖上方历史 H200 运行结论）

本节是当前唯一可启动的 full profile。详细说明、token 例子和 artifact 树见 [B200_REPRODUCTION.md](B200_REPRODUCTION.md)；上方第 3–9 节的 H200 参数、SDPA、G=4、5 次置换和 checkpoint-500 诊断均为归档，不能复制到 B200 命令中。

### 已验证状态

| 项目 | 已验证事实 | 证据边界 |
| --- | --- | --- |
| 硬件与 runtime | 集群 3，8×B200，隔离 Python 3.12 venv，CUDA 13.1；FA2 binary 含 `sm_100` | 最终 smoke 真实执行 FA2/Qwen rotary 前反向；full epoch 未运行 |
| 模型 | Qwen2.5-VL-7B-COT-SFT；checked-in exact-revision manifest 覆盖 config/index/weight shard/tokenizer/processor 等全部顶层常规文件，runtime gate 逐项重算 | gate 是 full 的必需契约；gate artifact 未生成前，不宣称当前机器独立 hash 已通过 |
| paired smoke | 8 条固定**视频**，baseline/KTR 各 1 optimizer step、8 ranks；外层 94.840 / 95.008 s；跨卡最大显存 47,858 / 45,870 MiB | 单步 video V/T 容量/链路证据；不是 image 或 mixed-modality 分布式 sampler 覆盖，更不是完整 epoch/收敛结论 |
| selector | `paper/per_completion`、E/V 对每条 completion 精确 top-20%；视频 completion 的 T 也精确 top-20%；absolute delta、每 rank/step 一个非恒等置换、不含 reverse | image completion 的 T 是设计上全零 mask（无帧序）；不等于全局所有 rank 共用同一置换 |
| 媒体 token | `image_token_id != video_token_id`，visual positions 取两者并集 | ID 分离由启动 preflight 覆盖；真实 mixed sampler 训练覆盖留给 full |
| Qwen FA2 兼容 | 仅在 pinned legacy Transformers + FA2 下启用上游等价 rotary dtype shim | 解析 attention 非 FA2 或 shim gate 失败即拒绝，不降到 SDPA |

最终 smoke 记录的 CoT token 数为 E=341、V=592、T=472、union records=768（类别可重叠）；例子包括 E 的 ` step`/` First`、V 的 ` step`/`'ll`、T 的 ` by`/` First`。它们由分数归因选中，不是人工词性分类。

### B200 实现、优化与妥协

| 类别 | 最终实现 | 明确边界 |
| --- | --- | --- |
| 上游参数 | 8 ranks、prompt=16384、completion=768、G=8、FA2、1 epoch、save step=100 | `save_total_limit=2` 是持久存储折中，记录在 `launch_config.txt` |
| mixed data | `grpo_prepare_dataset.py --data-type all` 后由 `grpo_verify_media_decode.py` 在 CUDA-hidden spawned CPU worker 解码 image/video，硬超时、rejections、吞吐/ETA | full 前预检可耗时；worker crash/0 条记录 fail-fast |
| data strictness | `grpo_validate_b200_dataset.py` 固定核验 Holmes-16k 模态计数 | standalone 120 秒 gate 曾为 16916/16916；首次 strict full 在 120 秒预检中只有 16914/16916、2 timeout，data-class fail-closed；修复后每次 full 仍须 16916/16916、0 rejection |
| smoke lineage | `grpo_validate_b200_smoke.py` 检查成功 marker、两个 completion、profile、clean Git、smoke ancestor 和关键训练源码 SHA | launcher/文档后续提交可存在；训练关键源码变化必须重跑 smoke |
| 保活 | 外层 `run_full_b200.sh` 使用官方 controller；CPU preflight 时保活运行，只有 GPU preflight/torchrun 时暂停 | cleanup 先停本任务、确认 GPU 空闲，再恢复且确保唯一实例；不使用 `pkill` |
| 可观测性 | terminal.log、GPU JSONL、heartbeat、torchrun logs、source/data/smoke provenance | full 必须实际结束才可声称 `training_complete.json`/summary 成功 |

`--max_pixels 401408` 保持为原始 launcher 的 CLI 请求值，但 vendored Qwen 文件型视频预处理的实际单帧上限约为 105369；最终 smoke 的观察值是 94080–98784。所有 B200 full artifact 都写入三者，不能声称实际逐帧使用 401408。若需要实际 401408，必须建独立 `patched-intent` profile、修改处理代码并重跑 smoke。

### 数据门禁与手动命令

当前数据状态：16,916 条 source（8,765 image + 8,151 video）；233 个缺失训练视频已补齐。standalone strict mixed decoder 在 120 秒阈值下曾通过 16916/16916；首次 strict KTR full 的同源、同参数重跑有两条长视频引用超时，`verified=16914`、`rejected=2`、`timed_out=2`，data-class gate exit=1。修复后的 CPU 单条预检上限为 600 秒，命令行值会覆盖本地配置并写入 `launch_config.txt`；该新上限的 full gate 尚待操作者运行验证。历史 15,365 条 run 保留 reduced 标签。

### 补齐 strict Holmes-16k 的匿名步骤

1. 以原始 16,916 条 source manifest 为唯一清单，从已获授权的数据源或共享交付补齐 233 个缺失的**相对媒体路径**；保留原目录结构，不改写标注，也不把媒体落到临时目录。
2. 在 `<共享持久卷>` 中重新执行 B200 launcher 的 path gate，核对总数 `16,916`、图像 `8,765`、视频 `8,151`，并保存新的 path manifest 与 SHA-256。
3. 以当前训练同一 Qwen/torchvision 配置做 mixed image/video decode preflight（`nframes=8`、CLI `max_pixels=401408`），保留 accepted dataset、manifest 和全部 rejection 证据。
4. 只有总数完整且 decode `0 rejection` 时，才可不带任何 reduced/filter opt-in 启动 strict full；否则继续使用明确标记的 reduced/filtered 路径。补齐、gate 与结论必须按第 0 节发一条新的 `KT-HANDOFF/v1` 记录。

```bash
# strict：本次 full 仍须 16,916 条全部通过 mixed decoder 后才进入 GPU
cd <REPO_ROOT>
B200_ALLOW_PAUSE_KEEPALIVE=1 \
  B200_ALLOW_REDUCED_HOLMES=0 \
  B200_ALLOW_DECODER_FILTERED=0 \
  B200_MEDIA_DECODE_TIMEOUT_SECONDS=600 \
  VARIANT=ktr \
  RUN_ROOT=<共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-<UTC> \
  ./run_full_b200.sh

# 历史 / 诊断 reduced：明确写入 reproduction_class=reduced-media-subset；不是当前 strict 路径
B200_ALLOW_PAUSE_KEEPALIVE=1 \
  B200_ALLOW_REDUCED_HOLMES=1 \
  VARIANT=ktr \
  RUN_ROOT=<共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-reduced-<UTC> \
  ./run_full_b200.sh
```

资源对照要用同一 clean commit、同一 strict data gate 和同一 `B200_MEDIA_DECODE_TIMEOUT_SECONDS=600` 串行再跑 baseline；不要同时占卡。full 的 GPU 阶段成功、失败、TERM/HUP 都会自动尝试恢复保活。仅当自动恢复失败且已确认 torchrun/rank 进程完全退出时，才使用：

```bash
KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start
```

当前 `RUNNING` 状态：首次 strict KTR full wrapper 已在 CPU-only decoder data-class gate exit=1（2 条记录、同一物理视频的 120 秒 timeout），未启动 `torchrun`；standalone strict 16,916 条的历史 PASS 不能覆盖此次失败。已验证外部保活主进程为唯一实例；尚无本项目 B200 `torchrun` / GPU full epoch。修复后的 full 仍会在 GPU 前重跑 gate，并仅由官方 controller 暂停/恢复保活。

### 最新 active 状态记录（可直接续写）

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-001
TIME_UTC: 2026-09-22T05:28:43Z
FROM / TO: AI / 远程协作者
TYPE: HANDOFF
STATUS: PASSED
STATE_CHANGE: 集群 3 最终 video-only B200 paired smoke 已通过；full 仍未启动。
SCOPE: 集群 3 的 B200 环境、smoke 证据和 strict/reduced 数据门禁
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=<当前已部署提交>；run=<共享持久卷>/video-ktr-b200/artifacts/grpo-smoke-b200/<UTC>
CLAIM: 8×B200 的 FA2、Qwen rotary、视频 E/V/T/union 与一次真实优化步已验证；不能据此宣称 mixed sampler 或 full epoch 已验证。
EVIDENCE: somke-b200.sh exit=0；baseline/KTR 外层 94.840/95.008 s；跨已监控 GPU 最大显存 47858/45870 MiB；training process=0；保活恢复为唯一实例。
CHANGES: 本分支已准备 B200 环境/launcher/文档；本记录不代表启动或部署 full。
EXECUTED: 最终 somke-b200.sh；未执行 run_full_b200.sh。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/grpo-smoke-b200/<UTC>/
RISK / ROLLBACK: 当前仅 15365/16916，默认 strict gate 必须拒绝；显式 reduced 才可运行且不得称严格 Holmes-16k。遇到保活恢复失败，先确认全部 rank 退出。
LAST_VERIFIED: smoke exit=0；FA2 解析 gate、token report、资源对比、训练进程清理和保活唯一实例检查均通过。
RUNNING: 无 B200 full training；保活应在运行；下次检查条件=操作者决定 strict 或 reduced 手动启动前。
RECOVERY: KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start（仅自动恢复失败且确认 torchrun/rank 已退出时）。
FIRST_COMMAND: cd <REPO_ROOT> && git status --short
NEXT: 操作者补齐媒体后复验 strict gate，或显式选择 reduced 路径手动启动单个变体；baseline/KTR 必须串行。
ASK: NONE
```

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-002
TIME_UTC: 2026-09-22T05:44:00Z
FROM / TO: AI / 远程协作者
TYPE: VERIFICATION
STATUS: PASSED
STATE_CHANGE: 集群 3 已部署模型/环境完整性门禁，并完成一次 CPU-only 实测；full 仍未启动。
SCOPE: 集群 3 checkpoint、隔离运行时与保活边界
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=bd87d79；GPU task=none
CLAIM: checked-in exact-revision manifest 与本地 checkpoint 的 19 个顶层常规文件、4 个 indexed weight shard 一致；Python/Torch/torchvision/PyAV/DeepSpeed 与环境 manifest 一致。
EVIDENCE: runtime gate exit=0；files=19；shards=4；CUDA_VISIBLE_DEVICES 为空；训练 process=0；保活唯一实例=1。
CHANGES: full 在长 CPU data preflight 前后均会运行 smoke/source 与 model/environment gate；未启动 torchrun。
EXECUTED: CPU-only runtime gate；未执行 somke-b200.sh 或 run_full_b200.sh。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/b200-runtime-integrity-<UTC>/runtime_gate.json
RISK / ROLLBACK: 当前数据仍为 15365/16916；strict full 仍须补齐媒体。若后续 gate 失败，不暂停保活、不启动训练，保留失败 artifact。
LAST_VERIFIED: runtime gate exit=0；commit=bd87d79；worktree clean。
RUNNING: 无 B200 training；保活运行且唯一；下次检查=操作者手动启动前。
RECOVERY: 无需操作；仅自动恢复失败且确认 torchrun/rank 已退出时才使用 `KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start`。
FIRST_COMMAND: cd <REPO_ROOT> && git status --short && git rev-parse --short HEAD
NEXT: 补齐媒体走 strict，或显式选择 reduced 后手动串行运行 KTR / baseline。
ASK: NONE
```

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-003
TIME_UTC: 2026-09-22T06:36:36Z
FROM / TO: AI / 远程协作者
TYPE: UPDATE
STATUS: RUNNING
STATE_CHANGE: 集群 3 已补齐 Holmes-16k 的 233 个缺失训练视频；strict path gate 已通过，strict decoder gate 待当前历史 reduced run 退出后执行。
SCOPE: 集群 3 Holmes-16k 媒体补齐、strict path 契约与正式 paired full 前置条件
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=ec02889；run=<共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>
CLAIM: 16,916 条的路径完整性已验证，但尚无全量 mixed decoder `rejected=0` 证据，不能启动或宣称 strict full。
EVIDENCE: transfer=233/233 non-empty、bytes=1892102611；path gate exit=0、class=strict-holmes-16k、available=16916/16916（8765 image、8151 video）。
CHANGES: 更新 B200 数据状态、strict 启动前置条件和本交接记录；未提交媒体、模型、artifact 或凭据。
EXECUTED: 传输 233 个已核对文件名的训练视频；在 CUDA-hidden 条件下执行 path verification 与 strict path data-class gate；未启动 strict KTR 或 baseline。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>/Holmes-16k-path-verified.json(.manifest.json)；data_path_gate.json；path_prepare.log；path_gate.log。
RISK / ROLLBACK: 当前 historical reduced run 的 manifest 已冻结为 15365 条，保留其 reduced 结论；全量 decoder 任一 rejection 都会阻止 strict 声明。KTR 与 baseline 不可并发。
LAST_VERIFIED: strict path gate exit=0；strict_holmes_16k=true；missing_records=0；外部保活未被本次 CPU-only 检查暂停。
RUNNING: historical reduced KTR 的 CPU-only decoder preflight；最近进度以该 run 的 terminal.log 为准；当前已出现 timeout，预期将 fail-closed，训练未开始。
RECOVERY: KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start（仅自动恢复失败且已确认全部 torchrun/rank 退出时）。
FIRST_COMMAND: tail -n 40 <共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-reduced-<UTC>/terminal.log
NEXT: AI 在 historical reduced run 退出后执行一次完整 strict mixed decode；仅 `verified=16916,rejected=0,timed_out=0` 时由操作者手动按 KTR→baseline 串行启动。
ASK: NONE
```

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-004
TIME_UTC: 2026-09-22T07:25:41Z
FROM / TO: AI / 远程协作者
TYPE: UPDATE
STATUS: RUNNING
STATE_CHANGE: historical reduced wrapper 已在 decoder data-class gate fail-closed；当前已启动新的 strict 16,916 条 canonical decoder gate，尚未启动 GPU 训练。
SCOPE: 集群 3 strict mixed-media decoder、历史 reduced failure 证据与正式 full 前置条件
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=ec02889；run=<共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>
CLAIM: 当前 strict decoder 已处理 2500/16916 且零 rejection；完成后仍须由 data-class validator 生成 `data_gate.json` 并 exit=0，才可 strict-ready。
EVIDENCE: historical wrapper exit=1、`torchrun`=0、decoder verified=15363/rejected=2/timed_out=2；strict decoder canonical 参数为 torchvision、nframes=8、CLI max_pixels=401408、workers=8、timeout=120，当前 2500/16916、rejected=0。
CHANGES: 更新 B200/开发/README/说明/交接文档的实际运行状态；未改训练源码、媒体、模型、保活或凭据。
EXECUTED: 对历史 timeout 的同一媒体做无竞争单 worker 诊断（120 秒失败、600 秒 112 秒通过）；启动 CUDA-hidden strict decoder；未启动 strict KTR 或 baseline。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-reduced-<UTC>/Holmes-16k-media-decoder-verified.json(.manifest.json/.rejections.json)；<共享持久卷>/video-ktr-b200/artifacts/decoder-diagnosis-<UTC>/；<共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>/decoder_terminal.log。
RISK / ROLLBACK: 历史 timeout 的媒体可在 112 秒单 worker 解码，但仍接近 120 秒 canonical 上限；strict full 只能以全量 canonical decoder `verified=16916,rejected=0,timed_out=0` 和随后的 data gate exit=0 为准。不得设 filtered opt-in；KTR 与 baseline 不可并发。
LAST_VERIFIED: transfer=233/233 non-empty、strict path gate exit=0、available=16916/16916；历史 wrapper fail-closed 且保活未暂停。
RUNNING: strict mixed decoder，CPU-only；2500/16916、rejected=0；下次检查=终态 decoder manifest 与 data gate。
RECOVERY: NONE；当前非 GPU 训练，保活不得手工停止。仅自动恢复失败且确认全部 torchrun/rank 退出时才使用官方 launcher。
FIRST_COMMAND: tail -n 40 <共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>/decoder_terminal.log
NEXT: AI 等待 decoder 完成后运行 strict data-class validator；仅 gate exit=0 时由操作者依次手动启动 strict KTR、待其成功结束后再启动 strict baseline，并生成对比 artifact。
ASK: NONE
```

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-005
TIME_UTC: 2026-09-22T08:27:03Z
FROM / TO: AI / 远程协作者
TYPE: UPDATE
STATUS: PASSED
STATE_CHANGE: 233 个缺失训练视频已补齐；standalone strict mixed decoder 与 strict data-class gate 均通过，正式 paired full 现在具备数据前置条件。
SCOPE: 集群 3 Holmes-16k 媒体补齐、strict 数据 gate、KTR→baseline 手动启动前置条件
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；commit=ec02889（运行源码；本记录所在文档提交另见 Git log）；run=<共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>
CLAIM: 完整 Holmes-16k 已通过 canonical CPU-only mixed decoder 与 data-class gate，类别为 strict-holmes-16k；尚未启动本项目 GPU full。
EVIDENCE: transfer=233/233 non-empty、bytes=1892102611；path gate exit=0、available=16916/16916（8765 image、8151 video）；decoder exit=0、status=passed、verification_complete=true、verified=16916、rejected=0、timed_out=0；decoder=torchvision、CUDA-hidden、nframes=8、CLI max_pixels=401408；data-class gate exit=0、strict_holmes_16k=true、reproduction_class=strict-holmes-16k、decoder_filtered=false。
CHANGES: 更新 B200/开发/README/说明/交接文档的 strict 终态、训练顺序和证据；未提交媒体、模型、artifact、保活配置或凭据。
EXECUTED: 传输并路径核验 233 个训练视频；执行 CUDA-hidden standalone mixed decoder 和 strict data-class validator；未启动 KTR、baseline、torchrun 或 GPU full。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>/Holmes-16k-path-verified.json(.manifest.json)；Holmes-16k-decoder-verified.json(.manifest.json/.rejections.json)；data_path_gate.json；data_gate.json；decoder_terminal.log；data_gate.log。
RISK / ROLLBACK: full 每次仍会在 GPU 前重新执行 path/decode gate；任一新的 rejection/timeout 都会 fail-closed，不得设置 reduced/filter opt-in 来冒充 strict。历史 15365 条 reduced run 与其失败证据保持冻结。KTR 与 baseline 不可并发。
LAST_VERIFIED: data-class gate exit=0；status=passed；class=strict-holmes-16k；完整模态计数=8765 image + 8151 video；外部保活主进程=1；本项目 torchrun=0。
RUNNING: 外部保活为唯一已验证实例；无本项目 decoder、torchrun 或 GPU full。下次检查=操作者手动启动 KTR 后的 terminal.log/heartbeat。
RECOVERY: 不需要手工停止保活；正式 full 仅使用官方 controller 管理暂停与恢复。仅自动恢复失败且确认全部 torchrun/rank 已退出时使用 `KEEP_ALIVE_DASHBOARD=0 bash <KEEPALIVE_LAUNCHER> start`。
FIRST_COMMAND: tail -n 40 <共享持久卷>/video-ktr-b200/artifacts/strict-data-gate-<UTC>/data_gate.log
NEXT: 操作者先手动启动 strict KTR；仅在其 exit=0、`training_complete.json` 与保活恢复均通过后，再以独立新 run-root 启动 strict baseline；最后运行对比脚本并要求通过。
ASK: NONE
```

```text
[KT-HANDOFF/v1]
ID: 20260922-AI-006
TIME_UTC: 2026-09-22T10:21:19Z
FROM / TO: AI / 远程协作者
TYPE: UPDATE
STATUS: PASSED
STATE_CHANGE: 首次 strict KTR full 在 120 秒 CPU 媒体门禁失败；已修复启动脚本的预检超时默认值、命令行覆盖优先级和配置记录。PASSED 仅指修复检查，尚未重跑 full。
SCOPE: 集群 3 strict KTR 预检失败诊断、600 秒超时策略与手动重试条件
ENV: 集群 3；branch=<owner>/video-ktr-repro-handoff；失败运行源码 commit=03fcc4c；修复提交=本记录所在 Git commit；run=grpo-full-b200/ktr-strict-20260922T083532Z
CLAIM: 失败由同一长视频的两条记录超过 120 秒预检阈值引起；strict data-class gate 正确拒绝，未进入 GPU 训练。修复后的 600 秒策略仍须在新 full run 中验证 16916/16916。
EVIDENCE: standalone 120 秒 gate=16916/16916、0 timeout；失败 full 同一 source SHA、8 workers、torchvision、CUDA-hidden、nframes=8、CLI max_pixels=401408，decoder=16914/16916、rejected=2、timed_out=2；两条记录均为同一视频，source indices=2166/5549；data-class exit=1、torchrun=0；此前单视频 600 秒诊断耗时 112 秒并通过；launcher override 测试 7/7 通过，`bash -n`、`git diff --check` 通过。
CHANGES: `run_full_b200.sh` 默认单条 CPU 预检超时设为 600 秒，保留命令行 `B200_MEDIA_DECODE_TIMEOUT_SECONDS` 对本地配置的优先级，终端和 `launch_config.txt` 记录实际超时与 workers；补充回归测试和本次交接文档。未改媒体、模型或训练目标。
EXECUTED: 只读核对失败 run 的 terminal/decoder/rejections、与 standalone manifest 对比、保活与训练进程检查；本地脚本语法和超时覆盖测试。未重启训练或操作保活。
ARTIFACTS: <共享持久卷>/video-ktr-b200/artifacts/grpo-full-b200/ktr-strict-20260922T083532Z/terminal.log；Holmes-16k-media-decoder-verified.json.manifest.json；Holmes-16k-media-decoder-verified.json.rejections.json；此前 standalone strict-data-gate-<UTC>/data_gate.json。
RISK / ROLLBACK: 600 秒是 CPU 预检容错上限，不是 120 秒配置的再次通过证据；重试若仍有任一 rejection，则继续 fail-closed。不得用 reduced/filter opt-in 把失败数据称为 strict。KTR 与 baseline 必须使用同一 600 秒策略串行执行。
LAST_VERIFIED: 失败 wrapper exit=1 且未启动 torchrun；外部保活主进程唯一；本地 launcher 覆盖测试通过。
RUNNING: 无本项目 full wrapper 或 torchrun；外部保活运行。下次检查=操作者手动重启 KTR 后的 decoder/data-class gate。
RECOVERY: 当前无需手工操作保活；仅正式训练结束后自动恢复失败、且全部 rank 退出时，才通过官方 controller 恢复。
FIRST_COMMAND: cd <REPO_ROOT> && git status --short && git rev-parse --short HEAD
NEXT: 操作者在已部署的干净修复提交上，以新 run root、`B200_MEDIA_DECODE_TIMEOUT_SECONDS=600` 和两个降级开关为 0 手动重启 strict KTR；KTR 成功并恢复保活后，用相同设置启动 baseline。
ASK: NONE
```
