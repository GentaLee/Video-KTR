# H200 状态（本分支唯一事实源）

PROFILE: h200
STATUS: KTR_COMPLETED
LAST_VERIFIED_UTC: 2026-09-24T09:45:00Z

## 最新全量结果（覆盖下方历史待启动状态）

- run `ktr-20260923T054132Z-2458824` 正常完成 2115/2115，exit=0；北京时间 2026-09-24 13:37:52 外层完成，无恢复/补算。
- 最终四片模型、checkpoint-2100 和 checkpoint-2115 保留原位；归档与校验见 [RESULTS.md](../reports/h200-ktr-full-20260924/RESULTS.md)。外层训练窗口 86065.144 秒，峰值 111489 MiB，Trainer train_loss 0.005958747757303179。
- 训练结束当时保活已恢复；本次实查保活不在运行，已调用原控制器重新启动并确认 GPU compute，无训练重启。
- B200 尾批错误未在本轮触发；不修改已完成运行训练源码。baseline 未自动启动，独立模型质量评测待做。
- 归档开发分支 `archive-20260924/video-ktr-h200`，运行仍按原始 `8a2e3cb` + dirty/untracked provenance 标识。只提交结果/状态及归档工具，其他既有未提交改动保留。

## 工作区责任

本工作区只维护集群 2 / H200；开发分支为 `<owner>/video-ktr-h200`。本次拆分只新增协作文件并从同一 commit 创建分支，保留全部既有未提交改动，不更改训练源码或进程。开发 HEAD 不是正在运行实验的源码身份。

## 历史运行与本次授权停止

- run：`paired-step0-ktr-20260922T103931Z`，相对产物根 `artifacts/grpo-full/`。
- 启动 provenance：`10bff9c` + dirty worktree；既有 provenance 保存了 diff SHA 与源码哈希，但不能仅凭 diff SHA 重建全部历史内容。不得将当前开发 HEAD 冒充其运行版本。
- 配置：4×H200；Video-R1 的 116,248 条核验视频；4096/512/G4；SDPA；8 帧；5 次时序置换（含 reverse）；每 500 步 checkpoint。
- 用户于本次任务明确授权停止旧训练并调优配置。2026-09-23 04:57:44 UTC 向已核对身份的 launcher 发送 TERM，torchrun/rank 正常退出；04:58:04 UTC 唯一保活恢复。末条 rank0 trace 为 global_step=351 的 generate_start；旧摘要 exit=130 是主动停止，不是新一次训练故障。旧产物保留，不续接到新数据协议。
- 该 run 的 baseline 尚未启动。与 B200 配置不同，不能跨集群直接作 KTR/baseline 对照。

## H200 对齐验收通过（正式训练待用户手动启动）

- 新入口：`bash run_ktr.sh`、结束后 `bash run_baseline.sh`，共用 `run_full_h200_aligned.sh`；历史 `run_full.sh` 不改作新实验入口。完整配置与验收说明见 [H200_ALIGNED_RUN.md](../H200_ALIGNED_RUN.md)。
- 对齐 Holmes 16,916 条（8,765 image / 8,151 video）、16384/768/G8、FA2、8 帧、单次非恒等时序置换、per-completion top20%、绝对 delta、checkpoint100、精确媒体缓存及 CPU 预取。
- H200 保留 4 卡，用 batch1/rank × gradacc2 匹配名义 8 prompts/update；不是 B200 8-rank 逐位等价轨迹。Torch/CUDA/FA 二进制按本机 ABI 独立校验。
- 本地 233 个缺失的 `videos_train/` 路径已从既有 Holmes 媒体补齐，未改标注或过滤数据；完整路径 gate 为 16,916/16,916。
- FA2 2.8.3 的 cu12/torch2.10/cp312/ABI-TRUE 社区 wheel 已核对 SHA256，四张 H200 的 FA2 + Qwen rotary 前反向通过；两种真实训练日志均确认实际使用FA2。
- 验收根：`artifacts/h200-aligned-runtime/artifacts/validation/paired-20260923T0521Z/`。KTR/baseline各7步通过，loss与梯度有限，最终模型及4-rank optimizer/model/RNG/scheduler checkpoint保存通过；paired gate PASS。完整epoch和新checkpoint恢复尚未验收。
- KTR外层361.148秒 / Trainer293.0245秒 / 峰值94,838MiB；baseline外层350.001秒 / Trainer285.7445秒 / 峰值84,148MiB。是短跑容量/功能数据，不是最终训练效果或稳定吞吐结论。
- 全数据缓存：16,916/16,916，0拒绝/超时；热缓存媒体gate约27秒；3条真实媒体缓存精确一致性通过。155项单元测试通过。
- 运行身份：开发HEAD `8a2e3cb` + dirty/untracked源码；关键文件哈希写入runtime gate，identity=`0e1a3921cf917893928f5ee1f15ba61cd6b29a30c9f2f8ebcf5b731830fe7dac`。关键源码快照保存在验收根；未自动commit/push。
- 最后检查：所有验收训练进程已退出，保活恢复且唯一。首次验收的参数拼写错误已修复并加实际参数解析门禁，失败产物保留，不计入成功结果。
- 机器私有配置 `.h200-aligned.env`、依赖、缓存和验证日志位于被忽略的持久路径，不提交 Git。B200 工作区与正在运行的实验均未操作。

## 边界与下一步

训练目录与集群 1 CPU 共享；旧训练停止后才新增启动源码。新验证运行时不覆盖关键源码，不切换 commit，不从 B200 整支合并。历史未提交改动原样保留，不能用 `git add .` 一并上传。

结束后整理本 run 的资源摘要、checkpoint 身份、评测与 token 示例到 `reports/<run-id>/`，脱敏后提交本分支；更新本文并 push，由 B200 负责者导入快照。

RECOVERY: 本轮 KTR 已正常完成，无需恢复或重跑。保活已重新启动。baseline 待用户另行启动；节点锁拒绝并发。历史运行产物保持原位，不用归档分支 HEAD 替换运行身份。
