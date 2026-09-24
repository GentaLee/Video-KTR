# 集群 3：B200 尾批故障与 checkpoint 恢复

> 目录归档后，当前启动以 [OPERATIONS.md](OPERATIONS.md) 及仓库根`run.sh`为准。本页的86317d0/b532e4c命令记录恢复当时的固定release，不作为最新交付入口。

## 故障与修复边界

原 run `ktr-20260923T031639Z-2981461` 在 2114/2115 后、最后一次 `compute_loss` 的日志指标聚合处失败：`shape '[-1, 8]' is invalid for input of size 4`。不是缺媒体或 CUDA OOM。原外层训练耗时 52,760.078 秒；退出后保活已自动恢复。

Holmes 有 16,916 条，混合模态 sampler 显式补齐 4 条后输出 16,920 条。Accelerate 的 `gather_for_metrics` 在末批按 prompt remainder=4 截断 **completion** 级 reward，再按 G=8 分组时失败。

修复提交 `86317d0` 将 completion 级训练统计改用不截断的 `accelerator.gather`，同时修复 `_distributed_mean`。KTR 和 baseline 共用此 trainer。没有改变训练 loss、E/V/T 选择、数据、采样顺序或超参数；指标口径包含 sampler 补齐项，不能声称是去重评测集指标。不是通过 drop_last、删样本或降低 G 绕过故障。

## 恢复协议

- 从原 run 的 `training/checkpoint-2100` 恢复；原输出保持不动，结果写新 run。
- 8 卡、原始 base/reference 模型不变；通过 `--resume_from_checkpoint` 恢复 policy/optimizer/scheduler/RNG/Trainer step。不是将 checkpoint 当作新的 base 模型。
- `B200_RESUME_FROM_CHECKPOINT` 为显式选项，默认新训练；入口在暂停保活前核对实验类型、完整数据字节哈希、2115 总步数及 8 卡状态分片。门禁为结构检查，实际恢复必须以训练日志为证。
- 保持完整 epoch 配置，不把 `max_steps` 设置为 15。2100→2115 才是本次 15 个 optimizer steps；其中 2101–2114 是故障后重算。
- 新版本发布到 `releases/<commit>/repo`，旧源码不覆盖。H200 分支与其运行进程未修改。
- 原数据 SHA256：`19025cd0a08fb6b01b495668867f6888d79336caa23c2fe379652cb16af46a1f`。

## 验证记录

- 新增 3 个尾批统计/入口测试、5 个恢复门禁测试，全部通过。
- 8 个 full wrapper、12 个保活生命周期、14 个数据/runtime/smoke 门禁测试通过（共 42 个）。
- 集群 3 新 paired smoke：`artifacts/grpo-smoke-b200/tail-fix-86317d0`；最终结果见下方完成记录。
- 冒烟/全量均使用既有节点锁和官方保活 controller：CPU 准备时保活，GPU 工作前停止，正常或异常退出后清理本任务 GPU 进程再恢复。
- 新 paired smoke 两者均 exit=0；baseline 外层 119.209 秒、峰值 47,859 MiB；KTR 外层 120.346 秒、峰值 56,269 MiB。仅为单步功能/容量验证，不是正式训练性能结论。
- 首次 release 续跑 `ktr-20260924T031422Z-386968` 在 GPU 工作前被环境门禁拒绝：旧 manifest 的 `repo_root` 仍指向旧目录。重新执行已有 `setup_b200_env.sh` 的只读导入/版本/来源校验和 manifest 生成段，为 release 创建独立 `environment-manifest.json`；未重装包、未修改旧 manifest 或 venv `.pth`。新源路径通过显式 `PYTHONPATH` 验证，运行命令显式设置 `B200_ENV_MANIFEST`。
- 正式恢复 run：`ktr-20260924T031544Z-389593`；checkpoint 门禁、全量缓存 16,916/16,916、19 个模型文件及 4 个权重 shard 的哈希复核通过。
- **完成：2100→2115，恰好15步，exit=0；最终模型和checkpoint-2115已保存，唯一保活已恢复。** 详细资源/数值/权重哈希见 [结果报告](../reports/b200-ktr-full-20260924/RESULTS.md)。新增baseline快捷入口测试后，相关测试共43项通过。

## baseline 启动原则

采用同一修复 release 和新版 paired smoke；清除 `B200_RESUME_FROM_CHECKPOINT`，从原始 SFT 模型开始，不能从 KTR checkpoint 开始。保持全 Holmes、2115 steps、8 卡、16384/768/G8/FA2，以及其他配对超参数，仅 `video_ktr=false`。正式 baseline 由用户手动启动。

后续交付的快捷入口为 `bash <已交付release>/repo/run_baseline_b200.sh`：自动解析项目根、release环境清单与已通过smoke，强制清空KTR恢复参数，再调用同一full启动器。该快捷入口只整理部署参数，没有改变训练关键源码；full仍强制验收smoke源哈希。

已交付并核验的baseline版本为 `b532e4c41d1b25abb12a6bab9c233f923c2f8fcd`，新环境manifest已生成，critical-source/smoke gate PASS；入口测试PASS，**没有启动正式baseline**。直接运行：

```bash
bash <项目持久根>/releases/b532e4c41d1b25abb12a6bab9c233f923c2f8fcd/repo/run_baseline_b200.sh
```

交付bundle SHA256：`002dd5f947c2e02d31e5a08ca3f816029fab6ff9b1af1cb8f65a11232774e3b2`。训练修复版本86317d0与交付版本b532e4c的所有critical training sources一致。后续纯文档提交不替换这一固定运行release。

在集群 3 上执行（用实际持久根替换占位符）：

```bash
KTR_ROOT=<共享持久卷>/video-ktr-b200
KTR_RELEASE="$KTR_ROOT/releases/86317d00619a56507f66236caa01eba1ebb8b459"
B200_ROOT="$KTR_ROOT" B200_ENV_FILE="$KTR_ROOT/b200.env" \
B200_ENV_MANIFEST="$KTR_RELEASE/environment-manifest.json" \
B200_SMOKE_RUN_ROOT="$KTR_ROOT/artifacts/grpo-smoke-b200/tail-fix-86317d0" \
B200_RESUME_FROM_CHECKPOINT= \
bash "$KTR_RELEASE/repo/launch_b200.sh" baseline
```

命令自动管理保活；CPU 阶段保持保活，GPU 阶段暂停，结束/失败后恢复。终端持续显示日志，Ctrl-C 只关闭日志查看器，SSH 断开不停止 detached 训练。节点锁拒绝同时启动另一训练。不要改用旧目录的启动命令，旧目录保留作历史。

## 结果解释边界

恢复段 Trainer 的速度/耗时不是完整 epoch 的速度/耗时；资源报告须区分原失败尝试、恢复段和重复计算。训练 reward 不是独立验证集准确率。KTR 与 baseline 的效果比较仍需同一独立评测集，完整训练成功不能替代评测结论。
