# B200 状态（本分支唯一事实源）

PROFILE: b200
STATUS: KTR_COMPLETED_BASELINE_READY
LAST_VERIFIED_UTC: 2026-09-24T03:28:00Z

## 工作区责任

本工作区只维护集群 3 / B200；联网开发分支为 `<owner>/video-ktr-b200`，与 H200 是独立 clone。运行集群的持久盘与开发端不同，Git 提交和 bundle 是交付边界，不存在共享目录自动更新。

## 当前结果（覆盖下方原运行历史）

- 原任务在2114/2115后的末批completion指标聚合失败；不是缺数据/OOM。
- 修复训练源码 `86317d00619a56507f66236caa01eba1ebb8b459`；新release不覆盖旧目录。KTR/baseline共享completion统计修复，训练loss和选择器不变。
- 新paired smoke `tail-fix-86317d0` 两者exit=0，保活恢复。43项相关测试通过（包含后续baseline快捷入口测试）。
- 从原checkpoint-2100恢复，run=`ktr-20260924T031544Z-389593`，**15步完成到2115，exit=0**；最终4-shard模型与checkpoint-2115的8卡optimizer/model/RNG/scheduler状态保存成功。训练进程已退出，唯一保活恢复。
- 恢复段外层502.566秒，峰值79,695MiB；原尝试52,760.078秒，峰值110,517MiB。前后窗口与重复计算不可混作完整成功epoch性能。
- baseline正式训练未启动；使用同修复release的`run_baseline_b200.sh`，从原始SFT开始，明确清空resume。该快捷入口经stub环境测试，不伪称正式baseline已完成。
- baseline交付release=`b532e4c41d1b25abb12a6bab9c233f923c2f8fcd`；独立environment manifest已生成，smoke/critical-source gate PASS，运行端Git干净。后续文档commit不改变固定运行release。
- 完整结果与模型哈希：[RESULTS.md](../reports/b200-ktr-full-20260924/RESULTS.md)；故障/命令：[B200_CHECKPOINT_RECOVERY.md](../B200_CHECKPOINT_RECOVERY.md)；踩坑：[B200_PITFALLS.md](../B200_PITFALLS.md)。

## 原运行历史

- run：`ktr-20260923T031639Z-2981461`，相对产物根 `artifacts/grpo-full-b200/`。
- 当前运行源码冻结在 `d967c23`、历史分支 `<owner>/video-ktr-repro-handoff`；新开发分支不追认或修改该 run 的身份。
- 配置：8×B200；Holmes-16k 全部 16,916 条（图像 8,765 / 视频 8,151）；16384/768/G8；FA2；8 帧；单次随机非恒等置换；per-completion top-20%、绝对 delta；每 100 步 checkpoint。
- 全数据缓存 0 拒绝/超时；8 卡 paired smoke、两种 7-step 回归、最终保存及保活故障恢复均已通过，见 B200_PIPELINE_VALIDATION.md。
- 最后检查：正式 KTR 152/2,115 步，训练日志及 GPU 运算持续推进；这不是实时进度。正式 baseline 尚未启动。
- 请求 max_pixels=401408；Qwen 原有有效视频上限约 105369，不应宣称逐帧实际401408。

## 边界与下一步

原运行端 `repo/` 保持不动；新版本通过bundle放到全新`releases/<commit>/repo/`。baseline保持同一实验定义与修复关键源码，核对critical-source/smoke gate，不夹入无关算法改动。H200分支及训练未修改；其同类尾批风险需该分支负责者单独处理。

full 不保存全量逐 token CoT 明细；已有 smoke token_examples；完整模型效果待同一独立评测集比较。训练完成后回传脱敏摘要和哈希到开发端 `reports/<run-id>/`，本分支显式 push，H200 再导入状态快照。

RECOVERY: KTR本轮已完成，无需重复启动。baseline由用户手动运行新版快捷入口，自动管理保活；发布版需有与路径匹配的environment-manifest.json。完整模型质量尚待baseline及共同独立评测，不能以训练reward代替评测。
