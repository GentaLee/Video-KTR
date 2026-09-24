# B200 操作说明

## 开发端和运行端

开发端`video-ktr-b200/repo`负责代码与Git；集群3使用独立盘的`video-ktr-b200/releases/<commit>/repo`运行。两端通过同一commit的bundle显式交付，不共享文件夹，不用rsync覆盖运行目录。旧`repo`和旧release只作历史，不作为新命令入口。

运行端`<项目持久根>/current`是指向本次已验收release仓库的导航软链接。只在无训练运行时按显式交付切换；启动器解析真实路径，运行中不更换源码。可用`git -C <项目持久根>/current rev-parse HEAD`核对与开发/GitHub版本一致。

## 当前结果

2026-09-25整理：KTR与baseline均已完成2115步，模型保存成功、保活恢复，详见[结果索引](../reports/README.md)。下列命令用于明确授权的新训练，本轮无需重跑。报告更新不自动切换运行端current或冻结release。

## 正常操作

进入已交付release的repo后：

```bash
bash run.sh --help
bash run.sh baseline
```

任意工作目录均可使用 `bash <项目持久根>/current/run.sh baseline`。这是需要新跑baseline时的入口，不是查看结果的命令。

baseline强制清空KTR恢复参数、从原始SFT模型开始，完整Holmes/2115步。GPU运行前暂停已识别保活，结束/失败后确认GPU进程释放再恢复。终端持续显示进度；Ctrl-C仅关闭detached训练的查看器，不会停止正式训练。不要重复提交同一个任务；节点锁拒绝并发。

`bash run.sh smoke`是双模式短验证；`bash run.sh ktr`用于明确授权的新KTR任务。本轮KTR和baseline均已完成，无需重复。KTR显式恢复可用`B200_RESUME_FROM_CHECKPOINT=<完整checkpoint路径> bash run.sh ktr`，当前门禁仅支持同variant、相同数据及2115总步数的8卡完整epoch。

## 脚本分层

| 文件 | 用途 |
| --- | --- |
| [run.sh](../run.sh) | 唯一日常命令入口 |
| [scripts/run_baseline_b200.sh](../scripts/run_baseline_b200.sh) | fresh baseline配置与防误恢复 |
| [scripts/launch_b200.sh](../scripts/launch_b200.sh) | detached提交与实时日志查看 |
| [scripts/run_full_b200.sh](../scripts/run_full_b200.sh) | 数据/模型/smoke门禁、checkpoint、训练、保活 |
| [scripts/somke-b200.sh](../scripts/somke-b200.sh) | B200 paired smoke及其保活管理 |
| [scripts/smoke.sh](../scripts/smoke.sh) | paired smoke共用实现，非日常入口 |
| [scripts/setup_b200_env.sh](../scripts/setup_b200_env.sh) | 环境配置，禁止为看状态随意重跑安装 |

新release须有其父目录的独立`environment-manifest.json`，与repo路径匹配；该私有文件不进Git。没有则门禁拒绝，不修改旧manifest凑通过。训练代码未变时可通过critical-source hash复用已通过smoke；训练代码变化必须重新smoke。

## 停止与日志

日志在项目根`artifacts/launches/<variant>-<run>.log`，退出码为同名`.exit`；详细产物在`artifacts/grpo-full-b200/<run>/`。停止任务必须先核对run和wrapper PID，再向该wrapper发送TERM，由其清理自己拥有的训练进程组并恢复保活；不要使用全局pkill或kill -9。停止后确认`.exit`、GPU占用和唯一保活。

2026-09-24，用户启动的`baseline-20260924T033709Z-424904`按明确授权停止，exit=130，保活恢复，产物保留。这是主动停止，不是新的算法故障，也不是正式baseline完成。
