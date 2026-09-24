# Video-KTR · B200 工作区

本分支只负责集群3/B200。**KTR与baseline均完成2115步，最终模型已保存；独立质量评测待进行。** 实时事实以 [状态交接](handoff/STATUS.md) 为准。

## 从这里开始

| 想了解什么 | 入口 |
| --- | --- |
| 启动与目录职责 | [操作说明](docs/OPERATIONS.md) |
| 数据搬运、混合组批、开发计划 | [开发记录](docs/DEVELOPMENT.md) |
| 故障与解决办法 | [踩坑索引](docs/B200_PITFALLS.md) |
| checkpoint恢复证据 | [恢复记录](docs/B200_CHECKPOINT_RECOVERY.md) |
| KTR训练结果、耗时、显存、模型哈希 | [结果报告](reports/b200-ktr-full-20260924/RESULTS.md) |
| 两分支与跨盘协作 | [协作协议](docs/COLLABORATION.md) |
| 原始需求与历史材料 | [需求](describe.md) · [历史归档](docs/archive/README.md) |

配对结论见 [对比报告](reports/b200-paired-20260925/COMPARISON.md)，所有run见 [结果归档索引](reports/README.md)。

## 唯一日常入口

在B200运行端已交付版本的仓库目录执行：

```bash
bash run.sh baseline  # 仅明确需要重跑时使用；本轮baseline已完成
# bash run.sh smoke   # 双模式冒烟，不是正式baseline
# bash run.sh ktr     # 仅明确需要重新跑KTR时使用；本轮KTR已完成
```

不要在联网控制端启动GPU训练。旧release及产物保留，不覆盖；发布过程不自动启动训练。实际集群路径见聊天交付命令，不写入公共Git文档。

```text
run.sh              统一命令入口
scripts/            B200运行脚本及共享smoke实现
docs/               当前操作、开发、恢复、踩坑与协作说明
docs/archive/       历史记录与停用脚本文本
handoff/            本分支状态及对方只读快照
reports/            脱敏结果和哈希
src/ tests/ tools/  训练实现、测试与工具
```

上游介绍、论文链接、原始README保留于 [历史README](docs/archive/README_UPSTREAM.md)。
