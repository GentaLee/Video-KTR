# B200 结果与运行归档索引

整理日期：2026-09-25。采用逻辑归档：**原始目录原位保留，不移动/删除 checkpoint，不复制大模型进 Git**。

| run | 分类 | 用途 |
| --- | --- | --- |
| `ktr-20260923T031639Z-2981461` | KTR原失败尝试，2114步 | 原1–2100日志 + checkpoint2100；保留尾批故障证据 |
| `ktr-20260924T031544Z-389593` | KTR成功恢复15步，最终2115 | KTR最终模型及checkpoint2115 |
| `baseline-20260924T033709Z-424904` | 用户主动停止，exit130 | 历史记录；不计正式baseline性能 |
| `baseline-20260924T035051Z-450240` | baseline完整2115步，exit0 | baseline最终模型及checkpoint2115 |

以上原始目录相对项目根为 `artifacts/grpo-full-b200/<run>/`；训练日志 `training.log`，资源摘要 `training_summary.json`，来源 `source_provenance.txt`，最终权重及完成记录 `training/`。launcher日志与退出码在 `artifacts/launches/<run>.log/.exit`。路径仅为定位，不是公共Git下载链接。

## 当前报告

- [baseline完整结果](b200-baseline-full-20260925/RESULTS.md)
- [KTR完整结果与恢复说明](b200-ktr-full-20260924/RESULTS.md)
- [两组对比与限制](b200-paired-20260925/COMPARISON.md)
- [脱敏机器证据](b200-paired-20260925/evidence.json)：运行源码、原始日志哈希、模型哈希、checkpoint结构、配置、采样计划与指标窗口。

证据JSON中的 duration_seconds 是各run外层训练计时；windows 是日志更新的非加权均值。KTR去重方法见对比报告。公开Git不含主机/账户路径、完整日志、数据和模型；运行源码commit与本次报告commit分别记录，不因文档更新重标运行身份。

历史开发材料在 [文档归档](../docs/archive/README.md)；当前事实以 [STATUS](../handoff/STATUS.md) 为准。后续评测应新增独立报告目录，不覆盖本轮训练证据。
