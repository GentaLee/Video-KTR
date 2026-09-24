# B200 状态（本分支唯一事实源）

PROFILE: b200
STATUS: KTR_AND_BASELINE_COMPLETED_EVALUATION_PENDING
REPORT_DATE: 2026-09-25
LAST_VERIFIED_UTC: 2026-09-24T18:25:09Z

## 当前事实

集群3两组已完成，训练进程退出，唯一保活已恢复。本次仅只读核验运行与归档报告，未重启训练、评测或操作保活。报告日期采用工作区日期；LAST_VERIFIED来自运行端时钟，模型完成时间以各run的UTC记录为准。

| 项目 | baseline | KTR |
| --- | --- | --- |
| 最终run | baseline-20260924T035051Z-450240 | ktr-20260924T031544Z-389593 |
| 运行源码 | ff1a1d03407bdec6382b6715429881eea223aaac | 86317d00619a56507f66236caa01eba1ebb8b459 |
| 完成时间UTC | 2026-09-24T17:45:36.180225Z | 2026-09-24T03:26:09.242377Z |
| global_step / exit | 2115 / 0 | 2115 / 0 |
| 训练段消耗 | 13.88小时 | 14.80小时，含原失败段与恢复重算 |
| 单卡峰值 | 58.40GiB | 107.93GiB |

两者最终4-shard模型、完整checkpoint2115的8卡optimizer/model/RNG及scheduler结构通过；未对最终checkpoint做重新加载验收。KTR权重重新哈希与原记录一致，未被baseline覆盖。两者最后100步训练奖励接近，尚无独立质量评测结论。

## 对齐和历史边界

8×B200、16916条全量Holmes、16384/768/G8、FA2、8帧。sampler显式补4条形成16920位置/2115步，三次run数据SHA及sampler plan一致。请求像素401408、原Qwen有效视频cap约105369。KTR每completion top20%、绝对delta、单次非恒等置换，image/video ID区分；baseline所有有效completion tokens参与训练。

KTR原run在2114后末批指标聚合失败；修复后从2100恢复15步，loss/selector不变。baseline从原始SFT开始，不续KTR。旧baseline-20260924T033709Z-424904是用户主动停止exit130，不是当前结果。

## 归档与部署

[归档索引](../reports/README.md) · [配对报告](../reports/b200-paired-20260925/COMPARISON.md) · [baseline报告](../reports/b200-baseline-full-20260925/RESULTS.md) · [KTR报告](../reports/b200-ktr-full-20260924/RESULTS.md)。

只维护B200分支；H200未操作。源码和产物位于不同集群独立盘，不共享目录。本次报告更新不替换运行端current（仍为ff1a1d0）或任何冻结release；原始产物原位保留。报告提交不是新的训练源码身份。

NEXT: 同一独立评测集比较SFT/baseline/KTR，并补最终模型CoT token示例；需要单独启动评测。本轮训练无需重复。
RECOVERY: NONE，两个最终训练任务均完成。
