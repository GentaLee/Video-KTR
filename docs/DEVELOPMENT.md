# B200 当前开发记录

更新：2026-09-24。状态以 [handoff/STATUS.md](../handoff/STATUS.md) 为准；历史详细过程保留于 [旧开发记录](archive/DEVELOPMENT_PLAN_20260924.md)。

## 已完成

- 全量Holmes路径和媒体缓存验证16916/16916，无删样本；8×B200、16384/768/G8、FA2、8帧。
- KTR按每completion精确top20%、绝对delta、单次非恒等时序置换、不同image/video id实现；full与baseline共用数据/运行流水线。
- 末批completion指标误截断已修复；checkpoint-2100恢复15步至2115成功，模型、完整checkpoint、资源摘要保存，保活恢复。
- [KTR结果](../reports/b200-ktr-full-20260924/RESULTS.md)与[踩坑索引](B200_PITFALLS.md)已整理。用户新启动的baseline已按要求停止，等待使用整理后的入口重新启动。

## 数据搬运和“拌匀”方式

1. 控制端准备/补齐媒体，经显式传输到集群3独立持久盘。源码用Git bundle+SHA交付；媒体、模型、缓存不上Git。不是跨集群共享目录。
2. 不改原始标注，以路径及媒体门禁确认全部16916条（image8765/video8151）。恢复时数据文件SHA及顺序一致；没有为通过门禁过滤坏样本。本次归档不移动原始数据或训练输出。
3. sampler先按image/video分组，以seed42+epoch在组内打乱。同模态构成全局8-prompt block，再随机打乱block顺序。因此整个epoch混合模态，但每一步8个rank同模态，避免ZeRO不同分支collective错序；不是任意逐条混排。
4. image补3、video补1，显式记录原始索引后得到16920个采样位置/2115步。保留16916个独立样本，额外4个是重复位置；不drop_last、不掩盖重复、不伪称新增数据。计划写入`modality_sampler_plan.jsonl`。
5. 训练统计包含补齐项。修复仅让completion级统计不被prompt remainder截断；loss和token选择不变。全量模型质量仍要独立评测，训练reward不能代替评测准确率。

## CPU/GPU/保活

冷启动先精确解码缓存；热启动全量缓存验证约几十秒。每rank2个CPU worker预取，GPU消费只读已验证缓存。CPU准备阶段保活，GPU训练前暂停，任务正常/异常退出释放GPU后恢复。冷缓存预检仍在正式训练前，不宣称全部预检与GPU训练完全重叠。

## 当前代码与文档归档

根目录只保留README、AGENTS、需求describe及统一run.sh。当前文档在docs，历史记录及停用脚本文本在docs/archive；运行脚本集中scripts，结果在reports，状态在handoff。归档保留历史，不重写旧run身份；运行端用新release，不覆盖旧release。

## 下一步

用户`bash run.sh baseline`→完整2115步及保存/保活验收→两模型使用同一独立评测集比较效果与资源。baseline不加载KTR权重，不自动续KTR checkpoint。H200工作区、训练和分支不在本次操作范围。
