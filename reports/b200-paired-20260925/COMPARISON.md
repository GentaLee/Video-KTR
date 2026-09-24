# 集群 3 / B200：baseline vs KTR 训练结果对比

整理日期：2026-09-25。两者均完成 2115 步、保存最终模型。**训练奖励接近；独立质量评测尚未进行，不能判定 KTR 优于 baseline。**

## 对齐与保留的差异

两者使用相同原始 SFT、8×B200、Holmes 全部 16916 条（image8765/video8151）、16384 prompt / 768 completion / G8、FA2、8帧；sampler seed42、时序 seed43。相同模态组内打乱并组成全局8-prompt block，再打乱 block。显式补齐 image3/video1，16920个采样位置对应2115步；没有过滤样本或隐瞒重复。

三个 run 的数据文件 SHA256 相同：`19025cd0a08fb6b01b495668867f6888d79336caa23c2fe379652cb16af46a1f`；sampler plan 除时间字段外完全一致。已记录的非路径 launch_config 字段仅 variant 不同；这不表示全部运行元数据、源码提交或随机生成轨迹相同。baseline 与 KTR 恢复版本的所有已记录 critical-source 哈希一致；KTR 原版与恢复版存在已披露的 completion 指标聚合修复。

- baseline：GRPO 对所有有效 completion tokens 计算更新。
- KTR：仅高熵 ∪ 视觉敏感 ∪ 时序敏感 token；每 completion ceil(20%)、绝对 delta、单次非恒等随机时序置换；image/video ID 已区分，图像无时序 token。
- 共同限制：请求 max_pixels401408，但原 Qwen 有效视频 cap约105369。不是逐帧实际401408，也不宣称逐行复刻上游算法。
- baseline fresh SFT；KTR 从 checkpoint2100 恢复，重算2101–2114并完成2115。不掩盖中断，不宣称逐位确定性。

## 资源对比

| 项目 | baseline | KTR |
| --- | ---: | ---: |
| 最终 global_step | 2115 | 2115 |
| 外层训练段实际消耗 | 13.88小时 | 14.80小时（原失败段+恢复段） |
| 单卡显存峰值（跨卡最大） | 58.40 GiB | 107.93 GiB |
| 最终退出码 | 0 | 0（恢复 run） |

KTR 实际训练段消耗比 baseline 多约6.6%，但含14步重复、额外加载/保存，排除停机间隔和前置门禁；**不能解读为纯算法开销或不中断 epoch 的耗时差**。KTR 单卡采样峰值约1.85倍，峰值受生成轨迹、样本及分配器缓存影响，不是模型张量静态占用。NVML采样也可能漏过短时尖峰。恢复段 Trainer 的5.235steps/s和极小 train_loss不能作全程性能指标。

## 同口径训练指标

baseline 使用2115条；KTR取原日志1–2100，加恢复日志2101–2115，去掉原2101–2114重复记录。两组已记录数值均有限。下表为最后100个更新的算术平均，不是独立样本去重评测。

| 指标 | baseline | KTR |
| --- | ---: | ---: |
| reward | 1.722093 | 1.717312 |
| accuracy_reward | 0.605343 | 0.605437 |
| format_reward | 0.999688 | 0.999219 |
| KL | 0.076885 | 0.136823 |
| completion 长度 | 350.902 | 367.737 |
| 日志 loss | 0.003077 | 0.005476 |

最后100步的训练奖励相近，KTR回答更长、KL更大。不能把 soft accuracy_reward 直接写成测试准确率；两种 token loss 掩码/归一化口径不同，不按 loss 大小评判模型优劣。窗口样本不同、单seed且无置信区间，首末窗口趋势也不是泛化提升的证据。首100/末100/全程统计均保留于 [证据 JSON](evidence.json)。

## 隔离、归档和未完成事项

baseline 与 KTR 使用独立 run 输出目录。本次重新计算 KTR 四个最终权重 SHA256，全部匹配之前 [KTR报告](../b200-ktr-full-20260924/RESULTS.md)，未被 baseline 覆盖；baseline 四个权重哈希均与 KTR 不同。两者 checkpoint2115 状态分片结构和最终模型 header/index 已核验，未执行最终checkpoint重载或模型推理。

原始日志、媒体、模型、缓存、checkpoint 留在原持久盘；Git仅保存报告、索引、脱敏摘要及哈希。失败/主动停止 run 明确保留并排除出正式 baseline。见 [归档索引](../README.md) 和 [baseline结果](../b200-baseline-full-20260925/RESULTS.md)。

下一步需另行运行：固定同一独立评测集、同一生成配置，对 SFT/baseline/KTR 做质量比较；从最终模型抽取同一批 CoT 的 E/V/T token 示例。full未保存全量逐token CoT，现有 smoke 示例只能证明筛选功能，不能代替最终模型分析。本次没有自动启动评测、训练或改动保活。
