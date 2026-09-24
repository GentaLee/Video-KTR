# 集群 3 / B200 KTR 完整 epoch 结果

核验日期：2026-09-24 UTC。**训练完成并保存成功；正式 baseline 与独立质量评测尚未运行。**

## 实验身份与完成证据

| 项目 | 已核验事实 |
| --- | --- |
| 原 run | `ktr-20260923T031639Z-2981461`；源码 `d967c23`；完成 2114 步后末批统计失败 |
| 恢复 run | `ktr-20260924T031544Z-389593`；训练源码 `86317d00619a56507f66236caa01eba1ebb8b459` |
| 恢复点 | 原 `training/checkpoint-2100`；新日志恰好 15 条 optimizer-step 记录 |
| 完成 | `training/training_complete.json`：global_step=2115；2026-09-24T03:26:09.242377Z |
| 退出 | launcher `.exit=0`，full wrapper completed successfully |
| 最终 checkpoint | `training/checkpoint-2115`：8 optimizer shards、8 model-state shards、8 RNG、scheduler、trainer_state(global_step=2115) |
| 最终模型 | `training/`：4 个 safetensors shard + index、config、tokenizer/processor；完整 shard SHA256 见下方 |
| 收尾 | 训练 GPU 进程全部退出，唯一保活进程已自动恢复 |

路径均相对 `<项目持久根>/artifacts/grpo-full-b200/<run>/`。开发文档后续 commit 不改变运行源码身份。checkpoint-2115 结构已核验，但本次没有再从 2115 加载训练，也未对最终模型执行独立推理评测。

最终4个safetensors文件另经CPU只读header/index检查，共729个tensor key，分片分别459/131/122/17，与索引逐项一致；不是模型生成质量验收。

## 配置与数据

8×B200、Qwen2.5-VL-7B-COT-SFT、Holmes 全部 16,916 条（8,765 image / 8,151 video）；prompt16384 / completion768 / G8；FA2；8 帧；单次非恒等随机时序置换；每 completion 精确 ceil(20%) top-k、绝对 delta、image/video token id 分离。请求 max_pixels=401408，但原 Qwen 视频有效上限约105369，不宣称每帧实用401408。

sampler 分模态组全局 block 后打乱 block 顺序；固定 seed42，epoch0；原和恢复 sampler plan 除记录时间外完全一致。额外4条是显式补齐（image3 / video1），不是新增独立数据。16920/8=2115 optimizer steps。数据文件 SHA256：`19025cd0a08fb6b01b495668867f6888d79336caa23c2fe379652cb16af46a1f`。

## 时长与显存（不可混用统计口径）

| 阶段 | 外层训练计时 | 单卡显存峰值（跨卡最大） | 说明 |
| --- | ---: | ---: | --- |
| 原失败尝试 | 52,760.078 秒（14.66 小时） | 110,517 MiB（107.93 GiB） | 2114 步及失败收尾；不是成功 epoch |
| 恢复 15 步 | 502.566 秒（8.38 分钟） | 79,695 MiB（77.83 GiB） | 含模型/状态加载及保存；不是全程峰值 |
| 两次训练段实际消耗合计 | 53,262.644 秒（14.80 小时） | 110,517 MiB | 包含重复的2101–2114，不含两次运行之间停机及前置门禁/冒烟 |

恢复 Trainer 内部计时404.0428秒；其输出`train_steps_per_second=5.235`和`train_loss=3.859e-05`使用累计global_step与局部恢复窗口，**不能作为15步或完整epoch的真实吞吐/平均loss**。实际15条日志平均loss=0.005433；15/404.0428≈0.0371更新/秒（含该内部窗口的等待/保存，不是稳态吞吐）。进度条恢复后也有相同的累计步数 ETA 偏差。

## 数值分析

去重方法：原日志仅保留1–2100，再拼接恢复日志2101–2115。共2115条更新记录，所有已记录数值均有限，无NaN/Inf。

| 训练指标 | 前100步均值 | 最后100步均值 |
| --- | ---: | ---: |
| reward | 1.51712 | 1.71731 |
| KL | 0.04406 | 0.13682 |
| completion tokens | 284.77 | 367.74 |
| union 更新比例 | 33.20% | 34.14% |
| loss | 0.001767 | 0.005476 |
| 梯度范数（日志值） | 5.3553 | 3.8392 |

训练reward提高、回答变长，同时KL增大；这是不同训练样本窗口的描述，**不能证明独立推理能力提升或KTR优于baseline**。GRPO loss包含KL项，不应按普通监督训练“loss必须单调下降”解读。梯度范数日志也不是裁剪后范数。

最后一步：loss0.0058、梯度范数3.28795、学习率0、reward1.775、KL0.14567；E/V/T各平均70.52 tokens，union130.23，更新比例37.13%。最后一批为视频，因此T非零；图像批次T为0是预期行为。

恢复的第2101步loss/reward/KL/学习率与原日志一致；后续重算存在小幅差异，未验证逐位确定性。checkpoint恢复、学习率轨迹、数据顺序通过；不声称重算权重与未中断运行逐位相同。

## 最终权重 SHA256

```text
57d39bcdde6379d1b4db04bdb66dc4ca7d466eaa78378284820362f941c49328  model-00001-of-00004.safetensors
196183595c0193074ff7bfd1dff9986ab2bb351e87104fdbf0438fb7e1a7b443  model-00002-of-00004.safetensors
2ffb4b2749bfaf6646ede74260a88cce195dc8fdcf0be6240def6bf12f5c838b  model-00003-of-00004.safetensors
41dafaec1265319e66fdf4205d037c03435fd138957991137c40b44ce2b8fa42  model-00004-of-00004.safetensors
```

## 下一步

由用户手动启动同修复源码的baseline，全量2115步，从相同原始SFT模型开始，不加载KTR状态；使用相同数据与采样器。入口见 [恢复与命令](../../B200_CHECKPOINT_RECOVERY.md)。结束后同一独立评测集评测两个最终模型，再比较质量、资源和token示例。

full未保存全量CoT逐token记录；已有paired smoke token_examples可作功能示例，不能当成最终训练模型的CoT分析。数据、模型、完整日志不上传GitHub；本报告只上传脱敏摘要和哈希。
