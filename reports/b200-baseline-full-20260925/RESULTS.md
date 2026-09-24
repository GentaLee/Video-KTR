# 集群 3 / B200 baseline 完整 epoch 结果

整理日期：2026-09-25。**2115/2115 步完成，exit=0，最终模型及 checkpoint 保存成功。**

## 身份和验收

- run：`baseline-20260924T035051Z-450240`；运行源码：`ff1a1d03407bdec6382b6715429881eea223aaac`。
- 完成记录 UTC：`2026-09-24T17:45:36.180225Z`（北京时间 9 月 25 日 01:45）。报告日期不替代运行时间。
- 从 Qwen2.5-VL-7B-COT-SFT 开始，video_ktr=false；不是从 KTR checkpoint 续训。
- 2115 条 optimizer-step 日志，已记录数值全部有限；末批通过，没有再次触发 completion 聚合错误。
- checkpoint-2115：global_step=2115，8 optimizer / 8 model-state / 8 RNG shards，scheduler 存在。
- 最终模型 4 个 safetensors shards，CPU header/index 对应 729 个 tensor keys；本次未做最终模型推理或 checkpoint-2115 重载测试。
- 核验时训练进程已退出，唯一保活已恢复；未操作保活或重新启动训练。

## 资源和指标

| 指标 | 结果 |
| --- | ---: |
| 外层训练计时（含训练段加载/保存，不含前置门禁） | 49,962.360 秒 / 13.88 小时 |
| Trainer 内部计时 | 49,883.417 秒 |
| 单卡显存峰值（跨卡最大） | 59,799 MiB / 58.40 GiB |
| GPU 利用率：各卡采样均值的平均 | 46.31% |
| Trainer train_loss | 0.002973741 |
| 最后 100 步平均 reward | 1.722093 |
| 最后 100 步平均 accuracy_reward | 0.605343 |
| 最后 100 步平均 format_reward | 0.999688 |
| 最后 100 步平均 KL | 0.076885 |
| 最后 100 步平均 completion 长度 | 350.902 tokens |

accuracy_reward 是训练奖励分量，不等同独立测试准确率。日志 loss 已舍入，其逐步平均与 Trainer 累积 train_loss 略有差异是不同统计口径。

## 权重 SHA256

```text
89a6ed8e4d9f6632dfae8a7120a8984b2dcbfdb7433f080539ab347e124b4a80  model-00001-of-00004.safetensors
cb724dc1e912b86bf1ef0c241899c7448e41033a526f25b92a386da9b5a7cbbd  model-00002-of-00004.safetensors
c84d0a253ec370e0e0a519be7e46351100a4ef4225f9a3633fbda7fbc7737dac  model-00003-of-00004.safetensors
b23b51056fb5b32bb75ddc417eeff631928e91ae0a41d71d40c23f8ca24d0ead  model-00004-of-00004.safetensors
```

原始文件保留在 `<项目持久根>/artifacts/grpo-full-b200/baseline-20260924T035051Z-450240/`，最终模型在 `training/`，完整 checkpoint 在 `training/checkpoint-2115/`。没有移动或覆盖 KTR 产物。

配对配置、共同限制与结论见 [对比报告](../b200-paired-20260925/COMPARISON.md)；机器可核验摘要见 [evidence.json](../b200-paired-20260925/evidence.json)；所有 run 的定位见 [归档索引](../README.md)。
