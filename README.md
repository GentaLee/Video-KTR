# 🦈 Video-KTR: Key-Token Reinforcement for Video Reasoning

## 本仓库复现分支与当前状态

更新日期：2026-09-25。以下为本仓库的复现实验记录，与下方原论文结果分开解读；训练完成不等于已复现论文评测成绩。

### 为什么使用 `multimodal-ktr` 命名

本次 Holmes 训练数据包含 **8,765 条图像样本和 8,151 条视频样本**，共 16,916 条，因此实验分支由 `video-ktr` 改为 `multimodal-ktr`，避免被理解为仅包含视频。`ktr` 表示复现项目范围，分支内同时维护 KTR 与 GRPO baseline；`h200` / `b200` 区分硬件、环境和运行记录。原项目名称 **Video-KTR**、论文名称、已有目录和冻结的运行源码身份不变。

### 长期分支与进展

| 分支 | 职责 | 已提交的最新状态 | 后续工作 |
| --- | --- | --- | --- |
| [`main`](https://github.com/GentaLee/Video-KTR/tree/main) | 上游基准与公共导航 | 保留原项目介绍；不混入两套实验运行代码 | 通过独立 PR 更新公共文档 |
| [`GentaLee-patch-1`](https://github.com/GentaLee/Video-KTR/tree/GentaLee-patch-1) | 原作者补充需求、复现起点 | 保留需求说明及历史 | 作为参考，不承担日常实验开发 |
| [`zhaoye/multimodal-ktr-h200`](https://github.com/GentaLee/Video-KTR/tree/zhaoye/multimodal-ktr-h200) | H200 实验与结果 | 已归档全量 KTR 2115/2115 步、exit=0、最终模型及 checkpoint；已提交记录尚无完整 baseline 结果 | 完成并归档配对 baseline、独立质量评测 |
| [`zhaoye/multimodal-ktr-b200`](https://github.com/GentaLee/Video-KTR/tree/zhaoye/multimodal-ktr-b200) | B200 实验与结果 | KTR 与 baseline 均完成 2115/2115 步并保存；KTR 经 checkpoint2100 恢复完成，baseline 从 SFT 新训；训练结果对比已归档 | 使用同一独立评测集比较质量，补最终模型 CoT token 示例 |

状态依据各分支已提交报告，不是实时进程监控。H200 使用 4 卡与梯度累积，B200 使用 8 卡；不能将跨硬件结果直接当作只改变 KTR 开关的严格对照。B200 最后 100 步训练奖励接近，尚不能据此判定 KTR 优于 baseline。

- H200：[状态记录](https://github.com/GentaLee/Video-KTR/blob/zhaoye/multimodal-ktr-h200/handoff/STATUS.md) · [KTR 结果](https://github.com/GentaLee/Video-KTR/blob/zhaoye/multimodal-ktr-h200/reports/h200-ktr-full-20260924/RESULTS.md)
- B200：[状态记录](https://github.com/GentaLee/Video-KTR/blob/zhaoye/multimodal-ktr-b200/handoff/STATUS.md) · [结果索引](https://github.com/GentaLee/Video-KTR/blob/zhaoye/multimodal-ktr-b200/reports/README.md) · [配对分析](https://github.com/GentaLee/Video-KTR/blob/zhaoye/multimodal-ktr-b200/reports/b200-paired-20260925/COMPARISON.md)

### 旧分支的去向与协作约定

旧 `zhaoye/video-ktr-h200` 和 `archive-20260924/video-ktr-h200` 已统一到新的 H200 分支，包含归档节点 `8fbaa46` 及全部祖先提交；旧 B200 分支已改名。拆分前的 `zhaoye/video-ktr-repro-handoff` 分支名已删除，其历史仍由两条实验分支保留，没有删除模型、数据或 checkpoint。

H200 / B200 各维护一条长期实验分支，仅交换经审阅的修复和脱敏交接记录，不为同步文档整支合并。公共 README 改动从 `main` 建立临时文档分支、通过 PR 合入，合并后删除该临时分支；不删除两条长期实验分支。原始日志、模型、数据和私有环境文件不上 Git。

---

> **Video-KTR** is a reinforcement learning framework designed for complex video reasoning.\
> It identifies and amplifies *critical visual--temporal tokens* via selective gradient reinforcement, significantly improving video reasoning performance.

<p align="center">
<a href="https://arxiv.org/abs/2601.19686">[📖 Paper]</a> &nbsp; <a href="https://huggingface.co/Video-KTR/Video-KTR-7B">[🤗 Video-KTR-7B-model]</a>
</p>
<p align="center">
<strong>[2026.01.26]</strong>      🎉Our work is accepted by ICLR 2026.
</p>


## 🌟 Highlights

-   🚀 **State-of-the-art performance** on multiple video reasoning benchmarks (Video-Holmes, VideoMMMU, MMVU, VideoMME)
-   🎯 **Key Token Reinforcement (KTR)**: amplifies signal on high-entropy / visual-aware / temporal-aware tokens
-   🔍 **Better temporal and causal reasoning** demonstrated by detailed case studies

<div align="center">
  <img src="images/method.png" width="90%">
</div>

Video-KTR improves video reasoning by identifying truly critical reasoning tokens. We use counterfactual probing (masking images or shuffling frames) to find
- 👀 visual-aware tokens
- ⏰ temporal-aware tokens
- and apply entropy filtering to select uncertain but informative tokens.

During training, non-critical tokens are masked, and gradients are reinforced only on key tokens, enabling more stable and accurate temporal–causal reasoning.

------------------------------------------------------------------------

### 📊 Main Results

- On **Video-Holmes**, Video-KTR reaches **42.7**, **nearly matching closed-source models** such as GPT-4o (**42.0**) and Gemini-2.5-Pro (**45.0**) 🎯.  
- At the 7B scale, it also **substantially outperforms all existing open-source baselines**, highlighting the effectiveness of our key-token reinforcement approach.

<div align="center">
  <img src="images/main_experiment.png" width="90%">
</div>


### ⚖️ Data Ablation on Different 

Results from applying multiple post-training methods on the same dataset show that our approach consistently delivers superior performance.

<div align="center">
  <img src="images/data_ablation.png" width="90%">
</div>

### 🔍 Analyze on Token Selecting

- We decompose the gradients of the final layer and show that the tokens we mask out contribute low-magnitude, highly scattered gradients, indicating weak and noisy supervision.
- The reduced training loss variance further confirms that our method leads to more stable and efficient optimization.

<div align="center">
  <img src="images/training_metrics.png" width="90%">
</div>

Qualitatively, our word-cloud and POS analyses further confirm that masked tokens are largely function words while the selected tokens are informative.

<div align="center">
  <img src="images/tokens.png" width="90%">
</div>

### 🎬 Case Studies on Event Causality Reasoning & Temporal Ordering 

<div align="center">
  <img src="images/demo_case_1.png" width="90%">
</div>

<div align="center">
  <img src="images/demo_case_2.png" width="90%">
</div>

------------------------------------------------------------------------

## 🔧 Installation

``` bash
# build environment
git clone https://github.com/ziyue1999/Video-KTR.git
cd Video-KTR

conda create -n video-r1 python=3.11 
conda activate video-r1
bash setup.sh

# download training dataset
git lfs install
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
```
Please put the downloaded dataset to src/r1-v/Video-R1-data/. The `Video-R1-260k.json` file is for RL training.
Then, unzip the data
``` bash
python ./src/unzip.py
```

Qwen2.5-VL has been frequently updated in the Transformers library, which may cause version-related bugs or inconsistencies. Our code is compatible with the following version, please download at [here](https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view?usp=sharing)

Then install our provided version of transformers

```bash
unzip transformers-main.zip
cd ./transformers-main
pip install .
```

------------------------------------------------------------------------

## 🚀 Training

Our training pipeline builds upon **Video-R1** and **Qwen2.5-VL-SFT**.

The script for GRPO training is as follows
```bash
cd src/r1-v
bash ../scripts/run_grpo_video_ktr.sh
```

------------------------------------------------------------------------

## 🖍 Inference & Evaluation

Run evaluation on Video-Holmes / VideoMMMU / MMVU:

``` bash
bash ./src/eval_bench.sh
```

For infernce on a single example, you may use:

```bash
python ./src/inference_example.py
```

------------------------------------------------------------------------

## 📑 Citation
If you find our work helpful for your research, please consider citing our work.
```bash
@misc{wang2026videoktrreinforcingvideoreasoning,
      title={Video-KTR: Reinforcing Video Reasoning via Key Token Attribution}, 
      author={Ziyue Wang and Sheng Jin and Zhongrong Zuo and Jiawei Wu and Han Qiu and Qi She and Hao Zhang and Xudong Jiang},
      year={2026},
      eprint={2601.19686},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.19686}, 
}
```
