# 🦈 Video-KTR: Key-Token Reinforcement for Video Reasoning

> 本工作区专用于 **B200**。先读 [本分支状态](handoff/STATUS.md) 和 [跨存储盘交付协议](COLLABORATION.md)；不要从这里修改 H200 工作区或覆盖正在运行的 B200 release。

> 本分支的集群 3 启动与最新流水线验证见 [B200_PIPELINE_VALIDATION.md](B200_PIPELINE_VALIDATION.md)：精确媒体缓存、CPU 预取、自动保活，以及 `bash launch_b200.sh ktr` / `baseline` 两个串行训练入口。

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

### Cluster 3 B200 reproducibility profile

The upstream command above remains the reference launcher.  This repository
also contains a separately gated, offline 8×B200 profile for reproducibility
work: 16,384 prompt tokens, 768 completion tokens, `G=8`,
FlashAttention-2, Holmes-16k, requested `max_pixels=401408`, and
`nframes=8`. Its B200 paired smoke has passed, but it is an 8-record
**video-only** smoke, not distributed mixed image/video sampler coverage; a
GPU full epoch has **not** been started by this repository workflow. The first
strict KTR wrapper stopped at its CPU decoder data-class gate after two
120-second media timeouts, before `torchrun`.

The B200 path uses `paper/per_completion`: E/V receive an exact top-20% mask
per completion; T receives the same exact top-20% selection for video
completions, while image completions have an all-zero T mask by design. Visual
and temporal scores use absolute delta, one non-identity temporal permutation
is used per rank/step, and image/video token IDs are handled separately. See [B200_REPRODUCTION.md](B200_REPRODUCTION.md)
for the environment, evidence, alignment table, data caveat, and recovery
procedure; remote collaboration messages follow
[REMOTE_COLLAB_HANDOFF.md](REMOTE_COLLAB_HANDOFF.md).

On the configured GPU machine, the operator—not the setup workflow—runs the
full job.  The strict command proceeds only after all 16,916 Holmes media
records are present and mixed-media decoder verification has no rejection:

```bash
B200_ALLOW_PAUSE_KEEPALIVE=1 \
B200_ALLOW_REDUCED_HOLMES=0 \
B200_ALLOW_DECODER_FILTERED=0 \
B200_MEDIA_DECODE_TIMEOUT_SECONDS=600 \
VARIANT=ktr ./run_full_b200.sh
```

The historical `15,365 / 16,916` media subset required an explicit opt-in and
is labeled `reduced-media-subset`; it is not a strict full-data result. The
233 missing training-video paths have since been restored. A standalone
120-second mixed-media gate passed all `16,916` records, but the first full
preflight verified only `16,914`: two references to the same long video timed
out. The launcher now allows 600 seconds per record in CPU preflight and
records that setting. Each full launch still requires all `16,916` records
with zero rejections before GPU work. Use the same 600-second setting for
baseline; the following reduced command is historical diagnostic-only:

```bash
B200_ALLOW_PAUSE_KEEPALIVE=1 \
B200_ALLOW_REDUCED_HOLMES=1 \
VARIANT=ktr ./run_full_b200.sh
```

`run_full_b200.sh` streams preflight ETA, training progress, and GPU heartbeat
to the terminal.  It keeps the external protection workload running during
CPU-only preparation, pauses it only for GPU work through its official
controller, and restores/verifies it after success, failure, or interruption.

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
