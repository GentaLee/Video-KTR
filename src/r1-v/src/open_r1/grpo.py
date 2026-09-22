# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

if __package__:
    from .trainer import Qwen2VLGRPOTrainer
else:  # Script-path invocation used by the upstream launchers.
    from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    video_ktr: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using Video-KTR-GRPO"},
    )
    entropy_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "entropy ratio"}
    )
    visual_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "visual-aware ratio"}
    )
    temporal_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "temporal-aware ratio"}
    )
    output_selected_token: Optional[bool] = field(
        default=False,
        metadata={"help": "write a bounded JSONL sample of E/V/T/union tokens per rank"},
    )
    token_record_limit: Optional[int] = field(
        default=16,
        metadata={"help": "maximum union-selected token examples saved for each completion"},
    )
    data_root: Optional[str] = field(
        default=None,
        metadata={"help": "absolute root containing paths recorded in dataset_name"},
    )
    nframes: Optional[int] = field(
        default=8,
        metadata={"help": "fixed frame count for video inputs; must be at least two for KTR"},
    )
    selection_mode: Optional[str] = field(
        default="paper",
        metadata={"help": "KTR selection protocol: paper (absolute) or repo (signed legacy)"},
    )
    selection_scope: Optional[str] = field(
        default=None,
        metadata={"help": "KTR ranking scope: per_completion or batch; default depends on mode"},
    )
    temporal_permutations: Optional[int] = field(
        default=5,
        metadata={"help": "number of distinct non-identity frame-order probes for KTR"},
    )
    temporal_include_reverse: Optional[bool] = field(
        default=True,
        metadata={"help": "include reverse frame order among KTR temporal probes"},
    )
    temporal_seed: Optional[int] = field(
        default=43,
        metadata={"help": "base seed for reproducible KTR temporal permutations"},
    )
    skip_final_model_save: Optional[bool] = field(
        default=False,
        metadata={"help": "smoke-only option: do not write the final full model checkpoint"},
    )


def legacy_deepspeed_resume_safe_globals(resume_from_checkpoint: Optional[str]):
    """Allow only the two known DeepSpeed classes in a legacy ZeRO resume.

    PyTorch 2.6 changed :func:`torch.load` to default to ``weights_only=True``.
    DeepSpeed 0.15.4 does not pass that argument in its checkpoint engine, and
    its locally generated ZeRO optimizer state serializes ``ZeroStageEnum``
    and ``LossScaler``.  Keep the secure default rather than globally forcing
    ``weights_only=False``: each torchrun rank allowlists precisely those
    DeepSpeed classes only while Trainer restores a checkpoint.

    This function has no effect for fresh runs.  The launcher additionally
    verifies a small model-state shard through the exact checkpoint-engine
    call before it starts torchrun.
    """

    if not resume_from_checkpoint:
        return nullcontext()

    import torch
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        # Older PyTorch versions retain their historical default
        # (weights_only=False), so no compatibility allowlist is necessary.
        return nullcontext()
    print(
        "[grpo] enabled restricted legacy DeepSpeed resume safe-globals "
        "(ZeroStageEnum, LossScaler); weights_only remains enabled",
        flush=True,
    )
    return safe_globals([ZeroStageEnum, LossScaler])



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # The shared direct implementation is selected explicitly for KTR, or for
    # a baseline that supplies ``data_root`` so it can use the same safe media
    # loader as KTR.  Keep a plain upstream baseline (``video_ktr=false`` and
    # no data root) on the original trainer instead of silently changing its
    # launcher semantics.
    direct_video_mode = bool(script_args.video_ktr or script_args.data_root)
    if training_args.use_vllm and direct_video_mode:
        raise ValueError(
            "Video-KTR and the comparable direct baseline require the direct "
            "trainer path. The checked-in vLLM trainer does not apply E/V/T token masks."
        )
    if direct_video_mode:
        root_text = script_args.data_root or os.environ.get("VIDEO_ROOT")
        if not root_text:
            raise ValueError(
                "direct Video-KTR/baseline requires --data_root (or VIDEO_ROOT) "
                "so media paths are resolved without a fallback"
            )
        if training_args.per_device_train_batch_size != 1:
            raise ValueError(
                "the direct Video-KTR/baseline trainer requires --per_device_train_batch_size 1"
            )
        if training_args.num_generations < 2:
            raise ValueError("GRPO requires --num_generations >= 2")
    if script_args.video_ktr:
        if script_args.nframes is None or script_args.nframes < 2:
            raise ValueError("Video-KTR requires --nframes >= 2")
        if script_args.temporal_permutations is None or script_args.temporal_permutations < 1:
            raise ValueError("Video-KTR requires --temporal_permutations >= 1")
        if script_args.token_record_limit is None or script_args.token_record_limit < 0:
            raise ValueError("--token_record_limit must be non-negative")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    if direct_video_mode:
        # Keep this import lazy: an ordinary upstream baseline must not need
        # the KTR helper module or repository-root PYTHONPATH entries.
        if __package__:
            from .trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer
        else:
            from trainer.video_ktr_grpo_trainer import VideoKTRGRPOTrainer

        trainer_cls = VideoKTRGRPOTrainer
    elif training_args.use_vllm:
        # Importing the optional vLLM trainer eagerly would import the host's
        # vLLM package even for direct KTR runs.  The H200 image's editable
        # vLLM expects a newer Transformers than this checkpoint supports.
        if __package__:
            from .trainer.vllm_grpo_trainer_modified import Qwen2VLGRPOVLLMTrainerModified
        else:
            from trainer.vllm_grpo_trainer_modified import Qwen2VLGRPOVLLMTrainerModified

        trainer_cls = Qwen2VLGRPOVLLMTrainerModified
    else:
        trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # ``attn_implementation`` is only a request until the checkpoint has
    # actually been loaded.  Strict accelerator profiles can opt into this
    # post-load gate so a missing FlashAttention kernel never silently turns a
    # purported FA2 reproduction into SDPA/eager attention.
    required_attention = os.environ.get("VIDEO_KTR_REQUIRE_ATTN_IMPLEMENTATION")
    attention_candidates = [getattr(trainer, "model", None)]
    seen_candidates: set[int] = set()
    resolved_attention = None
    while attention_candidates:
        candidate = attention_candidates.pop(0)
        if candidate is None or id(candidate) in seen_candidates:
            continue
        seen_candidates.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None:
            for attribute in ("_attn_implementation", "_attn_implementation_internal"):
                value = getattr(config, attribute, None)
                if isinstance(value, str) and value:
                    resolved_attention = value
                    break
        if resolved_attention is not None:
            break
        for attribute in ("module", "model", "base_model"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                attention_candidates.append(nested)
    if trainer.is_world_process_zero():
        print(
            "[grpo] "
            f"requested_attention={model_args.attn_implementation}; "
            f"resolved_attention={resolved_attention}",
            flush=True,
        )
    if required_attention and resolved_attention != required_attention:
        raise RuntimeError(
            "strict attention gate failed: "
            f"required {required_attention!r}, resolved {resolved_attention!r}"
        )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        with legacy_deepspeed_resume_safe_globals(checkpoint):
            train_output = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_output = trainer.train()

    # Smoke must exercise forward/backward/optimizer without unconditionally
    # writing a 7B checkpoint.  Full runs retain the original save behavior.
    if not script_args.skip_final_model_save:
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)

    if trainer.is_world_process_zero():
        completion = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "global_step": int(trainer.state.global_step),
            "train_metrics": dict(train_output.metrics),
            # ``Trainer`` clears its rolling log cache after every report.  Keep
            # the final distributed GRPO/KTR statistics explicitly so resource
            # summaries can compare baseline and E∪V∪T runs without parsing a
            # terminal log.
            "latest_step_metrics": dict(getattr(trainer, "latest_step_metrics", {})),
            "video_ktr": bool(script_args.video_ktr),
            "selection_mode": script_args.selection_mode,
            "selection_scope": script_args.selection_scope,
            "temporal_permutations": script_args.temporal_permutations,
            "skip_final_model_save": bool(script_args.skip_final_model_save),
            "dataset_name": script_args.dataset_name,
            "data_root": script_args.data_root,
            "model_name_or_path": model_args.model_name_or_path,
        }
        completion_path = Path(training_args.output_dir) / "training_complete.json"
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[grpo] wrote completion record: {completion_path}", flush=True)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
