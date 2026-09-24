"""Fail closed on incompatible full-epoch B200 resume; never unpickle weights."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(checkpoint, dataset, variant, max_steps):
    checkpoint = Path(checkpoint).resolve(strict=True)
    old_run = checkpoint.parent.parent
    config = dict(line.split("=", 1) for line in
                  (old_run / "launch_config.txt").read_text().splitlines() if "=" in line)
    expected = {"variant": variant, "nproc_per_node": "8", "max_steps": "-1",
                "max_prompt_length": "16384", "max_completion_length": "768",
                "num_generations": "8", "selection_scope": "per_completion",
                "temporal_permutations": "1", "delta": "absolute"}
    if max_steps != -1 or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("resume requires the same variant and full-epoch B200 profile")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    dataset_sha = digest(dataset)
    if dataset_sha != digest(config["training_dataset_json"]):
        raise ValueError("resume dataset bytes/order differ from the original run")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = state["global_step"]
    if not 0 < step < state["max_steps"] or state["max_steps"] != 2115:
        raise ValueError("not an unfinished strict Holmes checkpoint")
    tag = (checkpoint / "latest").read_text().strip()
    if tag != f"global_step{step}":
        raise ValueError("DeepSpeed latest tag disagrees with trainer step")
    required = [checkpoint / "scheduler.pt", checkpoint / "training_args.bin"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    required += [checkpoint / f for f in set(index["weight_map"].values())]
    for rank in range(8):
        required += [checkpoint / f"rng_state_{rank}.pth",
                     checkpoint / tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                     checkpoint / tag / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"]
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise ValueError("checkpoint has missing or empty recovery shards")
    return {"status": "passed", "checkpoint": str(checkpoint), "global_step": step,
            "max_steps": state["max_steps"], "remaining_steps": state["max_steps"] - step,
            "dataset_sha256": dataset_sha, "variant": variant,
            "validation": "metadata/shard presence; actual loading is verified by training"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=["ktr", "baseline"], required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.checkpoint, args.dataset, args.variant, args.max_steps)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
