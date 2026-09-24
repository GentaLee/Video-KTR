import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resume_gate", ROOT / "src/grpo_validate_b200_resume.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.dataset = root / "data.json"
        self.dataset.write_text('[{"id": 1}]')
        self.current = root / "current.json"
        self.current.write_bytes(self.dataset.read_bytes())
        self.checkpoint = root / "training/checkpoint-2100"
        self.checkpoint.mkdir(parents=True)
        config = dict(variant="ktr", nproc_per_node="8", max_steps="-1",
                      max_prompt_length="16384", max_completion_length="768",
                      num_generations="8", selection_scope="per_completion",
                      temporal_permutations="1", delta="absolute",
                      training_dataset_json=str(self.dataset))
        (root / "launch_config.txt").write_text("\n".join(f"{k}={v}" for k, v in config.items()))
        (self.checkpoint / "trainer_state.json").write_text(json.dumps(dict(global_step=2100, max_steps=2115)))
        (self.checkpoint / "latest").write_text("global_step2100")
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
        files = ["scheduler.pt", "training_args.bin", "model.safetensors"]
        for rank in range(8):
            files += [f"rng_state_{rank}.pth",
                      f"global_step2100/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                      f"global_step2100/zero_pp_rank_{rank}_mp_rank_00_model_states.pt"]
        for name in files:
            path = self.checkpoint / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fixture")

    def test_full_state_candidate(self):
        result = gate.validate(self.checkpoint, self.current, "ktr", -1)
        self.assertEqual(result["remaining_steps"], 15)

    def test_cannot_resume_ktr_as_baseline(self):
        with self.assertRaises(ValueError):
            gate.validate(self.checkpoint, self.current, "baseline", -1)

    def test_rejects_dataset_change(self):
        self.current.write_text("[]")
        with self.assertRaises(ValueError):
            gate.validate(self.checkpoint, self.current, "ktr", -1)

    def test_rejects_missing_optimizer_rank(self):
        (self.checkpoint / "global_step2100/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt").unlink()
        with self.assertRaises(ValueError):
            gate.validate(self.checkpoint, self.current, "ktr", -1)

    def test_rejects_remaining_steps_as_total(self):
        with self.assertRaises(ValueError):
            gate.validate(self.checkpoint, self.current, "ktr", 15)


if __name__ == "__main__":
    unittest.main()
