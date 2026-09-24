"""Run the real metric helper without importing the heavyweight trainer."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/r1-v/src/open_r1/trainer/video_ktr_grpo_trainer.py").read_text()


class Values:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def float(self):
        return self
    def mean(self):
        return SimpleNamespace(item=lambda: sum(self.values) / len(self.values))


class TailMetricsTests(unittest.TestCase):
    def test_completion_metrics_never_use_prompt_trimming(self):
        tree = ast.parse(SOURCE)
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in ("_distributed_mean", "compute_loss"):
            calls = [n.func.attr for n in ast.walk(methods[name])
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            self.assertNotIn("gather_for_metrics", calls)
            self.assertIn("gather", calls)

    def test_last_batch_keeps_all_completion_values(self):
        method = next(n for n in ast.walk(ast.parse(SOURCE))
                      if isinstance(n, ast.FunctionDef) and n.name == "_distributed_mean")
        method.args.args[1].annotation = None
        method.returns = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {}
        exec(compile(module, "trainer_metric", "exec"), namespace)
        def forbidden(_):
            raise AssertionError("prompt-remainder trimming must never run")
        accelerator = SimpleNamespace(gather=lambda x: x, gather_for_metrics=forbidden)
        trainer = SimpleNamespace(accelerator=accelerator)
        # 8 ranks x G8 completions; prompt remainder=4 would destroy grouping.
        values = Values(list(range(64)))
        self.assertEqual(namespace["_distributed_mean"](trainer, values), 31.5)

    def test_wrapper_forwards_resume_without_replacing_base_model(self):
        script = (ROOT / "scripts/run_full_b200.sh").read_text()
        self.assertIn('command+=(--resume_from_checkpoint "${B200_RESUME_FROM_CHECKPOINT}")', script)
        self.assertIn('--model_name_or_path "${model_path}"', script)
        self.assertIn('"checkpoint resume contract"', script)


if __name__ == "__main__":
    unittest.main()
