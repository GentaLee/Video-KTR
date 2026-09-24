import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BaselineEntryTests(unittest.TestCase):
    def test_release_root_manifest_and_no_ktr_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "releases/commit"
            repo = release / "repo/scripts"
            repo.mkdir(parents=True)
            (release / "environment-manifest.json").write_text("{}")
            (repo / "run_baseline_b200.sh").write_bytes((ROOT / "scripts/run_baseline_b200.sh").read_bytes())
            (repo / "launch_b200.sh").write_text(
                'printf "%s\\n" "$1" "$B200_ROOT" "$B200_ENV_MANIFEST" '
                '"resume=<$B200_RESUME_FROM_CHECKPOINT>" "$B200_SMOKE_RUN_ROOT"\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith("B200_")}
            env["B200_RESUME_FROM_CHECKPOINT"] = "/do/not/resume/ktr"
            result = subprocess.run(["bash", str(repo / "run_baseline_b200.sh")],
                                    env=env, check=True, capture_output=True, text=True)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[1:5], ["baseline", str(root),
                             str(release / "environment-manifest.json"), "resume=<>"])
            self.assertEqual(lines[5], str(root / "artifacts/grpo-smoke-b200/tail-fix-86317d0"))


if __name__ == "__main__":
    unittest.main()
