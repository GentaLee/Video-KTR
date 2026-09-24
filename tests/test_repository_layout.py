import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_current_document_links_exist(self):
        files = [ROOT / "README.md", ROOT / "describe.md", ROOT / "handoff/STATUS.md"]
        files += list((ROOT / "docs").glob("*.md"))
        files += list((ROOT / "reports").rglob("*.md"))
        for path in files:
            for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
                if re.match(r"[a-z]+:|#|/", target):
                    continue
                local = target.split("#")[0]
                self.assertTrue((path.parent / local).exists(), f"{path}: {target}")

    def test_shell_syntax(self):
        for path in [ROOT / "run.sh", *sorted((ROOT / "scripts").glob("*.sh"))]:
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_router_requires_explicit_command(self):
        for args in ([], ["unknown"]):
            result = subprocess.run(["bash", str(ROOT / "run.sh"), *args], capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_router_resolves_release_and_dispatches_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "releases/commit"
            repo = release / "repo"
            scripts = repo / "scripts"
            scripts.mkdir(parents=True)
            (repo / "run.sh").write_bytes((ROOT / "run.sh").read_bytes())
            (release / "environment-manifest.json").write_text("{}")
            for name in ("run_baseline_b200.sh", "launch_b200.sh", "somke-b200.sh"):
                (scripts / name).write_text('printf "%s\\n" "$B200_ROOT" "$B200_ENV_MANIFEST" "${1:-}"\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith("B200_")}
            for variant in ("baseline", "ktr", "smoke"):
                result = subprocess.run(["bash", str(repo / "run.sh"), variant],
                                        env=env, capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                self.assertEqual(lines[:2], [str(root), str(release / "environment-manifest.json")])
                self.assertEqual(lines[2], "ktr" if variant == "ktr" else "")


if __name__ == "__main__":
    unittest.main()
