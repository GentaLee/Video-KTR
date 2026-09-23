import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/sync_handoff.py"
spec = importlib.util.spec_from_file_location("sync_handoff", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HandoffSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.g("init", "-q")
        self.g("config", "user.email", "test@example.invalid")
        self.g("config", "user.name", "Test")
        self.g("switch", "-c", "test/video-ktr-h200")
        (self.root / "handoff").mkdir()
        (self.root / ".experiment-role").write_text("h200\n")
        (self.root / "handoff/STATUS.md").write_text("H200 status\n")
        (self.root / "training.py").write_text("unchanged\n")
        self.g("add", ".")
        self.g("commit", "-qm", "h200")
        self.own = self.g("rev-parse", "HEAD").strip()
        self.g("switch", "-c", "test/video-ktr-b200")
        (self.root / ".experiment-role").write_text("b200\n")
        (self.root / "handoff/STATUS.md").write_text("B200 status\n")
        self.g("commit", "-qam", "b200")
        self.peer = self.g("rev-parse", "HEAD").strip()
        self.g("switch", "test/video-ktr-h200")

    def g(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], stderr=subprocess.PIPE
        ).decode()

    def test_import_is_idempotent_and_does_not_change_head_or_sources(self):
        self.assertEqual(module.synchronize(self.root, self.peer), 0)
        target = self.root / "handoff/peers/b200.md"
        self.assertIn("SOURCE_SHA256", target.read_text())
        self.assertIn("B200 status", target.read_text())
        stamp = target.stat().st_mtime_ns
        self.assertEqual(module.synchronize(self.root, self.peer), 0)
        self.assertEqual(target.stat().st_mtime_ns, stamp)
        self.assertEqual(self.g("rev-parse", "HEAD").strip(), self.own)
        self.assertEqual((self.root / "training.py").read_text(), "unchanged\n")
        self.assertEqual((self.root / "handoff/STATUS.md").read_text(), "H200 status\n")

    def test_check_is_read_only(self):
        self.assertEqual(module.synchronize(self.root, self.peer, check=True), 1)
        self.assertFalse((self.root / "handoff/peers").exists())

    def test_refuses_own_role_and_wrong_branch(self):
        with self.assertRaises(ValueError):
            module.synchronize(self.root, self.own)
        self.g("switch", "-c", "wrong")
        with self.assertRaises(ValueError):
            module.synchronize(self.root, self.peer)

    def test_refuses_dirty_snapshot(self):
        target = self.root / "handoff/peers/b200.md"
        target.parent.mkdir()
        target.write_text("uncommitted human note\n")
        with self.assertRaises(ValueError):
            module.synchronize(self.root, self.peer)
        self.assertEqual(target.read_text(), "uncommitted human note\n")

    def test_refuses_symlink_destination(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.root / "handoff/peers").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(ValueError):
            module.synchronize(self.root, self.peer)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_committed_snapshot_can_update_from_new_peer_commit(self):
        module.synchronize(self.root, self.peer)
        self.g("add", "handoff/peers/b200.md")
        self.g("commit", "-qm", "import")
        self.g("switch", "test/video-ktr-b200")
        (self.root / "handoff/STATUS.md").write_text("B200 new status\n")
        self.g("commit", "-qam", "updated")
        new_peer = self.g("rev-parse", "HEAD").strip()
        self.g("switch", "test/video-ktr-h200")
        self.assertEqual(module.synchronize(self.root, new_peer), 0)
        self.assertIn("B200 new status", (self.root / "handoff/peers/b200.md").read_text())


if __name__ == "__main__":
    unittest.main()
