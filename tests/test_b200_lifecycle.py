"""Exercise the B200 launcher with local controllers and no GPU access."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_full_b200.sh"


def shell_function(source: str, name: str) -> str:
    header = f"{name}() {{"
    return header + source.split(header, 1)[1].split("\n}\n", 1)[0] + "\n}\n"


class B200LifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def functions(self, *names: str) -> str:
        return "\n".join(shell_function(self.source, name) for name in names)

    def run_shell(self, shell: str, *, timeout: float = 10) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + shell],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def test_preflight_finishes_when_terminal_output_is_not_consumed(self) -> None:
        """Fill the terminal pipe, but let the real producer and wrapper exit."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "producer-complete"
            producer_group = root / "producer-group"
            producer = (
                "import os, pathlib, stat, sys; "
                f"pathlib.Path({str(producer_group)!r}).write_text(str(os.getpgrp())); "
                "assert stat.S_ISREG(os.fstat(1).st_mode), 'producer stdout must be a regular file'; "
                "sys.stdout.write('x' * (1024 * 1024)); sys.stdout.flush(); "
                f"pathlib.Path({str(marker)!r}).write_text('complete')"
            )
            follower = self.source.split('touch "${run_root}/terminal.log"\n', 1)[1].split(
                "\nrequire_path() {", 1
            )[0]
            shell = "\n".join(
                [
                    "set -Eeuo pipefail",
                    f"run_root={shlex.quote(directory)}",
                    'preflight_pid=""; preflight_label=""; log_follower_pid=""',
                    self.functions("log", "process_is_live", "process_group_is_live", "stop_child", "stop_owned_process_group", "run_logged_preflight"),
                    "trap 'stop_child \"${log_follower_pid}\"' EXIT",
                    'touch "${run_root}/terminal.log"',
                    follower,
                    "run_logged_preflight backpressure "
                    + shlex.join([sys.executable, "-c", producer]),
                    # Give the observer time to fill stdout, without reading it.
                    "sleep 0.7",
                    '[[ -z "${preflight_pid}" ]]',
                ]
            )
            process = subprocess.Popen(
                ["bash", "-c", shell], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                process.wait(timeout=8)  # Deliberately do not drain stdout here.
                stdout, stderr = process.communicate(timeout=2)
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertEqual(marker.read_text(), "complete")
                self.assertGreater((root / "terminal.log").stat().st_size, 1024 * 1024)
                self.assertIn("backpressure: passed", (root / "terminal.log").read_text())
                self.assertGreater(len(stdout), 4096, "observer did not exercise a full terminal pipe")
                self.assertLess(len(stdout), 1024 * 1024, "test unexpectedly consumed producer output")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=2)
                if producer_group.exists():
                    try:
                        os.killpg(int(producer_group.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_failed_preflight_propagates_status_and_clears_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shell = "\n".join(
                [
                    f"run_root={shlex.quote(directory)}",
                    'preflight_pid=""; preflight_label=""',
                    self.functions("log", "process_is_live", "process_group_is_live", "stop_owned_process_group", "run_logged_preflight"),
                    "status=0",
                    "run_logged_preflight expected-failure bash -c 'exit 17' || status=$?",
                    '[[ "$status" == 17 && -z "$preflight_pid" && -z "$preflight_label" ]]',
                ]
            )
            result = self.run_shell(shell)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("failed with exit=17", (Path(directory) / "terminal.log").read_text())

    def controller_fixture(self, root: Path, initial_count: int) -> str:
        """Fake only the external controller and its process discovery."""
        (root / "count").write_text(str(initial_count))
        launcher = root / "controller.sh"
        launcher.write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            'printf "controller:%s\\n" "$1" >> "$EVENT_LOG"\n'
            'if [[ "$1" == start ]]; then printf 1 > "$COUNT_FILE"; fi\n'
        )
        return "\n".join(
            [
                f"run_root={shlex.quote(str(root))}",
                f"keepalive_launcher={shlex.quote(str(launcher))}",
                f"export EVENT_LOG={shlex.quote(str(root / 'events'))}",
                f"export COUNT_FILE={shlex.quote(str(root / 'count'))}",
                'keepalive_paused=1; keepalive_stop_requested=1; keepalive_pids=()',
                'find_keepalive_pids() { local count; count="$(<"$COUNT_FILE")"; keepalive_pids=(); '
                'for ((i=0; i<count; i++)); do keepalive_pids+=("$((1000+i))"); done; }',
                self.functions("log", "wait_for_keepalive_count", "start_keepalive"),
            ]
        )

    def test_keepalive_start_is_idempotent_and_rejects_duplicates(self) -> None:
        for initial_count in (0, 1, 2):
            with self.subTest(initial_count=initial_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_shell(
                    self.controller_fixture(root, initial_count)
                    + '\nstart_keepalive\nstart_keepalive\n[[ "$keepalive_paused" == 0 ]]'
                )
                events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
                self.assertEqual(result.returncode, 1 if initial_count == 2 else 0, result.stderr)
                self.assertEqual(events, ["controller:start"] if initial_count == 0 else [])

    def test_cpu_preflight_starts_missing_keepalive_and_captures_identity(self) -> None:
        phase = self.source.split(
            'log "phase 4/8: ensuring the external keep-alive is active during CPU-only data preparation"', 1
        )[1].split('log "phase 5/8:', 1)[0]
        for initial_count in (0, 1, 2):
            with self.subTest(initial_count=initial_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_shell(
                    self.controller_fixture(root, initial_count)
                    + '\ncapture_keepalive_identity() { printf "identity\\n" >> "$EVENT_LOG"; }\n'
                    + phase
                )
                events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
                self.assertEqual(result.returncode, 2 if initial_count == 2 else 0, result.stderr)
                self.assertEqual(events, {0: ["controller:start", "identity"], 1: ["identity"], 2: []}[initial_count])

    def cleanup_fixture(self, root: Path, *, stop_ok: bool = True, gpu_empty: bool = True) -> str:
        return "\n".join(
            [
                self.controller_fixture(root, 0),
                'preflight_pid=111; preflight_label=decoder; training_pid=222',
                'heartbeat_pid=""; monitor_pid=""; log_follower_pid=""',
                'training_started=""; training_exit=""; summary_written=0',
                'stop_child() { :; }',
                'sleep() { :; }',
                'stop_owned_process_group() { printf "stop:%s\\n" "$1" >> "$EVENT_LOG"; '
                + f"return {0 if stop_ok else 1}; " + "}",
                'query_gpu_compute_pids() { printf "gpu-query\\n" >> "$EVENT_LOG"; '
                + ('gpu_compute_pid_snapshot=();' if gpu_empty else 'gpu_compute_pid_snapshot=(4242);') + " }",
                self.functions("wait_for_gpu_quiescence", "cleanup"),
                "trap cleanup EXIT",
            ]
        )

    def test_success_and_failure_restore_only_after_owned_work_stops_and_gpu_is_empty(self) -> None:
        for status in (0, 17):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_shell(self.cleanup_fixture(root) + f"\nexit {status}")
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(
                    (root / "events").read_text().splitlines(),
                    ["stop:CPU preflight (decoder)", "stop:full torchrun", "gpu-query", "controller:start"],
                )

    def test_cleanup_keeps_protection_paused_if_owned_work_or_gpu_is_busy(self) -> None:
        for stop_ok, gpu_empty in ((False, True), (True, False)):
            with self.subTest(stop_ok=stop_ok, gpu_empty=gpu_empty), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_shell(self.cleanup_fixture(root, stop_ok=stop_ok, gpu_empty=gpu_empty) + "\nexit 17")
                self.assertEqual(result.returncode, 5, result.stderr)
                events = (root / "events").read_text().splitlines()
                self.assertNotIn("controller:start", events)
                if not stop_ok:
                    self.assertNotIn("gpu-query", events)
                self.assertEqual((root / "count").read_text(), "0")

    def test_concurrent_launch_is_rejected_and_lock_releases_on_exit(self) -> None:
        lock_section = self.source.split('exec 9>"${b200_root}/.b200-training.lock"', 1)[1].split("\nlog() {", 1)[0]
        lock_section = 'exec 9>"${b200_root}/.b200-training.lock"' + lock_section
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = f"b200_root={shlex.quote(directory)}\n"
            owner_shell = prefix + f"run_root={shlex.quote(str(root / 'owner'))}\n" + lock_section
            owner_shell += '\nprintf acquired > "${b200_root}/acquired"\nread -r release\n'
            owner = subprocess.Popen(
                ["bash", "-c", "set -Eeuo pipefail\n" + owner_shell],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not (root / "acquired").exists() and owner.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((root / "acquired").exists(), "first launcher failed to acquire its lock")
                contender_shell = prefix + f"run_root={shlex.quote(str(root / 'contender'))}\n" + lock_section
                blocked = self.run_shell(contender_shell)
                self.assertEqual(blocked.returncode, 2, blocked.stderr)
                self.assertIn("run variants serially", blocked.stderr)
                self.assertFalse((root / "contender").exists())
                owner.communicate("release\n", timeout=3)
                self.assertEqual(owner.returncode, 0)
                allowed = self.run_shell(contender_shell)
                self.assertEqual(allowed.returncode, 0, allowed.stderr)
                self.assertTrue((root / "contender").is_dir())
                repeated = self.run_shell(contender_shell)
                self.assertEqual(repeated.returncode, 2, repeated.stderr)
                self.assertIn("RUN_ROOT must name a new directory", repeated.stderr)
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
