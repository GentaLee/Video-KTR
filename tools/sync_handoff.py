#!/usr/bin/env python3
"""Import only the other experiment's committed status, without changing code."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROLES = {"h200", "b200"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def synchronize(root, ref, check=False):
    root = Path(root).resolve()
    role = (root / ".experiment-role").read_text().strip()
    if role not in ROLES:
        raise ValueError("invalid local experiment role")
    branch = git(root, "symbolic-ref", "--short", "HEAD").decode().strip()
    if not branch.endswith("/video-ktr-" + role):
        raise ValueError("development branch does not match local experiment role")
    commit = git(root, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    peer = git(root, "show", commit + ":.experiment-role").decode().strip()
    if peer not in ROLES or peer == role:
        raise ValueError("source must be the other experiment profile")
    content = git(root, "show", commit + ":handoff/STATUS.md")
    if not content or len(content) > 1024 * 1024:
        raise ValueError("peer status must be nonempty and at most 1 MiB")
    content.decode("utf-8")
    commit_time = git(root, "show", "-s", "--format=%cI", commit).decode().strip()
    digest = hashlib.sha256(content).hexdigest()
    output = (
        f"# Imported {peer.upper()} handoff\n\n"
        "> Generated snapshot; edit the source branch, not this file.\n\n"
        f"- SOURCE_COMMIT: `{commit}`\n"
        f"- SOURCE_COMMIT_TIME: `{commit_time}`\n"
        f"- SOURCE_PATH: `handoff/STATUS.md`\n"
        f"- SOURCE_SHA256: `{digest}`\n\n---\n\n"
    ).encode() + content
    relative = "handoff/peers/" + peer + ".md"
    target = root / relative
    for path in (root / "handoff", target.parent, target):
        if path.is_symlink():
            raise ValueError("refuse symlinked snapshot destination")
    if target.exists() and target.read_bytes() == output:
        print(f"unchanged {relative}; source={commit}")
        return 0
    if check:
        print(f"out of date {relative}; source={commit}")
        return 1
    if git(root, "status", "--porcelain", "--", relative).strip():
        raise ValueError("snapshot has uncommitted changes; review/commit before updating")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".peer-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(output)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"imported {relative}; source={commit}; sha256={digest}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("peer_ref", help="fetched peer branch or exact commit")
    parser.add_argument("--check", action="store_true", help="read-only freshness check")
    args = parser.parse_args()
    try:
        return synchronize(Path(__file__).resolve().parents[1], args.peer_ref, args.check)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"handoff sync refused: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
