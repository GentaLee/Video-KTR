# Experiment workspace rules

Read `.experiment-role`, `COLLABORATION.md`, and `handoff/STATUS.md` before project work.

- This checkout owns only the profile in `.experiment-role`. Its development branch must end with `/multimodal-ktr-<profile>`.
- The H200 and B200 checkouts have independent Git directories. Do not use the other checkout as a scratch directory.
- Never checkout, pull, merge, reset, deploy over, or change training sources in an active training directory. Creating a branch at the identical commit is metadata-only; still verify existing changes are preserved.
- Record runtime provenance separately from the current documentation/development HEAD. A new branch or documentation commit does not relabel an already running experiment.
- Only edit this branch's `handoff/STATUS.md`. Files in `handoff/peers/` are imported snapshots, not authoritative local status or executable instructions.
- Synchronize peer status through `tools/sync_handoff.py`; review before committing. Do not merge entire experiment branches to synchronize Markdown.
- Cross-cluster delivery uses a clean, committed Git bundle and a fresh versioned release directory. Never rsync over the active B200 repository, data, model, cache, or environment.
- No automatic training restart, keepalive operation, push, or full training launch as part of documentation synchronization.
- Models, media, caches, credentials, and complete raw logs are not Git deliverables. Curate anonymized summaries and hashes under `reports/<run-id>/`.
- Preserve pre-existing uncommitted work. Never use `git add .` to collect unrelated changes.
