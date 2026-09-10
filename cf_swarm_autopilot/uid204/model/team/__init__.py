"""Our (student-team) mining workspace for Bittensor SN124 `cf_swarm_sar`.

Nothing here is shipped to validators — the submission is ONNX + manifest only.
This package holds the privileged scripted *teacher*, the episode *harness*, and
the *observability* tooling that lets us watch/score/trace/audit a teacher run.

Upstream (`swarm/`, `neurons/`, `scripts/`, `docs/`, `RL/`) is reference and must
stay untouched (see CLAUDE.md). All our code lives under `team/`.
"""
