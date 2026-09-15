"""Which champion are we developing on? One import site for every probe.

SWARM_BASE selects it: "uid167" (default, the current champion) or "uid99".
The choice is read at CALL time, not import time, so a spawn worker that
inherits the environment picks up the same base the parent chose.

Importing a base module directly still works and is the right thing to do when
a script is deliberately pinned to one -- but a panel must go through here, or
it can quietly control against a champion that has been dethroned.
"""
import importlib
import os

BASES = {"uid99": "team.deploy_autopilot.uid99_base",
         "uid167": "team.deploy_autopilot.uid167_base"}


def which() -> str:
    name = os.environ.get("SWARM_BASE", "uid167").strip().lower()
    if name not in BASES:
        raise ValueError("SWARM_BASE=%r; expected one of %s"
                         % (name, sorted(BASES)))
    return name


def module():
    return importlib.import_module(BASES[which()])


def build_agent(*args, **kw):
    return module().build_agent(*args, **kw)


def assert_is_base(cfg) -> None:
    module().assert_is_base(cfg)
