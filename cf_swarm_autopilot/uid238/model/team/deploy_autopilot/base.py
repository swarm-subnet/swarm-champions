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
