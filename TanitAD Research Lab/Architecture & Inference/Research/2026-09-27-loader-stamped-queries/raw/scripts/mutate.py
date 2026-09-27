"""Apply / revert one named mutation in the MUTATION tree (C:/lgt/i3mut) only.

M1  the agent-head rule is gone: refc_v3_train.agent_queries_as_trained is renamed away, so
    refcv3_arm.rebuild_config's getattr(tr, "agent_queries_as_trained", None) reads None (the pre-I3 tree).
M2  the box-head rule is gone: refcv3_arm.rebuild_perception_branch no longer reads the stamped n_queries,
    so the branch rebuilds at PerceptionBranchConfig's default (the refcv6_loader behaviour).
"""
import hashlib
import sys
from pathlib import Path

TREE = Path("C:/lgt/i3mut")
assert TREE.resolve() != Path("C:/lgt/i3tree").resolve()
MUT = {
    "M1": ("stack/scripts/refc_v3_train.py",
           "def agent_queries_as_trained(config: dict, args=None) -> tuple:",
           "def _agent_queries_as_trained_MUTATED_AWAY(config: dict, args=None) -> tuple:"),
    "M2": ("taniteval/tools/refcv3_arm.py",
           '    for k in ("d_bev", "n_queries", "d_model", "stride", "bev_source", "presence_loss",',
           '    for k in ("d_bev", "d_model", "stride", "bev_source", "presence_loss",'),
}
name, action = sys.argv[1], sys.argv[2]
rel, old, new = MUT[name]
p = TREE / rel
b = p.read_bytes()
s = b.decode("utf-8")
nl = "\r\n" if b.count(b"\r\n") else "\n"
if action == "apply":
    assert s.count(old) == 1, (name, s.count(old))
    s = s.replace(old, new)
elif action == "revert":
    assert s.count(new) == 1, (name, s.count(new))
    s = s.replace(new, old)
else:
    raise SystemExit(action)
p.write_bytes(s.encode("utf-8"))
h = hashlib.sha1(b"blob %d\0" % len(s.encode()) + s.encode()).hexdigest()
print(f"{name} {action}: {rel} git-blob {h} (line endings {'CRLF' if nl == chr(13)+chr(10) else 'LF'})")
