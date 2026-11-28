"""E1b — build /workspace/e1b/e1a_horizon.py = /workspace/e1a_e2a/e1a_horizon.py
PLUS three purely ADDITIVE capture lines (pred2s / gt2s), so the lateral vs
longitudinal decomposition (taniteval/lateral.py) can be computed on the SAME
closed-loop rollout E1a used.

The E1a source is NOT mutated. The patch is applied by exact string replacement
and the unified diff is printed, so additivity is provable, not asserted: no
existing expression is edited, only new `.append(...)` / `out[...] =` statements
are inserted. Every metric E1a computes is bit-identical.
"""
import difflib
import hashlib
from pathlib import Path

SRC = Path("/workspace/e1a_e2a/e1a_horizon.py")
DST = Path("/workspace/e1b/e1a_horizon.py")

s = SRC.read_text()
orig = s

A_OLD = '''    rows = {k: [] for k in ("ade2s", "peak_lat", "mean_lat", "peak_yaw",
                            "hd2s", "hdK", "speed", "eid", "t0", "epi")}
'''
A_NEW = A_OLD + '''    _cap_pred, _cap_gt = [], []          # E1b ADDITIVE capture (no metric changed)
'''

B_OLD = '''            rows["ade2s"].append(
                torch.linalg.norm(ego_ego[:, WP_IDX] - gt2, dim=-1).mean(1))
'''
B_NEW = B_OLD + '''            _cap_pred.append(ego_ego[:, WP_IDX].clone())   # E1b ADDITIVE
            _cap_gt.append(gt2.clone())                    # E1b ADDITIVE
'''

C_OLD = '''    out["fixed_steps"] = fixed_steps
    return out
'''
C_NEW = '''    out["fixed_steps"] = fixed_steps
    out["pred2s"] = torch.cat(_cap_pred)   # E1b ADDITIVE [N,4,2] ego frame
    out["gt2s"] = torch.cat(_cap_gt)       # E1b ADDITIVE [N,4,2] ego frame
    return out
'''

for old, new in ((A_OLD, A_NEW), (B_OLD, B_NEW), (C_OLD, C_NEW)):
    assert s.count(old) == 1, f"anchor not unique ({s.count(old)}x):\n{old}"
    s = s.replace(old, new)

diff = list(difflib.unified_diff(orig.splitlines(True), s.splitlines(True),
                                 "e1a_e2a/e1a_horizon.py", "e1b/e1a_horizon.py"))
removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
assert not removed, f"PATCH IS NOT ADDITIVE — removed lines:\n{''.join(removed)}"

DST.write_text(s)
print("".join(diff))
print(f"[cap] src md5 {hashlib.md5(orig.encode()).hexdigest()}  "
      f"dst md5 {hashlib.md5(s.encode()).hexdigest()}")
print(f"[cap] additive-only verified: 0 removed lines, "
      f"+{sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} added")
