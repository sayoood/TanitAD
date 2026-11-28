"""Mutation check for test_pose_sync.py: re-introduce each plausible defect into a COPY of the overlaid test tree and
require the suite to go RED.  A guard that stays green under its own defect guards nothing.

    PYTHONPATH is set per mutation to the mutated copy.  Usage:  python mutation_check.py <overlaid tree> <scratch dir>
"""
import json
import os
import shutil
import subprocess
import sys

TREE, SCR = sys.argv[1], sys.argv[2]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

MUTATIONS = [
    ("M1 sign flipped inside shift_for (the correction goes the wrong way)",
     "stack/tanitad/data/pose_sync.py",
     "return self.sign * float(r.delta_s[int(raw_row)]) / r.dt_s",
     "return -self.sign * float(r.delta_s[int(raw_row)]) / r.dt_s"),
    ("M2 per-row shift instead of the window shift (the literal reading)",
     "stack/tanitad/data/pose_sync.py",
     "P = shift_rows(np.asarray(poses), s, angle_cols=(2,)).astype(np.float32)",
     "P = shift_rows_per_row(np.asarray(poses), np.asarray(ps.table[int(sid)].delta_s[int(raw_offset):][:np.asarray(poses).shape[0]]) / ps.table[int(sid)].dt_s, angle_cols=(2,)).astype(np.float32)"),
    ("M3 integer rows no longer returned bit-exactly (zero-offset identity lost)",
     "stack/tanitad/data/pose_sync.py",
     "    out[z] = x0[z]\n    out[o] = x1[o]\n",
     "    out[:, 3] = out[:, 3] + 1e-7\n"),
    ("M4 trainer hook drops the future poses (targets stay unshifted)",
     "stack/scripts/refc_v3_train.py",
     "        item[\"future_poses_ext\"] = pv[idx.clamp(max=T - 1)]\n",
     "        pass\n"),
    ("M5 trainer hook shifts a DIFFERENT row (NOW-1) than the one the image belongs to",
     "stack/scripts/refc_v3_train.py",
     "        got = _ps.window_shifted_tracks(self.pose_sync, sid, now, self._raw_offset(ep), ep.poses, ep.actions)",
     "        got = _ps.window_shifted_tracks(self.pose_sync, sid, now + 25, self._raw_offset(ep), ep.poses, ep.actions)"),
    ("M6 yaw interpolated linearly across the +-pi seam (no shortest arc)",
     "stack/tanitad/data/pose_sync.py",
     "out[:, c] = _wrap(x0[:, c] + f * _wrap(x1[:, c] - x0[:, c]))",
     "out[:, c] = _wrap(x0[:, c] + f * (x1[:, c] - x0[:, c]))"),
    ("M7 negative deltas accepted (the sign/unit guard removed)",
     "stack/tanitad/data/pose_sync.py",
     "np.all(np.isfinite(d)) and d.min() >= 0.0 and d.max() <= DELTA_US_MAX",
     "np.all(np.isfinite(d))"),
]

res = []
for name, rel, old, new in MUTATIONS:
    dst = os.path.join(SCR, "mut")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    ign = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(os.path.join(TREE, "stack"), os.path.join(dst, "stack"), ignore=ign)
    shutil.copytree(os.path.join(TREE, "taniteval", "taniteval"), os.path.join(dst, "taniteval", "taniteval"), ignore=ign)
    p = os.path.join(dst, rel)
    raw = open(p, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    o, n = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
    if raw.count(o) != 1:
        res.append({"mutation": name, "result": f"MUTATION NOT APPLIED (count {raw.count(o)})"})
        continue
    open(p, "wb").write(raw.replace(o, n).encode("utf-8"))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{dst}/stack;{dst}/stack/scripts;{dst}/taniteval"
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY, "-m", "pytest", "tests/test_pose_sync.py", "-q", "-x", "-p", "no:cacheprovider"],
                       cwd=f"{dst}/stack", env=env, capture_output=True, text=True)
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:] or [""]
    res.append({"mutation": name, "returncode": r.returncode, "result": "RED (caught)" if r.returncode != 0 else "GREEN (NOT caught)",
                "tail": tail[0][:160]})
    print(json.dumps(res[-1]), flush=True)
json.dump(res, open(os.path.join(SCR, "mutation_check.json"), "w"), indent=1)
print("caught", sum(1 for r in res if r["result"].startswith("RED")), "of", len(res))
