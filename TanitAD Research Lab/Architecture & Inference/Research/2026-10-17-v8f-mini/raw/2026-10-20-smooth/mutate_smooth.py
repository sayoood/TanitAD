"""Delete one guard at a time from v8f_smooth_read.py; each mutation must turn test_v8f_smooth_read.py RED. Source restored in `finally`."""
import json, os, subprocess, sys
T, P = sys.argv[1], "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_smooth_read.py"
TEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_smooth_read.py"]
src = open(os.path.join(T, P), encoding="utf-8").read()
MUT = {
    "D ignores the sign": ('ok=bool(v["delta"] < 0 and v["separated"])', 'ok=bool(v["separated"])'),
    "P bar ignored": ('blk["P_ok"] = bool(r["hi"] <= P_BAR_M)', 'blk["P_ok"] = True'),
    "control C ignored": ('    if not c_ok:\n        return "INVALID', '    if False:\n        return "INVALID'),
    "one seed decides": ('    if len(per_seed) < 2:\n        return "INTERIM', '    if False:\n        return "INTERIM'),
    "ade6 check dropped": ('        if dev > R.TOL_ADE:\n            R.refuse(f"eval_{t}', '        if False:\n            R.refuse(f"eval_{t}'),
    "order 1 in place of 2": ('WINDOW, ORDER, DOSE_WINDOW = 7, 2, 11', 'WINDOW, ORDER, DOSE_WINDOW = 7, 1, 11'),
}
def run():
    r = subprocess.run([sys.executable] + TEST, cwd=T, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    lines = (r.stdout or "").strip().splitlines()
    return r.returncode, (lines[-1] if lines else "")
res = {}
try:
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, name
        with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
            f.write(src.replace(old, new))
        rc, tail = run()
        res[name] = dict(rc=rc, RED=rc != 0, summary=tail)
finally:
    with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
rc, tail = run()
res["_restored"] = dict(rc=rc, summary=tail)
print(json.dumps(res, indent=1))
assert all(v["RED"] for k, v in res.items() if not k.startswith("_")) and res["_restored"]["rc"] == 0
