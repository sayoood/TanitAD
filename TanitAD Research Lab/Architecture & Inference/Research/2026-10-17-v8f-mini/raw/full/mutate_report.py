"""Delete one guard at a time from v8f_mini_full_report.py; each mutation must turn test_v8f_mini_full_report.py RED. Source restored in `finally`."""
import json, os, subprocess, sys
T, P = sys.argv[1], "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_mini_full_report.py"
TEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_mini_full_report.py"]
src = open(os.path.join(T, P), encoding="utf-8").read()
MUT = {
    "eid check dropped": ("            if not np.array_equal(z[\"eid\"], eid):\n                refuse", "            if False:\n                refuse"),
    "ade-vs-gt check dropped": ("            if dev > TOL_ADE:\n                refuse(f\"eval_", "            if False:\n                refuse(f\"eval_"),
    "floors P10 check dropped": ("        if dev > TOL_ADE:\n            refuse(f\"the P10", "        if False:\n            refuse(f\"the P10"),
    "WM-1 re-derivation dropped": ("        if abs(d - c[\"mean\"]) > TOL_VERDICT:\n            refuse(f\"verdict WM-1",
                                   "        if False:\n            refuse(f\"verdict WM-1"),
    "K-1 re-derivation dropped": ("            if abs(d - c[\"mean\"]) > TOL_VERDICT:\n                refuse(f\"verdict K-1",
                                  "            if False:\n                refuse(f\"verdict K-1"),
    "missing arm tolerated": ("                if arm in ARMS:\n                    refuse", "                if False:\n                    refuse"),
    "pairs M2b with itself": ("E.paired_family_block(pw[f\"M2b_s{s}\"], pw[f\"M1_s{s}\"]", "E.paired_family_block(pw[f\"M2b_s{s}\"], pw[f\"M2b_s{s}\"]"),
    "bar words re-decided": ("f\"**{'PASS' if r['K1_pass'] else 'FAIL'}** |\")", "f\"**{'PASS' if c[0]['separated'] else 'FAIL'}** |\")"),
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
